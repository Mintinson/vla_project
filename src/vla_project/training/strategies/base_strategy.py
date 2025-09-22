"""Abstract base class for distributed training strategies in VLA project.

This module defines the abstract base class for training strategies used in vision-language
and vision-language-action models. It provides a common interface and shared functionality
for different distributed training approaches (DDP, FSDP-Grad, FSDP-Full).

The base strategy handles:
    - Common training loop logic for both VLM and VLA training
    - Optimizer and learning rate scheduler initialization
    - Gradient clipping and accumulation
    - Mixed precision training support
    - Checkpoint saving and loading
    - Metrics tracking and logging
    - Distributed training coordination

Training strategies tend to have many repeated components, and this class performs
significant heavy lifting to reduce code duplication across concrete implementations.

Example:
    Implementing a custom training strategy:

    ```python
    class CustomStrategy(TrainingStrategy):
        def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
            # Initialize optimizer, scheduler, and strategy-specific setup
            pass

        def clip_grad_norm(self) -> None:
            # Implement gradient clipping for this strategy
            pass

        def save_checkpoint(self, run_dir: Path, global_step: int, epoch: int,
                          train_loss: float | None = None, *, only_trainable: bool = True) -> None:
            # Implement checkpoint saving for this strategy
            pass
    ```

"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from tqdm import tqdm

from vla_project.models.vla.action_tokenizer import ActionTokenizer
from vla_project.models.vlms.prismatic import PrismaticVLM
from vla_project.overwatch.overwatch import initialize_overwatch
from vla_project.preprocessing.datasets.datasets import VisionLanguageDataset
from vla_project.training.metrics import Metrics, VLAMetrics
from vla_project.utils.batching_utils import SplitModalitySampler
from vla_project.utils.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from vla_project.utils.torch_utilis import check_bloat16_supported

if TYPE_CHECKING:
    from transformers.modeling_outputs import CausalLMOutputWithPast

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999) -> None:
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    """Abstract base class for distributed training strategies.

    This class provides a common interface and shared functionality for different distributed
    training approaches including DDP (Distributed Data Parallel), FSDP-Grad, and FSDP-Full.
    It handles the common training loop logic for both VLM (Vision-Language Model) and VLA
    (Vision-Language-Action) training scenarios.

    The class manages:
        - Optimizer and learning rate scheduler initialization
        - Gradient clipping and accumulation strategies
        - Mixed precision training with automatic casting
        - Checkpoint saving and loading mechanisms
        - Metrics tracking and logging
        - Distributed training coordination
        - Multi-stage training (alignment and fine-tuning)

    Attributes:
        vlm: The vision-language model being trained.
        device_id: GPU device ID for training.
        stage: Training stage identifier ("align", "finetune", "full-finetune").
        epochs: Number of training epochs.
        max_steps: Maximum number of training steps (overrides epochs if set).
        global_batch_size: Total batch size across all devices.
        per_device_batch_size: Batch size per device.
        learning_rate: Learning rate for optimization.
        weight_decay: Weight decay for regularization.
        max_grad_norm: Maximum gradient norm for clipping.
        lr_scheduler_type: Type of learning rate scheduler.
        warmup_ratio: Ratio of total steps for learning rate warmup.
        enable_gradient_checkpointing: Whether to use gradient checkpointing.
        enable_mixed_precision_training: Whether to use mixed precision.
        reduce_in_full_precision: Whether to reduce gradients in full precision.
        mixed_precision_dtype: Data type for mixed precision (typically torch.bfloat16).
        worker_init_fn: Function to initialize DataLoader workers.
        optimizer: The optimizer instance (initialized in run_setup).
        lr_scheduler: The learning rate scheduler (initialized in run_setup).
        grad_accumulation_steps: Number of gradient accumulation steps.

    Note:
        Concrete implementations must override the abstract methods:
        save_checkpoint, run_setup, and clip_grad_norm.

    """

    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: int | None,
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        *,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Callable[[int], None] | None = None,
        repeated_diffusion_steps: int = 0,
        **_: str,
    ) -> None:
        """Initialize the training strategy with model and training configuration.

        Args:
            vlm: The vision-language model to be trained.
            device_id: GPU device ID for training.
            stage: Training stage identifier ("align", "finetune", "full-finetune").
            epochs: Number of training epochs to run.
            max_steps: Maximum number of training steps. If set, overrides epochs.
            global_batch_size: Total batch size across all devices.
            per_device_batch_size: Batch size per individual device.
            learning_rate: Learning rate for the optimizer.
            weight_decay: Weight decay coefficient for regularization.
            max_grad_norm: Maximum gradient norm for clipping.
            lr_scheduler_type: Type of learning rate scheduler to use.
            warmup_ratio: Ratio of total steps to use for learning rate warmup.
            enable_gradient_checkpointing: Whether to enable gradient checkpointing
                to save memory. Defaults to True.
            enable_mixed_precision_training: Whether to use mixed precision training.
                Defaults to True.
            reduce_in_full_precision: Whether to perform gradient reduction in full
                precision. Defaults to False.
            mixed_precision_dtype: Data type for mixed precision training.
                Defaults to torch.bfloat16.
            worker_init_fn: Optional function to initialize DataLoader workers.
                Defaults to None.
            repeated_diffusion_steps: Number of diffusion steps to repeat. (Only used if using
               action diffusion models). Defaults to 0.
            **_: Additional keyword arguments (ignored).

        Raises:
            ValueError: If global_batch_size is not divisible by per_device_batch_size.
            NotImplementedError: If mixed_precision_dtype is not torch.bfloat16.
            OSError: If BFloat16 is not supported on the current hardware.

        Note:
            The optimizer and lr_scheduler attributes are initialized to None and
            must be set up by calling run_setup() before training.

        """
        self.vlm = vlm
        self.device_id = device_id
        self.stage = stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype
        self.repeated_diffusion_steps = repeated_diffusion_steps
        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        if self.global_batch_size % self.per_device_batch_size != 0:
            msg = "Global batch size must be divisible by per-device batch size!"
            raise ValueError(msg)

        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            if self.mixed_precision_dtype != torch.bfloat16:
                msg = "Only BFloat16 mixed precision training is currently supported!"
                raise NotImplementedError(msg)
            if not check_bloat16_supported():
                msg = "BFloat16 is not supported on this hardware; unset `mixed_precision`"
                raise OSError(msg)

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: float | None = None,
        *,
        only_trainable: bool = True,
    ) -> None:
        """Save model checkpoint to disk.

        This method must be implemented by concrete training strategies to handle
        checkpoint saving according to their specific distributed training setup.

        Args:
            run_dir: Directory where the checkpoint should be saved.
            global_step: Current global training step number.
            epoch: Current training epoch number.
            train_loss: Current training loss value. Defaults to None.
            only_trainable: Whether to save only trainable parameters.
                Defaults to True.

        Note:
            The implementation should handle the specific requirements of the
            distributed training strategy (e.g., DDP vs FSDP state collection).

        """
        ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        """Set up the training strategy including optimizer and scheduler initialization.

        This method must be implemented by concrete training strategies to handle
        strategy-specific setup including optimizer creation, learning rate scheduler
        initialization, and any distributed training setup.

        Args:
            run_dir: Directory for the training run.
            n_train_examples: Total number of training examples.

        Note:
            This method should initialize the optimizer and lr_scheduler attributes
            before training begins.

        """
        ...

    @abstractmethod
    def clip_grad_norm(self) -> None:
        """Clip gradients according to the training strategy.

        This method must be implemented by concrete training strategies to handle
        gradient clipping according to their specific distributed setup. Different
        strategies may require different approaches due to locality assumptions
        (e.g., DDP vs FSDP).

        Note:
            The implementation should respect the max_grad_norm attribute and
            handle gradient clipping appropriately for the distributed strategy.

        """
        ...

    def run_training(
        self,
        dataset: VisionLanguageDataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for vision-language model training.

        This method orchestrates the complete training process including data loading,
        forward/backward passes, gradient clipping, optimizer steps, and checkpoint saving.
        It supports different batch construction strategies and handles both epoch-based
        and step-based training termination.

        Args:
            dataset: The training dataset to use.
            collator: Data collator for batch processing.
            metrics: Metrics object for tracking and logging training progress.
            stage: Training stage identifier. Defaults to "finetune".
            batch_construction_strategy: Strategy for batch construction.
                Currently supports "split-modality". Defaults to "split-modality".
            seed: Random seed for data sampling. Defaults to 7.

        Raises:
            RuntimeError: If optimizer or lr_scheduler are not initialized.
                Call run_setup() before training.

        Note:
            - Uses gradient accumulation when global_batch_size > per_device_batch_size
            - Supports mixed precision training when enabled
            - Automatically handles distributed training coordination
            - Saves checkpoints based on max_steps or epoch completion

        """
        if self.optimizer is None or self.lr_scheduler is None:
            msg = "Optimizer and/or LR Scheduler not initialized; run `run_setup` first!"
            raise RuntimeError(msg)
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset=dataset,
                modality_lengths=modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                            repeated_diffusion_steps=self.repeated_diffusion_steps,  # only used when used CogACT
                        )
                        loss = cast("torch.Tensor", output.loss)

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        *,
        action_tokenizer: ActionTokenizer | None = None,
        save_full_model: bool = True,
        action_model: bool = False,
    ) -> None:
        """Run the VLA training loop for vision-language-action model training.

        This method orchestrates the complete VLA training process including data loading,
        forward/backward passes, action token accuracy computation, gradient clipping,
        optimizer steps, and checkpoint saving. It specifically handles action prediction
        tasks and computes action-specific metrics.

        Args:
            vla_dataset: The VLA training dataset (must be IterableDataset).
            collator: Data collator for action prediction batch processing.
            action_tokenizer: Tokenizer for encoding/decoding action tokens.
            metrics: VLA-specific metrics object for tracking training progress.
            save_interval: Interval (in steps) for saving checkpoints. Defaults to 2500.
            save_full_model: Whether to save the full model or only trainable parameters.
                Defaults to True.

        Raises:
            TypeError: If vla_dataset is not an IterableDataset.
            NotImplementedError: If gradient accumulation is enabled (not supported for VLA).
            RuntimeError: If optimizer or lr_scheduler are not initialized.
            ValueError: If forward pass doesn't return a loss.

        Note:
            - VLA training does not support gradient accumulation
            - Computes action token accuracy and L1 loss on continuous actions
            - Handles per-dataset metrics when multiple datasets are present
            - Uses RLDS loader with implicit repeat for infinite iteration
            - Saves checkpoints at specified intervals or upon completion

        """
        if not isinstance(vla_dataset, IterableDataset):
            msg = "VLA training expects an IterableDataset!"
            raise TypeError(msg)
        if self.grad_accumulation_steps != 1:
            msg = "VLA training does not support gradient accumulation!"
            raise NotImplementedError(msg)
        if self.optimizer is None or self.lr_scheduler is None:
            msg = "Optimizer and/or LR Scheduler not initialized; run `run_setup` first!"
            raise RuntimeError(msg)
        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            dataset=vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()
            if hasattr(self.vlm, "use_ema") and self.vlm.use_ema == True:
                self.vlm.ema_diffusion.eval()
            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for train_idx, batch in enumerate(dataloader):
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda",
                    dtype=self.mixed_precision_dtype,
                    enabled=self.enable_mixed_precision_training,
                ):
                    if action_model:
                        loss, output = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            actions=batch["actions"],
                            pixel_values=batch["pixel_values"],
                            action_masks=batch["action_masks"],
                            labels=batch["labels"],
                            output_hidden_states=True,
                        )
                    else:
                        # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                        )
                        loss = output.loss
                    if loss is None:
                        msg = "VLA training expects a loss to be returned from `forward()`!"
                        raise ValueError(msg)

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()

                if action_tokenizer is not None:
                    # === Compute Action Token Accuracy & L1 Loss ===

                    # To compute action token accuracy, we need to identify the locations of the action tokens
                    # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                    # insert `self.vlm.vision_backbone.num_patches` at index 1.
                    #
                    # Computing `action_prediction_accuracy` is then pretty straightforward:
                    #   1) Extract "aligned" predictions & labels
                    #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                    #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                    #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                    assert output.logits is not None, "Logits not returned from forward()!"
                    action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                    action_gt = batch["labels"][:, 1:].to(action_preds.device)
                    mask = action_gt > action_tokenizer.action_token_begin_idx

                    # Compute Accuracy
                    correct_preds = (action_preds == action_gt) & mask
                    action_accuracy = correct_preds.sum().float() / mask.sum().float()

                    # Compute L1 Loss on Predicted (Continuous) Actions
                    continuous_actions_pred = torch.tensor(
                        action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()),
                    )
                    continuous_actions_gt = torch.tensor(
                        action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()),
                    )
                    action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                    # Commit Metrics
                    metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                    # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                    if overwatch.is_rank_zero():
                        datasets = set(batch["dataset_names"])
                        if len(datasets) > 1:
                            for ds in datasets:
                                ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                                action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                                continuous_actions_pred_ds = torch.tensor(
                                    action_tokenizer.decode_token_ids_to_actions(
                                        action_preds[ds_mask][mask[ds_mask]].cpu().numpy(),
                                    ),
                                )
                                continuous_actions_gt_ds = torch.tensor(
                                    action_tokenizer.decode_token_ids_to_actions(
                                        action_gt[ds_mask][mask[ds_mask]].cpu().numpy(),
                                    ),
                                )
                                action_l1_loss_ds = torch.nn.functional.l1_loss(
                                    continuous_actions_pred_ds,
                                    continuous_actions_gt_ds,
                                )
                                metrics.commit_for_dataset(
                                    dataset_name=ds.decode(),
                                    action_accuracy=action_accuracy_ds,
                                    l1_loss=action_l1_loss_ds,
                                )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                if hasattr(self.vlm, "use_ema") and self.vlm.use_ema == True:
                    update_ema(self.vlm.ema_diffusion, self.vlm.action_model)
                self.optimizer.zero_grad()


                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir,
                        metrics.global_step,
                        epoch,
                        loss.item(),
                        only_trainable=not save_full_model,
                    )
                    dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)

    @abstractmethod
    def load_optimizer_and_scheduler(self, checkpoint_path: str | Path) -> None:
        """Load the optimizer and learning rate scheduler state from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file from which to load the optimizer and scheduler states.

        Note:
            Concrete implementations should restore the optimizer and scheduler states as appropriate for the strategy.

        """
        ...
