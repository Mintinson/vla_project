"""Distributed Data Parallel (DDP) training strategy implementation.

This module implements the DDP training strategy for vision-language models using PyTorch's
DistributedDataParallel. DDP is a data parallel approach that replicates the model on each
GPU and synchronizes gradients across all processes during backpropagation.

Key features:
    - Model replication across all GPUs with gradient synchronization
    - Support for gradient checkpointing to reduce memory usage
    - Mixed precision training with automatic casting
    - Checkpoint saving with rank-zero coordination
    - Support for both constant and cosine learning rate schedules

The DDP strategy is suitable for:
    - Multi-GPU training on a single node or multiple nodes
    - Models that can fit entirely in GPU memory
    - Scenarios where communication overhead is manageable

Example:
    Creating and using a DDP training strategy:

    ```python
    from vla_project.training.strategies.ddp import DDPStrategy

    strategy = DDPStrategy(
        vlm=model,
        device_id=local_rank,
        stage="finetune",
        epochs=3,
        max_steps=None,
        global_batch_size=64,
        per_device_batch_size=8,
        learning_rate=1e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="linear-warmup+cosine-decay",
        warmup_ratio=0.05
    )
    ```

"""

import shutil
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.optim import AdamW
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup

from vla_project.overwatch import initialize_overwatch
from vla_project.training.strategies.base_strategy import TrainingStrategy

overwatch = initialize_overwatch(__name__)


class DDPStrategy(TrainingStrategy):
    """Distributed Data Parallel (DDP) training strategy implementation.

    This class implements the DDP training strategy that replicates the model across all
    available GPUs and synchronizes gradients during backpropagation. DDP is suitable for
    models that can fit entirely in GPU memory and provides efficient data parallelism.

    Key features:
        - Model replication with gradient synchronization
        - Support for gradient checkpointing to reduce memory usage
        - Mixed precision training with automatic casting
        - Rank-zero checkpoint saving coordination
        - AdamW optimizer with cosine decay and linear warmup

    The strategy handles:
        - Automatic device placement and DDP wrapping
        - Gradient checkpointing setup for LLM backbone
        - Optimizer and learning rate scheduler configuration
        - Checkpoint saving with training statistics
        - Gradient clipping using PyTorch's built-in utilities

    Note:
        - Currently does not support weight_decay > 0
        - Requires models to fit entirely in GPU memory
        - Uses gradient_as_bucket_view=True to optimize memory usage

    """

    @overwatch.rank_zero_only
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: float | None = None,
        *,
        only_trainable: bool = True,
    ) -> None:
        """Save model checkpoint to disk using DDP-specific state collection.

        This method saves the model and optimizer state dictionaries to disk. It extracts
        the underlying model from the DDP wrapper and saves only the specified module
        components. The checkpoint includes training statistics in the filename and
        creates a latest checkpoint symlink.

        Args:
            run_dir: Directory where the checkpoint should be saved.
            global_step: Current global training step number.
            epoch: Current training epoch number.
            train_loss: Current training loss value. If None, uses "inf" in filename.
                Defaults to None.
            only_trainable: Whether to save only trainable module parameters.
                If False, saves all module parameters. Defaults to True.

        Raises:
            TypeError: If the VLM is not wrapped in DDP.

        Note:
            - Only executes on rank zero due to @overwatch.rank_zero_only decorator
            - Checkpoint filename includes step, epoch, and loss for easy identification
            - Creates both named checkpoint and latest-checkpoint.pt symlink
            - Saves optimizer state for resuming training

        """
        if not isinstance(self.vlm, DDP):
            msg = "save_checkpoint assumes VLM is already wrapped in DDP!"
            raise TypeError(msg)

        model_state_dicts = {
            mkey: getattr(self.vlm.module, mkey).state_dict()
            for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
        }
        optimizer_state_dict = self.optimizer.state_dict()

        # Set Checkpoint Path =>> Embed *minimal* training statistics!
        checkpoint_dir = run_dir / "checkpoints"
        if train_loss is None:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
        else:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"

        # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
        torch.save({"model": model_state_dicts, "optimizer": optimizer_state_dict}, checkpoint_path)
        shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:  # noqa: ARG002
        """Set up the DDP training strategy with model wrapping and optimizer initialization.

        This method configures the DDP training environment including gradient checkpointing,
        device placement, DDP wrapping, and optimizer/scheduler initialization. It handles
        the DDP-specific setup requirements and validates configuration parameters.

        Args:
            run_dir: Directory for the training run (unused in DDP strategy).
            n_train_examples: Total number of training examples for scheduler configuration.

        Raises:
            NotImplementedError: If weight_decay > 0 is specified (not currently supported).
            ValueError: If an unsupported learning rate scheduler type is specified.

        Note:
            - Enables gradient checkpointing on LLM backbone if configured
            - Moves entire model to specified device before DDP wrapping
            - Uses gradient_as_bucket_view=True for memory optimization
            - Creates AdamW optimizer with only trainable parameters
            - Supports "linear-warmup+cosine-decay" and "constant" LR schedules
            - Logs comprehensive setup information for debugging

        """
        # Gradient Checkpointing Setup
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing --> we make the assumption that the "bulk" of activation memory is taken up
            #     by the LLM; because we also make the explicit assumption that each LLM is derived from a HF
            #     pretrained model, the only thing we *need* to do (technically) is call `gradient_checkpoint_enable`
            #     on `self.llm_backbone`.
            #
            # What does it actually do? --> runs the *generic* custom_forward + torch.utils.checkpoint.checkpoint logic
            #   => github.com/huggingface/transformers/.../models/llama/modeling_llama.py#L692-L706
            #
            # Additional Reference (to better understand gradient checkpointing in PyTorch writ large)
            #   => github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
            overwatch.info("Enabling Gradient Checkpointing on LLM Backbone", ctx_level=1)
            self.vlm.llm_backbone.gradient_checkpointing_enable()  # pyright: ignore[reportAttributeAccessIssue, reportCallIssue]

        # Move to Device =>> Note parameters are in full precision (*mixed precision* will only autocast as appropriate)
        overwatch.info("Placing Entire VLM (Vision Backbone, LLM Backbone, Projector Weights) on GPU", ctx_level=1)
        self.vlm.to(self.device_id)

        # Wrap with Distributed Data Parallel
        #   => Note: By default, wrapping naively with DDP(self.vlm) will initialize a *separate* buffer on GPU that
        #            is the same size/dtype as the model parameters; this will *double* GPU memory!
        # - stackoverflow.com/questions/68949954/model-takes-twice-the-memory-footprint-with-distributed-data-parallel
        overwatch.info("Wrapping VLM with Distributed Data Parallel", ctx_level=1)
        self.vlm = DDP(self.vlm, device_ids=[self.device_id], gradient_as_bucket_view=True)

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        trainable_params = [param for param in self.vlm.parameters() if param.requires_grad]
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            if self.weight_decay != 0:
                msg = "DDP training does not currently support `weight_decay` > 0!"
                raise NotImplementedError(msg)

            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        elif self.lr_scheduler_type == "constant":
            num_warmup_steps = 0

            if self.weight_decay != 0:
                msg = "DDP training does not currently support `weight_decay` > 0!"
                raise NotImplementedError(msg)

            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_constant_schedule(self.optimizer)

        else:
            msg = f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!"
            raise ValueError(msg)

        # Finalize Setup =>> Log
        overwatch.info(
            "DDP Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use Native AMP = {self.enable_mixed_precision_training} ({self.mixed_precision_dtype})\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n",
        )

    def clip_grad_norm(self) -> None:
        """Clip gradients using PyTorch's built-in gradient clipping utility.

        This method applies gradient clipping to all model parameters using PyTorch's
        torch.nn.utils.clip_grad_norm_ function. It uses the max_grad_norm value
        specified during strategy initialization.

        Note:
            - Uses PyTorch's standard gradient clipping implementation
            - Clips gradients of all model parameters
            - Respects the max_grad_norm attribute set during initialization
            - DDP handles gradient synchronization before clipping

        """
        torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), max_norm=self.max_grad_norm)

    def load_optimizer_and_scheduler(self, checkpoint_path: str | Path) -> None:
        """Load optimizer and scheduler state from checkpoint for DDP training.

        This method loads the optimizer and learning rate scheduler state from a previously
        saved checkpoint. Unlike FSDP, DDP doesn't require special state dictionary handling
        since the optimizer state is saved in standard PyTorch format.

        Args:
            checkpoint_path: Path to the main checkpoint file. The optimizer state is
                expected to be in a corresponding file with .optimizer extension.

        Raises:
            TypeError: If the VLM is not wrapped in DDP.

        Note:
            - Loads optimizer state directly without FSDP-specific transformations
            - Uses CPU mapping to avoid device conflicts during loading
            - Warns if optimizer checkpoint is missing but continues execution
            - LR scheduler state is loaded along with optimizer state

        """
        if not isinstance(self.vlm, DDP):
            msg = "DDPStrategy.load_optimizer_and_scheduler assumes VLM is already wrapped in DDP!"
            raise TypeError(msg)

        checkpoint_path = Path(checkpoint_path)

        # Check for traditional checkpoint format first (with optimizer in main checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "optimizer" in checkpoint:
            # Traditional format: optimizer state is in the main checkpoint file
            overwatch.info(f"Loading optimizer state from main checkpoint: {checkpoint_path}")
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            # Load scheduler state if available
            if "scheduler" in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
                overwatch.info("Loaded learning rate scheduler state")

        else:
            # FSDP-style format: look for separate optimizer file
            optimizer_path = self._get_optimizer_path(checkpoint_path)
            if not optimizer_path.exists():
                overwatch.warning(f"Optimizer checkpoint not found at {optimizer_path}!")
                return

            overwatch.info(f"Loading optimizer state from separate file: {optimizer_path}")
            optim_checkpoint = torch.load(optimizer_path, map_location="cpu")

            # Extract optimizer state from the checkpoint structure
            if "optimizer" in optim_checkpoint:
                self.optimizer.load_state_dict(optim_checkpoint["optimizer"])
            else:
                # Direct optimizer state dict
                self.optimizer.load_state_dict(optim_checkpoint)

            # Load scheduler state if available
            if "scheduler" in optim_checkpoint:
                self.lr_scheduler.load_state_dict(optim_checkpoint["scheduler"])
                overwatch.info("Loaded learning rate scheduler state")

        overwatch.info("Successfully loaded optimizer state for DDP training")

    def _get_optimizer_path(self, checkpoint_path: Path) -> Path:
        """Get the path to the optimizer checkpoint file."""
        return checkpoint_path.with_suffix(".optimizer")
