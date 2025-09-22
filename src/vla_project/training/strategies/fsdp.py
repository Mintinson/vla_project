"""Fully Sharded Data Parallel (FSDP) training strategy implementation.

This module implements the FSDP training strategy for vision-language models using PyTorch's
FullyShardedDataParallel. FSDP is an advanced data parallel approach that shards model parameters,
gradients, and optimizer states across multiple GPUs to enable training of very large models
that cannot fit in the memory of a single GPU.

Key features:
    - Parameter, gradient, and optimizer state sharding across GPUs
    - Mixed precision training with configurable precision policies
    - Support for gradient checkpointing with activation checkpointing
    - Flexible sharding strategies (shard-grad-op, full-shard)
    - Memory-efficient state dictionary handling
    - Advanced parameter grouping for weight decay

The FSDP strategy is suitable for:
    - Training very large models that exceed single GPU memory
    - Multi-GPU training with memory constraints
    - Scenarios requiring maximum memory efficiency
    - Advanced distributed training setups

Example:
    Creating and using an FSDP training strategy:

    ```python
    from vla_project.training.strategies.fsdp import FSDPStrategy

    strategy = FSDPStrategy(
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
        warmup_ratio=0.05,
        sharding_strategy="full-shard"
    )
    ```

"""

import math
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
from torch.optim import AdamW
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup

from vla_project.models.vlms.prismatic import PrismaticVLM
from vla_project.overwatch import initialize_overwatch
from vla_project.training.strategies.base_strategy import TrainingStrategy

overwatch = initialize_overwatch(__name__)


class FSDPStrategy(TrainingStrategy):
    """Fully Sharded Data Parallel (FSDP) training strategy implementation.

    This class implements the FSDP training strategy that shards model parameters, gradients,
    and optimizer states across multiple GPUs to enable training of very large models that
    cannot fit in the memory of a single GPU. FSDP provides advanced memory management and
    supports sophisticated sharding strategies.

    Key features:
        - Parameter, gradient, and optimizer state sharding across GPUs
        - Mixed precision training with configurable precision policies
        - Support for gradient checkpointing with activation checkpointing
        - Flexible sharding strategies (shard-grad-op, full-shard)
        - Memory-efficient state dictionary handling
        - Advanced parameter grouping for weight decay optimization

    The strategy handles:
        - Automatic FSDP wrapping with custom wrapping policies
        - Mixed precision configuration for different training stages
        - Non-reentrant gradient checkpointing for LLM transformers
        - Parameter grouping for selective weight decay application
        - Full state dictionary reconstruction for checkpoint saving

    Attributes:
        fsdp_sharding_strategy: The FSDP sharding strategy to use.
        fsdp_state_dict_type: Type of state dictionary for checkpoint saving.
        fsdp_save_policy: Configuration for full state dictionary saving.

    Note:
        - Supports "shard-grad-op" and "full-shard" sharding strategies
        - Currently only supports FULL_STATE_DICT for checkpoint saving
        - Automatically handles device placement and memory management
        - Requires careful coordination for checkpoint saving across ranks

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
        sharding_strategy: str = "shard-grad-op",
        state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT,
    ) -> None:
        """Initialize the FSDP training strategy with sharding configuration.

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
            sharding_strategy: FSDP sharding strategy to use. Options are "shard-grad-op"
                for gradient and optimizer state sharding, or "full-shard" for complete
                parameter sharding. Defaults to "shard-grad-op".
            state_dict_type: Type of state dictionary for checkpoint saving.
                Defaults to StateDictType.FULL_STATE_DICT.

        Raises:
            ValueError: If an unsupported sharding strategy is specified.
            AssertionError: If state_dict_type is not FULL_STATE_DICT (sharded saving
                not yet implemented).

        Note:
            - Inherits from TrainingStrategy and calls parent constructor
            - Sets up FSDP-specific parameters and configurations
            - Configures state dictionary saving policy for rank-zero checkpoint saving

        """
        super().__init__(
            vlm=vlm,
            device_id=device_id,
            stage=stage,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
        )

        # FSDP-Specific Parameters
        if sharding_strategy == "shard-grad-op":
            self.fsdp_sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2  # noqa: SLF001
        elif sharding_strategy == "full-shard":
            self.fsdp_sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            msg = f"FSDP Sharding Strategy {sharding_strategy} is not supported!"
            raise ValueError(msg)

        assert state_dict_type == StateDictType.FULL_STATE_DICT, "Sharded state saving is not yet implemented!"
        self.fsdp_state_dict_type = state_dict_type
        self.fsdp_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    def clip_grad_norm(self) -> None:
        """Clip gradients using FSDP's custom gradient clipping function.

        This method applies gradient clipping specifically designed for FSDP training.
        Unlike standard PyTorch gradient clipping, FSDP requires uniform gradient
        data types across all shards for proper gradient clipping functionality.

        Note:
            - Uses FSDP's custom clip_grad_norm_ function instead of PyTorch's standard version
            - Requires uniform gradient dtype across all model shards
            - Respects the max_grad_norm attribute set during strategy initialization
            - Handles gradient clipping in the context of FSDP's parameter sharding

        """
        # Note =>> FSDP uses a custom `clip_grad_norm_` function; requires *uniform grad dtype*
        self.vlm.clip_grad_norm_(max_norm=self.max_grad_norm)

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: float | None = None,
        *,
        only_trainable: bool = True,
    ) -> None:
        """Save model checkpoint to disk using FSDP-specific state collection.

        This method saves model state dictionaries to disk by first reconstructing the full
        state dictionary from FSDP shards, then extracting the specified module components.
        The checkpoint saving is coordinated to only occur on rank zero to avoid conflicts.

        Args:
            run_dir: Directory where the checkpoint should be saved.
            global_step: Current global training step number.
            epoch: Current training epoch number.
            train_loss: Current training loss value. If None, uses "inf" in filename.
                Defaults to None.
            only_trainable: Whether to save only trainable module parameters.
                If False, saves all module parameters. Defaults to True.

        Raises:
            TypeError: If the VLM is not wrapped in FSDP.

        Note:
            - Uses FSDP's state_dict_type context manager to reconstruct full state dict
            - Only saves on rank zero to avoid file system conflicts
            - Checkpoint filename includes step, epoch, and loss for identification
            - Does not save optimizer state (unlike DDP strategy)
            - Handles parameter name prefix splitting for module organization

        """
        if not isinstance(self.vlm, FSDP):
            msg = "FSDPStrategy.save_checkpoint assumes VLM is already wrapped in FSDP!"
            raise TypeError(msg)
        # Summon Full State Dictionary =>> Reconstitute from Shards
        with FSDP.state_dict_type(self.vlm, self.fsdp_state_dict_type, self.fsdp_save_policy):
            full_vlm_state_dict = self.vlm.state_dict()
            model_state_dicts = {
                mkey: OrderedDict() for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
            }
            # Iterate through `full_vlm_state_dict` and split `mkey.{full_dotted_path}` -> `mkey: {full_dotted_path}`
            for k, p in full_vlm_state_dict.items():
                for mkey, state_dict in model_state_dicts.items():
                    if k.startswith(mprefix := f"{mkey}."):
                        state_dict[k.removeprefix(mprefix)] = p
        # Save on rank zero *only*
        if overwatch.is_rank_zero():
            checkpoint_dir = run_dir / "checkpoints"
            if train_loss is None:
                checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
            else:
                checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"

            # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
            torch.save({"model": model_state_dicts}, checkpoint_path)

            # TODO (siddk) :: This breaks w/ Sagemaker default permissions (root vs. <user>)... skip?
            # shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:  # noqa: ARG002
        """Set up the FSDP training strategy with model wrapping and optimizer initialization.

        This method configures the FSDP training environment including mixed precision setup,
        FSDP wrapping with custom policies, gradient checkpointing configuration, and
        optimizer/scheduler initialization. It handles the complex FSDP-specific setup
        requirements for distributed training of large models.

        Args:
            run_dir: Directory for the training run (unused in FSDP strategy).
            n_train_examples: Total number of training examples for scheduler configuration.

        Raises:
            ValueError: If an unsupported learning rate scheduler type is specified.

        Note:
            - Configures mixed precision policy based on training stage and settings
            - Applies FSDP wrapping with custom policies from VLM
            - Sets up non-reentrant gradient checkpointing for transformer layers
            - Creates parameter groups for selective weight decay application
            - Supports "linear-warmup+cosine-decay" and "constant" LR schedules
            - Automatically handles device placement and memory management
            - Logs comprehensive setup information for debugging

        """
        # Iteratively Assemble FSDP Wrapping Policy by fetching the wrapping policies for each backbone/constituent
        vlm_fsdp_wrapping_policy = self.vlm.get_fsdp_wrapping_policy()

        # Assemble the Default FSDP Mixed Precision Policy
        if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:
            # MixedPrecision `param_dtype` specifies *compute* dtype (for forward/backward only)
            #   => Reference: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision
            reduce_buffer_dtype = torch.bfloat16 if not self.reduce_in_full_precision else torch.float32
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=reduce_buffer_dtype,
                buffer_dtype=reduce_buffer_dtype,
            )

            # When running FSDP with a frozen vision backbone --> move to half precision!
            if self.stage not in {"full-finetune", "vla-full-train", "vla-sandwich-train"}:
                overwatch.info("Casting Vision Backbone to *Half Precision* via `.to(dtype=...)`")
                self.vlm.vision_backbone.to(dtype=self.vlm.vision_backbone.half_precision_dtype)

        else:
            # If we're not using mixed precision, everything is in default full precision!
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )

        # <FSDP> => note that FSDP will automatically take care of device placement (similar to `autocast`)
        self.vlm = FSDP(
            self.vlm,
            auto_wrap_policy=vlm_fsdp_wrapping_policy,
            mixed_precision=fsdp_precision_policy,
            sharding_strategy=self.fsdp_sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
        )

        # Gradient Checkpoint Setup
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing under FSDP --> we make the same assumption as in the DDP/other strategies; the
            #   bulk of activation memory is taken up by the LLM activations. However, unlike other strategies, we
            #   cannot rely on the HF Transformers default `gradient_checkpointing_enable()` --> FSDP breaks semantics!
            #
            # Instead, we need to write our own *NO-REENTRANT* wrapper, and apply it to the LLM's Transformer Layer.
            non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

            def check_fn(submodule: nn.Module) -> bool:
                return isinstance(submodule, self.llm_transformer_layer_cls)

            # Note that the terms "activation checkpointing" and "gradient checkpointing" are synonymous!
            apply_activation_checkpointing(self.vlm, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

        # Barrier =>> Sharding takes a minute?
        dist.barrier()

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            # Default AdamW w/ specified LR & Linear Warmup / Cosine Decay & Weight Decay
            #   => Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed!
            decay, no_decay = [], []
            for name, param in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue

                # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            # Build Parameter Groups
            groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

            # Create Optimizer & LR Scheduler
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        elif self.lr_scheduler_type == "constant":
            num_warmup_steps = 0

            # Default AdamW w/ specified LR & Linear Warmup / Cosine Decay & Weight Decay
            #   => Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed!
            decay, no_decay = [], []
            for name, param in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue

                # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            # Build Parameter Groups
            groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

            # Create Optimizer & LR Scheduler
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_constant_schedule(self.optimizer)

        else:
            msg = f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!"
            raise ValueError(msg)

        # Finalize Setup =>> Log!
        overwatch.info(
            "FSDP Full-Shard Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone FSDP Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use FSDP Mixed Precision = {self.enable_mixed_precision_training}\n"
            f"                 |-> Parameter Precision = {fsdp_precision_policy.param_dtype}\n"
            f"                 |-> Reduction Precision = {fsdp_precision_policy.reduce_dtype}\n"
            f"                 |-> Buffer Precision = {fsdp_precision_policy.buffer_dtype}\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n",
        )

    def load_optimizer_and_scheduler(self, checkpoint_path: str | Path) -> None:
        """Load a checkpoint from the specified `checkpoint_path`."""
        if not isinstance(self.vlm, FSDP):
            msg = "FSDPStrategy.load_optimizer_and_scheduler assumes VLM is already wrapped in FSDP!"
            raise TypeError(msg)
        checkpoint_path = Path(checkpoint_path)
        optimizer_path = self._get_optimizer_path(checkpoint_path)
        if not optimizer_path.exists():
            overwatch.warning(f"Optimizer checkpoint not found at {optimizer_path}!")
            return
        # Load Checkpoint =>> Note that FSDP will automatically handle device placement!
        optim_state_dict = torch.load(optimizer_path, map_location="cpu")
        with FSDP.state_dict_type(
            self.vlm,
            self.fsdp_state_dict_type,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            optim_state_dict = FSDP.optim_state_dict_to_load(self.vlm, self.optimizer, optim_state_dict["optimizer"])
            self.optimizer.load_state_dict(optim_state_dict)
        overwatch.info(f"Loaded optimizer state dict from {optimizer_path}")

    def _get_optimizer_path(self, checkpoint_path: Path) -> Path:
        """Get the path to the optimizer checkpoint file."""
        return checkpoint_path.with_suffix(".optimizer")
