"""materialize.py.

Factory class defining functions for instantiating various Training Strategies, supporting different VLMs, backbones,
and strategy configurations.
"""

from collections.abc import Callable

import torch

from vla_project.models.vlms import PrismaticVLM
from vla_project.training.strategies import FSDPStrategy, TrainingStrategy

# Registry =>> Maps ID --> {cls(), kwargs} :: supports FSDP for now, but DDP handler is also implemented!
TRAIN_STRATEGIES = {
    "fsdp-shard-grad-op": {"cls": FSDPStrategy, "kwargs": {"sharding_strategy": "shard-grad-op"}},
    "fsdp-full-shard": {"cls": FSDPStrategy, "kwargs": {"sharding_strategy": "full-shard"}},
}


def get_train_strategy(
    train_strategy: str,
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
) -> TrainingStrategy:
    """Instantiate and return a TrainingStrategy object based on the specified training strategy and configuration.

    Parameters
    ----------
    train_strategy : str
        The identifier for the training strategy to use.
    vlm : PrismaticVLM
        The vision-language model to be trained.
    device_id : int
        The device ID to use for training.
    stage : str
        The current training stage.
    epochs : int
        Number of training epochs.
    max_steps : int or None
        Maximum number of training steps.
    global_batch_size : int
        Total batch size across all devices.
    per_device_batch_size : int
        Batch size per device.
    learning_rate : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay for the optimizer.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    lr_scheduler_type : str
        Type of learning rate scheduler.
    warmup_ratio : float
        Ratio of warmup steps.
    enable_gradient_checkpointing : bool, optional
        Whether to enable gradient checkpointing (default is True).
    enable_mixed_precision_training : bool, optional
        Whether to enable mixed precision training (default is True).
    reduce_in_full_precision : bool, optional
        Whether to reduce gradients in full precision (default is False).
    mixed_precision_dtype : torch.dtype, optional
        Data type for mixed precision training (default is torch.bfloat16).
    worker_init_fn : Callable[[int], None] or None, optional
        Function to initialize data loader workers.

    Returns
    -------
    TrainingStrategy
        An instance of the selected TrainingStrategy.

    Raises
    ------
    ValueError
        If the specified training strategy is not supported.

    """
    if train_strategy in TRAIN_STRATEGIES:
        strategy_cfg = TRAIN_STRATEGIES[train_strategy]
        return strategy_cfg["cls"](
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
            **strategy_cfg["kwargs"],
        )
    msg = f"Train Strategy `{train_strategy}` is not supported!"
    raise ValueError(msg)
