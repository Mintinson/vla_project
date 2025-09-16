"""General utilities for randomness, mixed precision training, and miscellaneous checks in PyTorch.

Random `set_global_seed` functionality is taken directly from PyTorch-Lighting:
    > Ref: https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/seed.py

This is pretty important to get right if we're every randomly generating our masks (or prefix dropout) inside our
Dataset __getitem__() with multiple workers... if not handled properly, we will get repeated augmentations anytime
we inject randomness from non-PyTorch sources (e.g., numpy, random)!
    > Ref: https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/

Terminology
    -> World Size :: Total number of processes distributed over (# nodes x # devices) -- assumed homogenous!
    -> Rank :: Integer index of current process in the total world size
    -> Local Rank :: Local index on given node in [0, Devices per Node]
"""

import os
import random
from collections.abc import Callable

import numpy as np
import torch

# === Randomness ===


def set_global_seed(seed: int, *, get_worker_init_fn: bool = False) -> Callable[[int], None] | None:
    """Set global random seed for reproducible training across all randomness libraries.

    Initializes random seeds for Python's random module, NumPy, and PyTorch to ensure
    reproducible results. Also sets up proper worker initialization for DataLoader
    multiprocessing to avoid repeated augmentations.

    This implementation is adapted from PyTorch Lightning and addresses the common
    issue where multiple DataLoader workers generate identical random augmentations
    due to improper seed handling.

    Args:
        seed (int): Random seed value. Must be within np.uint32 bounds.
        get_worker_init_fn (bool, optional): Whether to return a worker initialization
            function for DataLoader. Defaults to False.

    Returns:
        Callable[[int], None] | None: Worker initialization function if requested,
            otherwise None. The function should be passed to DataLoader's worker_init_fn.

    Raises:
        AssertionError: If seed is outside np.uint32 bounds.

    Example:
        >>> worker_init_fn = set_global_seed(42, get_worker_init_fn=True)
        >>> dataloader = DataLoader(dataset, worker_init_fn=worker_init_fn)

    Note:
        Sets the EXPERIMENT_GLOBAL_SEED environment variable for reference by
        worker processes in distributed training scenarios.

    """
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, "Seed outside the np.uint32 bounds!"

    # Set Seed as an Environment Variable
    os.environ["EXPERIMENT_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)

    return worker_init_function if get_worker_init_fn else None


def worker_init_function(worker_id: int) -> None:
    """Initialize random seeds for DataLoader worker processes.

    Ensures that each DataLoader worker process has a unique but deterministic
    random seed, preventing identical augmentations across workers while maintaining
    reproducibility. This function should be passed to DataLoader's worker_init_fn.

    The implementation uses NumPy's SeedSequence to create a hierarchy of seeds
    that incorporates the base seed, worker ID, and distributed training rank,
    ensuring each worker has a unique random state.

    Args:
        worker_id (int): Unique identifier for the DataLoader worker in range [0, num_workers).

    Note:
        This function is borrowed from PyTorch Lightning and inspired by PyTorch
        issue #5059. It requires the LOCAL_RANK environment variable to be set
        for distributed training scenarios.

    Reference:
        https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562

    Example:
        >>> dataloader = DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_function)

    """
    # Get current `rank` (if running distributed) and `process_seed`
    global_rank, process_seed = int(os.environ["LOCAL_RANK"]), torch.initial_seed()

    # Back out the "base" (original) seed - the per-worker seed is set in PyTorch:
    #   > https://pytorch.org/docs/stable/data.html#data-loading-randomness
    base_seed = process_seed - worker_id

    # "Magic" code --> basically creates a seed sequence that mixes different "sources" and seeds every library...
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])

    # Use 128 bits (4 x 32-bit words) to represent seed --> generate_state(k) produces a `k` element array!
    np.random.seed(seed_seq.generate_state(4))  # noqa: NPY002

    # Spawn distinct child sequences for PyTorch (reseed) and stdlib random
    torch_seed_seq, random_seed_seq = seed_seq.spawn(2)

    # Torch Manual seed takes 64 bits (so just specify a dtype of uint64
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])

    # Use 128 Bits for `random`, but express as integer instead of as an array
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)


# === BFloat16 Support ===


def check_bloat16_supported() -> bool:
    """Check if bfloat16 (BF16) mixed precision training is supported.

    Verifies that the current system supports bfloat16 training by checking:
    - CUDA availability and version (>= 11.0)
    - GPU hardware support for bfloat16
    - NCCL availability and version (>= 2.10) for distributed training

    Returns:
        bool: True if bfloat16 is fully supported, False otherwise.

    Note:
        This function safely handles import errors and CUDA unavailability,
        returning False in any error condition. BF16 support requires:
        - CUDA 11.0+
        - Hardware support (Ampere architecture or newer)
        - NCCL 2.10+ for distributed training

    Example:
        >>> if check_bloat16_supported():
        ...     autocast_dtype = torch.bfloat16
        ... else:
        ...     autocast_dtype = torch.float16

    """
    try:
        import packaging.version  # noqa: PLC0415
        import torch.distributed as dist  # noqa: PLC0415
        from torch.cuda import nccl  # noqa: PLC0415

        return (
            (torch.version.cuda is not None)  # pyright: ignore[reportAttributeAccessIssue]
            and torch.cuda.is_bf16_supported()
            and (packaging.version.parse(torch.version.cuda).release >= (11, 0))  # pyright: ignore[reportAttributeAccessIssue]
            and dist.is_nccl_available()
            and (nccl.version() >= (2, 10))
        )

    except Exception:
        return False
