"""Batching utilities for efficient multimodal training.

This module provides specialized samplers for handling multimodal datasets
where examples may have different modalities (unimodal vs multimodal) and
varying sequence lengths, optimizing for training efficiency.
"""

# ...existing code...

import math
from collections.abc import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


class SplitModalitySampler(Sampler):
    """Distributed sampler for multimodal datasets with length-based grouping.

    This sampler handles datasets containing both unimodal (text-only) and multimodal
    (text + image) examples. It groups examples by modality and sequence length to
    improve training efficiency by minimizing padding and ensuring similar computational
    loads across distributed training workers.

    The sampler performs the following operations:
    1. Separates examples by modality (unimodal vs multimodal)
    2. Groups examples within each modality by sequence length
    3. Creates length-balanced mini-batches for each GPU in distributed training
    4. Randomly shuffles batch order while preserving length grouping

    Attributes:
        num_replicas (int): Number of distributed training workers.
        rank (int): Rank of current worker in distributed training.
        seed (int): Random seed for reproducible sampling.
        epoch (int): Current training epoch (affects randomization).
        dataset (Dataset): The dataset to sample from.
        modality_lengths (list[tuple[bool, int]]): List of (is_multimodal, length) tuples.
        drop_last (bool): Whether to drop incomplete batches (always False).
        global_batch_size (int): Total batch size across all workers.
        total_size (int): Total number of samples after padding.
        num_samples (int): Number of samples per worker.

    """

    def __init__(
        self,
        *,
        dataset: Dataset,
        modality_lengths: list[tuple[bool, int]],
        global_batch_size: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """Initialize the SplitModalitySampler.

        Args:
            dataset (Dataset): Dataset to sample from.
            modality_lengths (list[tuple[bool, int]]): List of tuples where each tuple
                contains (is_multimodal, sequence_length) for each dataset example.
            global_batch_size (int): Total batch size across all distributed workers.
            num_replicas (int | None, optional): Number of distributed workers.
                If None, uses torch.distributed.get_world_size(). Defaults to None.
            rank (int | None, optional): Rank of current worker. If None, uses
                torch.distributed.get_rank(). Defaults to None.
            seed (int, optional): Random seed for reproducible sampling. Defaults to 0.
            drop_last (bool, optional): Whether to drop incomplete batches.
                Must be False for this sampler. Defaults to False.

        Raises:
            NotImplementedError: If drop_last is True (not supported).

        """
        super().__init__()
        self.num_replicas = num_replicas if num_replicas is not None else dist.get_world_size()
        self.rank = rank if rank is not None else dist.get_rank()
        self.seed = seed
        self.epoch = 0
        self.dataset = dataset
        self.modality_lengths = modality_lengths
        self.drop_last = drop_last
        self.global_batch_size = global_batch_size
        # For our purposes, `drop_last` is always False!
        if self.drop_last:
            msg = "drop_last=True is not supported for SplitModalitySampler"
            raise NotImplementedError(msg)
        self.total_size = math.ceil(len(self.dataset) / self.global_batch_size) * self.global_batch_size  # pyright: ignore[reportArgumentType]
        self.num_samples = self.total_size // self.num_replicas

    @staticmethod
    def reindex_batch(batch_idx: list[int], idx2lengths: list[int], n_buckets: int) -> list[list[int]]:
        """Redistribute batch indices into length-balanced buckets.

        Takes a batch of indices (sorted by length) and redistributes them into
        n_buckets such that each bucket has roughly equal total sequence length.
        This helps balance computational load across distributed workers.

        Args:
            batch_idx (list[int]): List of indices sorted by sequence length (descending).
            idx2lengths (list[int]): Mapping from index to sequence length.
            n_buckets (int): Number of buckets to create (typically num_replicas).

        Returns:
            list[list[int]]: List of n_buckets, each containing indices with
                roughly equal total sequence length.

        Raises:
            ValueError: If batch size is not divisible by number of buckets.

        Note:
            Uses a greedy algorithm that assigns each example to the bucket
            with currently smallest total length.

        """
        if len(batch_idx) % n_buckets != 0:
            msg = "Batch size must be divisible by number of buckets"
            raise ValueError(msg)
        # Establish initial buckets, capacities, and max number of elements per bucket
        n_examples_per_bucket = len(batch_idx) // n_buckets
        bucket_indices = [[] for _ in range(n_buckets)]
        bucket_lengths = [0.0 for _ in range(n_buckets)]
        # Note that `batch_idxs` is already sorted by corresponding length (in descending order)
        for idx in batch_idx:
            shortest_bucket_idx = bucket_lengths.index(min(bucket_lengths))
            bucket_indices[shortest_bucket_idx].append(idx)

            # Update `bucket_lengths` --> set length to infinity if at capacity!
            bucket_lengths[shortest_bucket_idx] += idx2lengths[idx]
            if len(bucket_indices[shortest_bucket_idx]) == n_examples_per_bucket:
                bucket_lengths[shortest_bucket_idx] = float("inf")

        return bucket_indices

    def get_modality_and_length_grouped_indices(self, generator: torch.Generator) -> list[int]:
        """Generate indices grouped by modality and sequence length.

        This is the core method that implements the sophisticated batching strategy.
        It separates examples by modality, groups them by length within batches,
        and redistributes them to ensure each distributed worker gets length-balanced
        mini-batches.

        The algorithm:
        1. Separate multimodal and unimodal examples
        2. Randomly shuffle each modality group
        3. Create global-sized batches and sort by length within each batch
        4. Redistribute each global batch into length-balanced buckets (one per worker)
        5. Merge and shuffle the final batch order
        6. Move the longest batch to index 0 for early OOM detection

        Args:
            generator (torch.Generator): Random number generator for reproducible sampling.

        Returns:
            list[int]: List of dataset indices arranged for optimal batching.

        Note:
            The returned indices are arranged so that each slice of global_batch_size
            consecutive indices forms a batch where each sub-slice of per_replica_batch_size
            indices (assigned to one worker) has similar sequence lengths.

        """
        multimodal_indices, multimodal_lengths = zip(
            *[(idx, length) for idx, (is_multimodal, length) in enumerate(self.modality_lengths) if is_multimodal],
            strict=False,
        )
        multimodal_indices = list(multimodal_indices)
        multimodal_lengths = list(multimodal_lengths)

        # Handle Special Case --> no "unimodal" inputs
        unimodal_split = [
            (idx, length) for idx, (is_multimodal, length) in enumerate(self.modality_lengths) if not is_multimodal
        ]
        if len(unimodal_split) == 0:
            unimodal_indices, unimodal_lengths = [], []
        else:
            unimodal_indices, unimodal_lengths = zip(*unimodal_split, strict=False)
            unimodal_indices = list(unimodal_indices)
            unimodal_lengths = list(unimodal_lengths)

        # Create a permutation of indices for each of the multimodal and unimodal data
        mm_shuffled_idxs = torch.randperm(len(multimodal_indices), generator=generator)
        uni_shuffled_idxs = torch.randperm(len(unimodal_indices), generator=generator)

        # We're going to be running sorting/grouping relative to `self.global_batch_size` and `self.num_replicas`
        g_bsz = self.global_batch_size

        # Break each of the permutations into batches of length `global_batch_size`
        mm_batch_idxs: list[list[int]] = [
            mm_shuffled_idxs[i : i + g_bsz].tolist() for i in range(0, len(mm_shuffled_idxs), g_bsz)
        ]
        uni_batch_idxs: list[list[int]] = [
            uni_shuffled_idxs[i : i + g_bsz].tolist() for i in range(0, len(uni_shuffled_idxs), g_bsz)
        ]

        # If "last" batch is not of length `g_bsz` --> PAD by stealing indices from the first batch!
        if len(mm_batch_idxs[-1]) < g_bsz:
            n_missing = g_bsz - len(mm_batch_idxs[-1])
            mm_batch_idxs[-1].extend(mm_batch_idxs[0][:n_missing])

        if len(uni_batch_idxs) > 0 and len(uni_batch_idxs[-1]) < g_bsz:
            n_missing = g_bsz - len(uni_batch_idxs[-1])
            uni_batch_idxs[-1].extend(uni_batch_idxs[0][:n_missing])

        # Now we're going to sort each batch by length --> this will aid in grouping by length by rank (efficiency!)
        mm_sorted_batch_idxs = [sorted(b, key=lambda i: multimodal_lengths[i], reverse=True) for b in mm_batch_idxs]
        uni_sorted_batch_idxs = [sorted(b, key=lambda i: unimodal_lengths[i], reverse=True) for b in uni_batch_idxs]

        # IMPORTANT :: At this point, for each modality, we have a list of "batches" (made up of indices) where indices
        # are sorted by example sequence length *within* each batch. To make this more concrete, consider the following:
        #   => World Size (`num_replicas`) = 2
        #   => Global Batch Size (`g_bsz`) = 4
        #   => `multimodal_indices` = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11]
        #      `multimodal_lengths` = [20, 90, 21, 22, 91, 18, 89, 19, 93, 88, 92, 17]
        #
        # At this point in the code, `mm_sorted_batch_idxs` might then look like the following (length in parenthesis):
        #   => `mm_sorted_batch_idxs`: [
        #       [4  (91), 3  (21), 0  (20), 5  (18)]    => Batch 1
        #       [6  (89), 9  (88), 7  (19), 11 (17)]    => Batch 2
        #       [8  (93), 10 (92), 1  (90), 2  (21)]    => Batch 3
        #   ]
        #
        # In practice: `g_bsz` is large (= 128), and for contiguous mini-batch "slices", length variance is low.

        # PROBLEM :: We want to split these "global batches" into equal-sized pieces, so that each "replica" (GPU)
        # sees a "mini-batch" of roughly the same sequence lengths; this is super useful for efficient training.

        # HOWEVER :: The default "access pattern" for splitting a large batch into mini-batches by a DistributedSampler
        # is akin to a "take every k" where `k` is equal to the number of replicas (GPUs) you're training on. Or, in
        # Python notation --> `rank_k_indices = flatten(mm_sorted_batch_idxs)[k::num_replicas].
        #
        # Naively translating this our example means each GPU (in our world of 2 total) sees the following indices
        # (grouped by "mini-batch" = `g_bsz / num_replicas` = 2 for convenience):
        #   => `rank_0_indices`: [ [4 (91), 0 (20)] =>> [6 (89), 7  (19)] =>> [8  (93), 1 (90)] ]
        #   => `rank_1_indices`: [ [3 (21), 5 (18)] =>> [9 (88), 11 (17)] =>> [10 (92), 2 (21)] ]
        #
        # We get lucky sometimes, but for the most part, each "mini-batch" has VASTLY DIFFERENT lengths! Bad!

        # FIX :: If we "undo" the access pattern with the following code and re-arrange the way we allocate batches
        # inside the __iter__ method below, we can allocate indices appropriately. Running the following code gives us
        # the following indices (grouped by "mini-batch" again for convenience):
        #   => `rank_0_indices`: [ [4 (91), 3 (21)] =>> [6  (89), 9 (88)] =>> [8 (93), 10 (92)] ]
        #   => `rank_1_indices`: [ [5 (18), 0 (20)] =>> [11 (17), 7 (19)] =>> [2 (21),  1 (90)] ]
        #
        # Much better! As `g_bsz` and `dataset` grow, we're more often than not getting *decent* groupings!
        mm_length_bucketed_idxs = [
            self.reindex_batch(batch, multimodal_lengths, self.num_replicas) for batch in mm_sorted_batch_idxs
        ]
        uni_length_bucketed_idxs = [
            self.reindex_batch(batch, unimodal_lengths, self.num_replicas) for batch in uni_sorted_batch_idxs
        ]

        # Note :: Because of the initial `randperm` --> we're indexing both sets from 0 (we're clobbering the range)
        #   => Flatten indices --> index into original `{modality}_indices` then re-batch!
        mm_output_idxs = [idx for batch in mm_length_bucketed_idxs for bucket in batch for idx in bucket]
        mm_reindexed = [multimodal_indices[idx] for idx in mm_output_idxs]
        mm_batches = [mm_reindexed[i : i + g_bsz] for i in range(0, len(mm_reindexed), g_bsz)]

        uni_output_idxs = [idx for batch in uni_length_bucketed_idxs for bucket in batch for idx in bucket]
        uni_reindexed = [unimodal_indices[idx] for idx in uni_output_idxs]
        uni_batches = [uni_reindexed[i : i + g_bsz] for i in range(0, len(uni_reindexed), g_bsz)]

        # Finally, randomly permute the multimodal & unimodal batches, merging into a single stream of indices
        merged_batches = mm_batches + uni_batches
        merge_idxs = torch.randperm(len(merged_batches), generator=generator)
        all_batches = [merged_batches[idx] for idx in merge_idxs]

        # [Quality of Life] Shift "max length" batch to index 0 --> if we OOM, it happens immediately!
        all_lengths = [length + ((_n_patches := 24 * 24) if is_mm else 0) for is_mm, length in self.modality_lengths]
        all_batches_max_lengths = [max(all_lengths[idx] for idx in batch) for batch in all_batches]

        # Identify Batch with "max length" --> Swap into Index 0
        longest_batch_idx = np.argmax(all_batches_max_lengths)
        all_batches[0], all_batches[longest_batch_idx] = all_batches[longest_batch_idx], all_batches[0]

        # Flatten & Return all Indices
        return [idx for batch in all_batches for idx in batch]

    def __iter__(self) -> Iterator[int]:
        """Return an iterator over dataset indices for the current worker.

        Generates the indices that this specific worker (rank) should process
        in the current epoch. The indices are arranged to provide length-balanced
        mini-batches for efficient training.

        Returns:
            Iterator[int]: Iterator over dataset indices for this worker.

        Raises:
            ValueError: If indices don't match expected dataset size or aren't
                properly divisible by batch size and number of replicas.

        Note:
            The iterator respects the distributed training setup, returning only
            the indices that this specific rank should process.

        """
        g = torch.Generator()
        g.manual_seed(seed=self.seed + self.epoch)
        indices = self.get_modality_and_length_grouped_indices(g)
        if not (len(set(indices)) == len(self.modality_lengths) == len(self.dataset)):  # pyright: ignore[reportArgumentType]
            msg = (
                f"Length mismatch: indices {len(set(indices))}, dataset {len(self.dataset)}"  # pyright: ignore[reportArgumentType]
                f" modality_lengths {len(self.modality_lengths)}"
            )
            raise ValueError(msg)
        if len(indices) % self.global_batch_size != 0 and (len(indices) % self.num_replicas) == 0:
            msg = (
                f"Total size {len(indices)} not divisible by global batch {self.global_batch_size}"
                f"and num_replicas {self.num_replicas}"
            )
            raise ValueError(msg)
        # Note :: We compute per-replica batch size as a function of `global_batch` and `num_replicas` to ensure that
        # gradient accumulation doesn't affect what indices are assigned a given rank.
        per_replica_batch_size = self.global_batch_size // self.num_replicas

        indices_t = torch.as_tensor(indices)
        per_replica_batch_indices_t = indices_t.view(-1, per_replica_batch_size)
        replica_indices_t = per_replica_batch_indices_t[self.rank :: self.num_replicas]

        replica_indices = replica_indices_t.flatten().tolist()
        return iter(replica_indices)

    def __len__(self) -> int:
        """Return the number of samples this worker will process.

        Returns:
            int: Number of samples for this worker in one epoch.

        """
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic shuffling.

        Must be called before creating a DataLoader for each new epoch to ensure
        different random shuffling across epochs while maintaining reproducibility.

        Args:
            epoch (int): Current epoch number.

        Note:
            This affects the random seed used for shuffling, ensuring that each
            epoch sees the data in a different but reproducible order.

        """
        self.epoch = epoch
