"""RLDS-specific data utilities for vision-language-action models.

This module provides comprehensive data processing utilities for RLDS (Recorded Learning
for Decision Making) datasets, specifically designed for robotic learning and vision-language-
action (VLA) model training. It includes functions for action normalization, gripper action
processing, dataset statistics computation, and trajectory transformations.

Key Features:
    - Action and proprioceptive state normalization with multiple schemes
    - Gripper action binarization and conversion utilities
    - Dataset statistics computation with caching
    - Bridge dataset specific transformations
    - Parallel processing utilities for large datasets

Example:
    Basic usage for action normalization:

    ```python
    from vla_project.models.vla.datasets.rlds.utils.data_utils import (
        normalize_action_and_proprio, NormalizationType
    )

    # Normalize trajectory actions and proprioceptive states
    normalized_traj = normalize_action_and_proprio(
        traj, metadata, NormalizationType.BOUNDS
    )
    ```

"""

import hashlib
import json
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import dlimp as dl
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from vla_project.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# Constants for gripper action thresholds
GRIPPER_OPEN_THRESHOLD = 0.95
GRIPPER_CLOSED_THRESHOLD = 0.05
GRIPPER_OPENING_THRESHOLD = -0.1
GRIPPER_CLOSING_THRESHOLD = 0.1


def tree_map(fn: Callable, tree: dict) -> dict:
    """Apply a function recursively to all non-dict values in a nested dictionary.

    Args:
        fn: Function to apply to each leaf value.
        tree: Nested dictionary structure to process.

    Returns:
        New dictionary with the same structure but transformed leaf values.

    Example:
        >>> data = {"a": 1, "b": {"c": 2}}
        >>> tree_map(lambda x: x * 2, data)
        {"a": 2, "b": {"c": 4}}

    """
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_merge(*trees: dict) -> dict:
    """Merge multiple nested dictionaries into a single dictionary.

    Recursively merges dictionaries, with later dictionaries taking precedence
    over earlier ones in case of key conflicts.

    Args:
        *trees: Variable number of nested dictionaries to merge.

    Returns:
        Merged dictionary containing all keys and values from input dictionaries.

    Example:
        >>> dict1 = {"a": 1, "b": {"c": 2}}
        >>> dict2 = {"b": {"d": 3}, "e": 4}
        >>> tree_merge(dict1, dict2)
        {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}

    """
    merged = {}
    for tree in trees:
        for k, v in tree.items():
            if isinstance(v, dict):
                merged[k] = tree_merge(merged.get(k, {}), v)
            else:
                merged[k] = v
    return merged


def to_padding(tensor: tf.Tensor) -> tf.Tensor:
    if tf.debugging.is_numeric_tensor(tensor):
        return tf.zeros_like(tensor)
    if tensor.dtype == tf.string:
        return tf.fill(tf.shape(tensor), "")
    msg = f"Cannot generate padding for tensor of type {tensor.dtype}."
    raise ValueError(msg)


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# === State / Action Processing Primitives ===


# ruff: noqa: B023
def normalize_action_and_proprio(traj: dict, metadata: dict, normalization_type: NormalizationType):
    """Normalize action and proprioception fields of a trajectory using metadata.

    Applies statistical normalization to action and proprioception data based on
    precomputed metadata statistics. Supports different normalization strategies.

    Args:
        traj: Dictionary containing trajectory data with action and observation keys.
        metadata: Dictionary containing statistical information for normalization,
            including means, standard deviations, min/max values, and quantiles.
        normalization_type: Type of normalization to apply. Options are:
            - NORMAL: Standard z-score normalization using mean and std
            - BOUNDS: Normalize to [-1, 1] range using min/max bounds
            - BOUNDS_Q99: Normalize using 1st and 99th percentile bounds

    Returns:
        Normalized trajectory dictionary with transformed action and proprioception values.

    Raises:
        ValueError: If normalization_type is not supported.

    Example:
        >>> metadata = {"action": {"mean": [0.1], "std": [0.5]}}
        >>> normalized_traj = normalize_action_and_proprio(traj, metadata, NormalizationType.NORMAL)

    """
    keys_to_normalize = {"action": "action", "proprio": "observation/proprio"}

    if normalization_type == NormalizationType.NORMAL:
        for key, traj_key in keys_to_normalize.items():
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["mean"], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(mask, (x - metadata[key]["mean"]) / (metadata[key]["std"] + 1e-8), x),
            )

        return traj

    if normalization_type in [NormalizationType.BOUNDS, NormalizationType.BOUNDS_Q99]:
        for key, traj_key in keys_to_normalize.items():
            if normalization_type == NormalizationType.BOUNDS:
                low = metadata[key]["min"]
                high = metadata[key]["max"]
            elif normalization_type == NormalizationType.BOUNDS_Q99:
                low = metadata[key]["q01"]
                high = metadata[key]["q99"]
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["min"], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(
                    mask,
                    tf.clip_by_value(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1),
                    x,
                ),
            )

            # Note (Moo Jin): Map unused action dimensions (i.e., dimensions where min == max) to all 0s.
            zeros_mask = metadata[key]["min"] == metadata[key]["max"]
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(zeros_mask, 0.0, x),
            )

        return traj

    error_msg = f"Unknown Normalization Type {normalization_type}"
    raise ValueError(error_msg)


def binarize_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """Convert gripper actions from continuous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near 1.0) or fully closed (near 0.0). As it
    transitions between the two, it sometimes passes through a few intermediate values. We relabel those intermediate
    values based on the state that is reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we give up on binarizing and relabel that
    chunk of intermediate values as the last action in the trajectory.

    The `scan_fn` implements the following logic:
        new_actions = np.empty_like(actions)
        carry = actions[-1]
        for i in reversed(range(actions.shape[0])):
            if in_between_mask[i]:
                carry = carry
            else:
                carry = float(open_mask[i])
            new_actions[i] = carry
    """
    open_mask, closed_mask = actions > GRIPPER_OPEN_THRESHOLD, actions < GRIPPER_CLOSED_THRESHOLD  # pyright: ignore[reportOperatorIssue]
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(in_between_mask[i], lambda: tf.cast(carry, tf.float32), lambda: is_open_float[i])

    return tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True)


def invert_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """Invert gripper actions by flipping open/closed states.

    Converts gripper actions where 0=closed, 1=open to 0=open, 1=closed
    or vice versa. Useful for datasets with different gripper conventions.

    Args:
        actions: Tensor of gripper actions in range [0, 1].

    Returns:
        Inverted gripper actions where original 0 becomes 1 and 1 becomes 0.

    Example:
        >>> actions = tf.constant([0.0, 1.0, 0.5])
        >>> inverted = invert_gripper_actions(actions)
        # Result: [1.0, 0.0, 0.5]

    """
    return 1 - actions  # pyright: ignore[reportOperatorIssue]


def rel2abs_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """Convert relative gripper actions to absolute gripper positions.

    Transforms relative gripper commands (+1 for closing, -1 for opening) into
    absolute gripper positions (0 = closed, 1 = open) by maintaining state
    throughout the trajectory.

    Args:
        actions: Tensor of relative gripper actions where:
            - Values > GRIPPER_CLOSING_THRESHOLD indicate closing commands
            - Values < GRIPPER_OPENING_THRESHOLD indicate opening commands
            - Values in between indicate no change

    Returns:
        Tensor of absolute gripper positions in range [0, 1] where:
            - 0.0 represents fully closed gripper
            - 1.0 represents fully open gripper

    Note:
        Assumes that the first relative gripper command is not redundant
        (i.e., doesn't close when already closed). If no relative commands
        are detected, assumes the gripper starts in the open position.

    Example:
        >>> rel_actions = tf.constant([-0.5, 0.0, 0.5, 0.0, -0.2])
        >>> abs_actions = rel2abs_gripper_actions(rel_actions)
        # Result: [1.0, 1.0, 0.0, 0.0, 0.0] (open -> open -> close -> close -> close)

    """
    # Note =>> -1 for closing, 1 for opening, 0 for no change
    opening_mask, closing_mask = actions < GRIPPER_OPENING_THRESHOLD, actions > GRIPPER_CLOSING_THRESHOLD  # pyright: ignore[reportOperatorIssue]
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(thresholded_actions[i] == 0, lambda: carry, lambda: thresholded_actions[i])

    # If no relative grasp, assumes open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)

    # Note =>> -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)
    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5

    return new_actions


# === Bridge-V2 =>> Dataset-Specific Transform ===
def relabel_bridge_actions(traj: dict[str, Any]) -> dict[str, Any]:
    """Relabel actions to use reached proprioceptive state for Bridge-V2 dataset.

    This function transforms the action labels to represent the actual state changes
    that occurred, using the proprioceptive state differences between consecutive
    timesteps. The last timestep is discarded as it contains no action.

    Args:
        traj: Trajectory dictionary containing 'observation' with 'state' key
            and 'action' key. The state should have shape [T, state_dim] where
            the first 6 dimensions represent movement states.

    Returns:
        Truncated trajectory dictionary with relabeled actions where:
            - Actions represent actual movement deltas from proprioceptive states
            - Last timestep is removed (no action to take)
            - Movement actions are concatenated with original gripper actions

    Example:
        >>> traj = {"observation": {"state": states}, "action": actions}
        >>> relabeled = relabel_bridge_actions(traj)
        # Actions now represent actual movement that occurred

    """
    movement_actions = traj["observation"]["state"][1:, :6] - traj["observation"]["state"][:-1, :6]
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
    traj_truncated["action"] = tf.concat([movement_actions, traj["action"][:-1, -1:]], axis=1)

    return traj_truncated


# === RLDS Dataset Initialization Utilities ===
def pprint_data_mixture(dataset_kwargs_list: list[dict[str, Any]], dataset_weights: list[int]) -> None:
    """Pretty print dataset mixture information with weights.

    Displays a formatted table showing all datasets in the mixture along with
    their corresponding sampling weights. Useful for debugging and monitoring
    dataset composition in multi-dataset training.

    Args:
        dataset_kwargs_list: List of dictionaries containing dataset configuration,
            each must have a 'name' key identifying the dataset.
        dataset_weights: List of integer weights corresponding to each dataset
            for sampling purposes.

    Note:
        This function uses print statements and should primarily be used for
        debugging or development purposes rather than production logging.

    Example:
        >>> datasets = [{"name": "dataset_a"}, {"name": "dataset_b"}]
        >>> weights = [3, 1]
        >>> pprint_data_mixture(datasets, weights)
        # Prints formatted table with dataset names and weights

    """
    print("\n######################################################################################")
    print(f"# Loading the following {len(dataset_kwargs_list)} datasets (incl. sampling weight):{'': >24} #")
    for dataset_kwargs, weight in zip(dataset_kwargs_list, dataset_weights, strict=False):
        pad = 80 - len(dataset_kwargs["name"])
        print(f"# {dataset_kwargs['name']}: {weight:=>{pad}f} #")
    print("######################################################################################\n")


def get_dataset_statistics(
    dataset: dl.DLataset,
    hash_dependencies: tuple[str, ...],
    save_dir: str | None = None,
) -> dict:
    """Compute or load cached dataset statistics for normalization.

    Either computes the statistics of a dataset or loads them from a cache file
    if this function has been called before with the same `hash_dependencies`.

    Args:
        dataset: DLataset instance to compute statistics for.
        hash_dependencies: Tuple of strings used to generate unique hash for
            caching. Should include dataset name, version, and any preprocessing
            parameters that affect statistics.
        save_dir: Optional directory to save statistics. If None, uses local
            cache directory at ~/.cache/orca/.

    Returns:
        Dictionary containing dataset statistics including:
            - min/max/mean/std of actions and proprioception
            - number of transitions and trajectories
            - other computed statistical measures

    Raises:
        ValueError: If dataset is infinite and statistics cannot be computed.

    Example:
        >>> hash_deps = ("dataset_name", "v1.0", "preprocess_config")
        >>> stats = get_dataset_statistics(dataset, hash_deps, "/tmp/cache")

    """
    unique_hash = hashlib.sha256("".join(hash_dependencies).encode("utf-8"), usedforsecurity=False).hexdigest()

    # Fallback local path for when data_dir is not writable or not provided
    local_path = (Path.home() / ".cache" / "orca" / f"dataset_statistics_{unique_hash}.json").expanduser()
    # local_path = os.path.expanduser(os.path.join("~", ".cache", "orca", f"dataset_statistics_{unique_hash}.json"))
    path = tf.io.gfile.join(save_dir, f"dataset_statistics_{unique_hash}.json") if save_dir is not None else local_path

    # check if cache file exists and load
    if tf.io.gfile.exists(path):
        overwatch.info(f"Loading existing dataset statistics from {path}.")
        with tf.io.gfile.GFile(path, "r") as f:
            return json.load(f)

    if local_path.exists():
        overwatch.info(f"Loading existing dataset statistics from {local_path}.")
        with local_path.open() as f:
            return json.load(f)

    dataset = dataset.traj_map(
        lambda traj: {
            "action": traj["action"],
            "proprio": (
                traj["observation"]["proprio"] if "proprio" in traj["observation"] else tf.zeros_like(traj["action"])
            ),
        },
    )

    cardinality = dataset.cardinality().numpy()
    if cardinality == tf.data.INFINITE_CARDINALITY:
        msg = "Cannot compute dataset statistics for infinite datasets."
        raise ValueError(msg)

    overwatch.info("Computing dataset statistics. This may take a bit, but should only need to happen once.")
    actions, proprios, num_transitions, num_trajectories = [], [], 0, 0
    for traj in tqdm(dataset.iterator(), total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None):
        actions.append(traj["action"])
        proprios.append(traj["proprio"])
        num_transitions += traj["action"].shape[0]
        num_trajectories += 1

    actions, proprios = np.concatenate(actions), np.concatenate(proprios)
    metadata = {
        "action": {
            "mean": actions.mean(0).tolist(),
            "std": actions.std(0).tolist(),
            "max": actions.max(0).tolist(),
            "min": actions.min(0).tolist(),
            "q01": np.quantile(actions, 0.01, axis=0).tolist(),
            "q99": np.quantile(actions, 0.99, axis=0).tolist(),
        },
        "proprio": {
            "mean": proprios.mean(0).tolist(),
            "std": proprios.std(0).tolist(),
            "max": proprios.max(0).tolist(),
            "min": proprios.min(0).tolist(),
            "q01": np.quantile(proprios, 0.01, axis=0).tolist(),
            "q99": np.quantile(proprios, 0.99, axis=0).tolist(),
        },
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }

    try:
        with tf.io.gfile.GFile(path, "w") as f:
            json.dump(metadata, f)
    except tf.errors.PermissionDeniedError:
        overwatch.warning(f"Could not write dataset statistics to {path}. Writing to {local_path} instead.")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("w") as f:
            json.dump(metadata, f)

    return metadata


def save_dataset_statistics(dataset_statistics: dict, run_dir: Path) -> None:
    """Save dataset statistics to a JSON file in the specified directory.

    Converts numpy arrays to lists for JSON serialization and saves the
    statistics to a 'dataset_statistics.json' file in the run directory.

    Args:
        dataset_statistics: Dictionary containing dataset statistics with
            nested structure including 'action' and optional 'proprio' keys.
        run_dir: Path object representing the directory to save the JSON file.

    Note:
        This function modifies the input dataset_statistics dictionary by
        converting numpy arrays to lists for JSON compatibility.

    """
    out_path = run_dir / "dataset_statistics.json"
    with out_path.open("w") as f_json:
        for stats in dataset_statistics.values():
            for k in stats["action"]:
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"]:
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats and isinstance(stats["num_trajectories"], np.ndarray):
                stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats and isinstance(stats["num_transitions"], np.ndarray):
                stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    overwatch.info(f"Saved dataset statistics file at path {out_path}")


def allocate_threads(n: int | None, weights: np.ndarray) -> np.ndarray:
    """Allocates an integer number of threads across datasets based on weights.

    The final array sums to `n`, but each element is no less than 1. If `n` is None, then every dataset is assigned a
    value of AUTOTUNE.
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

    assert np.all(weights >= 0), "Weights must be non-negative"
    assert len(weights) <= n, "Number of threads must be at least as large as length of weights"
    weights = np.array(weights) / np.sum(weights)

    allocation = np.zeros_like(weights, dtype=int)
    while True:
        # Give the remaining elements that would get less than 1 a 1
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)

        # Recompute the distribution over the remaining elements
        weights[mask] = 0
        weights = weights / weights.sum()

    # Allocate the remaining elements
    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= integral.sum()
    for i in np.argsort(fractional)[::-1][: int(n)]:
        allocation[i] += 1

    return allocation
