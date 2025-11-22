"""datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from vla_project.models.backbones.llm.prompting import PromptBuilder
from vla_project.models.backbones.vision import ImageTransform
from vla_project.models.vla.action_tokenizer import ActionTokenizer
from vla_project.models.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from vla_project.models.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from vla_project.models.vla.datasets.rlds.utils.data_utils import NormalizationType
from vla_project.utils.data_utils import tree_map

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class RLDSBatchTransform:
    """Transform RLDS batches to OpenVLA format.

    This class handles the conversion of RLDS (Robotics Learning Datasets) batches
    to the format expected by OpenVLA models, including tokenization, image processing,
    and prompt construction.

    Attributes:
        action_tokenizer: Tokenizer for converting actions to tokens.
        base_tokenizer: Base tokenizer for text processing.
        image_transform: Transform for processing images.
        prompt_builder_fn: Function for building conversation prompts.
        predict_stop_token: Whether to predict stop tokens in labels.

    """

    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: dict[str, Any]) -> dict[str, Any]:
        """Convert a RLDS batch to the format expected by the OpenVLA collator/models.

        Args:
            rlds_batch: Dictionary containing RLDS data with keys like 'dataset_name',
                       'action', 'observation', and 'task'.

        Returns:
            Dictionary with processed data containing:
                - pixel_values: Transformed image tensor
                - input_ids: Tokenized input sequence
                - labels: Target labels for training
                - dataset_name: Name of the source dataset

        """
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)


class RLDSDataset(IterableDataset):
    """Iterable dataset wrapper for RLDS TFDS pipeline.

    This class provides a PyTorch-compatible interface to RLDS (Robotics Learning Datasets)
    through a TFDS pipeline, handling data loading, preprocessing, and batching for training
    OpenVLA models.
    """

    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Initialize the RLDS dataset.

        Args:
            data_root_dir: Root directory containing the dataset files.
            data_mix: Name of the data mixture or single dataset to load.
            batch_transform: Transform to apply to each batch.
            resize_resolution: Target resolution (height, width) for image resizing.
            shuffle_buffer_size: Size of the shuffle buffer for data loading.
            train: Whether to load training data (vs. validation/test).
            image_aug: Whether to apply image augmentation during training.

        """
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = {
            "traj_transform_kwargs": {
                "window_size": 1,                                      # If we wanted to feed / predict more than one step
                "future_action_window_size": 0,                        # For action chunking
                "skip_unlabeled": True,                                # Skip trajectories without language labels
                "goal_relabeling_strategy": "uniform",                 # Goals are currently unused
            },
            "frame_transform_kwargs": {
                "resize_size": resize_resolution,
                "num_parallel_calls": 16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            },
            "dataset_kwargs_list": per_dataset_kwargs,
            "shuffle_buffer_size": shuffle_buffer_size,
            "sample_weights": weights,
            "balance_weights": True,
            "traj_transform_threads": len(mixture_spec),
            "traj_read_threads": len(mixture_spec),
            "train": train,
        }

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : {
                "random_resized_crop": {"scale": [0.9, 0.9], "ratio": [1.0, 1.0]},
                "random_brightness": [0.2],
                "random_contrast": [0.8, 1.2],
                "random_saturation": [0.8, 1.2],
                "random_hue": [0.05],
                "augment_order": [
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            }}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config: dict[str, Any]) -> tuple[Any, int, dict[str, Any]]:
        """Create the RLDS dataset from configuration.

        Args:
            rlds_config: Configuration dictionary for RLDS dataset creation.

        Returns:
            Tuple containing the dataset, dataset length, and dataset statistics.

        """
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> dict[str, Any]:
        """Iterate over the dataset, yielding transformed batches.

        Yields:
            Dictionary containing transformed data batch.

        """
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            Number of samples in the dataset.

        """
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        """Raise error for map-style access.

        Args:
            idx: Index (unused).

        Raises:
            NotImplementedError: Always raised as this is an IterableDataset.

        """
        msg = "IterableDataset does not implement map-style __getitem__; see __iter__ instead!"
        raise NotImplementedError(msg)


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config: dict[str, Any]) -> tuple[Any, int, dict[str, Any]]:
        """Create a single-dataset RLDS dataset for episodic data.

        Args:
            rlds_config: Configuration dictionary for RLDS dataset creation.

        Returns:
            Tuple containing the dataset, dataset length, and dataset statistics.

        Raises:
            ValueError: If more than one dataset is specified in the configuration.

        """
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        if len(per_dataset_kwargs) != 1:
            msg = "Only support single-dataset `mixes` for episodic datasets."
            raise ValueError(msg)

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> dict[str, Any]:
        """Iterate over the dataset, yielding full episodes as lists of steps.

        Yields:
            List of dictionaries, each containing a transformed step from an episode.

        """
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    """Dummy dataset for testing OpenVLA training pipeline.

    This class provides a simple dummy dataset implementation for testing the OpenVLA
    model training pipeline without requiring actual robotics data.

    Attributes:
        action_tokenizer: Tokenizer for converting actions to tokens.
        base_tokenizer: Base tokenizer for text processing.
        image_transform: Transform for processing images.
        prompt_builder_fn: Function for building conversation prompts.
        dataset_statistics: Statistics for action de-normalization.

    """

    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: type[PromptBuilder],
    ) -> None:
        """Initialize the dummy dataset.

        Args:
            action_tokenizer: Tokenizer for converting actions to tokens.
            base_tokenizer: Base tokenizer for text processing.
            image_transform: Transform for processing images.
            prompt_builder_fn: Function for building conversation prompts.

        """
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)},
            },
        }

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            Number of samples in the dummy dataset.

        """
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single item from the dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Dictionary containing processed data with pixel_values, input_ids, and labels.

        """
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return {"pixel_values": pixel_values, "input_ids": input_ids, "labels": labels}
