"""Data utilities for handling multimodal training data.

This module provides utilities for processing and collating multimodal data
including text, images, and actions for vision-language-action model training.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Apply a function to all values in a nested dictionary.

    Recursively traverses a nested dictionary structure and applies the given
    function to all leaf values, preserving the dictionary structure.

    Args:
        fn (Callable): Function to apply to each value.
        tree (dict): Nested dictionary to process.

    Returns:
        dict: New dictionary with function applied to each value.

    Example:
        >>> tree = {"a": {"b": 1, "c": 2}, "d": 3}
        >>> tree_map(lambda x: x * 2, tree)
        {"a": {"b": 2, "c": 4}, "d": 6}

    """
    return {k: fn(v) if not isinstance(v, dict) else tree_map(fn, v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Apply a function to all values in a nested dictionary, passing the key path.

    Similar to tree_map but provides the full key path to the function,
    allowing for key-dependent transformations.

    Args:
        fn (Callable): Function to apply to each value, taking the key path and value.
            Should accept (key_path, value) where key_path is a tuple of keys.
        tree (dict): Nested dictionary to process.
        keys (Sequence, optional): Current key path (used for recursion). Defaults to ().

    Returns:
        dict: New dictionary with function applied to each value.

    Example:
        >>> tree = {"a": {"b": 1}, "c": 2}
        >>> tree_map_with_key(lambda path, val: f"{'.'.join(path)}: {val}", tree)
        {"a": {"b": "a.b: 1"}, "c": "c: 2"}

    """
    return {
        k: fn((*keys, k), v) if not isinstance(v, dict) else tree_map_with_key(fn, v, (*keys, k))
        for k, v in tree.items()
    }


@dataclass
class PaddedCollatorOutput:
    """Output data structure for padded collation.

    Contains the batched and padded tensors returned by data collators,
    suitable for feeding into multimodal models.

    Attributes:
        pixel_values (torch.Tensor | dict[str, torch.Tensor]): Batched image data.
            Can be a single tensor or dictionary of tensors for different image types.
        input_ids (torch.Tensor): Padded token IDs of shape [batch_size, seq_len].
        attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_len].
        labels (torch.Tensor): Target labels of shape [batch_size, seq_len].
        multimodal_indices (torch.Tensor | None, optional): Indices of samples
            that contain both image and text data. Defaults to None.
        dataset_names (list[torch.Tensor] | None, optional): Names of source
            datasets for each sample. Defaults to None.

    """

    pixel_values: torch.Tensor | dict[str, torch.Tensor]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    multimodal_indices: torch.Tensor | None = None
    dataset_names: list[torch.Tensor] | None = None


@dataclass
class PaddedCollatorForLanguageModeling:
    """Data collator for language modeling with optional multimodal data.

    Handles batching and padding of text sequences with optional image data,
    supporting both unimodal (text-only) and multimodal training examples.

    Attributes:
        model_max_length (int): Maximum sequence length for truncation.
        pad_token_id (int): Token ID used for padding sequences.
        default_image_resolution (tuple[int, int, int]): Default image shape (C, H, W).
        padding_side (str): Side to pad sequences ("right" or "left"). Defaults to "right".
        pixel_values_dtype (torch.dtype): Data type for pixel values. Defaults to torch.float32.
        dummy_pixel_values (torch.Tensor): Dummy tensor for unimodal examples.

    """

    model_max_length: int
    pad_token_id: int
    default_image_resolution: tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        """Initialize dummy pixel values after dataclass creation.

        Creates a zero tensor with the default image resolution to use
        as placeholder for unimodal (text-only) examples.
        """
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[dict[str, torch.Tensor]]) -> PaddedCollatorOutput:
        """Collate a batch of training instances.

        Processes a sequence of training examples by padding text sequences
        and handling both unimodal and multimodal data appropriately.

        Args:
            instances (Sequence[dict[str, torch.Tensor]]): List of training examples,
                each containing "input_ids", "labels", and optionally "pixel_values".

        Returns:
            PaddedCollatorOutput: Batched and padded data ready for model input.

        Raises:
            ValueError: If pixel_values type is not supported.

        Note:
            Examples without pixel_values (unimodal) are padded with dummy_pixel_values
            and tracked via multimodal_indices for efficient processing.

        """
        input_ids = [instances["input_ids"] for instances in instances]
        labels = [instances["labels"] for instances in instances]
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None],
            dtype=torch.long,
        )
        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ],
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ],
                )
                for k in pv_example  # pyright: ignore[reportGeneralTypeIssues]
            }
        else:
            msg = f"Unsupported `pixel_values` type = {type(pixel_values)}"
            raise ValueError(msg)
        return PaddedCollatorOutput(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices if len(multimodal_indices) > 0 else None,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    """Data collator for action prediction tasks.

    Specialized collator for vision-language-action (VLA) training that requires
    all examples to have both visual and textual components. Handles batching
    and padding for action prediction datasets.

    Attributes:
        model_max_length (int): Maximum sequence length for truncation.
        pad_token_id (int): Token ID used for padding sequences.
        padding_side (str): Side to pad sequences. Defaults to "right".
        pixel_values_dtype (torch.dtype): Data type for pixel values. Defaults to torch.float32.

    """

    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[dict[str, torch.Tensor]]) -> PaddedCollatorOutput:
        """Collate a batch of VLA training instances.

        Processes a sequence of VLA training examples, ensuring all examples
        have visual components and properly batching multimodal data.

        Args:
            instances (Sequence[dict[str, torch.Tensor]]): List of VLA training examples,
                each containing "input_ids", "labels", "pixel_values", and optionally
                "dataset_name".

        Returns:
            PaddedCollatorOutput: Batched and padded VLA data ready for model input.

        Raises:
            ValueError: If padding_side is not "right" or if any example lacks pixel_values.
            TypeError: If pixel_values type is not supported.

        Note:
            Unlike the language modeling collator, this requires all examples to be
            multimodal (have pixel_values) as VLA training doesn't support text-only data.

        """
        input_ids = [instances["input_ids"] for instances in instances]
        labels = [instances["labels"] for instances in instances]
        pixel_values = [instance["pixel_values"] for instance in instances]
        dataset_names = [instance["dataset_name"] for instance in instances] if "dataset_name" in instances[0] else None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        if self.padding_side != "right":
            msg = f"Only `padding_side = 'right'` is supported during training, got `{self.padding_side = }`"
            raise ValueError(msg)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        if not all(pv is not None for pv in pixel_values):
            msg = "All examples must have `pixel_values` for VLA training!"
            raise ValueError(msg)

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))])
                for k in pixel_values[0]  # pyright: ignore[reportGeneralTypeIssues]
            }
        else:
            msg = f"Unsupported `pixel_values` type = {type(pixel_values)}"
            raise TypeError(msg)

        output = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return PaddedCollatorOutput(**output)
