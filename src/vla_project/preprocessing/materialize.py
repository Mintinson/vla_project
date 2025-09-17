"""Dataset materialization utilities for VLA project.

This module provides functionality to create and configure datasets and data collators
for different training stages (align, finetune, full-finetune) of vision-language models.
It acts as a factory for dataset creation based on configuration and training stage.

The module supports:
    - Alignment stage datasets for vision-language alignment training
    - Finetune stage datasets for instruction following training
    - Full finetune stage datasets for comprehensive fine-tuning

Example:
    Basic usage for creating a dataset and collator:

    ```python
    from vla_project.preprocessing.materialize import get_dataset_and_collator

    dataset, collator = get_dataset_and_collator(
        stage="finetune",
        dataset_cfg=dataset_config,
        image_transform=transform,
        tokenizer=tokenizer,
        prompt_builder_fn=PromptBuilder,
        default_image_resolution=(3, 224, 224)
    )
    ```
"""

from typing import cast

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from vla_project.config import DatasetConfig
from vla_project.models.backbones.llm.prompting.base_prompter import PromptBuilder
from vla_project.models.backbones.vision.vision_base import ImageTransform
from vla_project.preprocessing.datasets import AlignDataset, FinetuneDataset
from vla_project.utils.data_utils import PaddedCollatorForLanguageModeling

# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {"align": AlignDataset, "finetune": FinetuneDataset, "full-finetune": FinetuneDataset}
"""Dictionary mapping training stages to their corresponding dataset classes.

This mapping defines which dataset class should be used for each training stage:
    - "align": AlignDataset for vision-language alignment training
    - "finetune": FinetuneDataset for instruction following fine-tuning
    - "full-finetune": FinetuneDataset for comprehensive fine-tuning

Keys:
    Training stage identifier string.
    
Values:
    Dataset class constructor for the corresponding stage.
"""


def get_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: type[PromptBuilder],
    default_image_resolution: tuple[int, int, int],
    padding_side: str = "right",
) -> tuple[Dataset, PaddedCollatorForLanguageModeling]:
    """Create and configure a dataset and data collator for the specified training stage.

    This function serves as a factory for creating appropriate dataset instances and
    data collators based on the training stage (align, finetune, or full-finetune).
    It handles the stage-specific configuration and initialization logic.

    Args:
        stage: The training stage identifier. Must be one of "align", "finetune", or "full-finetune".
        dataset_cfg: Configuration object containing dataset paths and settings.
        image_transform: Transform function/callable for preprocessing images.
        tokenizer: Pre-trained tokenizer for text processing.
        prompt_builder_fn: Class constructor for building prompts (used in finetune stages).
        default_image_resolution: Default image resolution as (channels, height, width).
        padding_side: Side to pad sequences on, either "left" or "right". Defaults to "right".

    Returns:
        A tuple containing:
            - Dataset: The configured dataset instance for the specified stage
            - PaddedCollatorForLanguageModeling: The configured data collator

    Raises:
        ValueError: If the specified stage is not supported.
        KeyError: If the stage is not found in DATASET_INITIALIZER.

    Example:
        >>> from pathlib import Path
        >>> from vla_project.config import DatasetConfig
        >>>
        >>> # Create dataset and collator for alignment stage
        >>> dataset, collator = get_dataset_and_collator(
        ...     stage="align",
        ...     dataset_cfg=align_config,
        ...     image_transform=image_transform,
        ...     tokenizer=tokenizer,
        ...     prompt_builder_fn=PromptBuilder,
        ...     default_image_resolution=(3, 224, 224)
        ... )
        >>>
        >>> # Create dataset and collator for finetuning stage
        >>> dataset, collator = get_dataset_and_collator(
        ...     stage="finetune",
        ...     dataset_cfg=finetune_config,
        ...     image_transform=image_transform,
        ...     tokenizer=tokenizer,
        ...     prompt_builder_fn=PromptBuilder,
        ...     default_image_resolution=(3, 384, 384),
        ...     padding_side="left"
        ... )

    Note:
        - The "align" stage creates datasets for vision-language alignment training
        - The "finetune" and "full-finetune" stages both use FinetuneDataset but may
          have different configurations in practice
        - The collator is configured with the tokenizer's properties and image resolution
    """
    dataset_cls = DATASET_INITIALIZER[stage]
    dataset_root_dir = dataset_cfg.dataset_root_dir
    collator = PaddedCollatorForLanguageModeling(
        model_max_length=tokenizer.model_max_length,
        pad_token_id=cast("int", tokenizer.pad_token_id),
        default_image_resolution=default_image_resolution,
        padding_side=padding_side,
    )

    # Switch on `stage`
    if stage == "align":
        annotation_json, image_dir = dataset_cfg.align_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json, dataset_root_dir / image_dir, image_transform, tokenizer
        )
        return dataset, collator

    elif stage == "finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    elif stage == "full-finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    else:
        raise ValueError(f"Stage `{stage}` is not supported!")
