"""Dataset classes for vision-language model training.

This module provides dataset implementations for different training phases
including alignment and fine-tuning of vision-language models.
"""

import copy
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CodeGenTokenizerFast, LlamaTokenizerFast, PreTrainedTokenizerBase

from vla_project.models.backbones.llm.prompting.base_prompter import PromptBuilder
from vla_project.models.backbones.vision.vision_base import ImageTransform

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

class VisionLanguageDataset[T](Dataset[T], ABC):
    """Base class for vision-language datasets."""

    def __len__(self) -> int: ...

    @abstractmethod
    def get_modality_lengths(self) -> list[tuple[bool, int]]: ...


class AlignDataset(VisionLanguageDataset[dict[str, torch.Tensor | dict[str, torch.Tensor]]]):
    """Dataset for vision-language model alignment training.

    This dataset is designed for the alignment phase of vision-language model training,
    where the model learns to associate visual features with textual descriptions.
    It processes image-caption pairs from a JSON file and formats them for training.

    The dataset follows the LLaVa approach where during alignment, the model directly
    predicts captions from images without using the human prompt, focusing on learning
    the vision-language correspondence.

    Attributes:
        chat_json (Path): Path to JSON file containing conversations and image paths.
        image_dir (Path): Directory containing the image files.
        image_transform (ImageTransform): Transform pipeline for processing images.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for processing text.
        dataset_type (str): Type identifier for this dataset ("align").
        prompt_template (str): Template for formatting captions with EOS token.
        examples (list): Loaded conversation examples from JSON file.

    """

    def __init__(
        self,
        chat_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Initialize the AlignDataset.

        Args:
            chat_json (Path): Path to JSON file containing conversation data.
                Expected format: list of dicts with "image" and "conversations" keys.
            image_dir (Path): Directory containing image files referenced in chat_json.
            image_transform (ImageTransform): Image preprocessing pipeline.
            tokenizer (PreTrainedTokenizerBase): Tokenizer for text processing.

        """
        super().__init__()
        self.chat_json, self.image_dir = chat_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.dataset_type = "align"

        # Create Prompt Template
        self.prompt_template = "{caption}" + cast("str", self.tokenizer.eos_token)

        # Load Chat JSON
        with Path(chat_json).open("r") as f:
            self.examples = json.load(f)

    def __len__(self) -> int:
        """Get the number of examples in the dataset.

        Returns:
            int: Number of conversation examples in the dataset.

        """
        return len(self.examples)

    def get_modality_lengths(self, n_image_patches: int) -> list[tuple[bool, int]]:
        """Compute modality information and sequence lengths for each example.

        Analyzes each example to determine if it's multimodal (contains images)
        and calculates the total sequence length including both text tokens
        and image patches.

        Args:
            n_image_patches (int): Number of patches/tokens that each image
                will be tokenized into by the vision encoder.

        Returns:
            list[tuple[bool, int]]: List of tuples where each tuple contains:
                - bool: True if example is multimodal (has image), False otherwise
                - int: Total sequence length (text tokens + image patches if multimodal)

        Note:
            This information is used by samplers for efficient batching based on
            sequence length and modality type.

        """
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].replace("<image>", "").split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, (n_image_patches + n_words) if is_multimodal else n_words))
        return modality_lengths

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        r"""Retrieve and process a single training example.

        Processes an image-caption pair for alignment training. Following the LLaVa
        approach, this method discards the human prompt and formats the example to
        directly predict the caption from the image.

        The processing pipeline:
        1. Extracts the caption from the conversation (GPT response)
        2. Formats it with the EOS token
        3. Tokenizes the caption
        4. Creates labels (copying input_ids but ignoring BOS and image patch positions)
        5. Processes the image through the vision transform

        Args:
            idx (int): Index of the example to retrieve.

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: Dictionary containing:
                - "pixel_values": Processed image tensor or dict of tensors
                - "input_ids": Tokenized caption including BOS token
                - "labels": Target labels with IGNORE_INDEX for BOS token

        Raises:
            ValueError: If conversation doesn't have exactly 2 turns or if the
                last turn contains an <image> tag (which shouldn't happen in captions).

        Example:
            Given conversation:
            [
                {"from": "human", "value": "Describe this image.\n<image>"},
                {"from": "gpt", "value": "A red car on a street"}
            ]

            Returns tokenized version of: "A red car on a street</s>"

        Note:
            Image patches are conceptually inserted after the BOS token during
            model forward pass. The BOS token's label is set to IGNORE_INDEX
            since it will be replaced by image patch representations.

        """
        image_path, conversation = Path(self.examples[idx]["image"]), self.examples[idx]["conversations"]
        if len(conversation) != 2 or "<image>" not in conversation[-1]["value"]:  # noqa: PLR2004
            msg = "Each conversation must have exactly two turns and the last turn must not contain <image> tag!"
            raise ValueError(msg)

        # Format Caption --> {caption}{eos_token}
        caption = self.prompt_template.format(caption=conversation[-1]["value"].strip())

        # We treat image patches as "tokens = [p1 p2 p3, ...]"; we need to specify ordering of text/patch tokens.
        #   => Critically, we find that inserting *after* the BOS token leads to the strongest performance!
        #       - input_ids = "<s> p1 p2 p3 ... <caption_text> \n"
        #       - labels = "IGNORE IGNORE ..." (copy `input_ids` replacing <s> and p{1...K} with IGNORE)
        #
        # IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)

        # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
        labels[0] = IGNORE_INDEX

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
        pixel_values = self.image_transform(
            transforms.ToTensor()(Image.open(self.image_dir / image_path).convert("RGB")),
        )

        return {"pixel_values": pixel_values, "input_ids": input_ids, "labels": labels}


class FinetuneDataset(VisionLanguageDataset[dict[str, torch.Tensor | dict[str, torch.Tensor]]]):
    """Dataset for vision-language model fine-tuning with multi-turn conversations.

    This dataset handles the fine-tuning phase of vision-language model training,
    supporting multi-turn conversational data that may be either unimodal (text-only)
    or multimodal (text + image). It processes instruction-following conversations
    and formats them for training with proper prompt structure.

    Unlike the alignment phase, fine-tuning involves complete conversations with
    multiple turns, where only the assistant responses are used for loss computation
    while user prompts are ignored.

    Attributes:
        instruct_json (Path): Path to JSON file containing instruction conversations.
        image_dir (Path): Directory containing image files referenced in conversations.
        image_transform (ImageTransform): Transform pipeline for processing images.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for processing text.
        prompt_builder_fn (type[PromptBuilder]): Class for building conversation prompts.
        dataset_type (str): Type identifier for this dataset ("finetune").
        examples (list): Loaded conversation examples from JSON file.

    """

    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: type[PromptBuilder],
    ) -> None:
        """Initialize the FinetuneDataset.

        Args:
            instruct_json (Path): Path to JSON file containing instruction conversation data.
                Expected format: list of dicts with "conversations" and optionally "image" keys.
            image_dir (Path): Directory containing image files referenced in instruct_json.
            image_transform (ImageTransform): Image preprocessing pipeline.
            tokenizer (PreTrainedTokenizerBase): Tokenizer for text processing.
            prompt_builder_fn (type[PromptBuilder]): PromptBuilder class for formatting conversations.

        """
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"

        # Load Instruct JSON
        with Path(self.instruct_json).open("r") as f:
            self.examples = json.load(f)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        r"""Retrieve and process a single fine-tuning conversation example.

        Processes multi-turn conversations for instruction fine-tuning. Uses the prompt
        builder to format conversations properly, ensuring only assistant responses
        contribute to the loss while user prompts are ignored.

        The processing pipeline:
        1. Iterates through conversation turns using the prompt builder
        2. Tokenizes each turn with appropriate special token handling
        3. Sets labels to IGNORE_INDEX for user turns, keeps original tokens for assistant turns
        4. Handles both unimodal (text-only) and multimodal (text + image) examples
        5. Processes images if present and sets BOS token label to IGNORE_INDEX

        Args:
            idx (int): Index of the conversation example to retrieve.

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: Dictionary containing:
                - "pixel_values": Processed image tensor/dict or None for unimodal examples
                - "input_ids": Tokenized conversation sequence
                - "labels": Target labels with IGNORE_INDEX for user prompts and BOS token

        Raises:
            TypeError: If tokenizer type is not explicitly handled.

        Example:
            Given conversation:
            [
                {"from": "human", "value": "What's in this image?\n<image>"},
                {"from": "gpt", "value": "A cat sitting on a table"},
                {"from": "human", "value": "What color is the cat?"},
                {"from": "gpt", "value": "The cat is orange"}
            ]

            Only the GPT responses will have their tokens included in loss computation.

        Note:
            - User prompts (even indices) have labels set to IGNORE_INDEX
            - Assistant responses (odd indices) use original token IDs as labels
            - For multimodal examples, BOS token is ignored since image patches are inserted after it

        """
        conversation = self.examples[idx]["conversations"]

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="prismatic"), [], []
        for turn_idx, turn in enumerate(conversation):
            # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
            msg = prompt_builder.add_turn(turn["from"], turn["value"])

            # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
            if isinstance(self.tokenizer, LlamaTokenizerFast):
                msg = msg.rstrip()

            # Phi-2 Tokenizer == CodeGenTokenizer (Fast) -- no special handling!
            elif isinstance(self.tokenizer, CodeGenTokenizerFast):
                pass

            else:
                msg = f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!"
                raise TypeError(msg)

            # Tokenize Input IDs
            turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids

            # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
            turn_labels = (
                [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
            )

            # Add to Trackers
            input_ids.extend(turn_input_ids)
            labels.extend(turn_labels)

        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Handle Truncation (if necessary)
        input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image" in self.examples[idx]:
            image_path = Path(self.examples[idx]["image"])

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            labels[0] = IGNORE_INDEX

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            pixel_values = self.image_transform(
                transforms.ToTensor()(Image.open(self.image_dir / image_path).convert("RGB")),
            )

            return {"pixel_values": pixel_values, "input_ids": input_ids, "labels": labels}

        # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
        return {"pixel_values": None, "input_ids": input_ids, "labels": labels}

    def get_modality_lengths(self) -> list[tuple[bool, int]]:
        """Compute modality information and conversation lengths for each example.

        Analyzes each conversation to determine if it's multimodal (contains images)
        and calculates the total conversation length in words.

        Returns:
            list[tuple[bool, int]]: List of tuples where each tuple contains:
                - bool: True if example is multimodal (has image), False for unimodal
                - int: Total word count across all conversation turns

        Note:
            This information is used by samplers for efficient batching based on
            sequence length and modality type. Unlike alignment, this doesn't
            include image patch counts since those are added separately.

        """
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        """Get the number of conversation examples in the dataset.

        Returns:
            int: Number of conversation examples in the dataset.

        """
        return len(self.examples)
