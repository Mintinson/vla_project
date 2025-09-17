"""Dataset configuration definitions for VLA project.

This module defines dataset configurations for various vision-language training datasets,
primarily focused on LLaVA variants and related datasets. It provides a registry system
for dataset configurations and standardized access to dataset components.

The module includes:
    - Base dataset configuration class with common structure
    - Specific configurations for LLaVA v1.5 and its variants
    - Dataset registry enum for validation and reference
    - Support for multi-stage training (alignment and fine-tuning)

Example:
    Basic usage for accessing dataset configurations:

    ```python
    from vla_project.config.dataset_cfg import DatasetRegistry, DatasetConfig

    # Get a specific dataset configuration
    llava_config = DatasetRegistry.LLAVA_V15.value()

    # Access dataset components
    align_json, align_images = llava_config.align_stage_components
    finetune_json, finetune_images = llava_config.finetune_stage_components

    # Use with dataset registry
    config = DatasetConfig.get_choice_class("llava-v15")()
    ```
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path

from draccus import ChoiceRegistry

DATASET_ROOT_DIR = "/mnt/fsx/skaramcheti/datasets/prismatic-vlms"
"""Default root directory path for all datasets.

This constant defines the base path where all dataset files are expected to be stored.
Individual dataset configurations specify paths relative to this root directory.
"""


@dataclass
class DatasetConfig(ChoiceRegistry):
    """Base configuration class for vision-language datasets.

    This class serves as the base configuration for all dataset variants used in the VLA project.
    It provides a common structure for dataset definitions with support for multi-stage training
    (alignment and fine-tuning stages) and integrates with the ChoiceRegistry system for
    configuration management.

    Attributes:
        dataset_id: Unique identifier that fully specifies a dataset variant.
        align_stage_components: Tuple containing (annotation_file_path, images_directory_path)
            for the alignment training stage.
        finetune_stage_components: Tuple containing (annotation_file_path, images_directory_path)
            for the fine-tuning training stage.
        dataset_root_dir: Root directory path where all dataset files are stored. Other paths
            are specified relative to this root directory.

    Note:
        - All path attributes should be specified relative to dataset_root_dir
        - The class inherits from ChoiceRegistry to enable automatic registration
        - Subclasses should provide concrete values for all abstract attributes
    """

    dataset_id: str  # Unique ID that fully specifies a dataset variant

    # Dataset Components for each Stage in < align | finetune >
    align_stage_components: tuple[Path, Path]  # Path to annotation file and images directory for `align` stage
    finetune_stage_components: tuple[Path, Path]  # Path to annotation file and images directory for `finetune` stage

    dataset_root_dir: Path  # Path to dataset root directory; others paths are relative to root


# [Reproduction] LLaVa-v15 (exact dataset used in all public LLaVa-v15 models)
@dataclass
class LLaVa_V15_Config(DatasetConfig):
    """Configuration for the standard LLaVA v1.5 dataset.

    This configuration reproduces the exact dataset used in all public LLaVA v1.5 models.
    It includes both the alignment stage (LAION-CC-SBU-558K) and fine-tuning stage
    (LLaVA v1.5 Mix 665K) components.

    The dataset consists of:
        - Alignment: 558K image-caption pairs for vision-language alignment
        - Fine-tuning: 665K instruction-following examples with mixed data sources

    Attributes:
        dataset_id: Set to "llava-v15" for this configuration.
        align_stage_components: Points to LAION-CC-SBU-558K chat annotations and images.
        finetune_stage_components: Points to LLaVA v1.5 Mix 665K annotations and images.
        dataset_root_dir: Inherits from DATASET_ROOT_DIR constant.
    """

    dataset_id: str = "llava-v15"

    align_stage_components: tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_mix665k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path(DATASET_ROOT_DIR)


# [Multimodal-Only] LLava-v15 WITHOUT the Language-Only ShareGPT Data (No Co-Training)
@dataclass
class LLaVa_Multimodal_Only_Config(DatasetConfig):
    """Configuration for LLaVA v1.5 dataset without language-only ShareGPT data.

    This configuration provides the LLaVA v1.5 dataset excluding language-only ShareGPT
    data for training without co-training on pure text data. It focuses solely on
    multimodal examples for vision-language training.

    The dataset consists of:
        - Alignment: Same 558K image-caption pairs as standard LLaVA v1.5
        - Fine-tuning: Stripped 625K examples without ShareGPT language-only data

    Attributes:
        dataset_id: Set to "llava-multimodal" for this configuration.
        align_stage_components: Points to LAION-CC-SBU-558K chat annotations and images.
        finetune_stage_components: Points to stripped LLaVA v1.5 annotations without ShareGPT.
        dataset_root_dir: Inherits from DATASET_ROOT_DIR constant.
    """

    dataset_id: str = "llava-multimodal"

    align_stage_components: tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_stripped625k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path(DATASET_ROOT_DIR)


# LLaVa-v15 + LVIS-Instruct-4V
@dataclass
class LLaVa_LVIS4V_Config(DatasetConfig):
    """Configuration for LLaVA v1.5 dataset enhanced with LVIS-Instruct-4V data.

    This configuration extends the standard LLaVA v1.5 dataset with LVIS-Instruct-4V
    data for improved object detection and visual reasoning capabilities. The LVIS-Instruct-4V
    dataset provides additional instruction-following examples focused on object-level understanding.

    The dataset consists of:
        - Alignment: Same 558K image-caption pairs as standard LLaVA v1.5
        - Fine-tuning: 888K examples including original LLaVA v1.5 + LVIS-Instruct-4V data

    Attributes:
        dataset_id: Set to "llava-lvis4v" for this configuration.
        align_stage_components: Points to LAION-CC-SBU-558K chat annotations and images.
        finetune_stage_components: Points to LLaVA v1.5 + LVIS-4V mixed annotations.
        dataset_root_dir: Inherits from DATASET_ROOT_DIR constant.
    """

    dataset_id: str = "llava-lvis4v"

    align_stage_components: tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_mix888k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path(DATASET_ROOT_DIR)


# LLaVa-v15 + LRV-Instruct
@dataclass
class LLaVa_LRV_Config(DatasetConfig):
    """Configuration for LLaVA v1.5 dataset enhanced with LRV-Instruct data.

    This configuration extends the standard LLaVA v1.5 dataset with LRV-Instruct
    (Language-guided Robust Visual instruction) data for improved robustness and
    better handling of complex visual reasoning tasks.

    The dataset consists of:
        - Alignment: Same 558K image-caption pairs as standard LLaVA v1.5
        - Fine-tuning: 1008K examples including original LLaVA v1.5 + LRV-Instruct data

    Attributes:
        dataset_id: Set to "llava-lrv" for this configuration.
        align_stage_components: Points to LAION-CC-SBU-558K chat annotations and images.
        finetune_stage_components: Points to LLaVA v1.5 + LRV mixed annotations.
        dataset_root_dir: Inherits from DATASET_ROOT_DIR constant.
    """

    dataset_id: str = "llava-lrv"

    align_stage_components: tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lrv_mix1008k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path(DATASET_ROOT_DIR)


# LLaVa-v15 + LVIS-Instruct-4V + LRV-Instruct
@dataclass
class LLaVa_LVIS4V_LRV_Config(DatasetConfig):
    """Configuration for LLaVA v1.5 dataset enhanced with both LVIS-Instruct-4V and LRV-Instruct data.

    This configuration extends the standard LLaVA v1.5 dataset with both LVIS-Instruct-4V
    and LRV-Instruct data for comprehensive improvement in object detection, visual reasoning,
    and robustness. This represents the most complete dataset variant with all enhancement data.

    The dataset consists of:
        - Alignment: Same 558K image-caption pairs as standard LLaVA v1.5
        - Fine-tuning: 1231K examples including original LLaVA v1.5 + LVIS-4V + LRV data

    Attributes:
        dataset_id: Set to "llava-lvis4v-lrv" for this configuration.
        align_stage_components: Points to LAION-CC-SBU-558K chat annotations and images.
        finetune_stage_components: Points to LLaVA v1.5 + LVIS-4V + LRV mixed annotations.
        dataset_root_dir: Inherits from DATASET_ROOT_DIR constant.
    """

    dataset_id: str = "llava-lvis4v-lrv"

    align_stage_components: tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_mix1231k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path(DATASET_ROOT_DIR)


# === Define a Dataset Registry Enum for Reference & Validation =>> all *new* datasets must be added here! ===
@unique
class DatasetRegistry(Enum):
    """Registry of all available dataset configurations for validation and reference.

    This enum provides a centralized registry of all dataset configurations used in the VLA project.
    Each enum value corresponds to a specific dataset configuration class, enabling type-safe
    access to dataset configurations and validation of dataset IDs.

    The registry includes:
        - LLaVA v1.5 variants with different enhancement datasets
        - Multimodal-only configurations without language-only data
        - Combined configurations with multiple enhancement datasets

    Attributes:
        LLAVA_V15: Standard LLaVA v1.5 dataset configuration
        LLAVA_MULTIMODAL_ONLY: LLaVA v1.5 without ShareGPT language-only data
        LLAVA_LVIS4V: LLaVA v1.5 enhanced with LVIS-Instruct-4V
        LLAVA_LRV: LLaVA v1.5 enhanced with LRV-Instruct
        LLAVA_LVIS4V_LRV: LLaVA v1.5 enhanced with both LVIS-4V and LRV-Instruct

    Note:
        All new dataset configurations must be added to this registry to be
        recognized by the system.
    """

    # === LLaVa v1.5 ===
    LLAVA_V15 = LLaVa_V15_Config

    LLAVA_MULTIMODAL_ONLY = LLaVa_Multimodal_Only_Config

    LLAVA_LVIS4V = LLaVa_LVIS4V_Config
    LLAVA_LRV = LLaVa_LRV_Config

    LLAVA_LVIS4V_LRV = LLaVa_LVIS4V_LRV_Config

    @property
    def dataset_id(self) -> str:
        """Get the dataset ID for this registry entry.

        Returns:
            The dataset_id string from the corresponding configuration class.

        Example:
            >>> registry_entry = DatasetRegistry.LLAVA_V15
            >>> print(registry_entry.dataset_id)
            "llava-v15"
        """
        return self.value.dataset_id


# Register Datasets in Choice Registry
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)
    """Register each dataset configuration with the ChoiceRegistry system.
    
    This loop automatically registers all dataset configurations from the DatasetRegistry
    with the ChoiceRegistry system, enabling them to be accessed via their dataset_id
    strings. This allows for dynamic configuration selection based on string identifiers.
    
    The registration enables usage patterns like:
        config = DatasetConfig.get_choice_class("llava-v15")()
    """
