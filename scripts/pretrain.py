r"""Pretraining script for Prismatic Vision-Language Models.

This script implements distributed training for Prismatic VLMs using PyTorch's Fully-Sharded
Data Parallel (FSDP) to scale across multiple GPUs. It supports both alignment and fine-tuning
stages for vision-language model training.

The script handles the complete training pipeline including:
    - Model instantiation (vision backbone + LLM backbone)
    - Dataset loading and preprocessing
    - Distributed training setup with FSDP
    - Checkpoint management and logging
    - Integration with Weights & Biases for experiment tracking

Key Features:
    - Multi-stage training: alignment (projector-only) and fine-tuning (projector + LLM)
    - Support for gated models from HuggingFace Hub (e.g., LLaMA-2)
    - Mixed precision training with BF16 support
    - Gradient checkpointing for memory efficiency
    - Comprehensive logging and metric tracking

Prerequisites:
    - CUDA toolkit >= 11.0 for BF16 mixed precision support
    - HuggingFace authentication token for gated models
    - Proper distributed training setup for multi-GPU scenarios

Usage Examples:
    Single GPU debugging:
        torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py

    Multi-GPU training:
        torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py

    Custom configuration:
        torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py \\
            --model.model_id "prism-dinosiglip-controlled-7b" \\
            --stage "finetune" \\
            --seed 42

Environment Variables:
    HF_HOME: Custom location for HuggingFace/TIMM artifacts cache
    TOKENIZERS_PARALLELISM: Set to "false" for PyTorch DataLoader compatibility

Notes:
    For LLaMA-2 and other gated models:
    1. Get Meta approval for LLaMA-2 access
    2. Generate HuggingFace access token with "Read" permissions
    3. Set cfg.hf_token to token file path or environment variable

    Link: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import torch
import torch.distributed as dist
import yaml

from vla_project.config import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from vla_project.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
from vla_project.overwatch import initialize_overwatch
from vla_project.preprocessing import get_dataset_and_collator
from vla_project.training import Metrics, get_train_strategy
from vla_project.utils import set_global_seed

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class PretrainConfig:
    """Configuration dataclass for Prismatic VLM pretraining.

    This configuration class manages all hyperparameters and settings for training
    Prismatic Vision-Language Models. It supports different training stages (align,
    finetune) and automatically configures optimization parameters based on the
    selected stage.

    Attributes:
        model: Model configuration specifying architecture, backbones, and training settings.
        dataset: Dataset configuration for data loading and preprocessing.
        stage: Training stage, either "align" (projector-only) or "finetune" (projector + LLM).
        pretrained_checkpoint: Path to pretrained checkpoint for fine-tuning stage.
        run_id: Unique identifier for this training run.
        run_root_dir: Root directory for storing logs and checkpoints.
        seed: Random seed for reproducibility.
        hf_token: HuggingFace authentication token for gated models.
        trackers: Tuple of tracking systems to use (e.g., "jsonl", "wandb").
        wandb_project: Weights & Biases project name for experiment tracking.
        wandb_entity: Weights & Biases entity/organization name.

    Example:
        >>> cfg = PretrainConfig(
        ...     stage="finetune",
        ...     run_id="my_vlm_experiment",
        ...     seed=42
        ... )
        >>> # Configuration will automatically set optimization parameters for finetune stage

    """

    # fmt: off

    # ModelConfig (`prismatic/conf/models.py`); override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.PRISM_DINOSIGLIP_CONTROLLED_7B.model_id),
    )

    # DatasetConfig (`prismatic/conf/datasets.py`); override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.LLAVA_V15.dataset_id),
    )

    # Pretraining Stage in < align (projector-only) | finetune (projector + LLM) | full-finetune (all) >
    # ---
    stage: str = "finetune"                                         # Pretraining Stage in < align | finetune >
    pretrained_checkpoint: Path | None = None                    # Pretrained Checkpoint to Load (for `finetune`)
                                                                    #   if None =>> will match on (run_dir / `align`)

    # Run Arguments
    run_id: str | None = None                                    # Run ID for logging, Weights & Biases
    run_root_dir: Path = Path("runs")     # Path to directory to store logs & checkpoints
    seed: int = 7                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: str | Path = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "onyx-vlms"                                # Name of W&B project (default: `prismatic`)
    wandb_entity: str | None = "stanford-voltron"                # Name of W&B entity (default: None)

    def __post_init__(self) -> None:
        """Configure optimization parameters based on the training stage.

        Automatically sets learning rate, batch size, scheduler, and other optimization
        parameters based on the specified training stage. This ensures stage-appropriate
        hyperparameters are used without manual configuration.

        The method configures the following parameters for each stage:
            - Learning rate and weight decay
            - Batch sizes (global and per-device)
            - Learning rate scheduler type and warmup ratio
            - Maximum gradient norm for clipping
            - Training strategy (e.g., FSDP configuration)
            - Number of epochs and maximum steps

        Raises:
            ValueError: If the specified stage is not supported.

        Note:
            This method is called automatically after dataclass initialization
            and modifies the instance's optimization parameters in-place.

        """
        if self.stage == "align":
            self.epochs = self.model.align_epochs
            self.max_steps = self.model.align_max_steps
            self.global_batch_size = self.model.align_global_batch_size
            self.per_device_batch_size = self.model.align_per_device_batch_size

            self.learning_rate = self.model.align_learning_rate
            self.weight_decay = self.model.align_weight_decay
            self.max_grad_norm = self.model.align_max_grad_norm
            self.lr_scheduler_type = self.model.align_lr_scheduler_type
            self.warmup_ratio = self.model.align_warmup_ratio

            self.train_strategy = self.model.align_train_strategy

        elif self.stage.endswith("finetune"):
            self.epochs = self.model.finetune_epochs
            self.max_steps = self.model.finetune_max_steps
            self.global_batch_size = self.model.finetune_global_batch_size
            self.per_device_batch_size = self.model.finetune_per_device_batch_size

            self.learning_rate = self.model.finetune_learning_rate
            self.weight_decay = self.model.finetune_weight_decay
            self.max_grad_norm = self.model.finetune_max_grad_norm
            self.lr_scheduler_type = self.model.finetune_lr_scheduler_type
            self.warmup_ratio = self.model.finetune_warmup_ratio

            self.train_strategy = self.model.finetune_train_strategy

        else:
            msg = f"Stage `{self.stage}` is not supported!"
            raise ValueError(msg)

    # fmt: on


@draccus.wrap()
def pretrain(cfg: PretrainConfig) -> None:
    r"""Execute the complete Prismatic VLM training pipeline.

    This function orchestrates the entire training process for Prismatic Vision-Language
    Models, including model initialization, dataset preparation, distributed training
    setup, and experiment tracking. It supports both alignment and fine-tuning stages
    with automatic parameter configuration.

    The training pipeline includes:
        1. Distributed training setup with FSDP
        2. Vision and language backbone loading
        3. VLM model instantiation and checkpoint loading
        4. Dataset preparation with appropriate transforms
        5. Training strategy initialization
        6. Metrics tracking and logging setup
        7. Main training loop execution

    Args:
        cfg: Configuration object containing all training parameters, model settings,
            dataset configuration, and experiment tracking options.

    Raises:
        RuntimeError: If distributed training setup fails.
        FileNotFoundError: If required checkpoint files or tokens are missing.
        ValueError: If configuration parameters are invalid.

    Note:
        This function is designed to run under `torchrun` for distributed training.
        It automatically handles GPU device assignment, process group initialization,
        and cleanup.

    Example:
        Run with torchrun for distributed training:
        ```bash
        torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py \\
            --stage "finetune" \\
            --model.model_id "prism-dinosiglip-controlled-7b" \\
            --seed 42
        ```

    """
    overwatch.info("Prismatic VLM Training :: Gathering Light")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())  # pyright: ignore[reportAttributeAccessIssue]
    torch.cuda.empty_cache()

    # Create Unique Run Name & Save Directory
    model_id = cfg.model.model_id
    if (dataset_id := cfg.dataset.dataset_id) == "llava-v15":
        cfg.run_id = f"{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id
    else:
        cfg.run_id = f"{dataset_id}+{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Life is like a prism; what you see depends on how you turn the glass."', ctx_level=1)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    Path(run_dir := cfg.run_root_dir / cfg.run_id).mkdir(parents=True, exist_ok=True)
    Path(run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if overwatch.is_rank_zero():
        # Additionally save a JSON version of the config
        draccus.dump(cfg, (run_dir / "config.yaml").open("w"))
        with (run_dir / "config.yaml").open() as f_yaml, (run_dir / "config.json").open("w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load Vision Backbone --> on CPU, in Full Precision (initializing model, image_transform via TIMM)
    overwatch.info(f"Loading Vision Backbone [bold]{cfg.model.vision_backbone_id}[/] via TIMM ")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id,
        image_resize_strategy=cfg.model.image_resize_strategy,
    )

    # Load LLM Backbone --> on CPU, in Full Precision (initializing Tokenizer + handling special tokens if necessary)
    overwatch.info(f"Loading Pretrained LLM [bold]{cfg.model.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id,
        llm_max_length=cfg.model.llm_max_length,
        hf_token=hf_token,
    )

    # Create VLM => wraps `vision_backbone` and `llm`
    overwatch.info(f"Instantiating PrismaticVLM `{model_id}` for Training Stage = `{cfg.stage}`")
    vlm = get_vlm(
        model_id,
        cfg.model.arch_specifier,  # pyright: ignore[reportArgumentType]
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
    )

    # [Explicit] Call to `freeze_backbones` here for clarity => will log exactly what is frozen / what's not!
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.freeze_backbones(cfg.stage)

    # Load Weights from Checkpoint (depends on stage, config)
    overwatch.info(f"Invoking `VLM.load_checkpoint()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=cfg.pretrained_checkpoint)

    # Get Dataset for Specified Stage
    overwatch.info(f"Creating Dataset `{cfg.dataset.dataset_id}` => Stage: `{cfg.stage}`")
    train_dataset, collator = get_dataset_and_collator(
        cfg.stage,
        cfg.dataset,
        image_transform,
        tokenizer,
        prompt_builder_fn=llm_backbone.prompt_builder_fn,
        default_image_resolution=vision_backbone.default_image_resolution,
        padding_side=tokenizer.padding_side,
    )

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vlm,
        device_id=device_id,
        stage=cfg.stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.model.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.model.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(train_dataset))  # pyright: ignore[reportArgumentType]

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = Metrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        cfg.stage,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
    )

    # Run Training
    overwatch.info("Starting Training Loop")
    train_strategy.run_training(train_dataset, collator, metrics, stage=cfg.stage, seed=cfg.seed)  # pyright: ignore[reportArgumentType]

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    pretrain()  # pyright: ignore[reportCallIssue]
