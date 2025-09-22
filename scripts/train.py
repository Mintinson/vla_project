r"""Training script for Vision-Language-Action (VLA) policies.

This script implements the complete training pipeline for CogACT Vision-Language-Action
policies built on top of pretrained Vision-Language Models (VLMs). It supports distributed
training across multiple GPUs using PyTorch's Fully-Sharded Data Parallel (FSDP) and
trains on mixtures of the Open-X Embodiment dataset.

Key Features:
    - Multi-stage training: alignment, fine-tuning, and full fine-tuning
    - Support for action diffusion models with different architectures (DiT-S/B/L)
    - Action chunking for predicting future action sequences
    - Exponential Moving Average (EMA) support for action models
    - Comprehensive experiment tracking with Weights & Biases
    - Flexible backbone freezing configurations
    - Advanced routing mechanisms for improved performance

Architecture Support:
    - Vision backbones: DINOv2, SigLIP, and other TIMM models
    - Language backbones: LLaMA, Phi, and other HuggingFace models
    - Action models: Diffusion Transformer (DiT) variants
    - Training strategies: FSDP with mixed precision support

Environment Variables:
    TRAIN_ROUTE: Enable/disable routing mechanisms ("TRUE"/"FALSE")
    ADD_LSTM: Enable/disable LSTM components ("TRUE"/"FALSE")
    RANDOM_SEED: Random seed for reproducibility
    HF_HOME: Custom HuggingFace cache directory
    TF_CPP_MIN_LOG_LEVEL: Suppress TensorFlow logging (set to "3")

Usage Examples:
    Single GPU debugging:
        torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/train.py

    Multi-GPU training:
        torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/train.py \\
            --vla.vla_id "cogact-oxe-magic-soup" \\
            --seed 42 \\
            --image_aug

    Resume training from checkpoint:
        torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/train.py \\
            --pretrained_checkpoint "/path/to/checkpoint.pt" \\
            --is_resume \\
            --resume_step 10000 \\
            --resume_epoch 5

Prerequisites:
    - CUDA toolkit >= 11.0 for BF16 mixed precision support
    - Open-X Embodiment dataset properly formatted
    - Sufficient GPU memory for chosen batch size and model configuration

"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import torch
import torch.distributed as dist
import yaml

from vla_project.config import VLAConfig, VLARegistry
from vla_project.models.vla import get_vla_dataset_and_collator
from vla_project.models.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from vla_project.overwatch import initialize_overwatch
from vla_project.policy import CogACT, load, load_vla
from vla_project.training import VLAMetrics, get_train_strategy
from vla_project.utils import set_global_seed

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# RANDOM_SEED
import os

train_route = os.environ["TRAIN_ROUTE"].upper()
add_lstm = os.environ["ADD_LSTM"].upper()
random_seed = int(os.environ["RANDOM_SEED"])
print(f" random_seed: {random_seed} {type(random_seed)}")
# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainConfig:
    """Configuration dataclass for VLA training parameters.

    This configuration class manages all hyperparameters, paths, and settings for
    training Vision-Language-Action policies. It includes VLA-specific parameters,
    dataset configurations, training options, and experiment tracking settings.

    Attributes:
        vla: VLA model configuration specifying architecture and training parameters.
        data_root_dir: Root directory containing the Open-X Embodiment dataset.
        run_root_dir: Directory for storing training logs and checkpoints.
        pretrained_checkpoint: Path to checkpoint for resuming training.
        is_resume: Whether to resume from a previous training run.
        resume_step: Global step number to resume from (must match checkpoint).
        resume_epoch: Epoch number to resume from (must match checkpoint).
        run_id: Unique identifier for this training run.
        run_id_note: Additional note to append to run ID.
        save_interval: Number of steps between checkpoint saves.
        image_aug: Whether to enable image data augmentation.
        seed: Random seed for reproducibility.
        hf_token: HuggingFace authentication token for gated models.
        trackers: Tuple of tracking systems to use (e.g., "jsonl", "wandb").
        wandb_project: Weights & Biases project name.
        wandb_entity: Weights & Biases entity/organization name.
        repeated_diffusion_steps: Number of repeated steps for action diffusion training.
        load_all_data_for_training: Whether to load the complete training dataset.
        future_action_window_size: Number of future actions to predict (action chunking).
        past_action_window_size: Number of past actions to consider (currently unused).
        action_model_type: Diffusion Transformer variant ('DiT-S', 'DiT-B', 'DiT-L').
        use_ema: Whether to use Exponential Moving Average for action model.
        action_dim: Dimensionality of the action space.
        load_dit: Whether to load DiT components.

    Example:
        >>> cfg = TrainConfig(
        ...     run_id="my_vla_experiment",
        ...     image_aug=True,
        ...     future_action_window_size=10,
        ...     action_model_type="DiT-B"
        ... )
        >>> # Configuration will inherit optimization parameters from vla config

    """

    # fmt: off

    # VLAConfig (`conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.EXP_COGACT_OXE_MAGIC_SOUP_PLUS_MINUS.vla_id),
    )

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "datasets/open-x-embodiment",
    )
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints

    # Resume Run Parameters
    pretrained_checkpoint: str | Path | None = None                  # Absolute Path to Checkpoint
    is_resume: bool = True                                          # Whether we are continuing a prior training run
                                                                    # (only applicable given pretrained checkpoint)
    resume_step: int | None = None                               # Global Step to Resume (should match checkpoint)
    resume_epoch: int | None = None                              # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: str | None = None                                    # Run ID for logging, Weights & Biases
    run_id_note: str | None = None                               # Extra note for logging, Weights & Biases
    save_interval: int = 2500                                       # Interval for saving checkpoints (in steps)
    image_aug: bool = False                                         # Whether to enable image augmentations
    # seed: int = 42                                                  # Random seed (for reproducibility)
    seed: int = random_seed                                                 # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: str | Path = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    # trackers: Tuple[str, ...] = ("wandb")                  # Trackers to initialize (if W&B, add config!)
    trackers: tuple[str, ...] = ("jsonl")                         # Trackers to initialize (if W&B, add config!)
    wandb_project: str = ""                                         # Name of W&B project to log to (use default!)
    wandb_entity: str = ""                                          # Name of entity to log under
    repeated_diffusion_steps: int = 8                               # Repeated steps for training action model (a diffusion model)
    load_all_data_for_training: bool = True                         # Load all training data
    future_action_window_size: int = 15                             # Action chunking, predicting future actions + current action
    past_action_window_size: int = 0                                # Action history window size, not used now, set to be 0
    action_model_type: str = "DiT-B"                                # Action model type, chose from ['DiT-S', 'DiT-B', 'DiT-L']
    use_ema: bool = False                                           # EMA version of action model
    action_dim: int = 7                                             # Dimension of action space
    load_dit: bool = True

    def __post_init__(self) -> None:
        """Initialize optimization parameters from VLA config and validate world size.

        Extracts and sets training hyperparameters from the VLA configuration for
        convenient access during training. Also validates that the expected world
        size matches the actual number of available GPUs.

        Sets the following parameters from vla config:
            - Training schedule: epochs, max_steps
            - Batch sizes: global_batch_size, per_device_batch_size  
            - Optimization: learning_rate, weight_decay, max_grad_norm
            - Learning rate schedule: lr_scheduler_type, warmup_ratio
            - Training strategy: distributed training configuration

        Raises:
            AssertionError: If expected_world_size doesn't match actual GPU count.

        Note:
            This method is called automatically after dataclass initialization.

        """
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # [Validate] Assert on `expected_world_size`
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    r"""Execute the complete VLA training pipeline.

    This function orchestrates the entire training process for CogACT Vision-Language-Action
    policies, including model initialization, dataset preparation, distributed training setup,
    and experiment tracking. It supports various training stages and architectural configurations.

    The training pipeline includes:
        1. Distributed training setup with FSDP
        2. VLA model loading/creation with optional checkpoint resumption
        3. Backbone freezing configuration based on training stage
        4. Open-X dataset preparation with action chunking
        5. Training strategy initialization with optimizer setup
        6. Metrics tracking and checkpoint management
        7. Main VLA training loop execution

    Training Stages:
        - "align": Fine-tune projector only (frozen backbones)
        - "finetune": Fine-tune projector + LLM (frozen vision)
        - "full-finetune": Fine-tune all components
        - "vla-sandwich-train": Vision + projector + last LLM layer
        - "vla-last-layer-train": Last LLM layer only

    Args:
        cfg: Configuration object containing all training parameters, model settings,
            dataset configuration, and experiment tracking options.

    Raises:
        RuntimeError: If distributed training setup fails.
        ValueError: If backbone freezing configuration is not supported.
        FileNotFoundError: If checkpoint files or dataset paths are missing.
        AssertionError: If checkpoint validation fails during resume.

    Environment Dependencies:
        TRAIN_ROUTE: Controls routing mechanism activation
        ADD_LSTM: Controls LSTM component usage
        RANDOM_SEED: Sets reproducibility seed

    Note:
        This function is designed to run under `torchrun` for distributed training.
        It automatically handles GPU device assignment, process group management,
        and resource cleanup.

    Example:
        Run with torchrun for multi-GPU training:
        ```bash
        TRAIN_ROUTE=TRUE ADD_LSTM=FALSE RANDOM_SEED=42 \\
        torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/train.py \\
            --vla.vla_id "cogact-oxe-magic-soup" \\
            --image_aug \\
            --future_action_window_size 15
        ```

    """
    overwatch.info("CogACT-VLA Training :: Warming Up")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    hf_token = ""  ## cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    dist.barrier()
    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    # with open('/home/cx/4dvla_cx/temp/config.json','w') as f:
    #     json.dump(cfg,f)
    # print(cfg)
    if cfg.pretrained_checkpoint is not None:
        # [Validate] Pretrained Checkpoint `step` and `epoch` should match `resume_step` and `resume_epoch`
        #   =>> Note :: We make developers pass in `resume_*` arguments as an extra sanity check!
        if cfg.is_resume:
            assert int(re.search("step-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_step
            assert int(re.search("epoch-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_epoch
        overwatch.info("Loading VLA Checkpoint")
        if cfg.use_ema:
            overwatch.info("Loading EMA of Diffusion")
        vla = load_vla(
            cfg.pretrained_checkpoint,
            hf_token=hf_token,
            load_for_training=True,
            action_model_type=cfg.action_model_type,
            action_dim=cfg.action_dim,
            future_action_window_size=cfg.future_action_window_size,
            past_action_window_size=cfg.past_action_window_size,
            use_ema=cfg.use_ema,
            load_dit=cfg.load_dit,
        )

    else:
        vlm = load(cfg.vla.base_vlm, hf_token=hf_token, load_for_training=True)
        overwatch.info("Creating VLA from Base VLM")
        if cfg.use_ema:
            overwatch.info("Creating EMA for Diffusion")

        vla = CogACT(
            vlm,
            action_model_type=cfg.action_model_type,
            action_dim=cfg.action_dim,
            future_action_window_size=cfg.future_action_window_size,
            past_action_window_size=cfg.past_action_window_size,
            use_ema=cfg.use_ema,
            load_dit=cfg.load_dit,
        )
        # del this variable to avoid bugs. The vlm shouldn't be used anymore
        del vlm

    # [Validate] Model should be in Full Precision!
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # Determine training "stage" based on frozen vs unfrozen parameters --> supports different fine-tuning schemes!
    # import ipdb
    # ipdb.set_trace()
    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "full-finetune"  # Full fine-tuning
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "finetune"  # Frozen vision encoder
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        stage = "align"  # Fine-tuning projector
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone and cfg.vla.unfreeze_last_llm_layer:
        stage = "vla-sandwich-train"  # Fine-tuning vision encoder, projector, and LLM last layer
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone and cfg.vla.unfreeze_last_llm_layer:
        stage = "vla-last-layer-train"  # Fine-tuning LLM last layer only
    else:
        msg = (
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}"
            f"freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}"
            f"unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )
        raise ValueError(
            msg,
        )

    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vla.freeze_backbones(stage)

    if cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        for name, param in vla.vlm.named_parameters():
            param.requires_grad = False

    if train_route == "TRUE":
        # import pdb; pdb.set_trace()
        # vla.vlm.route1.requires_grad_(True)
        # vla.vlm.route2.requires_grad_(True)
        # vla.vlm.lstm.requires_grad_(True)
        vla.llm_backbone.llm.model.route1.requires_grad_(True)
        vla.llm_backbone.llm.model.route2.requires_grad_(True)
        # vla.llm_backbone.llm.model.route.requires_grad_(True)
        overwatch.info("`vla.llm_backbone.llm.model.route1 and route2` has been unfreezed.")
    else:
        # vla.llm_backbone.llm.model.route1.requires_grad_(True)
        # vla.llm_backbone.llm.model.route2.requires_grad_(True)
        # vla.llm_backbone.llm.model.lstm.requires_grad_(True)
        overwatch.info("`our train model is baseline")
        overwatch.info("`route1, route2 and lstm have been unfreezed.")
    if add_lstm == "TRUE" and train_route == "TRUE":
        # vla.lstm.vlm.requires_grad_(True)
        vla.llm_backbone.llm.model.lstm.requires_grad_(True)
        overwatch.info("`vla.llm_backbone.llm.model.lstm` has been unfreezed.")
    else:
        overwatch.info("This model not have lstm architecture.")

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable",
    )

    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    # import ipdb
    # ipdb.set_trace()
    vla_dataset, _, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=vla.vision_backbone.get_image_transform(),
        tokenizer=vla.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        default_image_resolution=vla.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        load_all_data_for_training=cfg.load_all_data_for_training,
        future_action_window_size=cfg.future_action_window_size,
        past_action_window_size=cfg.past_action_window_size,
    )
    # import pdb; pdb.set_trace()
    # data.keys() dict_keys(['pixel_values', 'input_ids', 'labels', 'dataset_name', 'actions', 'action_masks'])
    # data['pixel_values']['dino'].shape, data['pixel_values']['siglip'].shape: torch.Size([3, 224, 224]), torch.Size([3, 224, 224])
    # data['input_ids'].shape, data['labels'].shape: torch.Size([22]), torch.Size([22])
    # data['dataset_name'] : b'rlbench'
    # data['actions'].shape: torch.Size([16, 7])
    # data['action_masks'].shape : torch.Size([16]) e.g.tensor([ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False])
    # for data in vla_dataset:
    #     print(data)
    #     data = collator(data)
    #     print(data)
    #     break
    # import pdb; pdb.set_trace()
    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    dist.barrier()
    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    weight = torch.load(
        "/home/cx/chenhao/hub/models--CogACT--CogACT-Base/snapshots/ffc4db3bef7735ba7aa692d50b6454588a32b753/checkpoints/CogACT-Base.pt",
        map_location="cpu",
    )["model"]["llm_backbone"]
    # import pdb; pdb.set_trace()
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vla,
        device_id=device_id,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))
    if cfg.pretrained_checkpoint is not None and cfg.is_resume:
        train_strategy.load_optimizer_and_scheduler(cfg.pretrained_checkpoint)

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    # import pdb;pdb.set_trace()
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
    )
    # Run VLA Training
    overwatch.info("Starting VLA Training Loop")
    # import pdb; pdb.set_trace()
    train_strategy.run_vla_training(
        vla_dataset=vla_dataset,
        collator=collator,
        metrics=metrics,
        save_interval=cfg.save_interval,
        action_model=True,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()  # pyright: ignore[reportCallIssue]
