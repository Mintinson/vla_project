"""load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
import os
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download

from vla_project.config import ModelConfig
from vla_project.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform

# from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from vla_project.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from vla_project.models.vla.action_tokenizer import ActionTokenizer
from vla_project.models.vlms import PrismaticVLM
from vla_project.overwatch import initialize_overwatch
from vla_project.policy import CogACT
from vla_project.policy.registry import VLAProtocol, get_vla

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)
train_route = os.environ["TRAIN_ROUTE"].upper()


# === HF Hub Repository ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"


# === Available Models ===
def available_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> list[str]:
    # return list(GLOBAL_REGISTRY.items())
    return list(GLOBAL_REGISTRY.keys())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        msg = f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`"
        raise ValueError(msg)

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load(
    model_id_or_path: str | Path,
    hf_token: str | None = None,
    cache_dir: str | Path | None = None,
    *,
    load_for_training: bool = False,
) -> PrismaticVLM:
    """Loads a pretrained PrismaticVLM from either local disk or the HuggingFace Hub."""
    if os.path.isdir(model_id_or_path):  # noqa: PTH112
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # Get paths for `config.json` and pretrained checkpoint
        config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / "latest-checkpoint.pt"
        if not config_json.exists():
            msg = f"Missing `config.json` for `{run_dir = }`"
            raise ValueError(msg)
        if not checkpoint_pt.exists():
            msg = f"Missing checkpoint for `{run_dir = }`"
            raise ValueError(msg)
    else:
        if model_id_or_path not in GLOBAL_REGISTRY:
            msg = f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`"
            raise ValueError(msg)

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        with overwatch.local_zero_first():
            config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
            checkpoint_pt = hf_hub_download(
                repo_id=HF_HUB_REPO,
                filename=f"{model_id}/checkpoints/latest-checkpoint.pt",
                cache_dir=cache_dir,
            )
            config_json = Path(config_json)
            checkpoint_pt = Path(checkpoint_pt)

    # Load Model Config from `config.json`
    with config_json.open() as f:
        model_cfg = json.load(f)["model"]

    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]",
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, _ = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, _ = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=not load_for_training,
    )

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint")
    return PrismaticVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
        freeze_weights=not load_for_training,
    )


# === Load Pretrained VLA Model ===
def load_vla(
    model_id_or_path: str | Path,
    hf_token: str | None = None,
    cache_dir: str | Path | None = None,
    *,
    load_for_training: bool = False,
    model_type: str = "pretrained",
    **kwargs,
) -> CogACT:
    """Loads a pretrained CogACT from either local disk or the HuggingFace Hub."""
    # TODO (siddk, moojink) :: Unify semantics with `load()` above; right now, `load_vla()` assumes path points to
    #   checkpoint `.pt` file, rather than the top-level run directory!
    # import ipdb
    # ipdb.set_trace()
    if os.path.isfile(model_id_or_path):  # noqa: PTH113
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`")

        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        if checkpoint_pt.suffix != ".pt" or checkpoint_pt.parent.name != "checkpoints":
            msg = f"Invalid checkpoint path `{checkpoint_pt = }`; should be a `.pt` file in `checkpoints/`"
            raise ValueError(msg)
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        if not config_json.exists():
            msg = f"Missing `config.json` for `{run_dir = }`"
            raise ValueError(msg)
        if not dataset_statistics_json.exists():
            msg = f"Missing `dataset_statistics.json` for `{run_dir = }`"
            raise ValueError(msg)

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`model_id_or_path`)
    else:
        # Search HF Hub Repo via fsspec API
        overwatch.info(f"Checking HF for `{(hf_path := str(Path(model_id_or_path)))}`")
        if not (tmpfs := HfFileSystem()).exists(hf_path):
            msg = f"Couldn't find valid HF Hub Path `{hf_path = }`"
            raise ValueError(msg)

        valid_ckpts = tmpfs.glob(f"{hf_path}/checkpoints/*.pt")
        if (len(valid_ckpts) == 0) or (len(valid_ckpts) != 1):
            msg = f"Couldn't find a valid checkpoint to load from HF Hub Path `{hf_path}/checkpoints/"
            raise ValueError(msg)

        target_ckpt = Path(valid_ckpts[-1]).name
        model_id_or_path = str(model_id_or_path)  # Convert to string for HF Hub API
        overwatch.info(f"Downloading Model `{model_id_or_path}` Config & Checkpoint `{target_ckpt}`")
        with overwatch.local_zero_first():
            # relpath = Path(model_type) / model_id_or_path
            config_json = Path(
                hf_hub_download(
                    repo_id=model_id_or_path,
                    filename=f"{('config.json')!s}",
                    cache_dir=cache_dir,
                ),
            )
            dataset_statistics_json = Path(
                hf_hub_download(
                    repo_id=model_id_or_path,
                    filename=f"{('dataset_statistics.json')!s}",
                    cache_dir=cache_dir,
                ),
            )
            checkpoint_pt = Path(
                hf_hub_download(
                    repo_id=model_id_or_path,
                    filename=f"{(Path('checkpoints') / target_ckpt)!s}",
                    cache_dir=cache_dir,
                ),
            )

    # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
    with config_json.open() as f:
        vla_cfg = json.load(f)["vla"]
        model_cfg: ModelConfig = ModelConfig.get_choice_class(vla_cfg["base_vlm"])()

    # Load Dataset Statistics for Action Denormalization
    with dataset_statistics_json.open() as f:
        norm_stats = json.load(f)

    # = Load Individual Components necessary for Instantiating a VLA (via base VLM components) =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg.model_id}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg.vision_backbone_id}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg.llm_backbone_id}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg.arch_specifier}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]",
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg.vision_backbone_id}[/]")
    vision_backbone, _ = get_vision_backbone_and_transform(
        model_cfg.vision_backbone_id,
        model_cfg.image_resize_strategy,
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, _ = get_llm_backbone_and_tokenizer(
        model_cfg.llm_backbone_id,
        llm_max_length=model_cfg.llm_max_length,
        hf_token=hf_token,
        inference_mode=not load_for_training,
    )
    action_tokenizer = ActionTokenizer(llm_backbone.get_tokenizer()) if model_cfg.need_action_tokenization else None
    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLA [bold blue]{model_cfg.model_id}[/] from Checkpoint")
    # import pdb; pdb.set_trace()
    vla_model: VLAProtocol = get_vla(model_cfg.model_id)
    vla = vla_model.from_pretrained(
        checkpoint_pt,
        model_cfg.model_id,
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg.arch_specifier,
        freeze_weights=not load_for_training,
        norm_stats=norm_stats,
        action_tokenizer=action_tokenizer,
        **kwargs,
    )

    return vla
