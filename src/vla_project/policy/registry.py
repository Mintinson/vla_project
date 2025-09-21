from pathlib import Path
from typing import Protocol

from torch import nn

from vla_project.models.backbones.llm.llm_base import LLMBackbone
from vla_project.models.backbones.vision.vision_base import VisionBackbone


class VLAProtocol(Protocol):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        arch_specifier: str = "gelu-mlp",
        enable_mixed_precision_training: bool = True,
        freeze_weights: bool = True,
        **kwargs,
    ): ...


# ruff: noqa: PLC0415
def get_vla(vla_id: str) -> type[nn.Module]:
    if vla_id.lower() == "cogactvla":
        from .cogactvla import CogACT

        return CogACT
    elif vla_id.lower() == "openvla":
        from .openvla import OpenVLA

        return OpenVLA
    else:
        msg = f"VLA {vla_id} not found"
        raise ValueError(msg)
