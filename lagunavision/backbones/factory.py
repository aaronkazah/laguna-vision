from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from lagunavision.backbones.base import Backbone
from lagunavision.backbones.hf_causal import HfCausalBackbone

LAGUNA_XS2 = "poolside/Laguna-XS.2"
LLAMA_32_3B_INSTRUCT = "meta-llama/Llama-3.2-3B-Instruct"


@dataclass(frozen=True)
class BackboneSpec:
    build: Callable[..., Backbone]
    default_model_id: str
    trust_remote_code: bool = False


_REGISTRY: dict[str, BackboneSpec] = {
    "laguna": BackboneSpec(HfCausalBackbone, LAGUNA_XS2, trust_remote_code=True),
    "llama": BackboneSpec(HfCausalBackbone, LLAMA_32_3B_INSTRUCT),
    "hf": BackboneSpec(HfCausalBackbone, ""),
}


def register_backbone(
    name: str,
    build: Callable[..., Backbone],
    default_model_id: str = "",
    trust_remote_code: bool = False,
) -> None:
    """Register a backbone adapter under a logical name."""
    _REGISTRY[name] = BackboneSpec(build, default_model_id, trust_remote_code=trust_remote_code)


def available_backbones() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def build_backbone(name: str = "laguna", model_id: str = "", device: str = "auto", **kwargs) -> Backbone:
    """Construct a backbone by registered name, with an optional model_id override."""
    spec = _REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"unknown backbone '{name}'. available: {', '.join(_REGISTRY)}")
    resolved_id = model_id or spec.default_model_id
    if not resolved_id:
        raise ValueError(f"backbone '{name}' has no default model_id; pass --model-id")
    if spec.trust_remote_code:
        kwargs.setdefault("trust_remote_code", True)
    return spec.build(model_id=resolved_id, device=device, **kwargs)
