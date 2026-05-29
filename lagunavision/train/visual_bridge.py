from __future__ import annotations

import json
import math
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path

from lagunavision.backbones.base import LoraSettings
from lagunavision.backbones.factory import build_backbone
from lagunavision.data.manifest import load_manifest
from lagunavision.devices import resolve_torch_device
from lagunavision.encoders.factory import build_vision_encoder
from lagunavision.positions.normalized_2d import Normalized2DPositionEncoder
from lagunavision.projectors.features import stack_visual_features
from lagunavision.tiling.anyres import AnyResTiler
from lagunavision.train.batching import cosine_warmup_multiplier
from lagunavision.visual_pipeline import VisualProjectorSpec, build_projector


@dataclass(frozen=True)
class VisualBridgeTrainConfig:
    manifest: Path
    output_dir: Path
    backbone: str = "laguna"
    model_id: str = ""
    epochs: int = 1
    lr: float = 1e-3
    max_items: int = 0
    eval_manifest: Path | None = None
    encoder: str = "pil"
    encoder_id: str = ""
    max_tiles: int = 4
    projector: str = "mlp"
    visual_tokens: int = 64
    patch_px: int = 32
    device: str = "auto"
    vision_device: str = "auto"
    batch_size: int = 4
    grad_accum: int = 1
    warmup_ratio: float = 0.03
    num_workers: int = 0
    save_every: int = 0
    feature_cache_dir: Path | None = None
    init_checkpoint: Path | None = None
    init_lora_dir: Path | None = None
    grad_checkpointing: bool = True
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ()
    seed: int = 0


@dataclass(frozen=True)
class _Dist:
    world_size: int
    rank: int
    local_rank: int

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


async def train_visual_bridge(config: VisualBridgeTrainConfig) -> Path:
    """Train the connector with batched data parallelism.

    Runs single-process by default and as N-way data-parallel under ``torchrun``
    (it reads ``WORLD_SIZE``/``RANK``/``LOCAL_RANK``). The backbone is frozen and
    replicated per rank; only the small connector's gradients are averaged across
    ranks, so the expensive frozen forward scales linearly with GPU count.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, DistributedSampler
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with `python -m pip install -e '.[llama]'`.") from exc

    dist = _dist_from_env()
    device = _select_device(config.device, dist)
    if dist.distributed:
        torch.distributed.init_process_group(backend="nccl" if device.startswith("cuda") else "gloo")
        if device.startswith("cuda"):
            torch.cuda.set_device(dist.local_rank)
    torch.manual_seed(config.seed + dist.rank)

    items = load_manifest(config.manifest)
    if config.max_items > 0:
        items = items[: config.max_items]
    if not items:
        raise ValueError("manifest has no items")
    eval_items = load_manifest(config.eval_manifest) if config.eval_manifest else ()

    backbone = build_backbone(config.backbone, config.model_id, device=device, dtype=_compute_dtype(device))
    await backbone.load()
    backbone.freeze()
    lora_params: list = []
    if config.init_lora_dir is not None:
        backbone.load_adapter(config.init_lora_dir, trainable=True)
        lora_params = [parameter for parameter in backbone.torch_module.parameters() if parameter.requires_grad]
    elif config.lora_rank > 0:
        lora_params = backbone.enable_lora(
            LoraSettings(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                targets=tuple(config.lora_targets),
            )
        )
    model = backbone.torch_module
    if config.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    encoder = build_vision_encoder(config.encoder, config.encoder_id, config.patch_px, config.vision_device)
    tiler = AnyResTiler(max_tiles=config.max_tiles)
    positioner = Normalized2DPositionEncoder()
    if config.feature_cache_dir is not None and dist.distributed and not dist.is_main:
        torch.distributed.barrier()
    dataset = await _build_dataset(items, backbone, tiler, positioner, encoder, config.feature_cache_dir)
    if config.feature_cache_dir is not None and dist.distributed and dist.is_main:
        torch.distributed.barrier()
    input_dim = dataset.input_dim

    spec = VisualProjectorSpec(
        input_dim=input_dim,
        embedding_dim=backbone.hidden_size,
        projector=config.projector,
        visual_tokens=config.visual_tokens,
        encoder=config.encoder,
        encoder_id=config.encoder_id,
        max_tiles=config.max_tiles,
        patch_px=config.patch_px,
    )
    projector = build_projector(spec)
    if config.init_checkpoint is not None:
        projector.module.load_state_dict(torch.load(config.init_checkpoint, map_location="cpu"))
    projector.module.to(backbone.resolved_device)
    projector.module.train()
    trainable = list(projector.module.parameters()) + lora_params
    if dist.distributed:
        _broadcast_parameters(trainable)

    optimizer = torch.optim.AdamW(trainable, lr=config.lr)
    sampler = (
        DistributedSampler(dataset, num_replicas=dist.world_size, rank=dist.rank, shuffle=True, seed=config.seed)
        if dist.distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=_collate,
        num_workers=config.num_workers,
        drop_last=False,
    )

    steps_per_epoch = max(1, math.ceil(len(loader) / config.grad_accum))
    total_steps = max(1, steps_per_epoch * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_warmup_multiplier(step, warmup_steps, total_steps)
    )

    losses: list[float] = []
    global_step = 0
    for epoch in range(config.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for batch in loader:
            loss = _forward_loss(backbone, projector, batch)
            (loss / config.grad_accum).backward()
            losses.append(float(loss.detach().cpu()))
            pending += 1
            if pending == config.grad_accum:
                global_step += 1
                pending = 0
                _optimizer_step(trainable, optimizer, scheduler, dist)
                if dist.is_main and global_step % 10 == 0:
                    _log(epoch, global_step, total_steps, losses[-1], scheduler.get_last_lr()[0])
                if dist.is_main and config.save_every and global_step % config.save_every == 0:
                    _write_artifacts(
                        config,
                        projector,
                        spec,
                        backbone,
                        losses,
                        [],
                        len(items),
                        len(eval_items),
                        checkpoint_name=f"step_{global_step:06d}",
                    )
        if pending:
            global_step += 1
            _optimizer_step(trainable, optimizer, scheduler, dist)

    eval_losses: list[float] = []
    if eval_items and dist.is_main:
        eval_losses = await _evaluate(backbone, projector, eval_items, tiler, positioner, encoder)

    checkpoint = config.output_dir / "projector.pt"
    if dist.is_main:
        checkpoint = _write_artifacts(config, projector, spec, backbone, losses, eval_losses, len(items), len(eval_items))
    if dist.distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return checkpoint


def _forward_loss(backbone, projector, batch):
    import torch

    device = backbone.resolved_device
    visual_embeddings = [projector.module(vf.to(device=device, dtype=torch.float32)) for vf in batch["vf"]]
    model_batch = backbone.visual_training_batch(batch["prompt_ids"], batch["answer_ids"], visual_embeddings)
    return backbone.torch_module(**model_batch).loss


def _optimizer_step(params, optimizer, scheduler, dist: _Dist) -> None:
    if dist.distributed:
        _average_gradients(params, dist.world_size)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


async def _evaluate(backbone, projector, eval_items, tiler, positioner, encoder) -> list[float]:
    import torch

    projector.module.eval()
    losses: list[float] = []
    dataset = await _build_dataset(eval_items, backbone, tiler, positioner, encoder)
    with torch.no_grad():
        device = backbone.resolved_device
        for sample in dataset:
            visual = projector.module(sample["vf"].to(device=device, dtype=torch.float32))
            model_batch = backbone.visual_training_batch([sample["prompt_ids"]], [sample["answer_ids"]], [visual])
            losses.append(float(backbone.torch_module(**model_batch).loss.detach().cpu()))
    projector.module.train()
    return losses


def _write_artifacts(
    config,
    projector,
    spec,
    backbone,
    losses,
    eval_losses,
    train_items,
    eval_items,
    *,
    checkpoint_name: str | None = None,
) -> Path:
    import torch

    output_dir = config.output_dir / checkpoint_name if checkpoint_name else config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "projector.pt"
    torch.save(projector.module.state_dict(), checkpoint)
    if config.lora_rank > 0 or config.init_lora_dir is not None:
        backbone.save_adapter(output_dir / "lora")
    spec_row = {
        "input_dim": spec.input_dim,
        "embedding_dim": spec.embedding_dim,
        "hidden_dim": spec.hidden_dim,
        "backbone": config.backbone,
        "model_id": backbone.model_id,
        "encoder": config.encoder,
        "encoder_id": config.encoder_id,
        "projector": config.projector,
        "visual_tokens": config.visual_tokens,
        "max_tiles": config.max_tiles,
        "patch_px": config.patch_px,
        "lora_rank": config.lora_rank,
        "device": str(backbone.resolved_device),
        "vision_device": config.vision_device,
        "feature_cache_dir": str(config.feature_cache_dir) if config.feature_cache_dir else "",
        "init_checkpoint": str(config.init_checkpoint) if config.init_checkpoint else "",
        "init_lora_dir": str(config.init_lora_dir) if config.init_lora_dir else "",
        "checkpoint_name": checkpoint_name or "final",
    }
    (output_dir / "projector_spec.json").write_text(json.dumps(spec_row, indent=2), encoding="utf-8")
    (output_dir / "train_report.json").write_text(
        json.dumps(
            {
                "losses": losses,
                "eval_losses": eval_losses,
                "train_items": train_items,
                "eval_items": eval_items,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "grad_accum": config.grad_accum,
                "backbone": config.backbone,
                "model_id": backbone.model_id,
                "encoder": config.encoder,
                "encoder_id": config.encoder_id,
                "projector": config.projector,
                "visual_tokens": config.visual_tokens,
                "device": str(backbone.resolved_device),
                "vision_device": config.vision_device,
                "feature_cache_dir": str(config.feature_cache_dir) if config.feature_cache_dir else "",
                "init_checkpoint": str(config.init_checkpoint) if config.init_checkpoint else "",
                "init_lora_dir": str(config.init_lora_dir) if config.init_lora_dir else "",
                "checkpoint_name": checkpoint_name or "final",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return checkpoint


class _VisualBridgeDataset:
    def __init__(self, features: list, prompt_ids: list, answer_ids: list) -> None:
        self._features = features
        self._prompt_ids = prompt_ids
        self._answer_ids = answer_ids

    @property
    def input_dim(self) -> int:
        return int(_load_feature(self._features[0]).shape[1])

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, index: int) -> dict:
        return {
            "vf": _load_feature(self._features[index]),
            "prompt_ids": self._prompt_ids[index],
            "answer_ids": self._answer_ids[index],
        }


def _collate(samples: list[dict]) -> dict[str, list]:
    return {
        "vf": [sample["vf"] for sample in samples],
        "prompt_ids": [sample["prompt_ids"] for sample in samples],
        "answer_ids": [sample["answer_ids"] for sample in samples],
    }


async def _build_dataset(items, backbone, tiler, positioner, encoder, feature_cache_dir: Path | None = None) -> _VisualBridgeDataset:
    import torch

    features, prompt_ids, answer_ids = [], [], []
    cache_dir = feature_cache_dir.expanduser() if feature_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        tiles = _tiles_for_item(tiler, item)
        if cache_dir is not None:
            feature_path = _feature_cache_path(cache_dir, item)
            if not feature_path.exists():
                encoded = await encoder.encode(item.image, tiles)
                positions = positioner.encode_tiles(tiles)
                _save_feature_tensor(stack_visual_features(encoded, positions, "cpu"), feature_path)
            features.append(feature_path)
        else:
            encoded = await encoder.encode(item.image, tiles)
            positions = positioner.encode_tiles(tiles)
            features.append(stack_visual_features(encoded, positions, "cpu").float())
        prompt, answer = backbone.tokenize_example(item.question, _target_answer(item), context=item.ocr_text)
        prompt_ids.append(prompt)
        answer_ids.append(answer)
    return _VisualBridgeDataset(features, prompt_ids, answer_ids)


def _dist_from_env() -> _Dist:
    return _Dist(
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
    )


def _select_device(device: str, dist: _Dist) -> str:
    resolved = resolve_torch_device(device)
    if dist.distributed and resolved.startswith("cuda"):
        return f"cuda:{dist.local_rank}"
    return resolved


def _compute_dtype(device: str):
    import torch

    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _broadcast_parameters(params) -> None:
    import torch.distributed as distributed

    for parameter in params:
        distributed.broadcast(parameter.data, src=0)


def _average_gradients(params, world_size: int) -> None:
    import torch.distributed as distributed

    for parameter in params:
        if parameter.grad is not None:
            distributed.all_reduce(parameter.grad.data, op=distributed.ReduceOp.SUM)
            parameter.grad.data /= world_size


def _log(epoch: int, step: int, total_steps: int, loss: float, lr: float) -> None:
    print(
        json.dumps({"epoch": epoch + 1, "step": step, "total_steps": total_steps, "loss": loss, "lr": lr}),
        flush=True,
    )


def _target_answer(item) -> str:
    if getattr(item, "answer", ""):
        return item.answer
    key_text = ", ".join(item.must_include)
    if item.rubric == "vqa":
        return key_text
    if item.rubric == "description":
        context = ", ".join(item.accepted_fix_terms[:2])
        return f"The image shows {key_text}. It is about {context}."
    fix = item.accepted_fix_terms[0] if item.accepted_fix_terms else "fix the visible error"
    return f"The screenshot shows {key_text}. The minimal fix is to {fix}."


def _load_feature(feature):
    import torch

    if isinstance(feature, Path):
        return torch.load(feature, map_location="cpu")
    return feature


def _feature_cache_path(cache_dir: Path, item) -> Path:
    image_key = hashlib.sha1(str(item.image.resolve()).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{image_key}_{_safe_cache_name(item.image.stem)}.pt"


def _save_feature_tensor(feature, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(feature.half().cpu(), tmp)
    tmp.replace(path)


def _safe_cache_name(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"


def _tiles_for_item(tiler: AnyResTiler, item):
    from PIL import Image

    with Image.open(item.image) as image:
        width, height = image.size
    return tiler.tiles_for_size(width, height)
