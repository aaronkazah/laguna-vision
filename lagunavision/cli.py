from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from lagunavision.backbones.factory import available_backbones, build_backbone
from lagunavision.data.hf_materialize import DEFAULT_HF_DATASETS, materialize_hf_dataset, parse_dataset_requests
from lagunavision.data.llava import materialize_llava_hf, materialize_llava_json
from lagunavision.data.manifest import load_manifest
from lagunavision.data.sources import DATASET_SOURCES
from lagunavision.data.spatial_ocr import generate_spatial_ocr_manifest
from lagunavision.eval.ablation import AblationConfig, run_ablation
from lagunavision.eval.demo_set import generate_demo_eval
from lagunavision.eval.run_eval import run_text_eval
from lagunavision.eval.scene_probe import generate_scene_probe
from lagunavision.eval.scene_probe import generate_scene_dataset
from lagunavision.eval.score_eval import score_answer
from lagunavision.eval.web_probe import generate_web_probe
from lagunavision.model import LagunaVisionTextPipeline
from lagunavision.positions.normalized_2d import Normalized2DPositionEncoder
from lagunavision.projectors.features import stack_visual_features
from lagunavision.tiling.anyres import AnyResTiler
from lagunavision.train.visual_bridge import (
    VisualBridgeTrainConfig,
    _feature_cache_path,
    _save_feature_tensor,
    _tiles_for_item,
    train_visual_bridge,
)
from lagunavision.visual_pipeline import LagunaVisionImagePipeline, VisualProjectorSpec


def _add_backbone_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backbone", choices=available_backbones(), default="laguna")
    parser.add_argument("--model-id", default="")


def main() -> None:
    parser = argparse.ArgumentParser(prog="laguna-vision")
    subcommands = parser.add_subparsers(dest="command", required=True)

    datasets = subcommands.add_parser("datasets")
    datasets.set_defaults(func=_datasets)

    tile = subcommands.add_parser("tile")
    tile.add_argument("--width", type=int, required=True)
    tile.add_argument("--height", type=int, required=True)
    tile.set_defaults(func=_tile)

    spatial = subcommands.add_parser("spatial-ocr")
    spatial.add_argument("--output-dir", type=Path, required=True)
    spatial.add_argument("--count", type=int, default=200)
    spatial.set_defaults(func=_spatial_ocr)

    ask = subcommands.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--context", default="")
    _add_backbone_args(ask)
    ask.add_argument("--device", default="auto")
    ask.set_defaults(func=_ask)

    eval_cmd = subcommands.add_parser("eval-text")
    eval_cmd.add_argument("--manifest", type=Path, required=True)
    eval_cmd.add_argument("--output", type=Path, required=True)
    _add_backbone_args(eval_cmd)
    eval_cmd.add_argument("--device", default="auto")
    eval_cmd.add_argument("--ocr-context", action="store_true")
    eval_cmd.set_defaults(func=_eval_text)

    demo_eval = subcommands.add_parser("demo-eval")
    demo_eval.add_argument("--output-dir", type=Path, required=True)
    demo_eval.set_defaults(func=_demo_eval)

    web_probe = subcommands.add_parser("web-probe")
    web_probe.add_argument("--output-dir", type=Path, required=True)
    web_probe.add_argument("--limit", type=int, default=5)
    web_probe.add_argument("--width", type=int, default=1920)
    web_probe.add_argument("--height", type=int, default=1080)
    web_probe.set_defaults(func=_web_probe)

    scene_probe = subcommands.add_parser("scene-probe")
    scene_probe.add_argument("--output-dir", type=Path, required=True)
    scene_probe.add_argument("--limit", type=int, default=5)
    scene_probe.set_defaults(func=_scene_probe)

    scene_dataset = subcommands.add_parser("scene-dataset")
    scene_dataset.add_argument("--output-dir", type=Path, required=True)
    scene_dataset.add_argument("--train-count", type=int, default=40)
    scene_dataset.add_argument("--eval-count", type=int, default=10)
    scene_dataset.set_defaults(func=_scene_dataset)

    hf_dataset = subcommands.add_parser("hf-materialize")
    hf_dataset.add_argument("--output-dir", type=Path, required=True)
    hf_dataset.add_argument("--train-count", type=int, default=100)
    hf_dataset.add_argument("--eval-count", type=int, default=20)
    hf_dataset.add_argument("--dataset", action="append", default=[])
    hf_dataset.set_defaults(func=_hf_materialize)

    llava_dataset = subcommands.add_parser("llava-materialize")
    llava_dataset.add_argument("--output-dir", type=Path, required=True)
    llava_dataset.add_argument("--source-json", type=Path)
    llava_dataset.add_argument("--image-root", type=Path, action="append", default=[])
    llava_dataset.add_argument("--dataset", default="")
    llava_dataset.add_argument("--split", default="train")
    llava_dataset.add_argument("--limit", type=int, default=0)
    llava_dataset.add_argument("--eval-count", type=int, default=0)
    llava_dataset.add_argument("--image-mode", choices=("reference", "copy", "symlink"), default="reference")
    llava_dataset.set_defaults(func=_llava_materialize)

    cache_features = subcommands.add_parser("cache-visual-features")
    cache_features.add_argument("--manifest", type=Path, required=True)
    cache_features.add_argument("--output-dir", type=Path, required=True)
    cache_features.add_argument("--encoder", choices=("pil", "hf"), default="hf")
    cache_features.add_argument("--encoder-id", default="openai/clip-vit-large-patch14-336")
    cache_features.add_argument("--patch-px", type=int, default=32)
    cache_features.add_argument("--max-tiles", type=int, default=9)
    cache_features.add_argument("--device", default="cuda")
    cache_features.add_argument("--shard-index", type=int, default=0)
    cache_features.add_argument("--num-shards", type=int, default=1)
    cache_features.add_argument("--overwrite", action="store_true")
    cache_features.set_defaults(func=_cache_visual_features)

    score_cmd = subcommands.add_parser("score")
    score_cmd.add_argument("--manifest", type=Path, required=True)
    score_cmd.add_argument("--answers", type=Path, required=True)
    score_cmd.set_defaults(func=_score)

    train_bridge = subcommands.add_parser("train-visual-bridge")
    train_bridge.add_argument("--manifest", type=Path, required=True)
    train_bridge.add_argument("--output-dir", type=Path, required=True)
    _add_backbone_args(train_bridge)
    train_bridge.add_argument("--epochs", type=int, default=1)
    train_bridge.add_argument("--max-items", type=int, default=0)
    train_bridge.add_argument("--lr", type=float, default=1e-3)
    train_bridge.add_argument("--eval-manifest", type=Path)
    train_bridge.add_argument("--encoder", choices=("pil", "hf"), default="pil")
    train_bridge.add_argument("--encoder-id", default="")
    train_bridge.add_argument("--max-tiles", type=int, default=4)
    train_bridge.add_argument("--projector", choices=("mlp", "resampler"), default="mlp")
    train_bridge.add_argument("--visual-tokens", type=int, default=64)
    train_bridge.add_argument("--batch-size", type=int, default=4)
    train_bridge.add_argument("--grad-accum", type=int, default=1)
    train_bridge.add_argument("--warmup-ratio", type=float, default=0.03)
    train_bridge.add_argument("--num-workers", type=int, default=0)
    train_bridge.add_argument("--save-every", type=int, default=0)
    train_bridge.add_argument("--feature-cache-dir", type=Path)
    train_bridge.add_argument("--init-checkpoint", type=Path)
    train_bridge.add_argument("--init-lora-dir", type=Path)
    train_bridge.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false")
    train_bridge.add_argument("--lora-rank", type=int, default=0)
    train_bridge.add_argument("--lora-alpha", type=int, default=16)
    train_bridge.add_argument("--lora-dropout", type=float, default=0.05)
    train_bridge.add_argument("--lora-targets", nargs="*", default=[])
    train_bridge.add_argument("--device", default="auto")
    train_bridge.add_argument("--vision-device", default="auto")
    train_bridge.set_defaults(func=_train_visual_bridge, grad_checkpointing=True)

    ask_image = subcommands.add_parser("ask-image")
    ask_image.add_argument("--image", type=Path, required=True)
    ask_image.add_argument("--checkpoint", type=Path, required=True)
    ask_image.add_argument("--question", required=True)
    ask_image.add_argument("--backbone", choices=available_backbones(), default="")
    ask_image.add_argument("--model-id", default="")
    ask_image.add_argument("--device", default="auto")
    ask_image.add_argument("--vision-device", default="auto")
    ask_image.set_defaults(func=_ask_image)

    ablation = subcommands.add_parser("eval-ablation")
    ablation.add_argument("--manifest", type=Path, required=True)
    ablation.add_argument("--checkpoint", type=Path, required=True)
    ablation.add_argument("--output", type=Path, required=True)
    ablation.add_argument("--backbone", choices=available_backbones(), default="")
    ablation.add_argument("--model-id", default="")
    ablation.add_argument("--limit", type=int, default=0)
    ablation.add_argument("--threshold", type=float, default=0.15)
    ablation.add_argument("--device", default="auto")
    ablation.add_argument("--vision-device", default="auto")
    ablation.set_defaults(func=_eval_ablation)

    args = parser.parse_args()
    args.func(args)


def _datasets(_: argparse.Namespace) -> None:
    rows = [
        {
            "id": source.id,
            "stage": source.stage.value,
            "required": source.required,
            "use": source.use,
        }
        for source in DATASET_SOURCES
    ]
    print(json.dumps(rows, indent=2, sort_keys=True))


def _tile(args: argparse.Namespace) -> None:
    tiler = AnyResTiler()
    positions = Normalized2DPositionEncoder()
    tiles = tiler.tiles_for_size(args.width, args.height)
    rows = [
        {
            "id": tile.id,
            "crop": [tile.crop.left, tile.crop.top, tile.crop.right, tile.crop.bottom],
            "position": positions.encode_tile(tile).values,
        }
        for tile in tiles
    ]
    print(json.dumps(rows, indent=2))


def _spatial_ocr(args: argparse.Namespace) -> None:
    examples = generate_spatial_ocr_manifest(args.output_dir, args.count)
    print(json.dumps({"examples": len(examples), "manifest": str(args.output_dir / "manifest.jsonl")}))


def _ask(args: argparse.Namespace) -> None:
    async def run() -> None:
        backbone = build_backbone(args.backbone, args.model_id, device=args.device)
        pipeline = LagunaVisionTextPipeline(backbone)
        print(await pipeline.answer(args.question, extracted_context=args.context))

    asyncio.run(run())


def _eval_text(args: argparse.Namespace) -> None:
    async def run() -> None:
        passed, total = await run_text_eval(
            args.manifest,
            build_backbone(args.backbone, args.model_id, device=args.device),
            args.output,
            use_ocr_context=args.ocr_context,
        )
        print(json.dumps({"passed": passed, "total": total}))

    asyncio.run(run())


def _demo_eval(args: argparse.Namespace) -> None:
    manifest = generate_demo_eval(args.output_dir)
    print(json.dumps({"manifest": str(manifest), "items": 15}))


def _web_probe(args: argparse.Namespace) -> None:
    async def run() -> None:
        manifest = await generate_web_probe(args.output_dir, args.limit, width=args.width, height=args.height)
        print(json.dumps({"manifest": str(manifest), "items": args.limit}))

    asyncio.run(run())


def _scene_probe(args: argparse.Namespace) -> None:
    manifest = generate_scene_probe(args.output_dir, args.limit)
    print(json.dumps({"manifest": str(manifest), "items": args.limit}))


def _scene_dataset(args: argparse.Namespace) -> None:
    train_manifest, eval_manifest = generate_scene_dataset(
        args.output_dir,
        train_count=args.train_count,
        eval_count=args.eval_count,
    )
    print(
        json.dumps(
            {
                "train_manifest": str(train_manifest),
                "eval_manifest": str(eval_manifest),
                "train_items": args.train_count,
                "eval_items": args.eval_count,
            }
        )
    )


def _hf_materialize(args: argparse.Namespace) -> None:
    datasets = parse_dataset_requests(args.dataset) if args.dataset else None
    train_manifest, eval_manifest = materialize_hf_dataset(
        args.output_dir,
        train_count=args.train_count,
        eval_count=args.eval_count,
        datasets=datasets or DEFAULT_HF_DATASETS,
    )
    print(
        json.dumps(
            {
                "train_manifest": str(train_manifest),
                "eval_manifest": str(eval_manifest),
                "train_items": args.train_count,
                "eval_items": args.eval_count,
            }
        )
    )


def _llava_materialize(args: argparse.Namespace) -> None:
    if bool(args.source_json) == bool(args.dataset):
        raise SystemExit("provide exactly one of --source-json or --dataset")
    if args.source_json:
        result = materialize_llava_json(
            args.source_json,
            args.output_dir,
            image_roots=args.image_root,
            limit=args.limit,
            eval_count=args.eval_count,
            image_mode=args.image_mode,
        )
    else:
        result = materialize_llava_hf(
            args.dataset,
            args.output_dir,
            split=args.split,
            limit=args.limit,
            eval_count=args.eval_count,
        )
    print(
        json.dumps(
            {
                "train_manifest": str(result.train_manifest),
                "eval_manifest": str(result.eval_manifest) if result.eval_manifest else "",
                "train_items": result.train_count,
                "eval_items": result.eval_count,
            }
        )
    )


def _cache_visual_features(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be between 0 and --num-shards - 1")

    async def run() -> None:
        from lagunavision.encoders.factory import build_vision_encoder

        items = load_manifest(args.manifest)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        encoder = build_vision_encoder(args.encoder, args.encoder_id, args.patch_px, args.device)
        tiler = AnyResTiler(max_tiles=args.max_tiles)
        positioner = Normalized2DPositionEncoder()
        written = 0
        skipped = 0
        for index, item in enumerate(items):
            if index % args.num_shards != args.shard_index:
                continue
            feature_path = _feature_cache_path(args.output_dir, item)
            if feature_path.exists() and not args.overwrite:
                skipped += 1
                continue
            tiles = _tiles_for_item(tiler, item)
            encoded = await encoder.encode(item.image, tiles)
            positions = positioner.encode_tiles(tiles)
            _save_feature_tensor(stack_visual_features(encoded, positions, "cpu"), feature_path)
            written += 1
        print(json.dumps({"written": written, "skipped": skipped, "shard_index": args.shard_index, "num_shards": args.num_shards}))

    asyncio.run(run())


def _score(args: argparse.Namespace) -> None:
    items = {item.id: item for item in load_manifest(args.manifest)}
    total = 0
    passed = 0
    with args.answers.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            item_id = str(row["id"])
            score = score_answer(items[item_id], str(row["answer"]))
            total += 1
            passed += int(score.passed)
    print(json.dumps({"passed": passed, "total": total}))


def _train_visual_bridge(args: argparse.Namespace) -> None:
    async def run() -> None:
        checkpoint = await train_visual_bridge(
            VisualBridgeTrainConfig(
                manifest=args.manifest,
                output_dir=args.output_dir,
                backbone=args.backbone,
                model_id=args.model_id,
                epochs=args.epochs,
                max_items=args.max_items,
                lr=args.lr,
                eval_manifest=args.eval_manifest,
                encoder=args.encoder,
                encoder_id=args.encoder_id,
                max_tiles=args.max_tiles,
                projector=args.projector,
                visual_tokens=args.visual_tokens,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                warmup_ratio=args.warmup_ratio,
                num_workers=args.num_workers,
                save_every=args.save_every,
                feature_cache_dir=args.feature_cache_dir,
                init_checkpoint=args.init_checkpoint,
                init_lora_dir=args.init_lora_dir,
                grad_checkpointing=args.grad_checkpointing,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                lora_targets=tuple(args.lora_targets),
                device=args.device,
                vision_device=args.vision_device,
            )
        )
        print(json.dumps({"checkpoint": str(checkpoint)}))

    asyncio.run(run())


def _load_projector_spec(checkpoint: Path) -> tuple[VisualProjectorSpec, dict]:
    spec_row = json.loads((checkpoint.parent / "projector_spec.json").read_text(encoding="utf-8"))
    spec = VisualProjectorSpec(
        input_dim=int(spec_row["input_dim"]),
        embedding_dim=int(spec_row["embedding_dim"]),
        hidden_dim=int(spec_row["hidden_dim"]),
        projector=spec_row.get("projector", "mlp"),
        visual_tokens=int(spec_row.get("visual_tokens", 64)),
        encoder=spec_row.get("encoder", "pil"),
        encoder_id=spec_row.get("encoder_id", ""),
        max_tiles=int(spec_row.get("max_tiles", 4)),
        patch_px=int(spec_row.get("patch_px", 32)),
    )
    return spec, spec_row


def _lora_dir(checkpoint: Path, spec_row: dict) -> Path | None:
    directory = checkpoint.parent / "lora"
    return directory if int(spec_row.get("lora_rank", 0)) > 0 or directory.exists() else None


def _ask_image(args: argparse.Namespace) -> None:
    async def run() -> None:
        spec, spec_row = _load_projector_spec(args.checkpoint)
        pipeline = await LagunaVisionImagePipeline.from_checkpoint(
            checkpoint=args.checkpoint,
            spec=spec,
            backbone_name=args.backbone or spec_row.get("backbone", "laguna"),
            model_id=args.model_id or spec_row["model_id"],
            device=args.device,
            vision_device=args.vision_device,
            lora_dir=_lora_dir(args.checkpoint, spec_row),
        )
        print(await pipeline.answer_image(args.image, args.question))

    asyncio.run(run())


def _eval_ablation(args: argparse.Namespace) -> None:
    async def run() -> None:
        spec, spec_row = _load_projector_spec(args.checkpoint)
        summary = await run_ablation(
            AblationConfig(
                manifest=args.manifest,
                checkpoint=args.checkpoint,
                output=args.output,
                spec=spec,
                backbone_name=args.backbone or spec_row.get("backbone", "laguna"),
                model_id=args.model_id or spec_row["model_id"],
                device=args.device,
                vision_device=args.vision_device,
                limit=args.limit,
                capability_threshold=args.threshold,
                lora_dir=_lora_dir(args.checkpoint, spec_row),
            )
        )
        print(json.dumps(summary, indent=2))

    asyncio.run(run())


if __name__ == "__main__":
    main()
