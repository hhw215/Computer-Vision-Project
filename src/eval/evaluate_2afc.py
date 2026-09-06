import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.datasets.nights_triplet_dataset import NightsTripletDataset
from src.datasets.transforms import build_transforms
from src.models import BackboneConfig, VisionTransformerEncoder
from src.pipeline.dreamsim_like import DreamSimLikePipeline


def evaluate(
    split_csv: str,
    images_root: str,
    model_name: str,
    pretrained: bool,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
    max_batches: int | None = None,
    encoder: torch.nn.Module | None = None,
) -> Dict[str, float]:
    transform = build_transforms(image_size=image_size, train=False)
    dataset = NightsTripletDataset(
        split_csv=split_csv,
        images_root=images_root,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    if encoder is None:
        encoder = VisionTransformerEncoder(
            BackboneConfig(
                model_name=model_name,
                pretrained=pretrained,
                image_size=image_size,
            )
        )
    encoder = encoder.to(device)
    encoder.eval()

    pipeline = DreamSimLikePipeline(encoder).to(device)
    pipeline.eval()

    total = 0
    correct = 0
    elapsed_s = 0.0
    batches_ran = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            ref = batch["reference"].to(device, non_blocking=True)
            cand_a = batch["candidate_a"].to(device, non_blocking=True)
            cand_b = batch["candidate_b"].to(device, non_blocking=True)
            label = batch["choice"].to(device, non_blocking=True)

            start = time.perf_counter()
            out = pipeline(ref, cand_a, cand_b)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed_s += time.perf_counter() - start

            pred = out["pred"]
            correct += (pred == label).sum().item()
            total += label.numel()
            batches_ran += 1

    if total == 0:
        raise RuntimeError("No evaluation samples available. Check split CSV and image paths.")

    accuracy = correct / total
    images_processed = total * 3
    ms_per_image = (elapsed_s * 1000.0 / images_processed) if images_processed > 0 else float("nan")
    images_per_second = (images_processed / elapsed_s) if elapsed_s > 0 else float("inf")

    metrics = {
        "samples": total,
        "batches": batches_ran,
        "accuracy_2afc": accuracy,
        "ms_per_image": ms_per_image,
        "images_per_second": images_per_second,
        "elapsed_seconds": elapsed_s,
    }

    if device.startswith("cuda"):
        metrics["max_vram_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DreamSim-like 2AFC accuracy and speed.")
    parser.add_argument("--split_csv", type=str, default="dataset/splits/val.csv")
    parser.add_argument("--images_root", type=str, default="dataset/raw")
    parser.add_argument("--model_name", type=str, default="deit_small_patch16_224")
    parser.add_argument("--pretrained", action="store_true", help="Use timm pretrained weights.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--out_json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    metrics = evaluate(
        split_csv=args.split_csv,
        images_root=args.images_root,
        model_name=args.model_name,
        pretrained=args.pretrained,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_batches=args.max_batches,
    )

    print(json.dumps(metrics, indent=2))

    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
