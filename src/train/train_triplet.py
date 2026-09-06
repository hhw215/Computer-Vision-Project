import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.datasets.nights_triplet_dataset import NightsTripletDataset
from src.datasets.transforms import build_transforms
from src.models import (
    BackboneConfig,
    VisionTransformerEncoder,
    make_attention_factory,
    parse_replace_layers,
    replace_attention_modules,
    resolve_block_indices,
)
from src.pipeline.dreamsim_like import DreamSimLikePipeline


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy_2afc: float
    epoch_seconds: float
    lr_backbone: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DreamSim-like triplet model with optional MoH attention.")

    parser.add_argument("--train_csv", type=str, default="dataset/splits/train.csv")
    parser.add_argument("--val_csv", type=str, default="dataset/splits/val.csv")
    parser.add_argument("--images_root", type=str, default="dataset/raw")

    parser.add_argument("--model_name", type=str, default="deit_small_patch16_224")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--drop_path_rate", type=float, default=0.0)

    parser.add_argument("--attention_type", type=str, choices=["standard", "moh"], default="standard")
    parser.add_argument("--replace_layers", type=str, default="none", help="none | all | comma-separated block indices")
    parser.add_argument("--top_k", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--router_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs, 0 disables periodic saves")
    return parser.parse_args()


def build_encoder(args: argparse.Namespace) -> VisionTransformerEncoder:
    encoder = VisionTransformerEncoder(
        BackboneConfig(
            model_name=args.model_name,
            pretrained=args.pretrained,
            image_size=args.image_size,
            drop_path_rate=args.drop_path_rate,
        )
    )

    replace_layers = parse_replace_layers(args.replace_layers)
    block_indices = resolve_block_indices(len(encoder.backbone.blocks), replace_layers)

    if args.attention_type != "standard":
        replace_attention_modules(
            encoder,
            make_attention_factory(args.attention_type, top_k=args.top_k),
            block_indices=block_indices,
        )

    return encoder


def build_loaders(args: argparse.Namespace, device: str):
    train_ds = NightsTripletDataset(
        split_csv=args.train_csv,
        images_root=args.images_root,
        transform=build_transforms(image_size=args.image_size, train=True),
    )
    val_ds = NightsTripletDataset(
        split_csv=args.val_csv,
        images_root=args.images_root,
        transform=build_transforms(image_size=args.image_size, train=False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    return train_loader, val_loader


def batch_ranking_loss(sim_a: torch.Tensor, sim_b: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    sign = torch.where(label == 0, torch.ones_like(sim_a), -torch.ones_like(sim_a))
    margin = sim_a - sim_b
    return F.softplus(-sign * margin).mean()


def forward_triplet(pipeline: DreamSimLikePipeline, batch: dict, device: str):
    ref = batch["reference"].to(device, non_blocking=True)
    cand_a = batch["candidate_a"].to(device, non_blocking=True)
    cand_b = batch["candidate_b"].to(device, non_blocking=True)
    label = batch["choice"].to(device, non_blocking=True)
    out = pipeline(ref, cand_a, cand_b)
    return out, label


def evaluate_val(
    pipeline: DreamSimLikePipeline,
    loader: DataLoader,
    device: str,
    amp_enabled: bool,
) -> tuple[float, float]:
    pipeline.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in loader:
            with autocast(enabled=amp_enabled):
                out, label = forward_triplet(pipeline, batch, device)
                loss = batch_ranking_loss(out["sim_a"], out["sim_b"], label)

            total_loss += loss.item() * label.numel()
            total += label.numel()
            correct += (out["pred"] == label).sum().item()

    return total_loss / max(total, 1), correct / max(total, 1)


def build_optimizer(
    encoder: nn.Module,
    attention_type: str,
    lr: float,
    router_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if attention_type == "moh":
        router_params = []
        base_params = []
        for name, param in encoder.named_parameters():
            if not param.requires_grad:
                continue
            if ".router." in name:
                router_params.append(param)
            else:
                base_params.append(param)

        param_groups = [{"params": base_params, "lr": lr, "weight_decay": weight_decay}]
        if router_params:
            param_groups.append({"params": router_params, "lr": router_lr, "weight_decay": weight_decay})
        return torch.optim.AdamW(param_groups)

    return torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    run_name = args.run_name.strip() or (
        f"{args.model_name}_{args.attention_type}_layers-{args.replace_layers}_topk-{args.top_k}_seed-{args.seed}"
    )

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    logs_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    history_csv = logs_dir / f"train_history_{run_name}.csv"
    best_json = logs_dir / f"best_metrics_{run_name}.json"
    best_ckpt = ckpt_dir / f"best_{run_name}.pt"
    last_ckpt = ckpt_dir / f"last_{run_name}.pt"

    encoder = build_encoder(args).to(args.device)
    pipeline = DreamSimLikePipeline(encoder).to(args.device)

    train_loader, val_loader = build_loaders(args, args.device)
    optimizer = build_optimizer(
        encoder=encoder,
        attention_type=args.attention_type,
        lr=args.lr,
        router_lr=args.router_lr,
        weight_decay=args.weight_decay,
    )

    total_steps = max(1, args.epochs * max(1, len(train_loader)))
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        floor = args.min_lr / max(args.lr, 1e-12)
        return max(floor, cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    amp_enabled = args.device.startswith("cuda")
    scaler = GradScaler(enabled=amp_enabled)

    best_acc = -1.0
    best_row = None
    global_step = 0

    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(EpochMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)).keys()))
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            pipeline.train()
            epoch_start = time.perf_counter()
            train_loss_sum = 0.0
            train_count = 0

            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=amp_enabled):
                    out, label = forward_triplet(pipeline, batch, args.device)
                    loss = batch_ranking_loss(out["sim_a"], out["sim_b"], label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss_sum += loss.item() * label.numel()
                train_count += label.numel()
                global_step += 1

            train_loss = train_loss_sum / max(train_count, 1)
            val_loss, val_acc = evaluate_val(
                pipeline=pipeline,
                loader=val_loader,
                device=args.device,
                amp_enabled=amp_enabled,
            )
            epoch_s = time.perf_counter() - epoch_start

            row = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy_2afc=val_acc,
                epoch_seconds=epoch_s,
                lr_backbone=optimizer.param_groups[0]["lr"],
            )
            writer.writerow(asdict(row))
            f.flush()

            print(
                f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                f"val_acc={val_acc:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                best_row = asdict(row)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": pipeline.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "args": vars(args),
                        "best_val_accuracy_2afc": best_acc,
                    },
                    best_ckpt,
                )

            if args.save_every > 0 and epoch % args.save_every == 0:
                periodic_path = ckpt_dir / f"epoch{epoch:03d}_{run_name}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": pipeline.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "args": vars(args),
                        "best_val_accuracy_2afc": best_acc,
                    },
                    periodic_path,
                )

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": pipeline.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
            "best_val_accuracy_2afc": best_acc,
        },
        last_ckpt,
    )

    best_payload = {
        "run_name": run_name,
        "best_val_accuracy_2afc": best_acc,
        "best_epoch": best_row["epoch"] if best_row is not None else None,
        "best_row": best_row,
        "history_csv": str(history_csv),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
    }
    best_json.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    print(json.dumps(best_payload, indent=2))


if __name__ == "__main__":
    main()
