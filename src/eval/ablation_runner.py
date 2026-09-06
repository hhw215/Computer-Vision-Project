import csv
from pathlib import Path
from typing import Dict, List, Optional

from src.eval.evaluate_2afc import evaluate
from src.models import BackboneConfig, VisionTransformerEncoder, replace_attention_modules
from src.models.variants import make_attention_factory, resolve_block_indices


EXPERIMENTS: List[Dict] = [
    # Old baseline-only set (kept for reference):
    # {
    #     "name": "baseline_deit_tiny",
    #     "model_name": "deit_tiny_patch16_224",
    #     "pretrained": False,
    #     "attention_type": "standard",
    #     "replace_layers": None,
    # },
    # {
    #     "name": "baseline_deit_small",
    #     "model_name": "deit_small_patch16_224",
    #     "pretrained": False,
    #     "attention_type": "standard",
    #     "replace_layers": None,
    # },
    # {
    #     "name": "baseline_deit_base",
    #     "model_name": "deit_base_patch16_224",
    #     "pretrained": False,
    #     "attention_type": "standard",
    #     "replace_layers": None,
    # },

    # Active MoH ablations:
    {
        "name": "baseline_deit_small",
        "model_name": "deit_small_patch16_224",
        "pretrained": False,
        "attention_type": "standard",
        "replace_layers": None,
    },
    {
        "name": "moh_all_deit_small",
        "model_name": "deit_small_patch16_224",
        "pretrained": False,
        "attention_type": "moh",
        "replace_layers": "all",
    },
    {
        "name": "moh_last6_deit_small",
        "model_name": "deit_small_patch16_224",
        "pretrained": False,
        "attention_type": "moh",
        "replace_layers": [6, 7, 8, 9, 10, 11],
    },
    {
        "name": "moh_first6_deit_small",
        "model_name": "deit_small_patch16_224",
        "pretrained": False,
        "attention_type": "moh",
        "replace_layers": [0, 1, 2, 3, 4, 5],
    },
    {
        "name": "moh_every_other_deit_small",
        "model_name": "deit_small_patch16_224",
        "pretrained": False,
        "attention_type": "moh",
        "replace_layers": [0, 2, 4, 6, 8, 10],
    },
    {
        "name": "pyra_all_deit_small",
        "model_name": "deit_small_patch16_224",
        "pretrained": False,
        "attention_type": "pyra",
        "replace_layers": "all",
    },
    {
        "name": "meta_all_deit_small",
        "model_name": "deit_small_patch16_224",
        "pretrained": False,
        "attention_type": "meta",
        "replace_layers": "all",
    },
]


def format_replace_layers(value: Optional[object]) -> str:
    if value is None:
        return "none"
    if value == "all":
        return "all"
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)

def build_encoder(exp: Dict) -> VisionTransformerEncoder:
    encoder = VisionTransformerEncoder(
        BackboneConfig(
            model_name=exp["model_name"],
            pretrained=exp["pretrained"],
            image_size=224,
        )
    )

    block_indices = resolve_block_indices(len(encoder.backbone.blocks), exp["replace_layers"])
    if exp["attention_type"] != "standard":
        replace_attention_modules(
            encoder,
            make_attention_factory(exp["attention_type"], top_k=4),
            block_indices=block_indices,
        )

    return encoder

def run_experiment(exp: Dict) -> Dict:
    encoder = build_encoder(exp)

    metrics = evaluate(
        split_csv="dataset/splits/val.csv",
        images_root="dataset/raw",
        model_name=exp["model_name"],
        pretrained=exp["pretrained"],
        image_size=224,
        batch_size=16,
        num_workers=4,
        device="cpu",
        max_batches=None,
        encoder=encoder,
    )

    row = {
        "experiment": exp["name"],
        "model_name": exp["model_name"],
        "pretrained": exp["pretrained"],
        "attention_type": exp["attention_type"],
        "replace_layers": format_replace_layers(exp["replace_layers"]),
        **metrics,
    }
    return row

def main() -> None:
    out_path = Path("outputs/tables/ablation_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for exp in EXPERIMENTS:
        row = run_experiment(exp)
        rows.append(row)
        print(row)

    if not rows:
        raise RuntimeError("No experiments configured.")

    fieldnames = list(rows[0].keys())
    file_exists = out_path.exists()

    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()