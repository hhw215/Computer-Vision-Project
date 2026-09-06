import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REQUIRED_TRIPLET_COLUMNS = (
    "reference",
    "candidate_a",
    "candidate_b",
    "choice",
)


def _validate_triplet_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_TRIPLET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Triplet CSV is missing required columns "
            f"{missing}. Found: {list(df.columns)}"
        )


def _validate_binary_choice(series: pd.Series) -> pd.Series:
    labels = pd.to_numeric(series, errors="raise").astype(int)
    invalid = sorted(v for v in labels.unique().tolist() if v not in (0, 1))
    if invalid:
        raise ValueError(f"Choice labels must be binary 0/1. Found invalid values: {invalid}")
    return labels


def _stratified_indices(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for label in np.unique(labels):
        cls_indices = np.where(labels == label)[0]
        rng.shuffle(cls_indices)

        n = len(cls_indices)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val

        train_idx.extend(cls_indices[:n_train])
        val_idx.extend(cls_indices[n_train : n_train + n_val])
        test_idx.extend(cls_indices[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def _group_split_indices(
    groups: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)

    n = len(unique_groups)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train : n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val :])

    all_idx = np.arange(len(groups))
    train_idx = all_idx[np.array([g in train_groups for g in groups])]
    val_idx = all_idx[np.array([g in val_groups for g in groups])]
    test_idx = all_idx[np.array([g in test_groups for g in groups])]

    return train_idx, val_idx, test_idx


def _print_split_stats(name: str, df: pd.DataFrame) -> None:
    total = len(df)
    label_counts = df["choice"].value_counts().sort_index().to_dict() if total > 0 else {}
    print(f"{name}: {total} rows | labels: {label_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible triplet splits for NIGHTS-like datasets.")
    parser.add_argument("--triplets_csv", type=str, required=True, help="Input CSV containing triplet paths and labels.")
    parser.add_argument("--out_dir", type=str, default="dataset/splits", help="Output directory for train/val/test CSV files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument(
        "--group_col",
        type=str,
        default=None,
        help="Optional group column for leakage-safe group split (for example scene_id).",
    )
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}.")

    df = pd.read_csv(args.triplets_csv)
    _validate_triplet_columns(df)

    clean = pd.DataFrame(
        {
            "reference": df["reference"].astype(str),
            "candidate_a": df["candidate_a"].astype(str),
            "candidate_b": df["candidate_b"].astype(str),
            "choice": _validate_binary_choice(df["choice"]),
        }
    )

    if args.group_col is not None:
        if args.group_col not in df.columns:
            raise ValueError(f"group_col '{args.group_col}' not found in CSV columns.")
        clean["group"] = df[args.group_col].astype(str)
        train_idx, val_idx, test_idx = _group_split_indices(
            groups=clean["group"].to_numpy(),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        train_idx, val_idx, test_idx = _stratified_indices(
            labels=clean["choice"].to_numpy(),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    train_df = clean.iloc[train_idx].reset_index(drop=True)
    val_df = clean.iloc[val_idx].reset_index(drop=True)
    test_df = clean.iloc[test_idx].reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print(f"Saved splits to: {out_dir.resolve()}")
    _print_split_stats("train", train_df)
    _print_split_stats("val", val_df)
    _print_split_stats("test", test_df)


if __name__ == "__main__":
    main()
