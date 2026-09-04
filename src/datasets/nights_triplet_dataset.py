from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


COLUMN_ALIASES = {
    "reference": ["reference", "ref", "img_ref", "anchor", "image_ref"],
    "candidate_a": ["candidate_a", "a", "img_a", "left", "image_a"],
    "candidate_b": ["candidate_b", "b", "img_b", "right", "image_b"],
    "choice": ["choice", "label", "target", "human_choice", "winner"],
}


class NightsTripletDataset(Dataset):
    """Triplet dataset returning (reference, candidate_a, candidate_b, choice)."""

    def __init__(
        self,
        split_csv: str,
        images_root: Optional[str] = None,
        transform=None,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.images_root = Path(images_root) if images_root else None
        self.transform = transform

        df = pd.read_csv(self.split_csv)
        self.columns = self._resolve_columns(df)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        ref_path = self._resolve_path(row[self.columns["reference"]])
        a_path = self._resolve_path(row[self.columns["candidate_a"]])
        b_path = self._resolve_path(row[self.columns["candidate_b"]])

        ref_img = self._load_image(ref_path)
        a_img = self._load_image(a_path)
        b_img = self._load_image(b_path)

        if self.transform is not None:
            ref_img = self.transform(ref_img)
            a_img = self.transform(a_img)
            b_img = self.transform(b_img)

        label = int(row[self.columns["choice"]])

        return {
            "reference": ref_img,
            "candidate_a": a_img,
            "candidate_b": b_img,
            "choice": label,
            "reference_path": str(ref_path),
            "candidate_a_path": str(a_path),
            "candidate_b_path": str(b_path),
        }

    @staticmethod
    def _resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
        df_cols = {c.lower(): c for c in df.columns}
        out = {}

        for canonical, aliases in COLUMN_ALIASES.items():
            match = next((df_cols[a] for a in aliases if a in df_cols), None)
            if match is None:
                raise ValueError(
                    f"Missing required column for '{canonical}'. "
                    f"Expected one of: {aliases}. Found: {list(df.columns)}"
                )
            out[canonical] = match

        return out

    def _resolve_path(self, raw_path: str) -> Path:
        p = Path(str(raw_path))
        if p.is_absolute():
            return p
        if self.images_root is None:
            return p
        return self.images_root / p

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def check_missing_files(self) -> pd.DataFrame:
        """Return rows with at least one missing image path."""
        rows = []
        for i, row in self.df.iterrows():
            ref_p = self._resolve_path(row[self.columns["reference"]])
            a_p = self._resolve_path(row[self.columns["candidate_a"]])
            b_p = self._resolve_path(row[self.columns["candidate_b"]])
            if not (ref_p.exists() and a_p.exists() and b_p.exists()):
                rows.append(
                    {
                        "row_index": i,
                        "reference_exists": ref_p.exists(),
                        "candidate_a_exists": a_p.exists(),
                        "candidate_b_exists": b_p.exists(),
                    }
                )

        return pd.DataFrame(rows)
