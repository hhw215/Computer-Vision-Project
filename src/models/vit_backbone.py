from dataclasses import dataclass
from typing import Optional, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_BACKBONES: Sequence[str] = (
    "deit_tiny_patch16_224",
    "deit_small_patch16_224",
    "deit_base_patch16_224",
)


@dataclass(frozen=True)
class BackboneConfig:
    model_name: str = "deit_small_patch16_224"
    pretrained: bool = True
    image_size: int = 224
    proj_dim: Optional[int] = None
    drop_path_rate: float = 0.0
    l2_normalize: bool = True


class VisionTransformerEncoder(nn.Module):
    """Thin wrapper around timm ViTs that returns image embeddings."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            num_classes=0,
            img_size=config.image_size,
            drop_path_rate=config.drop_path_rate,
        )

        self.embedding_dim = int(self.backbone.num_features)
        output_dim = config.proj_dim or self.embedding_dim
        self.projection = (
            nn.Identity()
            if output_dim == self.embedding_dim
            else nn.Linear(self.embedding_dim, output_dim)
        )
        self.output_dim = output_dim

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)

        if isinstance(features, (tuple, list)):
            features = features[0]

        if features.ndim == 3:
            embeddings = features[:, 0]
        elif features.ndim == 2:
            embeddings = features
        else:
            raise ValueError(f"Unsupported feature shape: {tuple(features.shape)}")

        embeddings = self.projection(embeddings)

        if self.config.l2_normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(images)


def list_vit_backbones() -> Sequence[str]:
    return DEFAULT_BACKBONES