from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TripletScores:
    sim_a: torch.Tensor
    sim_b: torch.Tensor
    pred: torch.Tensor


class DreamSimLikePipeline(nn.Module):
    """Compute triplet similarity decisions from image embeddings."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def score_triplet(
        self,
        reference: torch.Tensor,
        candidate_a: torch.Tensor,
        candidate_b: torch.Tensor,
    ) -> TripletScores:
        e_ref = self.embed(reference)
        e_a = self.embed(candidate_a)
        e_b = self.embed(candidate_b)

        sim_a = F.cosine_similarity(e_ref, e_a, dim=-1)
        sim_b = F.cosine_similarity(e_ref, e_b, dim=-1)
        pred = torch.where(sim_a >= sim_b, 0, 1).long()

        return TripletScores(sim_a=sim_a, sim_b=sim_b, pred=pred)

    def forward(
        self,
        reference: torch.Tensor,
        candidate_a: torch.Tensor,
        candidate_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out = self.score_triplet(reference, candidate_a, candidate_b)
        return {"sim_a": out.sim_a, "sim_b": out.sim_b, "pred": out.pred}
