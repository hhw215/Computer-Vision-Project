from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn


class MoHAttention(nn.Module):
    """Mixture-of-Heads attention wrapper with optional top-k head routing."""

    def __init__(self, old_attn: nn.Module, top_k: int | None = 4):
        super().__init__()
        self.num_heads = old_attn.num_heads
        self.head_dim = old_attn.head_dim
        self.attn_dim = old_attn.attn_dim
        self.scale = old_attn.scale

        self.qkv = copy.deepcopy(old_attn.qkv)
        self.q_norm = copy.deepcopy(old_attn.q_norm)
        self.k_norm = copy.deepcopy(old_attn.k_norm)
        self.attn_drop = copy.deepcopy(old_attn.attn_drop)
        self.norm = copy.deepcopy(old_attn.norm)
        self.proj = copy.deepcopy(old_attn.proj)
        self.proj_drop = copy.deepcopy(old_attn.proj_drop)

        dim = old_attn.qkv.in_features
        self.router = nn.Linear(dim, self.num_heads)
        self.top_k = top_k

    def forward(self, x, attn_mask=None, is_causal=False):
        bsz, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(
            bsz, seq_len, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        head_out = attn @ v

        route_logits = self.router(x.mean(dim=1))

        if self.top_k is not None and self.top_k < self.num_heads:
            topk_vals, topk_idx = route_logits.topk(self.top_k, dim=-1)
            masked_logits = route_logits.new_full(route_logits.shape, float("-inf"))
            masked_logits.scatter_(1, topk_idx, topk_vals)
            route_logits = masked_logits

        head_weights = route_logits.softmax(dim=-1)
        head_out = head_out * head_weights[:, :, None, None]

        out = head_out.transpose(1, 2).reshape(bsz, seq_len, self.attn_dim)
        out = self.norm(out)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class PyraAttention(nn.Module):
    """Pyramidal head-routing attention wrapper with optional top-k routing."""

    def __init__(self, old_attn: nn.Module, top_k: int | None = 4):
        super().__init__()
        self.num_heads = old_attn.num_heads
        self.head_dim = old_attn.head_dim
        self.attn_dim = old_attn.attn_dim
        self.scale = old_attn.scale

        self.qkv = copy.deepcopy(old_attn.qkv)
        self.q_norm = copy.deepcopy(old_attn.q_norm)
        self.k_norm = copy.deepcopy(old_attn.k_norm)
        self.attn_drop = copy.deepcopy(old_attn.attn_drop)
        self.norm = copy.deepcopy(old_attn.norm)
        self.proj = copy.deepcopy(old_attn.proj)
        self.proj_drop = copy.deepcopy(old_attn.proj_drop)

        dim = old_attn.qkv.in_features
        self.router = nn.Linear(dim, self.num_heads)
        self.top_k = top_k

    def forward(self, x, attn_mask=None, is_causal=False):
        bsz, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(
            bsz, seq_len, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        head_out = attn @ v

        route_logits = self.router(x.mean(dim=1))
        # Encourage lower-index heads to carry slightly more mass (pyramidal prior).
        head_prior = torch.linspace(
            1.0,
            0.0,
            self.num_heads,
            device=route_logits.device,
            dtype=route_logits.dtype,
        )
        route_logits = route_logits + head_prior.unsqueeze(0)

        if self.top_k is not None and self.top_k < self.num_heads:
            topk_vals, topk_idx = route_logits.topk(self.top_k, dim=-1)
            masked_logits = route_logits.new_full(route_logits.shape, float("-inf"))
            masked_logits.scatter_(1, topk_idx, topk_vals)
            route_logits = masked_logits

        head_weights = route_logits.softmax(dim=-1)
        head_out = head_out * head_weights[:, :, None, None]

        out = head_out.transpose(1, 2).reshape(bsz, seq_len, self.attn_dim)
        out = self.norm(out)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MetaAttention(nn.Module):
    """Meta-routed attention wrapper using a small MLP router."""

    def __init__(self, old_attn: nn.Module, top_k: int | None = 4):
        super().__init__()
        self.num_heads = old_attn.num_heads
        self.head_dim = old_attn.head_dim
        self.attn_dim = old_attn.attn_dim
        self.scale = old_attn.scale

        self.qkv = copy.deepcopy(old_attn.qkv)
        self.q_norm = copy.deepcopy(old_attn.q_norm)
        self.k_norm = copy.deepcopy(old_attn.k_norm)
        self.attn_drop = copy.deepcopy(old_attn.attn_drop)
        self.norm = copy.deepcopy(old_attn.norm)
        self.proj = copy.deepcopy(old_attn.proj)
        self.proj_drop = copy.deepcopy(old_attn.proj_drop)

        dim = old_attn.qkv.in_features
        hidden = max(32, dim // 4)
        self.meta_router = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_heads),
        )
        self.top_k = top_k

    def forward(self, x, attn_mask=None, is_causal=False):
        bsz, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(
            bsz, seq_len, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        head_out = attn @ v

        route_logits = self.meta_router(x.mean(dim=1))

        if self.top_k is not None and self.top_k < self.num_heads:
            topk_vals, topk_idx = route_logits.topk(self.top_k, dim=-1)
            masked_logits = route_logits.new_full(route_logits.shape, float("-inf"))
            masked_logits.scatter_(1, topk_idx, topk_vals)
            route_logits = masked_logits

        head_weights = route_logits.softmax(dim=-1)
        head_out = head_out * head_weights[:, :, None, None]

        out = head_out.transpose(1, 2).reshape(bsz, seq_len, self.attn_dim)
        out = self.norm(out)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


def parse_replace_layers(value: str) -> str | list[int] | None:
    lowered = value.strip().lower()
    if lowered in {"none", "", "null"}:
        return None
    if lowered == "all":
        return "all"
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def resolve_block_indices(num_blocks: int, replace_layers: str | Iterable[int] | None) -> list[int]:
    if replace_layers is None:
        return []
    if replace_layers == "all":
        return list(range(num_blocks))
    return list(replace_layers)


def make_attention_factory(attention_type: str, top_k: int | None = 4):
    if attention_type == "standard":
        return lambda old_attn, block_idx: old_attn
    if attention_type == "moh":
        return lambda old_attn, block_idx: MoHAttention(old_attn, top_k=top_k)
    if attention_type == "pyra":
        return lambda old_attn, block_idx: PyraAttention(old_attn, top_k=top_k)
    if attention_type == "meta":
        return lambda old_attn, block_idx: MetaAttention(old_attn, top_k=top_k)
    raise ValueError(f"Unknown attention_type: {attention_type}")
