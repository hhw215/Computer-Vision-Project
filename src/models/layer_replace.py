from collections.abc import Iterable
from typing import Callable, List, Optional

import torch.nn as nn


AttentionFactory = Callable[[nn.Module, int], nn.Module]


def _resolve_vit_blocks(model: nn.Module):
    backbone = getattr(model, "backbone", model)
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise AttributeError("Model does not expose a ViT-style 'blocks' attribute.")
    return blocks


def replace_attention_modules(
    model: nn.Module,
    attention_factory: AttentionFactory,
    block_indices: Optional[Iterable[int]] = None,
) -> List[str]:
    """Replace block attention modules and return the replaced block paths."""
    blocks = _resolve_vit_blocks(model)

    if block_indices is None:
        block_indices = range(len(blocks))

    replaced_paths: List[str] = []
    for block_index in block_indices:
        old_attention = blocks[block_index].attn
        blocks[block_index].attn = attention_factory(old_attention, block_index)
        replaced_paths.append(f"blocks.{block_index}.attn")

    return replaced_paths