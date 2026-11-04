"""Subpixel trace decoding and confidence estimates."""

from __future__ import annotations

from typing import NamedTuple

import torch


class DecodedTrace(NamedTuple):
    y: torch.Tensor
    variance: torch.Tensor
    entropy: torch.Tensor
    existence_probability: torch.Tensor
    accepted: torch.Tensor


def decode_ordinate_logits(
    ordinate_logits: torch.Tensor,
    existence_logits: torch.Tensor,
    top_k: int = 5,
    existence_threshold: float = 0.5,
    maximum_variance: float | None = None,
) -> DecodedTrace:
    """Decode column distributions without forcing low-confidence points."""
    if ordinate_logits.ndim != 4 or ordinate_logits.shape[1] != 1:
        raise ValueError("ordinate_logits must have shape [B, 1, H, W]")
    if existence_logits.shape != (
        ordinate_logits.shape[0], 1, ordinate_logits.shape[3]
    ):
        raise ValueError("existence_logits must have shape [B, 1, W]")
    height = ordinate_logits.shape[2]
    selected = min(max(int(top_k), 1), height)
    values, indices = ordinate_logits[:, 0].topk(selected, dim=1)
    probabilities = values.softmax(dim=1)
    rows = indices.to(ordinate_logits.dtype)
    y = (probabilities * rows).sum(dim=1)
    variance = (probabilities * (rows - y[:, None]).square()).sum(dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
    existence = existence_logits[:, 0].sigmoid()
    accepted = existence >= existence_threshold
    if maximum_variance is not None:
        accepted &= variance <= maximum_variance
    return DecodedTrace(y, variance, entropy, existence, accepted)
