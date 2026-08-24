from __future__ import annotations

from typing import Mapping

import torch

from ncls.core.representations.legacy_ltc_k2.torch_eval import (
    LegacyLtcK2Tensors,
    eval_direct_top,
)


def layer_stack_direct_top_tensors(
    batch: Mapping[str, torch.Tensor],
    *,
    repeat_count: int = 1,
) -> LegacyLtcK2Tensors:
    """把显式 LayerStack adapter batch 转成 direct-top analytic core 输入。"""

    if repeat_count < 1:
        raise ValueError("direct-top repeat_count must be positive")

    def values(name: str) -> torch.Tensor:
        tensor = batch[name]
        return tensor if repeat_count == 1 else tensor.repeat_interleave(repeat_count, dim=0)

    count = len(batch["top_kind"]) * repeat_count
    device = batch["top_alpha"].device
    return LegacyLtcK2Tensors(
        interface_kind=values("top_kind").long(),
        alpha=values("top_alpha").float(),
        relative_ior=values("top_relative_ior").float(),
        eta=values("top_eta").float(),
        k=values("top_k").float(),
        color=values("top_color").float(),
        tangent_rotation=values("top_rotation").float(),
        amplitude=torch.zeros((count, 2, 3), dtype=torch.float32, device=device),
        inverse_scale=torch.ones((count, 2, 2), dtype=torch.float32, device=device),
        shear=torch.zeros((count, 2, 3), dtype=torch.float32, device=device),
        angle=torch.zeros((count, 2), dtype=torch.float32, device=device),
    )


def evaluate_layer_stack_direct_top(
    batch: Mapping[str, torch.Tensor],
    view: torch.Tensor,
    lights: torch.Tensor,
    *,
    repeat_count: int = 1,
) -> torch.Tensor:
    return eval_direct_top(
        layer_stack_direct_top_tensors(batch, repeat_count=repeat_count),
        view,
        lights,
    )
