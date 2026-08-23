from __future__ import annotations

from torch import nn

from .legacy_ltc_k2_p1 import ARCHITECTURE_ID, LegacyLtcK2P1Compiler


def create_model(architecture_id: str, *, width: int) -> nn.Module:
    if architecture_id == ARCHITECTURE_ID:
        return LegacyLtcK2P1Compiler(width=width)
    raise ValueError(f"unsupported learning architecture {architecture_id!r}")
