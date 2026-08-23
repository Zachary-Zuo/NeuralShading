from __future__ import annotations

import torch
from torch import nn

from ncls.core.material import MAX_INTERFACES
from ncls.core.representations.legacy_ltc_k2.torch_eval import RAW_RESIDUAL_DIMENSION
from ncls.learning.features import CONTINUOUS_FEATURE_COUNT


ARCHITECTURE_ID = "legacy-ltc-k2-p1@2"


class LegacyLtcK2P1Compiler(nn.Module):
    """迁移后的历史 P1 基线；名称明确表明它不是最终网络。"""

    architecture_id = ARCHITECTURE_ID
    representation_id = "legacy-ltc-k2@1"

    def __init__(self, width: int = 64, type_width: int = 8) -> None:
        super().__init__()
        if width < 8 or type_width < 1:
            raise ValueError("compiler widths are too small")
        self.width = width
        self.type_width = type_width
        self.type_embedding = nn.Embedding(4, type_width)
        self.interface_encoder = nn.Sequential(
            nn.Linear(CONTINUOUS_FEATURE_COUNT + type_width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.compose = nn.GRUCell(width, width)
        self.view_encoder = nn.Sequential(nn.Linear(3, width // 2), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(width + width // 2, width),
            nn.SiLU(),
            nn.Linear(width, RAW_RESIDUAL_DIMENSION),
        )
        final = self.head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[[0, 1, 2, 9, 10, 11]] = -5.0

    def encode_stack(
        self,
        interface_kinds: torch.Tensor,
        continuous: torch.Tensor,
        interface_counts: torch.Tensor,
    ) -> torch.Tensor:
        if interface_kinds.shape[1] != MAX_INTERFACES or continuous.shape[1] != MAX_INTERFACES:
            raise ValueError(f"compiler expects exactly {MAX_INTERFACES} padded interface slots")
        tokens = self.interface_encoder(
            torch.cat((self.type_embedding(interface_kinds), continuous), dim=-1)
        )
        state = torch.zeros(
            (len(interface_kinds), self.width),
            dtype=continuous.dtype,
            device=continuous.device,
        )
        for interface_index in range(MAX_INTERFACES):
            candidate = self.compose(tokens[:, interface_index], state)
            active = (interface_index < interface_counts)[:, None]
            state = torch.where(active, candidate, state)
        return state

    def forward(
        self,
        interface_kinds: torch.Tensor,
        continuous: torch.Tensor,
        interface_counts: torch.Tensor,
        view_directions: torch.Tensor,
    ) -> torch.Tensor:
        stack = self.encode_stack(interface_kinds, continuous, interface_counts)
        view = self.view_encoder(view_directions)
        return self.head(torch.cat((stack, view), dim=-1))
