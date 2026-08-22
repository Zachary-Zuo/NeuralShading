from __future__ import annotations

import torch
from torch import nn

from closures.torch_eval import RAW_RESIDUAL_DIMENSION
from model.features import CONTINUOUS_FEATURE_COUNT
from schema import MAX_LAYERS


class RecurrentCompilerBaseline(nn.Module):
    """Small order-sensitive P1 baseline from layer tokens and view to LTC-K2."""

    def __init__(self, width: int = 64, type_width: int = 8) -> None:
        super().__init__()
        self.width = width
        self.type_embedding = nn.Embedding(4, type_width)
        self.layer_encoder = nn.Sequential(
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
        layer_types: torch.Tensor,
        continuous: torch.Tensor,
        layer_counts: torch.Tensor,
    ) -> torch.Tensor:
        if layer_types.shape[1] != MAX_LAYERS or continuous.shape[1] != MAX_LAYERS:
            raise ValueError(f"compiler expects exactly {MAX_LAYERS} padded layer slots")
        tokens = self.layer_encoder(
            torch.cat((self.type_embedding(layer_types), continuous), dim=-1)
        )
        state = torch.zeros(
            (len(layer_types), self.width), dtype=continuous.dtype, device=continuous.device
        )
        for layer_index in range(MAX_LAYERS):
            candidate = self.compose(tokens[:, layer_index], state)
            active = (layer_index < layer_counts)[:, None]
            state = torch.where(active, candidate, state)
        return state

    def forward(
        self,
        layer_types: torch.Tensor,
        continuous: torch.Tensor,
        layer_counts: torch.Tensor,
        view_directions: torch.Tensor,
    ) -> torch.Tensor:
        stack = self.encode_stack(layer_types, continuous, layer_counts)
        view = self.view_encoder(view_directions)
        return self.head(torch.cat((stack, view), dim=-1))
