from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ncls.learning.models.metal_fused_profile import MetalFusedProfile


ROLE_CLASS_NAMES = ("color", "normal", "scalar", "packed")


def semantic_role_class(semantic: str, channel_count: int) -> int:
    value = semantic.lower()
    if "normal" in value:
        return 1
    if channel_count == 1 or any(
        token in value
        for token in (
            "rough", "mask", "ao", "height", "bump", "dirt", "weight",
            "lookup", "variation", "noise", "scrape", "scratch", "wear",
        )
    ):
        return 2
    if channel_count in {2, 4} or any(
        token in value for token in ("packed", "metallic", "transition")
    ):
        return 3
    return 0


class _ResidualConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.input = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm0 = nn.GroupNorm(min(16, out_channels), out_channels)
        self.hidden = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(16, out_channels), out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm0(self.input(value)))
        hidden = self.norm1(self.hidden(hidden))
        return F.silu(hidden + self.skip(value))


class _SharedUNet(nn.Module):
    def __init__(self, input_width: int, widths: tuple[int, ...]) -> None:
        super().__init__()
        if widths != (64, 128, 192, 256):
            raise ValueError("Metal full codec requires the frozen U-Net widths")
        self.encoders = nn.ModuleList()
        for index, width in enumerate(widths):
            self.encoders.append(
                _ResidualConv(input_width if index == 0 else width, width)
            )
        self.downsamples = nn.ModuleList(
            nn.Conv2d(widths[index], widths[index + 1], 3, stride=2, padding=1)
            for index in range(len(widths) - 1)
        )
        self.upsamples = nn.ModuleList(
            nn.Conv2d(widths[index], widths[index - 1], 3, padding=1)
            for index in range(len(widths) - 1, 0, -1)
        )
        self.decoders = nn.ModuleList(
            _ResidualConv(2 * widths[index - 1], widths[index - 1])
            for index in range(len(widths) - 1, 0, -1)
        )

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = value.shape[-2:]
        target_height = max(8, int(original_shape[0]))
        target_width = max(8, int(original_shape[1]))
        if (target_height, target_width) != original_shape:
            value = F.interpolate(
                value, size=(target_height, target_width), mode="bilinear", align_corners=False
            )
        skips = []
        hidden = value
        for index, encoder in enumerate(self.encoders):
            hidden = encoder(hidden)
            skips.append(hidden)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)
        bottleneck = hidden
        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips[:-1]), strict=True
        ):
            hidden = F.interpolate(
                hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            hidden = upsample(hidden)
            hidden = decoder(torch.cat((hidden, skip), dim=1))
        if hidden.shape[-2:] != original_shape:
            hidden = F.interpolate(
                hidden, size=original_shape, mode="bilinear", align_corners=False
            )
        return hidden, bottleneck


class _DecoderBlock(nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.linear0 = nn.Conv2d(width, width, 1)
        self.linear1 = nn.Conv2d(width, width, 1)
        self.adapter_scale = nn.Linear(rank, width)
        self.adapter_bias = nn.Linear(rank, width)
        self.lora_down = nn.Conv2d(width, rank, 1, bias=False)
        self.lora_up = nn.Conv2d(rank, width, 1, bias=False)

    def forward(self, value: torch.Tensor, adapter: torch.Tensor) -> torch.Tensor:
        scale = 0.25 * torch.tanh(self.adapter_scale(adapter))[:, :, None, None]
        bias = 0.25 * torch.tanh(self.adapter_bias(adapter))[:, :, None, None]
        hidden = F.silu(self.linear0(value))
        hidden = self.linear1(hidden)
        lora = self.lora_up(self.lora_down(value))
        return F.silu(value + hidden * (1.0 + scale) + lora * scale + bias)


@dataclass(frozen=True)
class MetalCodecLevel:
    high_grid: torch.Tensor
    low_grid: torch.Tensor
    high_quantized: torch.Tensor
    low_quantized: torch.Tensor
    adapter: torch.Tensor
    structured: torch.Tensor
    semantic: torch.Tensor
    qat_error: torch.Tensor
    trace: Mapping[str, torch.Tensor]


class MetalTextureCodec(nn.Module):
    """Role-aware shared encoder/decoder with independent per-mip QAT grids."""

    def __init__(self, profile: MetalFusedProfile, asset_count: int = 52) -> None:
        super().__init__()
        if asset_count != 52:
            raise ValueError("Metal full codec preserves the complete 52-asset table")
        self.profile = profile
        self.role_stems = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(4, profile.encoder_role_width, 3, padding=1),
                nn.GroupNorm(8, profile.encoder_role_width),
                nn.SiLU(),
                nn.Conv2d(
                    profile.encoder_role_width,
                    profile.encoder_role_width,
                    3,
                    padding=1,
                ),
                nn.SiLU(),
            )
            for _ in ROLE_CLASS_NAMES
        )
        self.role_embedding = nn.Embedding(len(ROLE_CLASS_NAMES), profile.encoder_role_width)
        self.asset_embedding = nn.Embedding(asset_count, profile.encoder_role_width)
        self.mip_embedding = nn.Sequential(
            nn.Linear(3, profile.encoder_role_width),
            nn.SiLU(),
            nn.Linear(profile.encoder_role_width, profile.encoder_role_width),
        )
        self.bundle_attention = nn.MultiheadAttention(
            profile.encoder_role_width,
            4,
            batch_first=True,
        )
        self.bundle_norm = nn.LayerNorm(profile.encoder_role_width)
        self.encoder = _SharedUNet(profile.encoder_role_width, profile.encoder_widths)
        self.high_head = nn.Conv2d(
            profile.encoder_widths[0], profile.grid_high_channels, 1
        )
        self.low_head = nn.Conv2d(
            profile.encoder_widths[0], profile.grid_low_channels, 1
        )
        self.adapter_head = nn.Sequential(
            nn.Linear(profile.encoder_widths[-1], profile.encoder_widths[1]),
            nn.SiLU(),
            nn.Linear(profile.encoder_widths[1], profile.asset_adapter_rank),
            nn.Tanh(),
        )
        self.high_log_scale = nn.Parameter(torch.full((profile.grid_high_channels,), -2.0))
        self.low_log_scale = nn.Parameter(torch.full((profile.grid_low_channels,), -2.0))
        decoder_input = (
            profile.grid_high_channels
            + profile.grid_low_channels
            + 2 * profile.encoder_role_width
        )
        self.decoder_input = nn.Conv2d(decoder_input, profile.decoder_width, 1)
        self.decoder_blocks = nn.ModuleList(
            _DecoderBlock(profile.decoder_width, profile.asset_adapter_rank)
            for _ in range(profile.decoder_blocks)
        )
        self.structured_head = nn.Conv2d(
            profile.decoder_width, profile.structured_width, 1
        )
        self.semantic_heads = nn.ModuleList(
            nn.Conv2d(profile.decoder_width, 4, 1) for _ in ROLE_CLASS_NAMES
        )

    @staticmethod
    def _fake_quantize(
        value: torch.Tensor, log_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = F.softplus(log_scale).to(value.dtype) + 1e-6
        scale = scale[None, :, None, None]
        integer = torch.clamp(torch.round(value / scale), -127.0, 127.0)
        dequantized = integer * scale
        straight_through = value + (dequantized - value).detach()
        return straight_through, torch.mean(torch.abs(value - dequantized))

    def _role_stem(self, value: torch.Tensor, role_class: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([stem(value) for stem in self.role_stems], dim=1)
        selector = F.one_hot(role_class, len(self.role_stems)).to(outputs.dtype)
        return torch.sum(outputs * selector[:, :, None, None, None], dim=1)

    def encode_level(
        self,
        patches: torch.Tensor,
        slot_mask: torch.Tensor,
        role_class: torch.Tensor,
        asset_index: torch.Tensor,
        mip_level: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Mapping[str, torch.Tensor],
    ]:
        if patches.ndim != 5 or patches.shape[2] != 4:
            raise ValueError("Metal codec patches must have shape [batch,slot,4,height,width]")
        batch, slots, _, height, width = patches.shape
        if slots > self.profile.maximum_texture_slots:
            raise ValueError("Metal codec input exceeds the nine-slot profile")
        if slot_mask.shape != (batch, slots) or role_class.shape != (batch, slots):
            raise ValueError("Metal codec slot metadata shape mismatch")
        if asset_index.shape != (batch,) or mip_level.shape != (batch,):
            raise ValueError("Metal codec asset/mip metadata shape mismatch")
        if torch.any((role_class < 0) | (role_class >= len(ROLE_CLASS_NAMES))):
            raise ValueError("Metal codec role class is outside the registered domain")
        flat = patches.reshape(batch * slots, 4, height, width)
        flat_roles = role_class.reshape(-1)
        stem = self._role_stem(flat, flat_roles).reshape(
            batch, slots, self.profile.encoder_role_width, height, width
        )
        token = stem.mean(dim=(-2, -1))
        token = token + self.role_embedding(role_class)
        token = token + self.asset_embedding(asset_index)[:, None, :]
        mip_features = torch.stack(
            (
                mip_level / 16.0,
                torch.sin(mip_level),
                torch.cos(mip_level),
            ),
            dim=1,
        )
        token = token + self.mip_embedding(mip_features)[:, None, :]
        attention, weights = self.bundle_attention(
            token,
            token,
            token,
            key_padding_mask=~slot_mask,
            need_weights=True,
        )
        token = self.bundle_norm(token + attention)
        stem = stem + token[:, :, :, None, None]
        encoded, bottleneck = self.encoder(
            stem.reshape(batch * slots, self.profile.encoder_role_width, height, width)
        )
        high = self.high_head(encoded)
        high = F.adaptive_avg_pool2d(
            high,
            ((height + 1) // 2, (width + 1) // 2),
        )
        low = self.low_head(encoded)
        low = F.adaptive_avg_pool2d(
            low,
            ((height + 7) // 8, (width + 7) // 8),
        )
        high_q, high_error = self._fake_quantize(high, self.high_log_scale)
        low_q, low_error = self._fake_quantize(low, self.low_log_scale)
        pooled = bottleneck.mean(dim=(-2, -1)).reshape(batch, slots, -1)
        mask = slot_mask.to(pooled.dtype)[:, :, None]
        adapter = self.adapter_head(
            torch.sum(pooled * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        )
        trace = {
            "role_stems": stem.square().mean(),
            "bundle_attention": attention.square().mean(),
            "encoder": encoded.square().mean(),
            "adapter": adapter.square().mean(),
            "quantization": high_error + low_error,
        }
        return high, low, high_q, low_q, adapter, trace

    def decode_level(
        self,
        high_q: torch.Tensor,
        low_q: torch.Tensor,
        adapter: torch.Tensor,
        role_class: torch.Tensor,
        asset_index: torch.Tensor,
        slot_mask: torch.Tensor,
        output_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, torch.Tensor]]:
        batch, slots = role_class.shape
        count = batch * slots
        high = F.interpolate(
            high_q, size=output_shape, mode="bilinear", align_corners=False
        )
        low = F.interpolate(
            low_q, size=output_shape, mode="bilinear", align_corners=False
        )
        role = self.role_embedding(role_class).reshape(
            count, self.profile.encoder_role_width, 1, 1
        ).expand(-1, -1, *output_shape)
        asset = self.asset_embedding(asset_index)[:, None, :].expand(-1, slots, -1)
        asset = asset.reshape(count, self.profile.encoder_role_width, 1, 1).expand(
            -1, -1, *output_shape
        )
        hidden = self.decoder_input(torch.cat((high, low, role, asset), dim=1))
        flat_adapter = adapter[:, None, :].expand(-1, slots, -1).reshape(
            count, self.profile.asset_adapter_rank
        )
        for block in self.decoder_blocks:
            hidden = block(hidden, flat_adapter)
        structured = self.structured_head(hidden).reshape(
            batch, slots, self.profile.structured_width, *output_shape
        )
        all_semantic = torch.stack(
            [head(hidden) for head in self.semantic_heads], dim=1
        )
        selector = F.one_hot(role_class.reshape(-1), len(self.semantic_heads)).to(
            all_semantic.dtype
        )
        semantic = torch.sum(
            all_semantic * selector[:, :, None, None, None], dim=1
        ).reshape(batch, slots, 4, *output_shape)
        mask = slot_mask.to(structured.dtype)[:, :, None, None, None]
        structured_aggregate = torch.sum(structured * mask, dim=1) / torch.clamp(
            mask.sum(dim=1), min=1.0
        )
        trace = {
            "decoder": hidden.square().mean(),
            "structured_head": structured_aggregate.square().mean(),
            "semantic_heads": semantic.square().mean(),
        }
        return structured_aggregate, semantic, trace

    def forward_level(
        self,
        patches: torch.Tensor,
        slot_mask: torch.Tensor,
        role_class: torch.Tensor,
        asset_index: torch.Tensor,
        mip_level: torch.Tensor,
    ) -> MetalCodecLevel:
        high, low, high_q, low_q, adapter, encode_trace = self.encode_level(
            patches, slot_mask, role_class, asset_index, mip_level
        )
        structured, semantic, decode_trace = self.decode_level(
            high_q,
            low_q,
            adapter,
            role_class,
            asset_index,
            slot_mask,
            patches.shape[-2:],
        )
        qat_error = encode_trace["quantization"]
        return MetalCodecLevel(
            high,
            low,
            high_q,
            low_q,
            adapter,
            structured,
            semantic,
            qat_error,
            {**encode_trace, **decode_trace},
        )


__all__ = [
    "MetalCodecLevel",
    "MetalTextureCodec",
    "ROLE_CLASS_NAMES",
    "semantic_role_class",
]
