from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ncls.core.identity import sha256_bytes, sha256_json
from ncls.learning.models.metal_fused import MetalFusedNeuralMaterialModel
from ncls.learning.models.metal_texture_codec import semantic_role_class
from ncls.learning.source_adaptation import (
    NativeAssetCollection,
    NativeAssetDomain,
    NativeAssetTile,
    NativeAssetTileRequest,
)


MetalAssetCookMode = Literal[
    "encoder-only", "encoder-bounded-refinement", "direct-optimized-control"
]


@dataclass(frozen=True)
class MetalAssetLevelRecord:
    domain_id: str
    mip_level: int
    role_class: int
    source_shape: tuple[int, int]
    high_shape: tuple[int, int]
    low_shape: tuple[int, int]
    high_offset: int
    low_offset: int
    scale_offset: int

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "domain_id": self.domain_id,
            "mip_level": self.mip_level,
            "role_class": self.role_class,
            "source_shape": list(self.source_shape),
            "high_shape": list(self.high_shape),
            "low_shape": list(self.low_shape),
            "high_offset": self.high_offset,
            "low_offset": self.low_offset,
            "scale_offset": self.scale_offset,
        }


@dataclass(frozen=True)
class MetalCompiledAssetState:
    profile_id: str
    mode: MetalAssetCookMode
    source_collection_identity: str
    asset_id: str
    asset_schema_id: str
    records: tuple[MetalAssetLevelRecord, ...]
    high_grid_int8: torch.Tensor
    low_grid_int8: torch.Tensor
    grid_scales: torch.Tensor
    adapter_fp16: torch.Tensor
    refinement_steps: int
    refinement_bound: float

    def __post_init__(self) -> None:
        if self.profile_id != "metal_fused_full_v1":
            raise ValueError("compiled Metal asset uses an unknown profile")
        if self.mode not in {
            "encoder-only",
            "encoder-bounded-refinement",
            "direct-optimized-control",
        }:
            raise ValueError("compiled Metal asset has an unknown cook mode")
        if (
            self.high_grid_int8.dtype != torch.int8
            or self.low_grid_int8.dtype != torch.int8
            or self.grid_scales.dtype != torch.float16
            or self.adapter_fp16.dtype != torch.float16
            or self.adapter_fp16.shape != (8,)
        ):
            raise ValueError("compiled Metal asset packing dtype/shape is invalid")
        high_cursor = low_cursor = scale_cursor = 0
        for record in self.records:
            if (
                record.high_offset != high_cursor
                or record.low_offset != low_cursor
                or record.scale_offset != scale_cursor
            ):
                raise ValueError("compiled Metal asset offsets are not dense")
            high_cursor += record.high_shape[0] * record.high_shape[1] * 8
            low_cursor += record.low_shape[0] * record.low_shape[1] * 8
            scale_cursor += 16
        if (
            high_cursor != self.high_grid_int8.numel()
            or low_cursor != self.low_grid_int8.numel()
            or scale_cursor != self.grid_scales.numel()
        ):
            raise ValueError("compiled Metal asset flat storage disagrees with records")

    @property
    def identity(self) -> str:
        return sha256_json(
            {
                "schema": "ncls.metal-fused-compiled-asset@1",
                "profile_id": self.profile_id,
                "mode": self.mode,
                "source_collection_identity": self.source_collection_identity,
                "asset_id": self.asset_id,
                "asset_schema_id": self.asset_schema_id,
                "records": [record.to_dict() for record in self.records],
                "high_grid_sha256": sha256_bytes(
                    self.high_grid_int8.contiguous().numpy().tobytes()
                ),
                "low_grid_sha256": sha256_bytes(
                    self.low_grid_int8.contiguous().numpy().tobytes()
                ),
                "grid_scales": self.grid_scales.float().tolist(),
                "adapter": self.adapter_fp16.float().tolist(),
                "refinement_steps": self.refinement_steps,
                "refinement_bound": self.refinement_bound,
            }
        )


@dataclass
class _FloatLevel:
    domain: NativeAssetDomain
    mip_level: int
    role_class: int
    high: torch.Tensor
    low: torch.Tensor


class MetalAssetCooker:
    """三条新资产路径共用的full-profile encoder/decoder cook。"""

    def __init__(
        self,
        model: MetalFusedNeuralMaterialModel,
        assets: NativeAssetCollection,
        *,
        max_core_texels: int = 16_384,
        encoder_halo: int = 32,
        encoder_batch_tiles: int = 8,
    ) -> None:
        if (
            max_core_texels < 1024
            or encoder_halo < 32
            or encoder_halo % 8 != 0
            or not 1 <= encoder_batch_tiles <= 32
        ):
            raise ValueError("Metal full U-Net cook tile/halo/batch settings are invalid")
        self.model = model
        self.assets = assets
        self.max_core_texels = max_core_texels
        self.encoder_halo = encoder_halo
        self.encoder_batch_tiles = encoder_batch_tiles

    @staticmethod
    def _domain_role_class(domain: NativeAssetDomain) -> int:
        if len(domain.roles) != 1:
            return 3
        role = domain.roles[0]
        return semantic_role_class(role.semantic, role.channel_count)

    @staticmethod
    def _tile_input(tile: NativeAssetTile) -> torch.Tensor:
        channels = int(tile.values.shape[2])
        if channels > 4:
            raise ValueError("Metal codec domain exceeds four packed source channels")
        result = torch.zeros(
            (1, 1, 4, tile.values.shape[0], tile.values.shape[1]),
            dtype=tile.values.dtype,
            device=tile.values.device,
        )
        result[0, 0, :channels] = tile.values.permute(2, 0, 1)
        return result

    @staticmethod
    def _accumulate(
        destination: torch.Tensor,
        weights: torch.Tensor,
        tile_grid: torch.Tensor,
        origin_yx: tuple[int, int],
        core_shape: tuple[int, int],
        halo: int,
        divisor: int,
    ) -> None:
        origin_y, origin_x = origin_yx
        core_height, core_width = core_shape
        destination_y = origin_y // divisor
        destination_x = origin_x // divisor
        core_grid_height = (core_height + divisor - 1) // divisor
        core_grid_width = (core_width + divisor - 1) // divisor
        source_y = halo // divisor
        source_x = halo // divisor
        values = tile_grid[
            0,
            :,
            source_y : source_y + core_grid_height,
            source_x : source_x + core_grid_width,
        ]
        target_height = min(core_grid_height, destination.shape[1] - destination_y)
        target_width = min(core_grid_width, destination.shape[2] - destination_x)
        destination[
            :, destination_y : destination_y + target_height,
            destination_x : destination_x + target_width,
        ] += values[:, :target_height, :target_width]
        weights[
            :, destination_y : destination_y + target_height,
            destination_x : destination_x + target_width,
        ] += 1.0

    def _encode_asset(
        self, asset_index: int, device: torch.device
    ) -> tuple[list[_FloatLevel], torch.Tensor]:
        descriptor = self.assets.descriptors[asset_index]
        levels: list[_FloatLevel] = []
        adapters = []
        self.model.texture_codec.eval()
        with torch.no_grad():
            for domain in descriptor.domains:
                role_class = self._domain_role_class(domain)
                for mip_level, (height, width) in enumerate(domain.level_shapes):
                    high = torch.zeros(
                        (8, (height + 1) // 2, (width + 1) // 2),
                        dtype=torch.float32,
                        device=device,
                    )
                    low = torch.zeros(
                        (8, (height + 7) // 8, (width + 7) // 8),
                        dtype=torch.float32,
                        device=device,
                    )
                    high_weights = torch.zeros_like(high[:1])
                    low_weights = torch.zeros_like(low[:1])
                    requests = tuple(
                        request
                        for request in self.assets.iter_tile_requests(
                            asset_index,
                            domain.domain_id,
                            self.max_core_texels,
                            self.encoder_halo,
                        )
                        if request.mip_level == mip_level
                    )
                    pending: list[tuple[NativeAssetTileRequest, NativeAssetTile]] = []

                    def flush() -> None:
                        if not pending:
                            return
                        try:
                            batch = torch.cat(
                                [self._tile_input(tile) for _, tile in pending], dim=0
                            )
                            count = batch.shape[0]
                            _, _, encoded_high, encoded_low, adapter, _ = (
                                self.model.texture_codec.encode_level(
                                    batch,
                                    torch.ones((count, 1), dtype=torch.bool, device=device),
                                    torch.full((count, 1), role_class, dtype=torch.int64, device=device),
                                    torch.full((count,), asset_index, dtype=torch.int64, device=device),
                                    torch.full((count,), float(mip_level), dtype=torch.float32, device=device),
                                )
                            )
                            for index, (request, _) in enumerate(pending):
                                self._accumulate(
                                    high,
                                    high_weights,
                                    encoded_high[index : index + 1],
                                    request.origin_yx,
                                    request.core_shape,
                                    request.halo,
                                    2,
                                )
                                self._accumulate(
                                    low,
                                    low_weights,
                                    encoded_low[index : index + 1],
                                    request.origin_yx,
                                    request.core_shape,
                                    request.halo,
                                    8,
                                )
                                adapters.append(adapter[index])
                        finally:
                            for _, tile in pending:
                                tile.release()
                            pending.clear()

                    for request in requests:
                        tile = self.assets.acquire_tile(request, device)
                        if pending and tile.values.shape != pending[0][1].values.shape:
                            flush()
                        pending.append((request, tile))
                        if len(pending) == self.encoder_batch_tiles:
                            flush()
                    flush()
                    if torch.any(high_weights == 0) or torch.any(low_weights == 0):
                        raise RuntimeError("Metal tiled encoder failed to cover a compiled grid")
                    levels.append(
                        _FloatLevel(
                            domain,
                            mip_level,
                            role_class,
                            high / high_weights,
                            low / low_weights,
                        )
                    )
        if not adapters:
            raise RuntimeError("Metal asset encoder produced no adapter observations")
        return levels, torch.stack(adapters).mean(dim=0)

    @staticmethod
    def _grid_patch(
        grid: torch.Tensor,
        tile: NativeAssetTile,
        divisor: int,
        address_mode: str,
        source_shape: tuple[int, int],
    ) -> torch.Tensor:
        output_height = (tile.values.shape[0] + divisor - 1) // divisor
        output_width = (tile.values.shape[1] + divisor - 1) // divisor
        y = (
            tile.origin_yx[0]
            - tile.halo
            + (torch.arange(output_height, device=grid.device) + 0.5) * divisor
        ) / max(1, source_shape[0])
        x = (
            tile.origin_yx[1]
            - tile.halo
            + (torch.arange(output_width, device=grid.device) + 0.5) * divisor
        ) / max(1, source_shape[1])
        if address_mode == "wrap":
            y, x = torch.frac(y), torch.frac(x)
            padding_mode = "zeros"
        else:
            y, x = torch.clamp(y, 0.0, 1.0), torch.clamp(x, 0.0, 1.0)
            padding_mode = "border"
        yy, xx = torch.meshgrid(y * 2.0 - 1.0, x * 2.0 - 1.0, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=-1)[None, ...]
        return F.grid_sample(
            grid[None, ...],
            coordinates,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=False,
        )

    def _refine(
        self,
        asset_index: int,
        levels: list[_FloatLevel],
        adapter: torch.Tensor,
        *,
        mode: MetalAssetCookMode,
        steps: int,
        bound: float,
    ) -> tuple[list[_FloatLevel], torch.Tensor]:
        if steps < 1:
            raise ValueError("optimized Metal asset paths require positive steps")
        encoder_high = [level.high.detach().clone() for level in levels]
        encoder_low = [level.low.detach().clone() for level in levels]
        encoder_adapter = adapter.detach().clone()
        if mode == "direct-optimized-control":
            parameters_high = [nn.Parameter(torch.zeros_like(value)) for value in encoder_high]
            parameters_low = [nn.Parameter(torch.zeros_like(value)) for value in encoder_low]
            adapter_parameter = nn.Parameter(torch.zeros_like(encoder_adapter))
        else:
            parameters_high = [nn.Parameter(value.clone()) for value in encoder_high]
            parameters_low = [nn.Parameter(value.clone()) for value in encoder_low]
            adapter_parameter = nn.Parameter(encoder_adapter.clone())
        parameters = [*parameters_high, *parameters_low, adapter_parameter]
        optimizer = torch.optim.Adam(parameters, lr=2e-3, fused=adapter.is_cuda)
        decoder_parameters = tuple(self.model.texture_codec.decoder_input.parameters()) + tuple(
            self.model.texture_codec.decoder_blocks.parameters()
        ) + tuple(self.model.texture_codec.structured_head.parameters()) + tuple(
            self.model.texture_codec.semantic_heads.parameters()
        )
        previous_requires_grad = [value.requires_grad for value in decoder_parameters]
        for value in decoder_parameters:
            value.requires_grad_(False)
        try:
            for step in range(steps):
                level_index = step % len(levels)
                level = levels[level_index]
                requests = tuple(
                    request
                    for request in self.assets.iter_tile_requests(
                        asset_index,
                        level.domain.domain_id,
                        self.max_core_texels,
                        self.encoder_halo,
                    )
                    if request.mip_level == level.mip_level
                )
                request = requests[(step // len(levels)) % len(requests)]
                tile = self.assets.acquire_tile(request, adapter.device)
                try:
                    high = self._grid_patch(
                        parameters_high[level_index],
                        tile,
                        2,
                        level.domain.address_mode,
                        level.domain.level_shapes[level.mip_level],
                    )
                    low = self._grid_patch(
                        parameters_low[level_index],
                        tile,
                        8,
                        level.domain.address_mode,
                        level.domain.level_shapes[level.mip_level],
                    )
                    _, semantic, _ = self.model.texture_codec.decode_level(
                        high,
                        low,
                        adapter_parameter[None, :],
                        torch.tensor([[level.role_class]], dtype=torch.int64, device=adapter.device),
                        torch.tensor([asset_index], dtype=torch.int64, device=adapter.device),
                        torch.ones((1, 1), dtype=torch.bool, device=adapter.device),
                        tile.values.shape[:2],
                    )
                    target = self._tile_input(tile)
                    loss = F.smooth_l1_loss(torch.sigmoid(semantic), target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    if mode == "encoder-bounded-refinement":
                        with torch.no_grad():
                            for parameter, initial in zip(
                                parameters_high, encoder_high, strict=True
                            ):
                                parameter.copy_(
                                    initial + torch.clamp(parameter - initial, -bound, bound)
                                )
                            for parameter, initial in zip(
                                parameters_low, encoder_low, strict=True
                            ):
                                parameter.copy_(
                                    initial + torch.clamp(parameter - initial, -bound, bound)
                                )
                            adapter_parameter.copy_(
                                encoder_adapter
                                + torch.clamp(
                                    adapter_parameter - encoder_adapter,
                                    -0.5 * bound,
                                    0.5 * bound,
                                )
                            )
                finally:
                    tile.release()
        finally:
            for value, requires_grad in zip(
                decoder_parameters, previous_requires_grad, strict=True
            ):
                value.requires_grad_(requires_grad)
        refined = [
            _FloatLevel(
                level.domain,
                level.mip_level,
                level.role_class,
                high.detach(),
                low.detach(),
            )
            for level, high, low in zip(
                levels, parameters_high, parameters_low, strict=True
            )
        ]
        return refined, adapter_parameter.detach()

    def cook_asset(
        self,
        asset_index: int,
        *,
        mode: MetalAssetCookMode,
        refinement_steps: int = 0,
        refinement_bound: float = 0.25,
    ) -> MetalCompiledAssetState:
        if not 0 <= asset_index < len(self.assets.descriptors):
            raise ValueError("Metal asset cook index is out of range")
        if mode == "encoder-only" and refinement_steps != 0:
            raise ValueError("encoder-only Metal cook cannot run hidden optimization")
        if mode != "encoder-only" and not 0.0 < refinement_bound <= 0.5:
            raise ValueError("Metal refinement bound must lie in (0,0.5]")
        device = next(self.model.parameters()).device
        with self.assets.cook_session(asset_index):
            levels, adapter = self._encode_asset(asset_index, device)
        if mode != "encoder-only":
            levels, adapter = self._refine(
                asset_index,
                levels,
                adapter,
                mode=mode,
                steps=refinement_steps,
                bound=refinement_bound,
            )
        high_scale = (
            F.softplus(self.model.texture_codec.high_log_scale.detach()) + 1e-6
        )
        low_scale = (
            F.softplus(self.model.texture_codec.low_log_scale.detach()) + 1e-6
        )
        high_values = []
        low_values = []
        scales = []
        records = []
        high_offset = low_offset = scale_offset = 0
        for level in levels:
            high_q = torch.clamp(
                torch.round(level.high / high_scale[:, None, None]), -127, 127
            ).to(torch.int8)
            low_q = torch.clamp(
                torch.round(level.low / low_scale[:, None, None]), -127, 127
            ).to(torch.int8)
            high_values.append(high_q.permute(1, 2, 0).reshape(-1).cpu())
            low_values.append(low_q.permute(1, 2, 0).reshape(-1).cpu())
            scales.append(torch.cat((high_scale, low_scale)).to(torch.float16).cpu())
            records.append(
                MetalAssetLevelRecord(
                    level.domain.domain_id,
                    level.mip_level,
                    level.role_class,
                    level.domain.level_shapes[level.mip_level],
                    tuple(level.high.shape[1:]),
                    tuple(level.low.shape[1:]),
                    high_offset,
                    low_offset,
                    scale_offset,
                )
            )
            high_offset += high_values[-1].numel()
            low_offset += low_values[-1].numel()
            scale_offset += 16
        descriptor = self.assets.descriptors[asset_index]
        return MetalCompiledAssetState(
            "metal_fused_full_v1",
            mode,
            self.assets.collection_id,
            descriptor.asset_id,
            descriptor.schema_id,
            tuple(records),
            torch.cat(high_values),
            torch.cat(low_values),
            torch.cat(scales),
            adapter.to(torch.float16).cpu(),
            refinement_steps,
            0.0 if mode == "encoder-only" else refinement_bound,
        )


__all__ = [
    "MetalAssetCookMode",
    "MetalAssetCooker",
    "MetalAssetLevelRecord",
    "MetalCompiledAssetState",
]
