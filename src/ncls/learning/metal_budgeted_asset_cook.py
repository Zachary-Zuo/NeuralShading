from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from ncls.core.identity import sha256_bytes, sha256_json
from ncls.data import PipelineTrace
from ncls.learning.models.metal_budgeted import MetalBudgetedModel


def _mip_shapes(height: int, width: int) -> tuple[tuple[int, int], ...]:
    if min(height, width) < 1:
        raise ValueError("Metal budgeted asset extent must be positive")
    result = [(height, width)]
    while result[-1] != (1, 1):
        previous_height, previous_width = result[-1]
        result.append(
            (max(1, previous_height // 2), max(1, previous_width // 2))
        )
    return tuple(result)


@dataclass(frozen=True)
class MetalBudgetedCompiledAsset:
    profile_id: str
    mode: str
    source_collection_identity: str
    asset_id: str
    asset_schema_id: str
    address_mode: str
    detail_levels: tuple[np.ndarray, ...]
    context_levels: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if self.mode != "encoder-only@1":
            raise ValueError("deployable Metal budgeted asset requires encoder-only cook")
        if self.address_mode not in {"clamp", "wrap"}:
            raise ValueError("Metal budgeted asset address mode is invalid")
        for label, levels in (
            ("Detail", self.detail_levels),
            ("Context", self.context_levels),
        ):
            if not levels:
                raise ValueError(f"Metal budgeted {label} hierarchy is empty")
            height, width = levels[0].shape[:2]
            expected = _mip_shapes(height, width)
            if len(levels) != len(expected):
                raise ValueError(f"Metal budgeted {label} mip count is invalid")
            for level, shape in zip(levels, expected, strict=True):
                if level.dtype != np.int8 or level.shape != (*shape, 4):
                    raise ValueError(
                        f"Metal budgeted {label} mip shape/dtype is invalid"
                    )
        detail_height, detail_width = self.detail_levels[0].shape[:2]
        context_height, context_width = self.context_levels[0].shape[:2]
        if (context_height, context_width) != (
            max(1, detail_height // 4),
            max(1, detail_width // 4),
        ):
            raise ValueError("Metal budgeted Context base extent must be Detail/4")

    @property
    def identity(self) -> str:
        return sha256_json(
            {
                "schema": "ncls.metal-budgeted-compiled-asset@1",
                "profile_id": self.profile_id,
                "mode": self.mode,
                "source_collection_identity": self.source_collection_identity,
                "asset_id": self.asset_id,
                "asset_schema_id": self.asset_schema_id,
                "address_mode": self.address_mode,
                "detail": [
                    {
                        "shape": list(level.shape),
                        "sha256": sha256_bytes(level.tobytes()),
                    }
                    for level in self.detail_levels
                ],
                "context": [
                    {
                        "shape": list(level.shape),
                        "sha256": sha256_bytes(level.tobytes()),
                    }
                    for level in self.context_levels
                ],
            }
        )


class MetalBudgetedAssetCompiler:
    """把最多九个source-native slot烘焙为固定Detail/Context两张纹理。"""

    def __init__(
        self,
        model: MetalBudgetedModel,
        assets: Any,
        *,
        batch_size: int = 8192,
        source_patch_size: int = 8,
        residency_budget_bytes: int = 8 * 1024**3,
    ) -> None:
        if (
            batch_size < 1
            or source_patch_size < 8
            or source_patch_size > 32
            or residency_budget_bytes < 1
        ):
            raise ValueError("Metal budgeted asset compiler budget is invalid")
        self.model = model
        self.assets = assets
        self.batch_size = batch_size
        self.source_patch_size = source_patch_size
        self.residency_budget_bytes = residency_budget_bytes

    @staticmethod
    def _base_extent(descriptor: Any) -> tuple[int, int]:
        return (
            max(int(domain.level_shapes[0][0]) for domain in descriptor.domains),
            max(int(domain.level_shapes[0][1]) for domain in descriptor.domains),
        )

    def _encode_hierarchy(
        self,
        asset_index: int,
        shapes: tuple[tuple[int, int], ...],
        *,
        output: str,
        progress: tqdm[Any],
    ) -> tuple[np.ndarray, ...]:
        device = next(self.model.parameters()).device
        levels: list[np.ndarray] = []
        for mip_level, (height, width) in enumerate(shapes):
            packed = np.empty((height * width, 4), dtype=np.int8)
            for begin in range(0, height * width, self.batch_size):
                end = min(height * width, begin + self.batch_size)
                flat = torch.arange(begin, end, dtype=torch.int64, device=device)
                y = torch.div(flat, width, rounding_mode="floor")
                x = torch.remainder(flat, width)
                uv = torch.stack(
                    (
                        (x.to(torch.float32) + 0.5) / float(width),
                        (y.to(torch.float32) + 0.5) / float(height),
                    ),
                    dim=1,
                )
                asset_indices = torch.full(
                    (end - begin,), asset_index, dtype=torch.int64, device=device
                )
                mip = torch.full(
                    (end - begin,),
                    float(mip_level),
                    dtype=torch.float32,
                    device=device,
                )
                patches, mask, roles = self.assets.sample_local_patches(
                    asset_indices,
                    uv,
                    mip,
                    patch_size=self.source_patch_size,
                    active_asset_indices=(asset_index,),
                )
                choice = torch.zeros_like(asset_indices)
                detail, context, valid = self.model.asset._encode_source_patches(
                    {
                        "metal_texture_patches": patches,
                        "metal_texture_slot_mask": mask,
                        "metal_texture_role_class": roles,
                    },
                    choice,
                )
                if not bool(valid.all()):
                    raise RuntimeError("Metal budgeted asset encoder found an empty slot set")
                values = detail if output == "detail" else context
                quantized = torch.round(torch.clamp(values, -1.0, 1.0) * 127.0)
                packed[begin:end] = quantized.to(torch.int8).cpu().numpy()
                progress.update(end - begin)
            levels.append(packed.reshape(height, width, 4))
        return tuple(levels)

    def compile(self, asset_index: int) -> MetalBudgetedCompiledAsset:
        if not 0 <= asset_index < len(self.assets.descriptors):
            raise ValueError("Metal budgeted asset index is out of range")
        descriptor = self.assets.descriptors[asset_index]
        address_modes = {domain.address_mode for domain in descriptor.domains}
        if len(address_modes) != 1:
            raise ValueError(
                "Metal budgeted two-plane asset cannot merge mixed address modes"
            )
        detail_height, detail_width = self._base_extent(descriptor)
        detail_shapes = _mip_shapes(detail_height, detail_width)
        context_shapes = _mip_shapes(
            max(1, detail_height // 4), max(1, detail_width // 4)
        )
        device = next(self.model.parameters()).device
        if device.type == "cuda" and hasattr(self.assets, "enable_gpu_sampling"):
            self.assets.enable_gpu_sampling(
                device,
                budget_bytes=self.residency_budget_bytes,
                trace=PipelineTrace(),
            )
        total = sum(height * width for height, width in detail_shapes)
        total += sum(height * width for height, width in context_shapes)
        self.model.asset.eval()
        with torch.inference_mode(), tqdm(
            total=total,
            unit="texel",
            desc="metal-budgeted asset cook",
            leave=False,
        ) as progress:
            detail = self._encode_hierarchy(
                asset_index, detail_shapes, output="detail", progress=progress
            )
            context = self._encode_hierarchy(
                asset_index, context_shapes, output="context", progress=progress
            )
        return MetalBudgetedCompiledAsset(
            self.model.profile.profile_id,
            "encoder-only@1",
            self.assets.collection_id,
            descriptor.asset_id,
            descriptor.schema_id,
            next(iter(address_modes)),
            detail,
            context,
        )


__all__ = [
    "MetalBudgetedAssetCompiler",
    "MetalBudgetedCompiledAsset",
]
