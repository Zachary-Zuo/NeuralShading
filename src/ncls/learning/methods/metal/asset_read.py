from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def mip_shapes(height: int, width: int) -> tuple[tuple[int, int], ...]:
    if min(height, width) < 1:
        raise ValueError("asset extent must be positive")
    result = [(height, width)]
    while result[-1] != (1, 1):
        h, w = result[-1]
        result.append((max(1, h // 2), max(1, w // 2)))
    return tuple(result)


def snorm8_ste(value: torch.Tensor) -> torch.Tensor:
    bounded = value.clamp(-1.0, 1.0)
    rounded = torch.round(bounded * 127.0) / 127.0
    return bounded + (rounded - bounded).detach()


def fp16_ste(value: torch.Tensor) -> torch.Tensor:
    rounded = value.to(torch.float16).to(value.dtype)
    return value + (rounded - value).detach()


@dataclass(frozen=True)
class AssetReadPlan:
    uv: torch.Tensor
    jacobian: torch.Tensor
    lod: torch.Tensor
    level: torch.Tensor

    def query_features(self, wo: torch.Tensor) -> torch.Tensor:
        flat = self.jacobian.flatten(start_dim=1)
        normalized = flat / (1.0 + torch.linalg.vector_norm(flat, dim=1, keepdim=True))
        return torch.cat((wo, normalized, torch.frac(self.lod)[:, None]), dim=1)


def plan_asset_read(
    uv: torch.Tensor,
    uv_dx: torch.Tensor,
    uv_dy: torch.Tensor,
    affine: torch.Tensor,
    extent_xy: torch.Tensor,
    maximum_level: torch.Tensor,
    filter_random: torch.Tensor,
) -> AssetReadPlan:
    """所有数值均使用声明的坐标/尺寸，不从图像统计或 GPU readback 推导。"""
    batch = uv.shape[0]
    if (
        uv.shape != (batch, 2)
        or uv_dx.shape != uv.shape
        or uv_dy.shape != uv.shape
        or affine.shape != (batch, 2, 3)
        or extent_xy.shape != uv.shape
        or maximum_level.shape != (batch,)
        or filter_random.shape != (batch,)
    ):
        raise ValueError("asset read-plan tensor shapes disagree")
    linear = affine[:, :, :2]
    transformed = torch.einsum("bij,bj->bi", linear, uv) + affine[:, :, 2]
    derivatives = torch.stack((uv_dx, uv_dy), dim=-1)
    jacobian = extent_xy[:, :, None] * (linear @ derivatives)
    rho = torch.linalg.vector_norm(jacobian, dim=1).amax(dim=1)
    lod = torch.minimum(torch.log2(rho.clamp_min(1.0)), maximum_level)
    level = torch.floor(lod).to(torch.int64)
    level = level + (filter_random < torch.frac(lod)).to(torch.int64)
    return AssetReadPlan(transformed, jacobian, lod, level)


@dataclass(frozen=True)
class EncodedAssetTile:
    """某 mip 的连续全局矩形；origin 可含 seam 外的地址，边界已按域求值。"""
    values: torch.Tensor  # [1,4,H,W]，量化前的 texel
    level_shape: tuple[int, int]
    origin_yx: tuple[int, int] = (0, 0)


def read_bilinear(
    tile: EncodedAssetTile,
    uv: torch.Tensor,
    *,
    address_mode: str,
    qat: bool = True,
    full_level: bool = False,
) -> torch.Tensor:
    values = snorm8_ste(tile.values) if qat else tile.values
    if values.ndim != 4 or values.shape[0] != 1 or uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("bilinear read requires [1,C,H,W] and [B,2]")
    height, width = tile.level_shape
    p = uv * uv.new_tensor((width, height)) - 0.5
    floor = torch.floor(p)
    fraction = p - floor
    xy = floor.to(torch.int64)
    offsets = torch.tensor(((0, 0), (1, 0), (0, 1), (1, 1)), device=uv.device)
    corners = xy[:, None, :] + offsets[None]
    if full_level:
        if tile.origin_yx != (0, 0) or values.shape[-2:] != tile.level_shape:
            raise ValueError("full-level read must contain the complete canonical mip")
        if address_mode == "wrap":
            corners = torch.remainder(corners, corners.new_tensor((width, height)))
        elif address_mode == "clamp":
            corners = torch.minimum(corners.clamp_min(0), corners.new_tensor((width - 1, height - 1)))
        else:
            raise ValueError("unsupported asset address mode")
    else:
        corners = corners - corners.new_tensor((tile.origin_yx[1], tile.origin_yx[0]))
    local_height, local_width = values.shape[-2:]
    inside = (corners >= 0).all() & (corners[..., 0] < local_width).all() & (corners[..., 1] < local_height).all()
    if inside.device.type == "cuda":
        torch._assert_async(inside, "encoded tile does not contain all four read neighbours")
    elif not bool(inside):
        raise ValueError("encoded tile does not contain all four read neighbours")
    indices = corners[..., 1] * local_width + corners[..., 0]
    texels = values[0].flatten(start_dim=1)[:, indices].permute(1, 2, 0)
    fx, fy = fraction[:, 0:1], fraction[:, 1:2]
    weights = torch.cat(((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy), dim=1)
    return torch.sum(texels * weights[..., None], dim=1)


def read_hierarchy(
    levels: Sequence[EncodedAssetTile],
    plan: AssetReadPlan,
    *,
    address_mode: str,
    qat: bool = True,
    full_level: bool = False,
) -> torch.Tensor:
    if not levels:
        raise ValueError("asset hierarchy cannot be empty")
    selected = plan.level.clamp_max(len(levels) - 1)
    # 对所有 row 使用同一个 CPU mip 计划，不为选择层读回 GPU 索引。
    result = plan.uv.new_zeros((plan.uv.shape[0], levels[0].values.shape[1]))
    for level, tile in enumerate(levels):
        value = read_bilinear(tile, plan.uv, address_mode=address_mode, qat=qat, full_level=full_level)
        result = result + torch.where((selected == level)[:, None], value, 0.0)
    return result
