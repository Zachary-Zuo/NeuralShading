from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn

from ncls.learning.conditioning_resources import ConditioningResources
from ncls.learning.methods.metal.asset import MetalBudgetedAssetSample
from ncls.learning.methods.metal.asset_read import fp16_ste, plan_asset_read, read_hierarchy
from ncls.learning.methods.metal.compiler import MetalBudgetedProgramState
from ncls.learning.methods.metal.native_uv import native_hash22, native_to_texture_coordinates, native_uv_lookups
from ncls.learning.methods.metal.spatial_encoder import MetalSpatialEncoder


class MetalSpatialAsset(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = MetalSpatialEncoder()

    def encode_resources(self, resources: ConditioningResources):
        encoded = []
        for resource in resources.entries:
            bundle = resource.metadata["bundle"]
            parts = []
            for index, part in enumerate(bundle.parts):
                raw = {f"raw-{read}": resource.tensors[f"part-{index}/raw-{read}"] for read in range(len(part.plan.raw_reads))}
                parts.append(self.encoder.encode_tiles(part.plan, raw))
            encoded.append(tuple(parts))
        return tuple(encoded)

    def forward(
        self, tensors: Mapping[str, torch.Tensor], program: MetalBudgetedProgramState,
        *, resources: ConditioningResources, binding: torch.Tensor, encoded=None, qat: bool = True,
    ) -> MetalBudgetedAssetSample:
        if encoded is None:
            encoded = self.encode_resources(resources)
        uv = tensors["uv"]
        batch = uv.shape[0]
        group_latent = uv.new_zeros((batch, 9, 8))
        group_filter = uv.new_zeros((batch, 9, 5))
        presence = uv.new_zeros((batch, 9, 1))
        global_condition = uv.new_zeros((batch, 8))
        for resource_index, (resource, parts) in enumerate(zip(resources.entries, encoded, strict=True)):
            bundle = resource.metadata["bundle"]
            active = binding == resource_index
            local_uv = torch.where(active[:, None], uv, uv.new_tensor(bundle.anchor_uv)[None])
            dx = torch.where(active[:, None], tensors["uv_dx"], 0.)
            dy = torch.where(active[:, None], tensors["uv_dy"], 0.)
            lookups = [native_uv_lookups(group.mapping, local_uv, dx, dy, return_cells=True) for group in bundle.groups]
            for part, (detail, context) in zip(bundle.parts, parts, strict=True):
                if part.group < 0:
                    code = torch.cat((detail[0].values.flatten(), context[0].values.flatten()))[None]
                    global_condition = global_condition + torch.where(active[:, None], fp16_ste(code) if qat else code, 0.)
                    continue
                group_index = part.group
                mapping = bundle.groups[group_index].mapping
                samples, sample_dx, sample_dy, weights, cells = lookups[group_index]
                if part.cell is None:
                    coordinate, weight = samples[:, 0], weights[:, 0]
                else:
                    cell = torch.tensor(part.cell, dtype=torch.int64, device=uv.device)
                    matches = (cells == cell[None, None]).all(dim=-1)
                    weight = (weights * matches).sum(dim=1)
                    a = uv.new_tensor(mapping.affine).reshape(2, 3)
                    coordinate = (local_uv @ a[:, :2].T + a[:, 2]) * mapping.lookup_scale - native_hash22(cell[None])[0]
                coordinate, ddx, ddy = native_to_texture_coordinates(coordinate, sample_dx[:, 0], sample_dy[:, 0])
                height, width = part.plan.shape
                extent = uv.new_tensor((width, height))[None].expand(batch, -1)
                identity = uv.new_tensor(((1., 0., 0.), (0., 1., 0.)))[None].expand(batch, -1, -1)
                plan = plan_asset_read(coordinate, ddx, ddy, identity, extent,
                                       uv.new_full((batch,), math.floor(math.log2(max(height, width)))), tensors["filter_random"])
                d = read_hierarchy(detail, plan, address_mode=part.plan.address_mode, qat=qat)
                c = read_hierarchy(context, plan, address_mode=part.plan.address_mode, qat=qat)
                latent = torch.cat((d, c), dim=1)
                # 这里只混合同一 UV 组的原生 lookup；不同组的值一直分开送给 decoder。
                contribution = torch.where(active[:, None], latent * weight[:, None], 0.)
                group_latent[:, group_index] = group_latent[:, group_index] + contribution
                flat = plan.jacobian.flatten(start_dim=1)
                filtering = torch.cat((flat / (1. + torch.linalg.vector_norm(flat, dim=1, keepdim=True)), torch.frac(plan.lod)[:, None]), dim=1)
                group_filter[:, group_index] = torch.where(active[:, None], filtering, group_filter[:, group_index])
                presence[:, group_index] = torch.where(active[:, None], 1., presence[:, group_index])
        spatial = fp16_ste(program.spatial_scale_bias) if qat else program.spatial_scale_bias
        scale = torch.exp(0.25 * spatial[:, :4])
        bias = 0.1 * spatial[:, 4:]
        decoded = group_latent * scale.repeat(1, 2)[:, None] + bias.repeat(1, 2)[:, None]
        decoded = decoded * presence
        features = torch.cat((decoded, group_filter, presence), dim=-1)
        valid = torch.isfinite(features).all(dim=(1, 2)) & (presence.sum(dim=(1, 2)) > 0)
        return MetalBudgetedAssetSample(
            decoded[:, 0, :4], decoded[:, 0, 4:], torch.zeros(batch, dtype=torch.int64, device=uv.device), valid,
            {"asset_detail": decoded[..., :4].square().mean(), "asset_context": decoded[..., 4:].square().mean(),
             "asset_group_latent": decoded.square().mean(), "asset_group_count": presence.sum(dim=1).mean(),
             "asset_global_condition": global_condition.square().mean()}, features, decoded, global_condition,
        )
