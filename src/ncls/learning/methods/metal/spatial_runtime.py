from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from ncls.learning.methods.metal.asset import MetalBudgetedAssetSample
from ncls.learning.methods.metal.asset_read import EncodedAssetTile, fp16_ste, plan_asset_read, read_hierarchy
from ncls.learning.methods.metal.native_uv import native_to_texture_coordinates, native_uv_lookups


SPATIAL_COMPILED_WORD_COUNT = 192
SPATIAL_COMPILED_LAYOUT_VERSION = 2


def spatial_material_payload(snapshot_id, asset):
    from ncls.bundle import RGBA8_SNORM_DDS_DTYPE, encode_rgba8_snorm_dds
    from ncls.core.scattering import MaterialPayload

    textures, layouts, samplers = {}, {}, {}
    # 共享 program 支持任意已注册组数；空 binding 用 1×1 零纹理，prepare 不读取它。
    for index in range(9):
        group = asset.groups[index].asset if index < len(asset.groups) else None
        for label in ("Detail", "Context"):
            levels = getattr(group, label.lower()+"_levels") if group is not None else (np.zeros((1, 1, 4), dtype=np.int8),)
            name = f"metal-spatial-{index}-{label.lower()}.dds"
            textures[name] = encode_rgba8_snorm_dds(levels)
            h, w = levels[0].shape[:2]
            layouts[name] = {"dtype": RGBA8_SNORM_DDS_DTYPE, "shape": [w, h, len(levels), 4],
                             "stride": 4, "alignment": 16, "usage": f"gNclsMetalSpatial{label}{index}"}
        samplers[f"metal-spatial-{index}"] = {"kind": "sampler", "usage": f"gNclsMetalSpatialSampler{index}",
            "filter": "linear", "address_mode": group.address_mode if group is not None else "clamp"}
    return MaterialPayload(snapshot_id, {}, {}, textures, layouts, samplers)


def spatial_program_condition(program, asset):
    condition = torch.as_tensor(asset.global_condition, dtype=program.compiler_condition.dtype,
                                device=program.compiler_condition.device)[None]
    return replace(program, compiler_condition=program.compiler_condition + 0.25 * condition)


def pack_spatial_compiled_material(program, asset):
    from ncls.learning.methods.metal.runtime import _program_floats

    program = spatial_program_condition(program, asset)
    values = _program_floats(program)
    if values.shape != (1, 64) or program.resource_and_flags.shape != (1, 8):
        raise ValueError("spatial compiled material requires one program")
    payload = values.detach().cpu().numpy().astype("<f2").tobytes()
    payload += program.resource_and_flags.detach().cpu().numpy().astype("<u4").tobytes()
    payload += np.asarray((2, len(asset.groups), 0, 0, 0, 0, 0, 0), dtype="<u4").tobytes()
    for index in range(9):
        if index >= len(asset.groups):
            payload += bytes(64)
            continue
        group = asset.groups[index]
        mapping, grids = group.group.mapping, group.asset
        floats = (*mapping.affine, mapping.cell_scale, mapping.lookup_scale)
        h, w = grids.detail_levels[0].shape[:2]
        ch, cw = grids.context_levels[0].shape[:2]
        flags = (int(mapping.mode == "nonrepeat"), mapping.hash_offset & 0xffffffff,
                 w, h, len(grids.detail_levels), cw, ch, len(grids.context_levels))
        payload += np.asarray(floats, dtype="<f4").tobytes() + np.asarray(flags, dtype="<u4").tobytes()
    if len(payload) != 4 * SPATIAL_COMPILED_WORD_COUNT:
        raise RuntimeError("spatial compiled material stride drifted")
    return payload


def sample_spatial_cooked_asset(asset, program, uv, uv_dx, uv_dy, filter_random):
    batch = uv.shape[0]
    latent = uv.new_zeros((batch, 9, 8))
    features = uv.new_zeros((batch, 9, 14))
    for index, group in enumerate(asset.groups):
        samples, dx, dy, weights = native_uv_lookups(group.group.mapping, uv, uv_dx, uv_dy)
        shape = group.asset.detail_levels[0].shape[:2]
        identity = uv.new_tensor(((1., 0., 0.), (0., 1., 0.)))[None].expand(batch, -1, -1)
        def levels(arrays):
            return tuple(EncodedAssetTile(torch.as_tensor(value, device=uv.device).float().permute(2, 0, 1)[None]/127., value.shape[:2]) for value in arrays)
        detail, context = levels(group.asset.detail_levels), levels(group.asset.context_levels)
        for lookup in range(group.group.mapping.lookup_count):
            coordinate, ddx, ddy = native_to_texture_coordinates(samples[:, lookup], dx[:, lookup], dy[:, lookup])
            plan = plan_asset_read(coordinate, ddx, ddy, identity,
                                   uv.new_tensor((shape[1], shape[0]))[None].expand(batch, -1),
                                   uv.new_full((batch,), len(detail)-1), filter_random)
            values = torch.cat((read_hierarchy(detail, plan, address_mode=group.asset.address_mode, qat=False, full_level=True),
                                read_hierarchy(context, plan, address_mode=group.asset.address_mode, qat=False, full_level=True)), dim=1)
            latent[:, index] += values * weights[:, lookup, None]
        features[:, index, 8:13] = plan.query_features(uv.new_zeros((batch, 3)))[:, 3:]
        features[:, index, 13] = 1
    spatial = fp16_ste(program.spatial_scale_bias)
    latent = (latent * torch.exp(0.25 * spatial[:, :4]).repeat(1, 2)[:, None]
              + 0.1 * spatial[:, 4:].repeat(1, 2)[:, None]) * features[:, :, 13:14]
    features[:, :, :8] = latent
    return MetalBudgetedAssetSample(latent[:, 0, :4], latent[:, 0, 4:],
        torch.zeros(batch, dtype=torch.int64, device=uv.device), torch.isfinite(features).all(dim=(1, 2)),
        {}, features, latent, uv.new_tensor(asset.global_condition)[None].expand(batch, -1))


def prepare_spatial_cooked_asset(model, asset, tensors, *, uv, wo, filter_random, uv_dx=(0., 0.), uv_dy=(0., 0.)):
    device = next(model.parameters()).device
    program = model.compile_program_state(tensors)
    sampled = sample_spatial_cooked_asset(asset, program,
        torch.tensor([uv], dtype=torch.float32, device=device),
        torch.tensor([uv_dx], dtype=torch.float32, device=device),
        torch.tensor([uv_dy], dtype=torch.float32, device=device),
        torch.tensor([filter_random], dtype=torch.float32, device=device))
    return model.prepare_from_components(program, sampled, torch.tensor([wo], dtype=torch.float32, device=device))
