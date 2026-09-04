from __future__ import annotations

from dataclasses import dataclass
import math
import re
import struct
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from ncls.learning.artifact_packing import pack_fp16_parameters
from ncls.learning.metal_asset_cook import MetalCompiledAssetState
from ncls.learning.models.metal_fused import (
    MetalModel,
    MetalSpatialState,
)
from ncls.learning.models.metal_typed_compiler import MetalMaterialProgramState


METAL_RAW_WORD_COUNT = 336
METAL_COMPILED_WORD_COUNT = 384
METAL_COMPILED_LAYOUT_VERSION = 1

METAL_RAW_OFFSETS = {
    "global": 0,
    "semantic": 8,
    "type": 40,
    "responsibility": 72,
    "discrete": 104,
    "presence": 136,
    "continuous": 168,
    "optical": 296,
    "access": 312,
    "frame": 328,
}

METAL_COMPILED_OFFSETS = {
    "compiler_latent": 0,
    "spatial_modulation": 64,
    "core_state": 128,
    "residual_state": 184,
    "block_condition": 212,
    "proposal_logits": 244,
    "proposal_modulation": 256,
    "tail_and_bounds": 292,
    "identity_and_flags": 300,
    "access": 316,
    "frame": 332,
}


_NON_RUNTIME_PREFIXES = (
    "texture_codec.role_stems.",
    "texture_codec.mip_embedding.",
    "texture_codec.bundle_attention.",
    "texture_codec.bundle_norm.",
    "texture_codec.encoder.",
    "texture_codec.high_head.",
    "texture_codec.low_head.",
    "texture_codec.adapter_head.",
    "texture_codec.semantic_heads.",
    "optimized_teacher.",
)
_NON_RUNTIME_NAMES = {
    "texture_codec.high_log_scale",
    "texture_codec.low_log_scale",
}


def metal_runtime_parameter_names(model: torch.nn.Module) -> tuple[str, ...]:
    result = tuple(
        name
        for name in model.state_dict()
        if name not in _NON_RUNTIME_NAMES
        and not name.startswith(_NON_RUNTIME_PREFIXES)
    )
    required_prefixes = (
        "texture_codec.role_embedding.",
        "texture_codec.asset_embedding.",
        "texture_codec.decoder_input.",
        "texture_codec.decoder_blocks.",
        "texture_codec.structured_head.",
        "typed_compiler.",
        "prepared_model.",
        "directional.angular_bank.",
        "hybrid.",
    )
    orphan = [name for name in result if not name.startswith(required_prefixes)]
    if orphan:
        raise ValueError(f"Metal runtime parameter classification has orphans: {orphan}")
    if not result:
        raise ValueError("Metal runtime parameter set is empty")
    return result


def metal_weight_define(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    return f"NCLS_METAL_W_{value}"


@dataclass(frozen=True)
class MetalPackedProgram:
    payload: bytes
    layout: Mapping[str, Mapping[str, Any]]
    defines: Mapping[str, str]


def pack_metal_program(model: MetalModel) -> MetalPackedProgram:
    names = metal_runtime_parameter_names(model)
    payload, layout = pack_fp16_parameters(model, names)
    if len(payload) % 4:
        payload += bytes(4 - len(payload) % 4)
    defines = {
        metal_weight_define(name): str(int(layout[name]["offset_elements"]))
        for name in names
    }
    if len(defines) != len(names):
        raise ValueError("Metal runtime weight define collision")
    return MetalPackedProgram(payload, layout, defines)


def quantize_runtime_model(
    model: MetalModel,
) -> MetalModel:
    """Apply the deployed FP16 master pack while retaining FP32 accumulations."""

    state = model.state_dict()
    for name in metal_runtime_parameter_names(model):
        value = state[name]
        state[name] = value.to(torch.float16).to(value.dtype)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def fake_quantize_fp16_ste(value: torch.Tensor) -> torch.Tensor:
    """Round a floating tensor to deployed FP16 while retaining master gradients."""

    if not value.is_floating_point():
        raise TypeError("Metal runtime fake quantization requires a floating tensor")
    rounded = value.to(torch.float16).to(value.dtype)
    return value + (rounded - value).detach()


def _copy_f32(words: np.ndarray, offset: int, value: torch.Tensor | np.ndarray) -> None:
    array = (
        value.detach().cpu().float().contiguous().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value, dtype=np.float32)
    ).reshape(-1)
    if not np.isfinite(array).all() or offset + array.size > words.size:
        raise ValueError("Metal packed float field is non-finite or out of bounds")
    words[offset : offset + array.size] = array.view(np.uint32)


def _copy_u32(words: np.ndarray, offset: int, value: torch.Tensor | np.ndarray | Sequence[int]) -> None:
    array = (
        value.detach().cpu().to(torch.int64).contiguous().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value, dtype=np.int64)
    ).reshape(-1)
    if np.any(array < 0) or np.any(array > np.iinfo(np.uint32).max) or offset + array.size > words.size:
        raise ValueError("Metal packed integer field is out of bounds")
    words[offset : offset + array.size] = array.astype(np.uint32)


def pack_metal_raw_parameters(tensors: Mapping[str, torch.Tensor]) -> bytes:
    words = np.zeros(METAL_RAW_WORD_COUNT, dtype="<u4")
    global_values = (
        tensors["metal_graph_index"],
        tensors["metal_schema_index"],
        tensors["metal_recipe_index"],
        tensors["metal_identity_index"],
        tensors["metal_finish_index"],
        tensors["metal_asset_index"],
    )
    _copy_u32(words, METAL_RAW_OFFSETS["global"], [int(value.item()) for value in global_values])
    for output, tensor_name in (
        ("semantic", "metal_typed_semantic_id"),
        ("type", "metal_typed_type_id"),
        ("responsibility", "metal_typed_responsibility_id"),
        ("discrete", "metal_typed_discrete"),
        ("presence", "metal_typed_presence"),
    ):
        _copy_u32(words, METAL_RAW_OFFSETS[output], tensors[tensor_name][0])
    _copy_f32(words, METAL_RAW_OFFSETS["continuous"], tensors["metal_typed_continuous"][0])
    _copy_f32(words, METAL_RAW_OFFSETS["optical"], tensors["metal_canonical_optical"][0])
    _copy_f32(words, METAL_RAW_OFFSETS["access"], tensors["metal_access_state"][0])
    _copy_f32(words, METAL_RAW_OFFSETS["frame"], tensors["metal_frame_state"][0])
    return words.tobytes()


def pack_metal_compiled_material(
    state: MetalMaterialProgramState,
    tensors: Mapping[str, torch.Tensor],
    asset: MetalCompiledAssetState,
    *,
    domain_count: int,
    maximum_extent: int,
    maximum_mip: int,
) -> bytes:
    if state.compiler_latent.shape[0] != 1:
        raise ValueError("Metal package compiler state requires exactly one material")
    words = np.zeros(METAL_COMPILED_WORD_COUNT, dtype="<u4")
    _copy_f32(words, METAL_COMPILED_OFFSETS["compiler_latent"], state.compiler_latent[0])
    _copy_f32(words, METAL_COMPILED_OFFSETS["spatial_modulation"], state.spatial_modulation[0])
    _copy_f32(words, METAL_COMPILED_OFFSETS["core_state"], state.core_state[0])
    _copy_f32(words, METAL_COMPILED_OFFSETS["residual_state"], state.residual_state[0])
    _copy_f32(words, METAL_COMPILED_OFFSETS["block_condition"], state.block_condition[0])
    _copy_f32(words, METAL_COMPILED_OFFSETS["proposal_logits"], state.proposal_logits[0])
    _copy_f32(words, METAL_COMPILED_OFFSETS["proposal_modulation"], state.proposal_modulation[0])
    _copy_f32(
        words,
        METAL_COMPILED_OFFSETS["tail_and_bounds"],
        torch.cat((state.correction_bound[0], state.tail_scale[0], state.frame_strength[0])),
    )
    _copy_u32(
        words,
        METAL_COMPILED_OFFSETS["identity_and_flags"],
        (
            METAL_COMPILED_LAYOUT_VERSION,
            int(tensors["metal_asset_index"].item()),
            domain_count,
            len(asset.records),
            maximum_extent,
            maximum_mip,
            METAL_RAW_WORD_COUNT,
            METAL_COMPILED_WORD_COUNT,
        ),
    )
    _copy_f32(words, METAL_COMPILED_OFFSETS["access"], tensors["metal_access_state"][0])
    _copy_f32(words, METAL_COMPILED_OFFSETS["frame"], tensors["metal_frame_state"][0])
    return words.tobytes()


@dataclass(frozen=True)
class MetalPackedAsset:
    blobs: Mapping[str, bytes]
    descriptors: Mapping[str, Mapping[str, Any]]
    domain_count: int
    maximum_extent: int
    maximum_mip: int


def _pack_int8x4(value: torch.Tensor) -> bytes:
    payload = value.detach().cpu().contiguous().numpy().astype(np.int8, copy=False).tobytes()
    if len(payload) % 4:
        payload += bytes(4 - len(payload) % 4)
    return payload


def _pack_half2(value: torch.Tensor) -> bytes:
    array = value.detach().cpu().contiguous().numpy().astype("<f2", copy=False).reshape(-1)
    if array.size % 2:
        array = np.pad(array, (0, 1))
    return array.tobytes()


def pack_metal_asset(
    asset: MetalCompiledAssetState,
    *,
    address_modes: Mapping[str, str] | None = None,
) -> MetalPackedAsset:
    records: list[int] = []
    domains: list[int] = []
    first = 0
    maximum_extent = 1
    maximum_mip = 0
    cursor = 0
    while cursor < len(asset.records):
        domain_id = asset.records[cursor].domain_id
        begin = cursor
        while cursor < len(asset.records) and asset.records[cursor].domain_id == domain_id:
            record = asset.records[cursor]
            if record.mip_level != cursor - begin:
                raise ValueError("Metal compiled asset domain mip records are not dense")
            records.extend(
                (
                    record.source_shape[0], record.source_shape[1],
                    record.high_shape[0], record.high_shape[1],
                    record.low_shape[0], record.low_shape[1],
                    record.high_offset, record.low_offset, record.scale_offset,
                    record.role_class, record.mip_level, 0,
                )
            )
            maximum_extent = max(maximum_extent, *record.source_shape)
            maximum_mip = max(maximum_mip, record.mip_level)
            cursor += 1
        count = cursor - begin
        address_mode = "wrap" if address_modes is None else address_modes[domain_id]
        if address_mode not in {"clamp", "wrap"}:
            raise ValueError("Metal compiled asset has an unsupported address mode")
        domains.extend(
            (first, count, asset.records[begin].role_class, 1 if address_mode == "wrap" else 0)
        )
        first += count
    if first != len(asset.records):
        raise ValueError("Metal compiled asset domain table does not cover records")
    record_bytes = np.asarray(records, dtype="<u4").tobytes()
    domain_bytes = np.asarray(domains, dtype="<u4").tobytes()
    high = _pack_int8x4(asset.high_grid_int8)
    low = _pack_int8x4(asset.low_grid_int8)
    scales = _pack_half2(asset.grid_scales)
    adapter = _pack_half2(asset.adapter_fp16)
    blobs = {
        "metal-domains": domain_bytes,
        "metal-records": record_bytes,
        "metal-high-grid": high,
        "metal-low-grid": low,
        "metal-grid-scales": scales,
        "metal-asset-adapter": adapter,
    }
    descriptions = {
        "metal-domains": ("uint32", [len(domains)], 4, "gNclsMetalDomains"),
        "metal-records": ("uint32", [len(records)], 4, "gNclsMetalRecords"),
        "metal-high-grid": ("packed-int8x4-uint32@1", [len(high) // 4], 4, "gNclsMetalHighGrid"),
        "metal-low-grid": ("packed-int8x4-uint32@1", [len(low) // 4], 4, "gNclsMetalLowGrid"),
        "metal-grid-scales": ("packed-float16x2-uint32@1", [len(scales) // 4], 4, "gNclsMetalGridScales"),
        "metal-asset-adapter": ("packed-float16x2-uint32@1", [len(adapter) // 4], 4, "gNclsMetalAssetAdapter"),
    }
    descriptors = {
        name: {
            "kind": "structured-buffer",
            "dtype": dtype,
            "shape": shape,
            "stride": stride,
            "alignment": 16,
            "usage": usage,
        }
        for name, (dtype, shape, stride, usage) in descriptions.items()
    }
    return MetalPackedAsset(
        blobs,
        descriptors,
        len(domains) // 4,
        maximum_extent,
        maximum_mip,
    )


def _sample_grid(
    grid: torch.Tensor,
    uv: torch.Tensor,
    *,
    address_mode: str,
) -> torch.Tensor:
    coordinate = torch.frac(uv) if address_mode == "wrap" else torch.clamp(uv, 0.0, 1.0)
    sampled = F.grid_sample(
        grid[None],
        (coordinate * 2.0 - 1.0)[None, None, None],
        mode="bilinear",
        padding_mode="zeros" if address_mode == "wrap" else "border",
        align_corners=False,
    )
    return sampled


def decode_metal_cooked_asset(
    model: MetalModel,
    asset: MetalCompiledAssetState,
    tensors: Mapping[str, torch.Tensor],
    uv: torch.Tensor,
    mip_level: float,
    *,
    address_modes: Mapping[str, str],
) -> torch.Tensor:
    device = next(model.parameters()).device
    high_flat = asset.high_grid_int8.to(device)
    low_flat = asset.low_grid_int8.to(device)
    scales = asset.grid_scales.to(device=device, dtype=torch.float32)
    adapter = asset.adapter_fp16.to(device=device, dtype=torch.float32)[None]
    asset_index = tensors["metal_asset_index"].to(device)
    by_domain: dict[str, list[Any]] = {}
    for record in asset.records:
        by_domain.setdefault(record.domain_id, []).append(record)
    adjacent: list[torch.Tensor] = []
    base = int(math.floor(mip_level))
    for adjacent_index in range(2):
        decoded_domains = []
        for domain_id, records in by_domain.items():
            record = records[min(base + adjacent_index, len(records) - 1)]
            high_count = record.high_shape[0] * record.high_shape[1] * 8
            low_count = record.low_shape[0] * record.low_shape[1] * 8
            high = high_flat[record.high_offset : record.high_offset + high_count].reshape(
                record.high_shape[0], record.high_shape[1], 8
            ).permute(2, 0, 1).float()
            low = low_flat[record.low_offset : record.low_offset + low_count].reshape(
                record.low_shape[0], record.low_shape[1], 8
            ).permute(2, 0, 1).float()
            scale = scales[record.scale_offset : record.scale_offset + 16]
            high *= scale[:8, None, None]
            low *= scale[8:, None, None]
            high_sample = _sample_grid(high, uv, address_mode=address_modes[domain_id])
            low_sample = _sample_grid(low, uv, address_mode=address_modes[domain_id])
            structured, _, _ = model.texture_codec.decode_level(
                high_sample,
                low_sample,
                adapter,
                torch.tensor([[record.role_class]], dtype=torch.int64, device=device),
                asset_index,
                torch.ones((1, 1), dtype=torch.bool, device=device),
                (1, 1),
            )
            decoded_domains.append(structured[:, :, 0, 0])
        adjacent.append(torch.stack(decoded_domains).mean(dim=0))
    return torch.lerp(adjacent[0], adjacent[1], mip_level - base)


def evaluate_metal_cooked_asset(
    model: MetalModel,
    asset: MetalCompiledAssetState,
    tensors: Mapping[str, torch.Tensor],
    *,
    uv: Sequence[float],
    mip_level: float,
    wo: Sequence[float],
    wi: Sequence[Sequence[float]],
    address_modes: Mapping[str, str],
) -> torch.Tensor:
    device = next(model.parameters()).device
    uv_tensor = torch.tensor(uv, dtype=torch.float32, device=device)
    access_uv, access_dx, access_dy, access_valid = model.prepared_model.execute_spatial_access(
        uv_tensor[None],
        torch.tensor([[2.0 ** mip_level / 1024.0, 0.0]], device=device),
        torch.tensor([[0.0, 2.0 ** mip_level / 1024.0]], device=device),
        tensors["metal_access_state"].to(device),
    )
    structured = decode_metal_cooked_asset(
        model, asset, tensors, access_uv[0], mip_level, address_modes=address_modes
    )
    spatial = MetalSpatialState(
        structured,
        access_uv,
        access_dx,
        access_dy,
        torch.tensor([mip_level - math.floor(mip_level)], device=device),
        access_valid,
        {},
    )
    program = model.typed_compiler({name: value.to(device) for name, value in tensors.items()})
    wo_tensor = torch.tensor([wo], dtype=torch.float32, device=device)
    prepared = model.prepare_from_components(program, spatial, tensors, wo=wo_tensor)
    wi_tensor = torch.tensor([wi], dtype=torch.float32, device=device)
    return model.evaluate_prepared(prepared, wo_tensor, wi_tensor).f.detach().cpu()[0]


__all__ = [
    "METAL_COMPILED_LAYOUT_VERSION",
    "METAL_COMPILED_OFFSETS",
    "METAL_COMPILED_WORD_COUNT",
    "METAL_RAW_OFFSETS",
    "METAL_RAW_WORD_COUNT",
    "MetalPackedAsset",
    "MetalPackedProgram",
    "decode_metal_cooked_asset",
    "evaluate_metal_cooked_asset",
    "metal_runtime_parameter_names",
    "metal_weight_define",
    "fake_quantize_fp16_ste",
    "pack_metal_asset",
    "pack_metal_compiled_material",
    "pack_metal_program",
    "pack_metal_raw_parameters",
    "quantize_runtime_model",
]
