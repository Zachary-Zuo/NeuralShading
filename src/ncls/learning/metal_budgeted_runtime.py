from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from ncls.learning.artifact_packing import pack_fp16_parameters
from ncls.learning.metal_budgeted_asset_cook import MetalBudgetedCompiledAsset
from ncls.learning.models.metal_budgeted import MetalBudgetedModel
from ncls.learning.models.metal_budgeted_asset import MetalBudgetedAssetSample
from ncls.learning.models.metal_budgeted_compiler import MetalBudgetedProgramState


METAL_BUDGETED_COMPILED_WORD_COUNT = 48
METAL_BUDGETED_COMPILED_LAYOUT_VERSION = 1
METAL_BUDGETED_PROGRAM_HALF_COUNT = 64
METAL_BUDGETED_PROGRAM_FLAG_COUNT = 8


def metal_budgeted_runtime_parameter_names(
    model: MetalBudgetedModel,
) -> frozenset[str]:
    names = {
        name
        for name, _ in model.named_parameters()
        if name.startswith("typed_compiler.")
        or name.startswith("prepared_model.")
        or name.startswith("evaluator.")
        or name == "asset.variant_scale_bias.weight"
    }
    if not names or any(name.startswith("asset.detail_encoder.") for name in names):
        raise ValueError("Metal budgeted runtime parameter classification drifted")
    return frozenset(names)


def metal_budgeted_weight_define(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    return f"NCLS_METAL_BUDGETED_W_{value}"


@dataclass(frozen=True)
class MetalBudgetedPackedProgram:
    payload: bytes
    layout: Mapping[str, Mapping[str, Any]]
    defines: Mapping[str, str]


def pack_metal_budgeted_program(
    model: MetalBudgetedModel,
) -> MetalBudgetedPackedProgram:
    names = tuple(sorted(metal_budgeted_runtime_parameter_names(model)))
    payload, layout = pack_fp16_parameters(model, names)
    if len(payload) % 4:
        payload += bytes(4 - len(payload) % 4)
    defines = {
        metal_budgeted_weight_define(name): str(int(layout[name]["offset_elements"]))
        for name in names
    }
    if len(defines) != len(names):
        raise ValueError("Metal budgeted runtime weight define collision")
    return MetalBudgetedPackedProgram(payload, layout, defines)


def quantize_metal_budgeted_runtime_model(
    model: MetalBudgetedModel,
) -> MetalBudgetedModel:
    state = model.state_dict()
    for name in metal_budgeted_runtime_parameter_names(model):
        value = state[name]
        state[name] = value.to(torch.float16).to(value.dtype)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _program_floats(program: MetalBudgetedProgramState) -> torch.Tensor:
    values = torch.cat(
        (
            program.compiler_condition,
            program.primary_lobe,
            program.secondary_lobe,
            program.spatial_scale_bias,
            program.proposal_prior,
            torch.zeros(
                (program.compiler_condition.shape[0], 5),
                dtype=program.compiler_condition.dtype,
                device=program.compiler_condition.device,
            ),
            program.access_state,
            program.frame_state,
        ),
        dim=1,
    )
    if values.shape[1] != METAL_BUDGETED_PROGRAM_HALF_COUNT:
        raise RuntimeError("Metal budgeted ProgramState float width drifted")
    return values


def quantize_metal_budgeted_program_state(
    program: MetalBudgetedProgramState,
) -> MetalBudgetedProgramState:
    q = lambda value: value.to(torch.float16).to(value.dtype)
    return MetalBudgetedProgramState(
        compiler_condition=q(program.compiler_condition),
        primary_lobe=q(program.primary_lobe),
        secondary_lobe=q(program.secondary_lobe),
        spatial_scale_bias=q(program.spatial_scale_bias),
        proposal_prior=q(program.proposal_prior),
        resource_variant=program.resource_variant,
        resource_and_flags=program.resource_and_flags,
        access_state=q(program.access_state),
        frame_state=q(program.frame_state),
        trace=program.trace,
    )


def pack_metal_budgeted_compiled_material(
    program: MetalBudgetedProgramState,
    asset: MetalBudgetedCompiledAsset,
) -> bytes:
    if (
        program.compiler_condition.shape[0] != 1
        or program.resource_and_flags.shape
        != (1, METAL_BUDGETED_PROGRAM_FLAG_COUNT)
    ):
        raise ValueError("Metal budgeted package requires one ProgramState")
    floats = (
        _program_floats(program)[0]
        .detach()
        .cpu()
        .numpy()
        .astype("<f2", copy=False)
    )
    flags = (
        program.resource_and_flags[0]
        .detach()
        .cpu()
        .numpy()
        .astype("<u4", copy=False)
    )
    detail_height, detail_width = asset.detail_levels[0].shape[:2]
    context_height, context_width = asset.context_levels[0].shape[:2]
    profile_mode = 0 if model_profile_mode(asset.profile_id) == "hybrid" else 1
    metadata = np.asarray(
        (
            METAL_BUDGETED_COMPILED_LAYOUT_VERSION,
            detail_width,
            detail_height,
            len(asset.detail_levels),
            context_width,
            context_height,
            len(asset.context_levels),
            profile_mode,
        ),
        dtype="<u4",
    )
    payload = floats.tobytes() + flags.tobytes() + metadata.tobytes()
    if len(payload) != 4 * METAL_BUDGETED_COMPILED_WORD_COUNT:
        raise RuntimeError("Metal budgeted compiled material stride drifted")
    return payload


def model_profile_mode(profile_id: str) -> str:
    if profile_id == "metal_budgeted_hybrid_v3":
        return "hybrid"
    if profile_id == "metal_budgeted_direct_control_v3":
        return "direct"
    raise ValueError(f"unsupported Metal budgeted deployment profile {profile_id!r}")


def _sample_level(
    level: np.ndarray,
    uv: torch.Tensor,
    *,
    address_mode: str,
) -> torch.Tensor:
    device = uv.device
    grid = torch.as_tensor(level, dtype=torch.int8, device=device)
    grid = grid.permute(2, 0, 1).to(torch.float32) / 127.0
    coordinate = (
        torch.remainder(uv, 1.0)
        if address_mode == "wrap"
        else torch.clamp(uv, 0.0, 1.0)
    )
    sampled = F.grid_sample(
        grid[None],
        (coordinate * 2.0 - 1.0)[None, None, None],
        mode="bilinear",
        padding_mode="zeros" if address_mode == "wrap" else "border",
        align_corners=False,
    )
    return sampled[0, :, 0, 0][None]


def _runtime_asset_sample(
    model: MetalBudgetedModel,
    program: MetalBudgetedProgramState,
    asset: MetalBudgetedCompiledAsset,
    *,
    uv: torch.Tensor,
    mip_level: float,
    filter_random: float,
) -> MetalBudgetedAssetSample:
    lower = int(np.floor(mip_level))
    fraction = float(mip_level - lower)
    selected = lower + int(filter_random < fraction)
    detail_index = min(max(selected, 0), len(asset.detail_levels) - 1)
    context_index = min(max(selected, 0), len(asset.context_levels) - 1)
    raw_detail = _sample_level(
        asset.detail_levels[detail_index], uv, address_mode=asset.address_mode
    )
    raw_context = _sample_level(
        asset.context_levels[context_index], uv, address_mode=asset.address_mode
    )
    variants = program.resource_variant
    table = model.asset.variant_scale_bias(variants)
    program_scale = program.spatial_scale_bias[:, :4]
    program_bias = program.spatial_scale_bias[:, 4:]
    detail_scale = torch.exp(
        0.25 * torch.tanh(table[:, 0:4]) + 0.25 * program_scale
    )
    detail_bias = 0.25 * torch.tanh(table[:, 4:8]) + 0.1 * program_bias
    context_scale = torch.exp(
        0.25 * torch.tanh(table[:, 8:12]) + 0.25 * program_scale
    )
    context_bias = 0.25 * torch.tanh(table[:, 12:16]) + 0.1 * program_bias
    detail = raw_detail * detail_scale + detail_bias
    context = raw_context * context_scale + context_bias
    valid = torch.isfinite(detail).all(dim=1) & torch.isfinite(context).all(dim=1)
    return MetalBudgetedAssetSample(
        detail,
        context,
        torch.full_like(program.resource_variant, selected),
        valid,
        {},
    )


def evaluate_metal_budgeted_cooked_asset(
    model: MetalBudgetedModel,
    asset: MetalBudgetedCompiledAsset,
    tensors: Mapping[str, torch.Tensor],
    *,
    uv: Sequence[float],
    mip_level: float,
    filter_random: float,
    wo: Sequence[float],
    wi: Sequence[Sequence[float]],
) -> torch.Tensor:
    device = next(model.parameters()).device
    runtime_tensors = {name: value.to(device) for name, value in tensors.items()}
    with torch.no_grad():
        program = quantize_metal_budgeted_program_state(
            model.compile_program_state(runtime_tensors)
        )
        access = program.access_state
        surface_uv = torch.tensor([uv], dtype=torch.float32, device=device)
        scaled = surface_uv * access[:, 0:2]
        cosine, sine = access[:, 4:5], access[:, 5:6]
        transformed = torch.stack(
            (
                cosine[:, 0] * scaled[:, 0] - sine[:, 0] * scaled[:, 1],
                sine[:, 0] * scaled[:, 0] + cosine[:, 0] * scaled[:, 1],
            ),
            dim=1,
        ) + access[:, 2:4]
        transformed = torch.where(
            access[:, 6:7] > 0.5, torch.remainder(transformed, 1.0), transformed
        )
        sampled = _runtime_asset_sample(
            model,
            program,
            asset,
            uv=transformed[0],
            mip_level=mip_level,
            filter_random=filter_random,
        )
        wo_tensor = torch.tensor([wo], dtype=torch.float32, device=device)
        wi_tensor = torch.tensor([wi], dtype=torch.float32, device=device)
        prepared = model.prepare_from_components(program, sampled, wo_tensor)
        return model.evaluate_prepared(
            prepared, wo_tensor, wi_tensor
        ).f.detach().cpu()[0]


__all__ = [
    "METAL_BUDGETED_COMPILED_LAYOUT_VERSION",
    "METAL_BUDGETED_COMPILED_WORD_COUNT",
    "MetalBudgetedPackedProgram",
    "evaluate_metal_budgeted_cooked_asset",
    "metal_budgeted_runtime_parameter_names",
    "metal_budgeted_weight_define",
    "pack_metal_budgeted_compiled_material",
    "pack_metal_budgeted_program",
    "quantize_metal_budgeted_program_state",
    "quantize_metal_budgeted_runtime_model",
]
