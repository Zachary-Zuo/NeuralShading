from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_LAYOUT_PATH = Path(__file__).with_name("abi") / "scattering_contract_v1.json"
LAYOUT: dict[str, Any] = json.loads(_LAYOUT_PATH.read_text(encoding="utf-8"))
CONTRACT_NAME = str(LAYOUT["contract_name"])
CONTRACT_VERSION = int(LAYOUT["contract_version"])


def _enum_lines(values: dict[str, int], *, prefix: str) -> str:
    return "\n".join(f"    {name} = {int(value)}u," for name, value in values.items())


def render_slang_contract() -> str:
    transport = _enum_lines(LAYOUT["transport_mode"], prefix="")
    events = _enum_lines(LAYOUT["event_flags"], prefix="")
    capabilities = _enum_lines(LAYOUT["capabilities"], prefix="")
    return f"""// 此文件由 ncls.core.scattering.abi_layout 生成；不要手工编辑。
#ifndef NCLS_SCATTERING_CONTRACT_SLANG
#define NCLS_SCATTERING_CONTRACT_SLANG

static const uint NCLS_SCATTERING_CONTRACT_VERSION = {CONTRACT_VERSION}u;

enum NclsTransportMode : uint
{{
{transport}
}}

enum NclsScatteringEvent : uint
{{
    None = 0u,
{events}
}}

enum NclsBackendCapability : uint
{{
    None = 0u,
{capabilities}
}}

struct NclsShadingFrame
{{
    float3 normal;
    float3 tangent;
    float3 bitangent;
}};

struct NclsSurfaceInteraction
{{
    float3 position;
    float3 geometricNormal;
    NclsShadingFrame shadingFrame;
    float2 uv;
    float2 uvDx;
    float2 uvDy;
    uint materialInstanceId;
    uint primitiveId;
    uint frontFacing;
}};

struct NclsScatteringContext
{{
    NclsSurfaceInteraction surface;
    float3 woWorld;
    uint transportMode;
    uint componentMask;
    float filterRandom;
}};

struct NclsScatteringPdf
{{
    float forward;
    float reverse;
}};

struct NclsScatteringEval
{{
    float3 f;
    NclsScatteringPdf pdf;
    uint eventFlags;
    uint valid;
}};

struct NclsScatteringSample
{{
    float3 wiWorld;
    float3 weight;
    NclsScatteringPdf pdf;
    float eta;
    uint eventFlags;
    uint valid;
}};

#endif
"""
