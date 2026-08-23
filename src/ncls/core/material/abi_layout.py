from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any


_LAYOUT_PATH = Path(__file__).with_name("abi") / "layer_stack_ir_v1.json"
_LAYOUT: dict[str, Any] = json.loads(_LAYOUT_PATH.read_text(encoding="utf-8"))

ABI_NAME = str(_LAYOUT["abi_name"])
ABI_VERSION = int(_LAYOUT["abi_version"])
ABI_MAGIC = int(_LAYOUT["magic"])
MAX_INTERFACES = int(_LAYOUT["max_interfaces"])
MAX_MEDIA = int(_LAYOUT["max_media"])
HEADER_STRUCT = struct.Struct(str(_LAYOUT["header"]["struct_format"]))
INTERFACE_STRUCT = struct.Struct(str(_LAYOUT["interface"]["struct_format"]))
MEDIUM_STRUCT = struct.Struct(str(_LAYOUT["medium"]["struct_format"]))
BINARY_SIZE = int(_LAYOUT["binary_size"])

assert HEADER_STRUCT.size == int(_LAYOUT["header"]["size"]) == 16
assert INTERFACE_STRUCT.size == int(_LAYOUT["interface"]["size"]) == 64
assert MEDIUM_STRUCT.size == int(_LAYOUT["medium"]["size"]) == 32
assert BINARY_SIZE == HEADER_STRUCT.size + MAX_INTERFACES * INTERFACE_STRUCT.size + MAX_MEDIA * MEDIUM_STRUCT.size


def render_slang_header() -> str:
    """生成与 JSON ABI 描述严格对应的 Slang 声明。"""

    return f"""// 此文件由 ncls.core.material.abi_layout 生成；不要手工编辑。
#ifndef NCLS_LAYER_STACK_IR_SLANG
#define NCLS_LAYER_STACK_IR_SLANG

static const uint NCLS_LAYER_STACK_IR_MAGIC = 0x{ABI_MAGIC:08x}u;
static const uint NCLS_LAYER_STACK_IR_VERSION = {ABI_VERSION}u;
static const uint NCLS_MAX_INTERFACES = {MAX_INTERFACES}u;
static const uint NCLS_MAX_MEDIA = {MAX_MEDIA}u;

enum NclsInterfaceKind : uint
{{
    RoughDielectric = 0,
    RoughConductor = 1,
    Diffuse = 2,
    Sheen = 3,
}}

// {INTERFACE_STRUCT.size} bytes. 字段是类型明确的 tagged union；未使用字段必须为零。
struct NclsLayerInterfaceIR
{{
    uint kind;
    uint flags;
    float alphaX;
    float alphaY;

    float relativeIor;
    float etaR;
    float etaG;
    float etaB;

    float kR;
    float kG;
    float kB;
    float colorR;

    float colorG;
    float colorB;
    float tangentRotation;
    float reserved;
}};

// {MEDIUM_STRUCT.size} bytes.
struct NclsHomogeneousMediumIR
{{
    float sigmaAR;
    float sigmaAG;
    float sigmaAB;
    float sigmaSR;
    float sigmaSG;
    float sigmaSB;
    float g;
    float thickness;
}};

// {BINARY_SIZE} bytes.
struct NclsLayerStackIR
{{
    uint magic;
    uint abiVersion;
    uint interfaceCount;
    uint mediumCount;
    NclsLayerInterfaceIR interfaces[NCLS_MAX_INTERFACES];
    NclsHomogeneousMediumIR media[NCLS_MAX_MEDIA];
}};

#endif
"""
