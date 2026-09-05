"""真实 viewer 的轻量同 BSDF 控制；不加载或训练任何学习模型。"""
from dataclasses import replace
import json
from pathlib import Path
import struct

import numpy as np

from ncls.bundle import ScatteringPackage, write_scattering_package
from ncls.core.identity import sha256_bytes, sha256_json
from ncls.core.material import DiffuseInterface, LayerStackIR, material_program_from_layer_stack
from ncls.references.programs.layer_stack import REFERENCE_PROGRAM_DEFINITION
from ncls.source_materials.families.layer_stack import LayerStackFamilyDefinition


def make_geometry(shaderball: Path, output: Path) -> None:
    """保留 shaderball，分离底座材质，并加入第三材质的接影地面。"""
    raw = shaderball.read_bytes()
    json_size = struct.unpack_from("<I", raw, 12)[0]
    doc = json.loads(raw[20:20 + json_size])
    binary = bytearray(raw[28 + json_size:])
    doc["materials"] = [
        {"name": name, "pbrMetallicRoughness": {"baseColorFactor": color, "metallicFactor": 0}}
        for name, color in (("Subject", [0.6, 0.4, 0.2, 1]),
                            ("Pedestal", [0.3, 0.5, 0.7, 1]), ("Ground", [0.35, 0.35, 0.35, 1]))
    ]
    for i, mesh in enumerate(doc["meshes"]):
        for primitive in mesh["primitives"]:
            primitive["material"] = i

    def accessor(values, kind):
        array = np.asarray(values, dtype="<f4")
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        binary.extend(array.tobytes())
        view = len(doc["bufferViews"])
        doc["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": array.nbytes})
        index = len(doc["accessors"])
        doc["accessors"].append({"bufferView": view, "componentType": 5126,
            "count": len(array), "type": kind, "min": array.min(axis=0).tolist(),
            "max": array.max(axis=0).tolist()})
        return index

    positions = [[-4, 0, -4], [-4, 0, 4], [4, 0, 4], [-4, 0, -4], [4, 0, 4], [4, 0, -4]]
    attrs = {"POSITION": accessor(positions, "VEC3"),
             "NORMAL": accessor([[0, 1, 0]] * 6, "VEC3"),
             "TEXCOORD_0": accessor([[0, 0], [0, 1], [1, 1], [0, 0], [1, 1], [1, 0]], "VEC2")}
    mesh_id = len(doc["meshes"])
    doc["meshes"].append({"name": "Ground", "primitives": [{"attributes": attrs, "material": 2}]})
    node_id = len(doc["nodes"])
    doc["nodes"].append({"name": "Ground", "mesh": mesh_id})
    doc["scenes"][doc.get("scene", 0)]["nodes"].append(node_id)
    doc["buffers"][0]["byteLength"] = len(binary)
    encoded = json.dumps(doc, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(binary))
        + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
        + struct.pack("<II", len(binary), 0x004E4942) + binary)


def make_package(root: Path, *, capabilities: int = 15, broken: bool = False):
    root.mkdir(parents=True, exist_ok=True)
    material = root / "diffuse.json"
    material.write_text(material_program_from_layer_stack(
        LayerStackIR((DiffuseInterface((0.6, 0.4, 0.2)),), ())).to_json(), encoding="utf-8")
    source = LayerStackFamilyDefinition().load_snapshot({"kind": "material-program", "path": str(material)})
    reference = REFERENCE_PROGRAM_DEFINITION
    runtime = reference.compile_runtime()
    # SceneReferenceProgram 已导入 canonical source，包只包装同一公开四入口。
    # 在独立 parity shader 中原 source module 自带 alias；在 scene 中 guard 已阻止该 alias。
    original = runtime.module_closure[runtime.program_module]
    closure = dict(runtime.module_closure)
    closure[runtime.program_module] = original.replace(
        b"#ifndef NCLS_REFERENCE_BACKEND_NO_PACKAGE_BINDING",
        b"#if !defined(NCLS_REFERENCE_BACKEND_NO_PACKAGE_BINDING) && !defined(NCLS_TEST_WRAPPER)")
    wrapper = b'''#define NCLS_TEST_WRAPPER 1
#include "shaders/ncls/reference_backends/layer_stack.slang"
typedef NclsLayerStackIR NclsPackageCompiledMaterial;
StructuredBuffer<NclsLayerStackIR> gNclsCompiledMaterials;
struct NclsPackageState : INclsScatteringState
{
    NclsScatteringContext context;
    float3 color;
    NclsScatteringPdf pdf(float3 wiWorld)
    {
        float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        return {max(wi.z, 0.0f) / NCLS_PI, max(wo.z, 0.0f) / NCLS_PI};
    }
    NclsScatteringEval evaluate<S : ISampleGenerator>(float3 wiWorld, inout S sg)
    {
        NclsScatteringEval result = {};
        float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        if (wo.z <= NCLS_MIN_COS || wi.z <= NCLS_MIN_COS) return result;
        sampleNext2D(sg); // Match the source query's random stream.
        result.f = color / NCLS_PI;
        result.pdf = pdf(wiWorld);
        result.eventFlags = (uint)NclsScatteringEvent::Reflection | (uint)NclsScatteringEvent::Glossy;
        result.valid = 1u;
        return result;
    }
    bool sample<S : ISampleGenerator>(out NclsScatteringSample result, inout S sg)
    {
        result = {};
        float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        float2 u = sampleNext2D(sg);
        NclsRng rng = NclsRng(uint2(asuint(u.x), asuint(u.y)));
        NclsLayerInterfaceIR layer = {};
        layer.kind = (uint)NclsInterfaceKind::Diffuse;
        layer.colorR = color.x; layer.colorG = color.y; layer.colorB = color.z;
        LayerBsdfSample sampled = nclsSampleInterfaceTransport(layer, wo, 1.0f, 1.0f,
            NCLS_SAMPLE_REFLECTION, true, rng);
        if (sampled.valid == 0u) return false;
        result.wiWorld = nclsFrameToWorld(context.surface.shadingFrame, sampled.direction);
        NclsScatteringEval value = evaluate(result.wiWorld, sg);
        if (value.valid == 0u || value.pdf.forward <= 0.0f) return false;
        result.weight = value.f * abs(sampled.direction.z) / value.pdf.forward;
        result.pdf = value.pdf;
        result.eta = 1.0f;
        result.eventFlags = value.eventFlags;
        result.valid = 1u;
        return true;
    }
};
struct NclsPackageBackend : INclsScatteringBackend
{
    typedef NclsPackageCompiledMaterial CompiledMaterial;
    typedef NclsPackageState State;
    State prepare(NclsScatteringContext context, CompiledMaterial material)
    {
        NclsLayerInterfaceIR layer = material.interfaces[0];
        return {context, float3(layer.colorR, layer.colorG, layer.colorB)};
    }
};
NclsPackageBackend nclsCreatePackageBackend() { return {}; }
NclsPackageCompiledMaterial nclsLoadPackageMaterial(uint index) { return gNclsCompiledMaterials[index]; }
'''
    if broken:
        wrapper += b"invalid_shader_for_transaction_test\n"
    closure["control.slang"] = wrapper
    runtime = replace(runtime, program_module="control.slang", module_closure=closure, capabilities=capabilities)
    signature = sha256_json({"source": source.snapshot_id, "capabilities": capabilities,
        "modules": {name: sha256_bytes(data) for name, data in closure.items()}})
    package_root = root / signature[:12]
    if package_root.exists():
        return material, package_root, ScatteringPackage.open(package_root).manifest
    manifest = write_scattering_package(package_root, program_kind="method",
        program_key="viewer-same-bsdf-control", program_version=1,
        program_descriptor_sha256=sha256_json({"fixture": "viewer-same-bsdf-control", "version": 1}),
        runtime_abi=reference.descriptor.runtime_abi, source=source,
        program_payload=runtime, asset_payload=reference.compile_material(source),
        validation={"status": "passed", "parity": {"view": [0, 0, 1], "lights": [[0, 0, 1]],
            "expected_f": [[c / np.pi for c in (0.6, 0.4, 0.2)]],
            "relative_tolerance": 1e-5, "absolute_tolerance": 1e-6}},
        provenance={"test": True})
    return material, package_root, manifest
