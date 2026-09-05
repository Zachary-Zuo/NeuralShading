from __future__ import annotations

import argparse
from pathlib import Path
import struct
import tempfile
from typing import Any, Mapping

import numpy as np

from ncls.bundle import ScatteringPackage
from ncls.references.backend import (
    close_reference_backend_devices,
    create_reference_backend,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在真实 ScatteringPackage 资源上执行 Metal budgeted Slang parity。"
    )
    parser.add_argument("package", type=Path)
    return parser.parse_args()


def _dds_levels(
    payload: bytes, descriptor: Mapping[str, Any]
) -> tuple[np.ndarray, ...]:
    width, height, mip_count, channels = (int(value) for value in descriptor["shape"])
    if channels != 4 or payload[:4] != b"DDS ":
        raise ValueError("Metal budgeted parity 只接受四通道 DDS 资源")
    dtype_name = str(descriptor["dtype"])
    if dtype_name == "texture2d-rgba8-snorm-dds@1":
        dtype = np.dtype(np.int8)
        bytes_per_texel = 4
    elif dtype_name == "texture2d-rgba16float-dds@1":
        dtype = np.dtype("<f2")
        bytes_per_texel = 8
    else:
        raise ValueError(f"Metal budgeted parity 不支持资源类型 {dtype_name!r}")
    header_width = struct.unpack_from("<I", payload, 16)[0]
    header_height = struct.unpack_from("<I", payload, 12)[0]
    header_mips = struct.unpack_from("<I", payload, 28)[0]
    if (header_width, header_height, header_mips) != (width, height, mip_count):
        raise ValueError("DDS header 与 ScatteringPackage descriptor 不一致")
    levels = []
    offset = 148
    mip_width, mip_height = width, height
    for _ in range(mip_count):
        size = mip_width * mip_height * bytes_per_texel
        level = np.frombuffer(payload[offset : offset + size], dtype=dtype).reshape(
            mip_height, mip_width, 4
        )
        levels.append(np.array(level, copy=True, order="C"))
        offset += size
        mip_width, mip_height = max(1, mip_width // 2), max(1, mip_height // 2)
    if offset != len(payload):
        raise ValueError("DDS mip payload 长度不一致")
    return tuple(levels)


def _entry_source(
    package: ScatteringPackage,
    *,
    light_count: int,
    compiled_material_index: int,
) -> str:
    manifest = package.manifest
    defines = [
        f"#define {name} {value}"
        for name, value in sorted(manifest.program["defines"].items())
    ]
    defines.append(f"#define NCLS_PACKAGE_PARITY_LIGHT_COUNT {light_count}")
    module = package.file(str(manifest.program["module"])).as_posix()
    defines.append(f'#include "{module}"')
    defines.append(
        f"""
struct NclsPackageParityGenerator : ISampleGenerator
{{
    uint state;
    [mutating] uint next()
    {{
        state = state * 1664525u + 1013904223u;
        return state;
    }}
}};
StructuredBuffer<float4> gParityLights;
StructuredBuffer<float4> gParityContext;
RWStructuredBuffer<float4> gParityOutput;
[numthreads(1, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{{
    NclsScatteringContext context = {{}};
    context.surface.shadingFrame.normal = float3(0.0f, 0.0f, 1.0f);
    context.surface.shadingFrame.tangent = float3(1.0f, 0.0f, 0.0f);
    context.surface.shadingFrame.bitangent = float3(0.0f, 1.0f, 0.0f);
    context.surface.geometricNormal = context.surface.shadingFrame.normal;
    context.surface.uv = gParityContext[0].xy;
    context.surface.uvDx = float2(0.0f);
    context.surface.uvDy = float2(0.0f);
    context.surface.frontFacing = 1u;
    context.woWorld = normalize(gParityContext[1].xyz);
    context.transportMode = (uint)NclsTransportMode::Radiance;
    context.componentMask = (uint)NclsScatteringEvent::Reflection;
    context.filterRandom = 0.0f;
    const NclsPackageState prepared = nclsCreatePackageBackend().prepare(
        context, nclsLoadPackageMaterial({compiled_material_index}u));
    const NclsMethodPackedState packed = nclsPackMethodState(prepared);
    const NclsPackageState state = nclsUnpackMethodState(
        context, nclsLoadPackageMaterial({compiled_material_index}u), packed,
        gNclsRuntimeWeights);
    [unroll]
    for (uint index = 0u; index < NCLS_PACKAGE_PARITY_LIGHT_COUNT; ++index)
    {{
        NclsPackageParityGenerator generator = {{index + 1u}};
        const NclsScatteringEval evaluation = state.evaluate(
            normalize(gParityLights[index].xyz), generator);
        gParityOutput[index] = float4(evaluation.f, float(evaluation.valid));
        NclsPackageParityGenerator sampleRng = {{index + 1u}};
        NclsScatteringSample sampled;
        prepared.sample(sampled, sampleRng);
        const NclsScatteringPdf pdf = prepared.pdf(sampled.wiWorld);
        const NclsScatteringEval f = prepared.evaluate(sampled.wiWorld, generator);
        const uint base = NCLS_PACKAGE_PARITY_LIGHT_COUNT + 4u * index;
        gParityOutput[base] = float4(sampled.wiWorld, float(sampled.valid));
        gParityOutput[base + 1u] = float4(sampled.pdf.forward, sampled.pdf.reverse, pdf.forward, pdf.reverse);
        gParityOutput[base + 2u] = float4(sampled.weight, 0.0f);
        gParityOutput[base + 3u] = float4(f.f, 0.0f);
    }}
}}
"""
    )
    return "\n".join(defines) + "\n"


def _texture(
    device: Any,
    falcor: Any,
    levels: tuple[np.ndarray, ...],
    dtype: str,
) -> Any:
    resource_format = (
        falcor.ResourceFormat.RGBA8Snorm
        if dtype == "texture2d-rgba8-snorm-dds@1"
        else falcor.ResourceFormat.RGBA16Float
    )
    texture = device.create_texture(
        width=levels[0].shape[1],
        height=levels[0].shape[0],
        format=resource_format,
        mip_levels=len(levels),
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    for index, level in enumerate(levels):
        texture.from_numpy(level, mip_level=index)
    return texture


def _sampler(device: Any, falcor: Any, descriptor: Mapping[str, Any]) -> Any:
    filters = {
        "point": falcor.TextureFilteringMode.Point,
        "linear": falcor.TextureFilteringMode.Linear,
    }
    addresses = {
        "clamp": falcor.TextureAddressingMode.Clamp,
        "wrap": falcor.TextureAddressingMode.Wrap,
    }
    filtering = str(descriptor["filter"])
    if filtering not in filters:
        raise ValueError(f"parity 暂不支持 sampler filter {filtering!r}")
    address = addresses[str(descriptor["address_mode"])]
    return device.create_sampler(
        mag_filter=filters[filtering],
        min_filter=filters[filtering],
        mip_filter=falcor.TextureFilteringMode.Point,
        address_mode_u=address,
        address_mode_v=address,
        address_mode_w=address,
    )


def validate(package_path: Path) -> None:
    import falcor

    package = ScatteringPackage.open(package_path)
    manifest = package.manifest
    if (
        manifest.program_key != "metal-budgeted-neural-material"
        or manifest.validation.get("status") != "gpu-parity-required"
    ):
        raise ValueError("输入不是需要 GPU parity 的 Metal ScatteringPackage")
    parity = dict(manifest.validation["parity"])
    lights = np.asarray(parity["lights"], dtype=np.float32)
    expected = np.asarray(parity["expected_f"], dtype=np.float32)
    view = np.asarray(parity["view"], dtype=np.float32)
    uv = np.asarray(parity["uv"], dtype=np.float32)
    if lights.ndim != 2 or lights.shape[1] != 3 or expected.shape != lights.shape:
        raise ValueError("package parity light/expected shape 不合法")

    with tempfile.TemporaryDirectory(prefix="ncls-metal-package-parity-") as temp:
        entry = Path(temp) / "package_parity.cs.slang"
        entry.write_text(
            _entry_source(
                package,
                light_count=len(lights),
                compiled_material_index=int(
                    manifest.instance["parameters"]["compiled_material_index"]
                ),
            ),
            encoding="utf-8",
        )
        device = create_reference_backend()._create_device(falcor)  # noqa: SLF001
        compute = falcor.ComputePass(device, file=entry, cs_entry="main")
        srv = falcor.ResourceBindFlags.ShaderResource
        uav = srv | falcor.ResourceBindFlags.UnorderedAccess

        for section in (
            manifest.program["blobs"],
            manifest.asset["blobs"],
            manifest.instance["blobs"],
        ):
            for logical_name, descriptor in section.items():
                payload = package.file(logical_name).read_bytes()
                stride = int(descriptor["stride"])
                if stride != 4:
                    raise ValueError(
                        "Metal budgeted parity 当前只接受 stride=4 的 typed blob"
                    )
                resource = device.create_structured_buffer(
                    struct_size=stride,
                    element_count=len(payload) // stride,
                    bind_flags=srv,
                )
                resource.from_numpy(np.frombuffer(payload, dtype=np.uint32).copy())
                compute.globals[str(descriptor["usage"])] = resource

        for logical_name, descriptor in manifest.asset["resources"].items():
            payload = package.file(logical_name).read_bytes()
            compute.globals[str(descriptor["usage"])] = _texture(
                device,
                falcor,
                _dds_levels(payload, descriptor),
                str(descriptor["dtype"]),
            )
        for descriptors in (
            manifest.program["samplers"],
            manifest.asset["samplers"],
        ):
            for descriptor in descriptors.values():
                compute.globals[str(descriptor["usage"])] = _sampler(
                    device, falcor, descriptor
                )

        light_values = np.concatenate(
            (lights, np.zeros((len(lights), 1), dtype=np.float32)), axis=1
        )
        light_buffer = device.create_structured_buffer(
            struct_size=16, element_count=len(lights), bind_flags=srv
        )
        light_buffer.from_numpy(light_values)
        context_values = np.asarray(
            [[uv[0], uv[1], float(parity["mip_level"]), 0.0], [*view, 0.0]],
            dtype=np.float32,
        )
        context_buffer = device.create_structured_buffer(
            struct_size=16, element_count=2, bind_flags=srv
        )
        context_buffer.from_numpy(context_values)
        output = device.create_structured_buffer(
            struct_size=16, element_count=5 * len(lights), bind_flags=uav
        )
        compute.globals.gParityLights = light_buffer
        compute.globals.gParityContext = context_buffer
        compute.globals.gParityOutput = output
        compute.execute(threads_x=1)
        values = output.to_numpy().view(np.float32).reshape(5 * len(lights), 4).copy()
        actual = values[:len(lights)]
        device.end_frame()

    np.testing.assert_allclose(
        actual[:, :3],
        expected,
        rtol=float(parity["relative_tolerance"]),
        atol=float(parity["absolute_tolerance"]),
    )
    np.testing.assert_array_equal(
        actual[:, 3], np.ones(len(lights), dtype=np.float32)
    )
    maximum_error = float(np.max(np.abs(actual[:, :3] - expected)))
    sampling = manifest.validation["sampling"]
    samples = values[len(lights):].reshape(len(lights), 4, 4)
    np.testing.assert_array_equal(samples[:, 0, 3], 1.0)
    rtol, atol = float(sampling["relative_tolerance"]), float(sampling["absolute_tolerance"])
    for actual_values, expected_values in (
        (samples[:, 0, :3], sampling["expected_wi"]),
        (samples[:, 1, :2], sampling["expected_pdf"]),
        (samples[:, 2, :3], sampling["expected_weight"]),
    ):
        np.testing.assert_allclose(actual_values, expected_values, rtol=rtol, atol=atol)
    np.testing.assert_allclose(samples[:, 1, :2], samples[:, 1, 2:], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(samples[:, 2, :3],
        samples[:, 3, :3] * samples[:, 0, 2:3] / samples[:, 1, 0:1], rtol=2e-5, atol=2e-6)
    print(
        f"GPU 四入口 parity 通过：package={manifest.package_id} lights={len(lights)} "
        f"max_abs_error={maximum_error:.8g}"
    )


if __name__ == "__main__":
    try:
        validate(_parse_args().package)
    finally:
        close_reference_backend_devices()
