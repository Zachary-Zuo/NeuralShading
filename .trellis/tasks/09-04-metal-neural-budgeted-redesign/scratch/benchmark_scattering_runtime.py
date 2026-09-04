"""Matched GPU runtime benchmark for frozen scattering packages.

Run this file through ``scripts/run_falcor_python.ps1``.  It deliberately lives
under the task scratch directory: raw timings are machine/run evidence and are
not a stable project API.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SHADER = Path(__file__).with_suffix(".cs.slang")
DEFAULT_PACKAGES = {
    "nvidia_faithful": PROJECT_ROOT
    / "artifacts/nvidia-faithful/materialx-recorded-200k/package",
    "metal_full": PROJECT_ROOT
    / "artifacts/viewer/metal-step00020000-tungsten/packages"
    / "003698fccac260627379da2383403f54f9b239b09f6489a4b0267e1ce483feb8",
}
DEFAULT_MDL_ARTIFACT = (
    PROJECT_ROOT
    / "artifacts/viewer/metal-step00020000-tungsten/reference"
    / "003698fccac260627379da2383403f54f9b239b09f6489a4b0267e1ce483feb8"
)
WORKLOAD_SEED = 2026090403
THREAD_GROUP_SIZE = 64
WINDOWS_SMOKE_LIMITS = {"count": 1024, "warmup": 2, "measurements": 3}


@dataclass(frozen=True)
class PackageInput:
    label: str
    root: Path
    manifest: Mapping[str, Any]
    module: Path
    defines: Mapping[str, str]
    adapter_define: str
    prepared_buffer_stride: int
    reported_prepared_state_bytes: int
    compiled_material_index: int
    blobs: tuple[tuple[Path | bytes, Mapping[str, Any]], ...]
    textures: tuple[tuple[Path, Mapping[str, Any]], ...]
    samplers: tuple[Mapping[str, Any], ...]
    generated_modules: tuple[tuple[str, str], ...]
    program_bytes: int
    asset_bytes: int
    instance_bytes: int


@dataclass
class PackageGpuRuntime:
    package: PackageInput
    passes: dict[str, Any]
    resources: list[Any]
    prepared: Any
    output: Any
    load_seconds: float
    compile_seconds: float


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor_files(
    root: Path,
    manifest: Mapping[str, Any],
    groups: Iterable[Mapping[str, Any]],
) -> tuple[tuple[Path, Mapping[str, Any]], ...]:
    files = manifest["files"]
    result: list[tuple[Path, Mapping[str, Any]]] = []
    for group in groups:
        for logical_name, descriptor in group.items():
            result.append((root / str(files[logical_name]), descriptor))
    return tuple(result)


def _sum_bytes(items: Iterable[tuple[Path | bytes, Mapping[str, Any]]]) -> int:
    return sum(
        source.stat().st_size if isinstance(source, Path) else len(source)
        for source, _ in items
    )


def load_package(label: str, root: Path) -> PackageInput:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("format_name") != "ncls.scattering-package":
        raise ValueError(f"{label}: not a ScatteringPackage manifest")
    files = manifest.get("files")
    hashes = manifest.get("content_hashes")
    if not isinstance(files, dict) or not isinstance(hashes, dict):
        raise ValueError(f"{label}: package file/hash maps are missing")
    for uri, expected in hashes.items():
        path = root / str(uri)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"{label}: missing or modified package member {uri}")

    version = int(manifest["format_version"])
    program = manifest["program"]
    program_blobs = _descriptor_files(root, manifest, (program.get("blobs", {}),))
    defines = {str(key): str(value) for key, value in program.get("defines", {}).items()}
    module = root / str(files[str(program["module"])])
    if version == 1:
        material = manifest["material"]
        material_blobs = _descriptor_files(root, manifest, (material.get("blobs", {}),))
        material_resources = _descriptor_files(
            root, manifest, (material.get("resources", {}),)
        )
        textures = tuple(
            item
            for item in material_resources
            if str(item[1].get("dtype", "")).startswith("texture2d-")
        )
        samplers = tuple(
            descriptor
            for _, descriptor in material_resources
            if str(descriptor.get("dtype", "")).startswith("sampler-")
        )
        if manifest.get("program_key") != "nvidia-neural-appearance":
            raise ValueError(f"{label}: unsupported historical package adapter")
        return PackageInput(
            label=label,
            root=root,
            manifest=manifest,
            module=module,
            defines=defines,
            adapter_define="NCLS_BENCH_ADAPTER_NVIDIA_V1",
            prepared_buffer_stride=96,
            reported_prepared_state_bytes=96,
            compiled_material_index=0,
            blobs=program_blobs + material_blobs,
            textures=textures,
            samplers=samplers,
            generated_modules=(),
            program_bytes=_sum_bytes(program_blobs),
            asset_bytes=_sum_bytes(material_blobs + material_resources),
            instance_bytes=0,
        )
    if version != 2:
        raise ValueError(f"{label}: unsupported package version {version}")

    asset = manifest["asset"]
    instance = manifest["instance"]
    asset_blobs = _descriptor_files(root, manifest, (asset.get("blobs", {}),))
    asset_resources = _descriptor_files(root, manifest, (asset.get("resources", {}),))
    instance_blobs = _descriptor_files(root, manifest, (instance.get("blobs", {}),))
    samplers = tuple(asset.get("samplers", {}).values()) + tuple(
        program.get("samplers", {}).values()
    )
    program_key = str(manifest.get("program_key"))
    if program_key == "metal-fused-neural-material":
        adapter = "NCLS_BENCH_ADAPTER_METAL_FULL_V1"
        # This is sizeof(NclsMetalPrepared) in the frozen Slang module.  The
        # profile's ABI reservation is 2816 B and is reported separately.
        storage_stride, reported_stride = 1880, 2816
    elif program_key == "metal-budgeted-neural-material":
        adapter = "NCLS_BENCH_ADAPTER_COMMON_PACKED"
        storage_stride = reported_stride = int(
            manifest.get("prepared_state_bytes", 160)
        )
    else:
        raise ValueError(f"{label}: no lossless state adapter for {program_key!r}")
    return PackageInput(
        label=label,
        root=root,
        manifest=manifest,
        module=module,
        defines=defines,
        adapter_define=adapter,
        prepared_buffer_stride=storage_stride,
        reported_prepared_state_bytes=reported_stride,
        compiled_material_index=int(instance["parameters"]["compiled_material_index"]),
        blobs=program_blobs + asset_blobs + instance_blobs,
        textures=tuple(
            item
            for item in asset_resources
            if str(item[1].get("dtype", "")).startswith("texture2d-")
        ),
        samplers=samplers,
        generated_modules=(),
        program_bytes=_sum_bytes(program_blobs),
        asset_bytes=_sum_bytes(asset_blobs + asset_resources),
        instance_bytes=_sum_bytes(instance_blobs),
    )


def load_mdl_artifact(label: str, root: Path) -> PackageInput:
    """Adapt the frozen SDK target-code artifact to the public source ABI."""

    root = root.resolve()
    manifest_path = root / "manifest.json"
    artifact = _read_json(manifest_path)
    if artifact.get("schema") != "ncls.mdl-compiled-artifact@1":
        raise ValueError(f"{label}: not a frozen MDL compiled artifact")
    declared = artifact.get("files_sha256", {})
    for uri, expected in declared.items():
        path = root / str(uri)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"{label}: missing or modified MDL artifact member {uri}")

    target_types = (
        PROJECT_ROOT
        / "external/MDL-SDK-2025.0.0-387700.1252-nt-x86-64"
        / "examples/mdl_sdk/dxr/content/mdl_target_code_types.hlsl"
    )
    runtime_source = PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_runtime.slangh"
    if not target_types.is_file() or not runtime_source.is_file():
        raise FileNotFoundError("the locked MDL target-code headers are unavailable")
    generated_path = root / str(artifact["code"])
    generated_source = "\n".join(
        (
            "#define MDL_NUM_TEXTURE_RESULTS 16",
            "#define MDL_DF_HANDLE_SLOT_MODE -1",
            "struct NclsMdlRendererState { float3 view_direction; };",
            "#define RENDERER_STATE_TYPE NclsMdlRendererState",
            f"#define NCLS_MDL_TEXTURE_COUNT {max(1, len(artifact.get('textures', ())))}",
            target_types.read_text(encoding="utf-8"),
            runtime_source.read_text(encoding="utf-8"),
            generated_path.read_text(encoding="utf-8"),
        )
    )

    argument = (root / str(artifact["argument_block"]["path"])).read_bytes()
    argument += bytes((-len(argument)) % 16)
    ro_segments = artifact.get("ro_data", ())
    ro_data = (
        (root / str(ro_segments[0]["path"])).read_bytes() if ro_segments else b""
    )
    ro_data += bytes(max(16, ((len(ro_data) + 15) // 16) * 16) - len(ro_data))
    records = struct.pack("<4I", 0, 0, 0, 0)
    blobs: tuple[tuple[Path | bytes, Mapping[str, Any]], ...] = (
        (
            argument,
            {
                "kind": "structured-buffer",
                "dtype": "float32",
                "shape": [len(argument) // 4],
                "stride": 16,
                "usage": "gMdlArgumentBlock",
            },
        ),
        (
            ro_data,
            {
                "kind": "structured-buffer",
                "dtype": "float32",
                "shape": [len(ro_data) // 4],
                "stride": 16,
                "usage": "gMdlRoData",
            },
        ),
        (
            records,
            {
                "kind": "structured-buffer",
                "dtype": "uint32",
                "shape": [1, 4],
                "stride": 16,
                "usage": "gMdlMaterialRecords",
            },
        ),
    )
    textures: list[tuple[Path, Mapping[str, Any]]] = []
    for texture in artifact.get("textures", ()):
        if texture.get("shape") != "2d":
            raise ValueError("the frozen MDL benchmark currently accepts 2D textures")
        channels = {"Sint8": 1, "Rgb": 3, "Rgba": 4}.get(
            str(texture.get("pixel_type"))
        )
        if channels is None:
            raise ValueError("the frozen MDL texture uses an unsupported pixel type")
        textures.append(
            (
                root / str(texture["data"]),
                {
                    "kind": "texture2d",
                    "dtype": "uint8",
                    "shape": [int(texture["height"]), int(texture["width"]), channels],
                    "stride": channels,
                    "usage": f"gMdlTexture2D{int(texture['index']) - 1}",
                    "source_layout": "mdl-decoded-texture@1",
                    "data_origin": texture.get("data_origin"),
                    "gamma": texture.get("gamma", "linear"),
                    "color_space": "linear",
                },
            )
        )
    sampler = {
        "kind": "sampler",
        "usage": "gMdlTextureSampler",
        "filter": "linear",
        "address_mode": "wrap",
    }
    pseudo_manifest = {
        "package_id": _sha256(manifest_path),
        "format_version": "source-abi@1",
        "program_key": "optimized-mdl-tungsten-reference",
        "runtime_abi": "ncls.scattering-backend@1",
    }
    return PackageInput(
        label=label,
        root=root,
        manifest=pseudo_manifest,
        module=PROJECT_ROOT / "shaders/ncls/reference_backends/mdl.slang",
        defines={},
        adapter_define="NCLS_BENCH_ADAPTER_MDL_REFERENCE",
        prepared_buffer_stride=16,
        reported_prepared_state_bytes=0,
        compiled_material_index=0,
        blobs=blobs,
        textures=tuple(textures),
        samplers=(sampler,),
        generated_modules=(("NclsMdlGenerated", generated_source),),
        program_bytes=(
            len(generated_source.encode("utf-8"))
            + (PROJECT_ROOT / "shaders/ncls/reference_backends/mdl.slang").stat().st_size
        ),
        asset_bytes=len(argument) + len(ro_data) + _sum_bytes(textures),
        instance_bytes=len(records),
    )


def _hemisphere(rng: np.random.Generator, count: int) -> np.ndarray:
    values = rng.normal(size=(count, 3)).astype(np.float32)
    values[:, 2] = np.abs(values[:, 2]) + np.float32(0.05)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    result = np.zeros((count, 4), dtype=np.float32)
    result[:, :3] = values
    return result


def make_workload(
    mode: str, count: int, maximum_direction_count: int
) -> dict[str, np.ndarray]:
    if count < 1 or maximum_direction_count < 1:
        raise ValueError("workload sizes must be positive")
    rng = np.random.default_rng(WORKLOAD_SEED)
    views = _hemisphere(rng, count)
    lights = _hemisphere(rng, count * maximum_direction_count)
    uv = rng.random((count, 2), dtype=np.float32)
    footprint = np.full((count, 2), 1.0 / 4096.0, dtype=np.float32)
    uv_and_dx = np.concatenate((uv, footprint * np.asarray((1.0, 0.0))), axis=1)
    dy_and_random = np.concatenate(
        (
            footprint * np.asarray((0.0, 1.0)),
            rng.random((count, 1), dtype=np.float32),
            np.zeros((count, 1), dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    if mode == "coherent":
        views[:] = views[0]
        uv_and_dx[:] = uv_and_dx[0]
        dy_and_random[:] = dy_and_random[0]
        first = lights[:maximum_direction_count].copy()
        lights = np.tile(first, (count, 1))
    elif mode != "divergent":
        raise ValueError(f"unknown workload mode {mode!r}")
    return {
        "views": np.ascontiguousarray(views),
        "lights": np.ascontiguousarray(lights),
        "uv_and_dx": np.ascontiguousarray(uv_and_dx.astype(np.float32)),
        "dy_and_random": np.ascontiguousarray(dy_and_random),
    }


def _dds_levels(path: Path) -> tuple[int, int, list[np.ndarray]]:
    payload = path.read_bytes()
    if len(payload) < 148 or payload[:4] != b"DDS ":
        raise ValueError(f"unsupported benchmark texture: {path}")
    import struct

    _, _, height, width, _, _, mip_count = struct.unpack_from("<7I", payload, 4)
    levels: list[np.ndarray] = []
    offset = 148
    mip_width, mip_height = width, height
    for _ in range(mip_count):
        size = mip_width * mip_height * 4
        level = np.frombuffer(payload, dtype="<f2", count=size, offset=offset)
        levels.append(level.reshape(mip_height, mip_width, 4).copy())
        offset += size * 2
        mip_width, mip_height = max(1, mip_width // 2), max(1, mip_height // 2)
    if offset != len(payload):
        raise ValueError(f"DDS mip chain size mismatch: {path}")
    return width, height, levels


def _create_buffer(
    falcor: Any,
    device: Any,
    source: Path | bytes,
    descriptor: Mapping[str, Any],
):
    payload = source.read_bytes() if isinstance(source, Path) else source
    stride = int(descriptor["stride"])
    # The historical @1 manifest describes scalar FP16 storage.  Its public
    # shader consumes packed half2 words, so the host view is uint32 as it was
    # in the original faithful implementation.
    if descriptor.get("usage") == "gNclsRuntimeWeights" and stride == 2:
        stride = 4
    if not payload or len(payload) % stride:
        raise ValueError(f"typed buffer payload/stride mismatch: {source!r}")
    flags = falcor.ResourceBindFlags.ShaderResource
    if descriptor.get("usage") in {"gNclsMetalRawParameters", "gNclsCompiledMaterials"}:
        flags |= falcor.ResourceBindFlags.UnorderedAccess
    buffer = device.create_structured_buffer(
        struct_size=stride,
        element_count=len(payload) // stride,
        bind_flags=flags,
    )
    buffer.from_numpy(np.frombuffer(payload, dtype=np.uint8).copy())
    return buffer


def _create_texture(falcor: Any, device: Any, path: Path, descriptor: Mapping[str, Any]):
    if descriptor.get("dtype") != "texture2d-rgba16float-dds@1":
        from ncls.references.query import _create_texture_payload

        return _create_texture_payload(  # noqa: SLF001 - typed source ABI adapter
            falcor, device, path.name, path.read_bytes(), descriptor
        )
    width, height, levels = _dds_levels(path)
    texture = device.create_texture(
        width=width,
        height=height,
        format=falcor.ResourceFormat.RGBA16Float,
        mip_levels=len(levels),
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    for mip, level in enumerate(levels):
        texture.from_numpy(level, mip_level=mip)
    return texture


def _create_sampler(falcor: Any, device: Any, descriptor: Mapping[str, Any]):
    dtype = str(descriptor.get("dtype", ""))
    filter_name = str(descriptor.get("filter", "linear"))
    address = str(descriptor.get("address_mode", "wrap"))
    point = filter_name == "point"
    if dtype.startswith("sampler-"):
        point = "point" in dtype
        address = "clamp" if "clamp" in dtype else "wrap"
    filter_mode = (
        falcor.TextureFilteringMode.Point if point else falcor.TextureFilteringMode.Linear
    )
    address_mode = (
        falcor.TextureAddressingMode.Clamp
        if address == "clamp"
        else falcor.TextureAddressingMode.Wrap
    )
    return device.create_sampler(
        mag_filter=filter_mode,
        min_filter=filter_mode,
        mip_filter=falcor.TextureFilteringMode.Point,
        address_mode_u=address_mode,
        address_mode_v=address_mode,
        address_mode_w=address_mode,
    )


def _input_buffer(falcor: Any, device: Any, values: np.ndarray):
    buffer = device.create_structured_buffer(
        struct_size=int(values.shape[1] * values.dtype.itemsize),
        element_count=int(values.shape[0]),
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    buffer.from_numpy(values)
    return buffer


def create_gpu_runtime(
    falcor: Any,
    device: Any,
    package: PackageInput,
    workload: Mapping[str, np.ndarray],
    count: int,
    maximum_direction_count: int,
) -> PackageGpuRuntime:
    load_start = time.perf_counter()
    resources: list[Any] = []
    by_usage: dict[str, Any] = {}
    for source, descriptor in package.blobs:
        resource = _create_buffer(falcor, device, source, descriptor)
        resources.append(resource)
        by_usage[str(descriptor["usage"])] = resource
    for path, descriptor in package.textures:
        resource = _create_texture(falcor, device, path, descriptor)
        resources.append(resource)
        by_usage[str(descriptor["usage"])] = resource
    for descriptor in package.samplers:
        resource = _create_sampler(falcor, device, descriptor)
        resources.append(resource)
        by_usage[str(descriptor["usage"])] = resource

    inputs = {
        "gBenchmarkViews": _input_buffer(falcor, device, workload["views"]),
        "gBenchmarkLights": _input_buffer(falcor, device, workload["lights"]),
        "gBenchmarkUvAndDx": _input_buffer(falcor, device, workload["uv_and_dx"]),
        "gBenchmarkDyAndRandom": _input_buffer(
            falcor, device, workload["dy_and_random"]
        ),
    }
    resources.extend(inputs.values())
    uav = (
        falcor.ResourceBindFlags.ShaderResource
        | falcor.ResourceBindFlags.UnorderedAccess
    )
    prepared = device.create_structured_buffer(
        struct_size=package.prepared_buffer_stride,
        element_count=count,
        bind_flags=uav,
    )
    output = device.create_structured_buffer(
        struct_size=16, element_count=count, bind_flags=uav
    )
    resources.extend((prepared, output))
    device.end_frame()
    device.wait()
    load_seconds = time.perf_counter() - load_start

    defines = dict(package.defines)
    defines[package.adapter_define] = "1"
    defines["NCLS_BENCH_PACKAGE_HEADER"] = f'"{package.module.as_posix()}"'
    entries = ("prepareOnly", "evaluateOnly", "prepareEvaluate", "sampleOnly", "pdfOnly")
    compile_start = time.perf_counter()
    if package.generated_modules:
        passes = {}
        benchmark_source = SHADER.read_text(encoding="utf-8")
        for entry in entries:
            desc = falcor.ProgramDesc()
            for module_name, source in package.generated_modules:
                desc.add_shader_module(module_name).add_string(source, SHADER)
            desc.add_shader_module("NclsBenchmarkRuntime").add_string(
                benchmark_source, SHADER
            )
            desc.cs_entry(entry)
            passes[entry] = falcor.ComputePass(device, desc, defines=defines)
    else:
        passes = {
            entry: falcor.ComputePass(
                device, file=SHADER, cs_entry=entry, defines=defines
            )
            for entry in entries
        }
    for compute in passes.values():
        for usage, resource in by_usage.items():
            setattr(compute.globals, usage, resource)
        for usage, resource in inputs.items():
            setattr(compute.globals, usage, resource)
        compute.globals.gBenchmarkPrepared = prepared
        compute.globals.gBenchmarkOutput = output
        compute.globals.gBenchmarkCount = count
        compute.globals.gBenchmarkDirectionCount = maximum_direction_count
        compute.globals.gBenchmarkCompiledMaterialIndex = package.compiled_material_index
        compute.globals.gBenchmarkSeed = WORKLOAD_SEED
    device.end_frame()
    device.wait()
    compile_seconds = time.perf_counter() - compile_start
    return PackageGpuRuntime(
        package, passes, resources, prepared, output, load_seconds, compile_seconds
    )


def _capture_records(capture: Mapping[str, Any], event_name: str) -> list[float]:
    events = capture.get("events", {})
    lane = events.get(f"/{event_name}/gpu_time") or events.get(
        f"{event_name}/gpu_time"
    )
    if lane is None:
        raise RuntimeError(
            f"GPU profiler capture has no lane for {event_name!r}; "
            f"available={tuple(events)!r}"
        )
    values = [float(value) for value in lane["records"]]
    if not values or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise RuntimeError(f"GPU profiler returned invalid records for {event_name!r}")
    return values


def _bootstrap_median_interval(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    medians = np.empty(2000, dtype=np.float64)
    for index in range(medians.size):
        medians[index] = np.median(rng.choice(values, size=values.size, replace=True))
    return [float(x) for x in np.quantile(medians, (0.025, 0.975))]


def summarize(records_ms: list[float], seed: int) -> Mapping[str, Any]:
    values = np.asarray(records_ms, dtype=np.float64)
    return {
        "unit": "ms",
        "count": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "bootstrap_median_95ci": _bootstrap_median_interval(values, seed),
    }


def benchmark_dispatch(
    device: Any,
    compute: Any,
    *,
    entry: str,
    count: int,
    direction_count: int,
    warmup: int,
    measurements: int,
    progress: tqdm,
) -> Mapping[str, Any]:
    compute.globals.gBenchmarkDirectionCount = direction_count
    for _ in range(warmup):
        compute.execute(threads_x=count)
        device.end_frame()
    device.wait()

    profiler = device.profiler
    profiler.enabled = True
    profiler.reset_stats()
    profiler.start_capture(reserved_frames=measurements + 1)
    event_name = f"ncls_{entry}_n{direction_count}"
    # Falcor initializes capture lanes on their first observed frame and starts
    # recording on the next one. Submit one identical priming dispatch so the
    # output contains exactly ``measurements`` GPU records.
    for measurement_index in range(measurements + 1):
        with profiler.event(event_name):
            compute.execute(threads_x=count)
        profiler.end_frame()
        device.end_frame()
        if measurement_index > 0:
            progress.update(1)
    device.wait()
    # Resolve the final frame before finalizing the capture.
    profiler.end_frame()
    device.end_frame()
    device.wait()
    capture = profiler.end_capture()
    records = _capture_records(capture, event_name)
    # Capture may contain a trailing resolve frame; retain exactly the timed
    # dispatches and keep their unaggregated values in the output.
    records = records[:measurements]
    return {
        "entry": entry,
        "direction_count": direction_count,
        "records_ms": records,
        "summary": summarize(records, WORKLOAD_SEED ^ direction_count),
    }


def _parse_package(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("package must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("package must use non-empty LABEL=PATH")
    return label, Path(raw_path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        action="append",
        type=_parse_package,
        help="frozen ScatteringPackage as LABEL=PATH; defaults to NVIDIA and old Metal",
    )
    parser.add_argument(
        "--workload",
        action="append",
        choices=("coherent", "divergent"),
        help="workload mode; repeat to select both (default: both)",
    )
    parser.add_argument(
        "--mdl-artifact",
        type=Path,
        default=DEFAULT_MDL_ARTIFACT,
        help="frozen optimized MDL target-code artifact (default: Tungsten probe)",
    )
    parser.add_argument(
        "--skip-mdl",
        action="store_true",
        help="omit the optimized MDL source-ABI control",
    )
    parser.add_argument(
        "--only-mdl",
        action="store_true",
        help="benchmark only the optimized MDL source-ABI control",
    )
    parser.add_argument("--count", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--measurements", type=int, default=100)
    parser.add_argument(
        "--directions", type=int, nargs="+", default=(1, 4, 8), choices=(1, 4, 8)
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/metal-budgeted/runtime/frozen-controls.raw.json",
    )
    return parser.parse_args()


def enforce_platform_limits(*, count: int, warmup: int, measurements: int) -> None:
    if sys.platform != "win32":
        return
    values = {"count": count, "warmup": warmup, "measurements": measurements}
    exceeded = {
        name: (values[name], limit)
        for name, limit in WINDOWS_SMOKE_LIMITS.items()
        if values[name] > limit
    }
    if exceeded:
        detail = ", ".join(
            f"{name}={value} (maximum {limit})"
            for name, (value, limit) in exceeded.items()
        )
        raise RuntimeError(
            "Windows only permits bounded runtime-harness smoke; "
            f"{detail}. Run the full matched workload on Linux/headless."
        )


def main() -> int:
    args = _arguments()
    if args.count < 1 or args.warmup < 0 or args.measurements < 2:
        raise ValueError("count/measurements must be positive and warmup nonnegative")
    enforce_platform_limits(
        count=args.count, warmup=args.warmup, measurements=args.measurements
    )
    if args.skip_mdl and args.only_mdl:
        raise ValueError("--skip-mdl and --only-mdl are mutually exclusive")
    requested_packages = [] if args.only_mdl else (
        args.package or list(DEFAULT_PACKAGES.items())
    )
    packages = [load_package(label, path) for label, path in requested_packages]
    if not args.skip_mdl:
        packages.insert(0, load_mdl_artifact("optimized_mdl_tungsten", args.mdl_artifact))
    workloads = tuple(dict.fromkeys(args.workload or ("coherent", "divergent")))
    directions = tuple(dict.fromkeys(args.directions))

    import falcor
    from ncls.references.backend import create_reference_backend

    device_start = time.perf_counter()
    device = create_reference_backend()._create_device(falcor)  # noqa: SLF001
    device_seconds = time.perf_counter() - device_start
    operation_count = 4 + len(directions)
    total = len(packages) * len(workloads) * operation_count * args.measurements
    result: dict[str, Any] = {
        "schema": "ncls.matched-scattering-runtime@1",
        "status": "diagnostic-report-only",
        "protocol": {
            "seed": WORKLOAD_SEED,
            "count": args.count,
            "thread_group_size": THREAD_GROUP_SIZE,
            "packet_width": THREAD_GROUP_SIZE,
            "warmup": args.warmup,
            "measurements": args.measurements,
            "workloads": workloads,
            "prepare_evaluate_direction_counts": directions,
            "precision": "package-native",
            "synchronization": "Falcor GPU profiler frame fence plus device.wait",
        },
        "device": {"repr": str(device.info), "create_seconds": device_seconds},
        "controls": {},
        "unavailable_controls": {},
    }
    with tqdm(total=total, unit="dispatch", desc="matched runtime") as progress:
        for package in packages:
            label = package.label
            package_result: dict[str, Any] = {
                "package": {
                    "root": str(package.root),
                    "package_id": package.manifest["package_id"],
                    "manifest_sha256": _sha256(package.root / "manifest.json"),
                    "format_version": package.manifest["format_version"],
                    "program_key": package.manifest["program_key"],
                    "runtime_abi": package.manifest["runtime_abi"],
                },
                "static": {
                    "program_bytes": package.program_bytes,
                    "asset_bytes": package.asset_bytes,
                    "instance_bytes": package.instance_bytes,
                    "reported_prepared_state_bytes": package.reported_prepared_state_bytes,
                    "benchmark_storage_stride": package.prepared_buffer_stride,
                },
                "workloads": {},
            }
            for workload_name in workloads:
                workload = make_workload(workload_name, args.count, max(directions))
                runtime = create_gpu_runtime(
                    falcor, device, package, workload, args.count, max(directions)
                )
                # Populate the persistent prepared buffer outside every timed
                # evaluate/sample/pdf measurement.
                runtime.passes["prepareOnly"].execute(threads_x=args.count)
                device.end_frame()
                device.wait()
                workload_result: dict[str, Any] = {
                    "cpu_lifecycle": {
                        "load_seconds": runtime.load_seconds,
                        "compile_seconds": runtime.compile_seconds,
                    },
                    "measurements": {},
                }
                requests = [
                    ("prepare-only", "prepareOnly", 1),
                    ("evaluate-only", "evaluateOnly", 1),
                    ("sample-only", "sampleOnly", 1),
                    ("pdf-only", "pdfOnly", 1),
                    *(
                        (f"prepare+evaluate-{count}", "prepareEvaluate", count)
                        for count in directions
                    ),
                ]
                for name, entry, direction_count in requests:
                    workload_result["measurements"][name] = benchmark_dispatch(
                        device,
                        runtime.passes[entry],
                        entry=entry,
                        count=args.count,
                        direction_count=direction_count,
                        warmup=args.warmup,
                        measurements=args.measurements,
                        progress=progress,
                    )
                checksum_values = (
                    runtime.output.to_numpy().view(np.float32).reshape(args.count, 4).copy()
                )
                if not np.isfinite(checksum_values).all():
                    raise RuntimeError(f"{label}/{workload_name}: non-finite benchmark output")
                workload_result["last_output_checksum"] = float(
                    checksum_values.astype(np.float64).sum()
                )
                package_result["workloads"][workload_name] = workload_result
                del runtime
            result["controls"][label] = package_result

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "controls": list(result["controls"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
