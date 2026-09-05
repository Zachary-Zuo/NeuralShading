from __future__ import annotations

from dataclasses import dataclass, replace
from collections import OrderedDict
import io
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import pyexr
import torch

from ncls.core.identity import sha256_json
from ncls.core.scattering import read_resource_payload
from ncls.references.backend import ReferenceBackendDescriptor
from ncls.references.plan import ReferenceExecutionGroup, ReferenceExecutionPlan


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUERY_SHADER = PROJECT_ROOT / "shaders/ncls/reference_query/reference_query.cs.slang"


@dataclass(frozen=True)
class ScatteringQuery:
    source_index: torch.Tensor
    wo: torch.Tensor
    execution_group_id: str
    position: torch.Tensor | None = None
    geometric_normal: torch.Tensor | None = None
    shading_normal: torch.Tensor | None = None
    tangent: torch.Tensor | None = None
    uv: torch.Tensor | None = None
    uv_dx: torch.Tensor | None = None
    uv_dy: torch.Tensor | None = None
    filter_random: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not self.execution_group_id:
            raise ValueError("ScatteringQuery requires an execution_group_id")
        if self.source_index.ndim != 1 or self.source_index.dtype != torch.int64:
            raise ValueError("ScatteringQuery source_index must be int64 [batch]")
        count = int(self.source_index.shape[0])
        if self.wo.shape != (count, 3) or not self.wo.is_floating_point():
            raise ValueError("ScatteringQuery wo must be floating [batch,3]")
        if self.source_index.device != self.wo.device:
            raise ValueError("ScatteringQuery tensors must share one device")
        for name in (
            "position",
            "geometric_normal",
            "shading_normal",
            "tangent",
        ):
            value = getattr(self, name)
            if value is not None and (
                value.shape != (count, 3) or value.device != self.wo.device
            ):
                raise ValueError(f"ScatteringQuery {name} must be [batch,3] on one device")
        for name in ("uv", "uv_dx", "uv_dy"):
            value = getattr(self, name)
            if value is not None and (
                value.shape != (count, 2) or value.device != self.wo.device
            ):
                raise ValueError(f"ScatteringQuery {name} must be [batch,2] on one device")
        if self.filter_random is not None:
            value = self.filter_random
            if value.shape != (count,) or value.device != self.wo.device or not value.is_floating_point():
                raise ValueError("ScatteringQuery filter_random must be floating [batch] on one device")
            valid = (torch.isfinite(value) & (value >= 0.0) & (value < 1.0)).all()
            if value.device.type == "cuda":
                torch._assert_async(valid)
            elif not bool(valid):
                raise ValueError("ScatteringQuery filter_random must be in [0,1)")

    @property
    def batch_size(self) -> int:
        return int(self.source_index.shape[0])

    @property
    def device(self) -> torch.device:
        return self.wo.device


@dataclass
class ReferenceQueryLease:
    owner: Any
    slot_index: int
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.owner._release(self)
            self.released = True


@dataclass(frozen=True)
class ReferenceEvaluateResult:
    f: torch.Tensor
    pdf_forward: torch.Tensor
    pdf_reverse: torch.Tensor
    event_flags: torch.Tensor
    valid: torch.Tensor
    lease: ReferenceQueryLease


@dataclass(frozen=True)
class ReferenceSampleResult:
    wi: torch.Tensor
    weight: torch.Tensor
    pdf_forward: torch.Tensor
    pdf_reverse: torch.Tensor
    eta: torch.Tensor
    event_flags: torch.Tensor
    valid: torch.Tensor
    lease: ReferenceQueryLease


@dataclass(frozen=True)
class ReferencePdfResult:
    forward: torch.Tensor
    reverse: torch.Tensor
    valid: torch.Tensor
    lease: ReferenceQueryLease


@dataclass
class _QuerySlot:
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    tensors: dict[str, torch.Tensor]


def _texture_extent(kind: str, shape: Sequence[int]) -> dict[str, int]:
    dimensions = tuple(int(value) for value in shape)
    if kind == "texture2d" and len(dimensions) in {2, 3}:
        height, width = dimensions[:2]
        return {"width": width, "height": height}
    if kind == "texture3d" and len(dimensions) in {3, 4}:
        depth, height, width = dimensions[:3]
        return {"width": width, "height": height, "depth": depth}
    raise ValueError(f"{kind} payload has invalid shape {dimensions}")


def _create_texture_payload(
    falcor: Any,
    device: Any,
    name: str,
    payload: bytes,
    descriptor: Mapping[str, Any],
):
    """Materialize one typed reference texture on the concrete Falcor device."""

    kind = str(descriptor["kind"])
    dtype = str(descriptor["dtype"])
    if dtype == "float32":
        values = np.frombuffer(payload, dtype=np.float32).reshape(
            tuple(int(value) for value in descriptor["shape"])
        ).copy()
    elif dtype == "uint8":
        values = np.frombuffer(payload, dtype=np.uint8).reshape(
            tuple(int(value) for value in descriptor["shape"])
        ).copy()
    elif dtype == "uint16":
        values = np.frombuffer(payload, dtype=np.uint16).reshape(
            tuple(int(value) for value in descriptor["shape"])
        ).copy()
    elif dtype == "encoded-image":
        suffix = Path(name).suffix.lower()
        if suffix == ".exr":
            # OpenEXR cannot reopen a live NamedTemporaryFile on Windows.
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"texture{suffix}"
                path.write_bytes(payload)
                values = pyexr.read(path).astype(np.float32)
        else:
            with Image.open(io.BytesIO(payload)) as image:
                scalar = image.mode in {"1", "L", "I", "I;16", "F"}
                values = np.asarray(
                    image.convert("L" if scalar else "RGBA"), copy=True
                )
        if values.ndim == 2:
            values = values[..., None]
        if values.shape[-1] == 1:
            values = np.repeat(values, 4, axis=-1)
        elif values.shape[-1] == 3:
            values = np.concatenate(
                (values, np.ones((*values.shape[:2], 1), dtype=values.dtype)),
                axis=-1,
            )
        values = np.ascontiguousarray(values)
    else:
        raise ValueError(f"unsupported texture payload dtype {dtype!r}")
    if descriptor.get("source_layout") == "mdl-decoded-texture@1":
        if values.ndim != 3 or values.shape[2] not in {1, 2, 3, 4}:
            raise ValueError("MDL decoded texture source has an invalid channel layout")
        origin = descriptor.get("data_origin")
        if origin == "lower_left":
            values = values[::-1].copy()
        elif origin != "top_left":
            raise ValueError("MDL decoded texture source has an invalid row origin")
        if values.shape[2] == 1:
            values = values[..., 0]
        elif values.shape[2] < 4:
            one = (
                np.iinfo(values.dtype).max
                if np.issubdtype(values.dtype, np.integer)
                else 1.0
            )
            expanded = np.zeros((*values.shape[:2], 4), dtype=values.dtype)
            expanded[..., : values.shape[2]] = values
            expanded[..., 3] = one
            values = expanded
        gamma = str(descriptor.get("gamma", "linear"))
        scalar_source = values.ndim == 2
        hardware_srgb = (
            gamma == "srgb" and values.dtype == np.uint8 and not scalar_source
        )
        if gamma == "srgb" and not hardware_srgb:
            normalized = values.astype(np.float32)
            if np.issubdtype(values.dtype, np.integer):
                normalized /= np.float32(np.iinfo(values.dtype).max)
            if scalar_source:
                normalized = np.where(
                    normalized <= np.float32(0.04045),
                    normalized / np.float32(12.92),
                    ((normalized + np.float32(0.055)) / np.float32(1.055))
                    ** np.float32(2.4),
                )
            else:
                color = normalized[..., :3]
                normalized[..., :3] = np.where(
                    color <= np.float32(0.04045),
                    color / np.float32(12.92),
                    ((color + np.float32(0.055)) / np.float32(1.055))
                    ** np.float32(2.4),
                )
            values = normalized.astype(np.float32, copy=False)
    scalar = values.ndim == (3 if kind == "texture3d" else 2)
    resource_format = (
        falcor.ResourceFormat.R8Unorm
        if values.dtype == np.uint8 and scalar
        else falcor.ResourceFormat.R16Unorm
        if values.dtype == np.uint16 and scalar
        else falcor.ResourceFormat.R32Float
        if values.dtype == np.float32 and scalar
        else falcor.ResourceFormat.RGBA8UnormSrgb
        if values.dtype == np.uint8 and descriptor.get("color_space") == "srgb"
        else falcor.ResourceFormat.RGBA8Unorm
        if values.dtype == np.uint8
        else falcor.ResourceFormat.RGBA16Unorm
        if values.dtype == np.uint16
        else falcor.ResourceFormat.RGBA32Float
    )
    kwargs = {
        **_texture_extent(kind, values.shape),
        "format": resource_format,
        "mip_levels": 1,
        "bind_flags": falcor.ResourceBindFlags.ShaderResource,
    }
    texture = device.create_texture(**kwargs)
    texture.from_numpy(np.ascontiguousarray(values))
    return texture


class _ReferenceExecutionGroupSession:
    """一个 group 的 concrete runtime；只由 public plan session 构造。"""

    def __init__(
        self,
        group: ReferenceExecutionGroup,
        *,
        backend_descriptor: ReferenceBackendDescriptor,
        falcor: Any,
        device_handle: Any,
        query_capacity: int,
        device: torch.device | str = "cuda:0",
        slot_count: int = 2,
        requested_operations: Sequence[str] = ("evaluate", "sample", "pdf"),
    ) -> None:
        definition = group.definition
        values = group.snapshots
        if not values or query_capacity < 1 or slot_count < 2:
            raise ValueError("reference backend session requires snapshots, capacity and two slots")
        for snapshot in values:
            definition.validate_snapshot(snapshot)
        requested_device = torch.device(device)
        if requested_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("ReferenceBackendSession requires a CUDA device")
        self.backend_descriptor = backend_descriptor
        self.backend_identity = backend_descriptor.identity
        self.concurrency = backend_descriptor.concurrency
        self.definition = definition
        self.group = group
        self.snapshots = values
        self.query_capacity = int(query_capacity)
        self.requested_device = requested_device
        runtime_started = time.perf_counter()
        self.runtime = definition.compile_runtime()
        runtime_seconds = time.perf_counter() - runtime_started
        self.materials = tuple(record.material for record in group.records)
        source_modules: dict[str, tuple[bytes, Mapping[str, Any]]] = {}
        for material in self.materials:
            for name, payload in material.blobs.items():
                descriptor = material.blob_descriptors[name]
                if descriptor.get("kind") != "slang-module-source":
                    continue
                module_name = str(descriptor["module_name"])
                previous = source_modules.get(module_name)
                current = (payload, descriptor)
                if previous is not None and previous != current:
                    raise ValueError(
                        "one reference execution group contains conflicting generated modules"
                    )
                source_modules[module_name] = current
        self._source_modules = source_modules
        self.reference_program_identity = sha256_json(
            {
                "descriptor": definition.descriptor.to_dict(),
                "source_snapshot_ids": [value.snapshot_id for value in values],
                "reference_backend_identity": self.backend_identity,
            }
        )
        self._falcor = falcor
        self._device = device_handle
        operations = tuple(str(value) for value in requested_operations)
        if (
            not operations
            or len(set(operations)) != len(operations)
            or not set(operations).issubset({"evaluate", "sample", "pdf"})
        ):
            raise ValueError("reference requested_operations are invalid")
        self.requested_operations = operations
        pass_started = time.perf_counter()
        self._passes = {
            name: self._create_pass(entry)
            for name, entry in (
                ("evaluate", "evaluateReference"),
                ("sample", "sampleReference"),
                ("pdf", "pdfReference"),
            )
            if name in operations
        }
        pass_seconds = time.perf_counter() - pass_started
        self._resources: list[Any] = []
        resource_started = time.perf_counter()
        self._static_bindings = self._create_static_bindings()
        for compute in self._passes.values():
            for usage, resource in self._static_bindings.items():
                compute.globals[usage] = resource
        resource_seconds = time.perf_counter() - resource_started
        slot_started = time.perf_counter()
        self._slots = tuple(self._create_slot() for _ in range(slot_count))
        slot_seconds = time.perf_counter() - slot_started
        self.build_profile = {
            "runtime_compile_seconds": runtime_seconds,
            "pass_build_seconds": pass_seconds,
            "resource_bind_seconds": resource_seconds,
            "slot_build_seconds": slot_seconds,
        }
        self.device = next(iter(self._slots[0].tensors.values())).device
        if self.device != requested_device:
            raise RuntimeError(
                f"Falcor interop mapped {self.device}, expected {requested_device}"
            )
        self._active: dict[int, ReferenceQueryLease] = {}
        self._closed = False

    def _create_pass(self, entry: str):
        module = Path(self.runtime.program_module)
        relative = os.path.relpath(
            PROJECT_ROOT / module, QUERY_SHADER.parent
        ).replace("\\", "/")
        source = (
            f'#define NCLS_REFERENCE_PROGRAM_HEADER "{relative}"\n'
            + QUERY_SHADER.read_text(encoding="utf-8")
        )
        desc = self._falcor.ProgramDesc()
        for name, payload in self.runtime.blobs.items():
            descriptor = self.runtime.blob_descriptors[name]
            if descriptor.get("kind") == "slang-module-source":
                desc.add_shader_module(str(descriptor["module_name"])).add_string(
                    payload.decode("utf-8"), QUERY_SHADER
                )
        for module_name, (payload, _) in self._source_modules.items():
            desc.add_shader_module(module_name).add_string(
                payload.decode("utf-8"), QUERY_SHADER
            )
        desc.add_shader_module("NclsReferenceQuery").add_string(source, QUERY_SHADER)
        desc.cs_entry(entry)
        return self._falcor.ComputePass(self._device, desc)

    def _create_static_bindings(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        self._bind_payload_data(
            result,
            self.runtime.blobs,
            self.runtime.blob_descriptors,
            aggregate=False,
        )
        names = set().union(*(material.blobs.keys() for material in self.materials))
        for name in names:
            values = [
                (material.blobs.get(name), material.blob_descriptors.get(name))
                for material in self.materials
            ]
            if any(payload is None or descriptor is None for payload, descriptor in values):
                raise ValueError("execution group material blob table is not invariant")
            first_descriptor = values[0][1]
            if any(descriptor != first_descriptor for _, descriptor in values[1:]):
                raise ValueError("execution group contains conflicting material blob layouts")
            if first_descriptor.get("kind") == "slang-module-source":
                continue
            payloads = [payload for payload, _ in values]
            self._bind_payload_data(
                result,
                {name: b"".join(payloads)},
                {name: first_descriptor},
                aggregate=len(payloads) > 1,
            )
        resource_names = set().union(*(material.resources.keys() for material in self.materials))
        for name in resource_names:
            values = [
                (material.resources.get(name), material.resource_descriptors.get(name))
                for material in self.materials
            ]
            if any(payload is None or descriptor is None for payload, descriptor in values):
                raise ValueError("execution group resource table is not material-invariant")
            first_payload, first_descriptor = values[0]
            if any(
                payload != first_payload or descriptor != first_descriptor
                for payload, descriptor in values[1:]
            ):
                raise ValueError("execution group contains conflicting resource bindings")
            self._bind_payload_data(
                result,
                {name: read_resource_payload(first_payload)},
                {name: first_descriptor},
                aggregate=False,
            )
        layouts = tuple(
            (record.argument_block_offset, record.read_only_data_offset)
            for record in self.group.records
        )
        group_blobs, group_descriptors = self.definition.compile_execution_group_bindings(
            self.materials, layouts
        )
        self._bind_payload_data(
            result,
            group_blobs,
            group_descriptors,
            aggregate=False,
        )
        sampler_descriptors: dict[str, Mapping[str, Any]] = {}
        for name, descriptor in (
            *self.runtime.sampler_descriptors.items(),
            *(
                item
                for material in self.materials
                for item in material.sampler_descriptors.items()
            ),
        ):
            previous = sampler_descriptors.get(name)
            if previous is not None and previous != descriptor:
                raise ValueError("execution group contains conflicting sampler descriptors")
            sampler_descriptors[name] = descriptor
        sampler_usages: set[str] = set()
        for descriptor in sampler_descriptors.values():
            usage = str(descriptor["usage"])
            if usage in result or usage in sampler_usages:
                raise ValueError("typed reference sampler usage is duplicated")
            sampler_usages.add(usage)
            mode = str(descriptor.get("address_mode", "clamp"))
            address = (
                self._falcor.TextureAddressingMode.Wrap
                if mode == "wrap"
                else self._falcor.TextureAddressingMode.Clamp
            )
            filter_name = str(descriptor.get("filter", "linear"))
            filtering = (
                self._falcor.TextureFilteringMode.Point
                if filter_name == "point"
                else self._falcor.TextureFilteringMode.Linear
            )
            sampler = self._device.create_sampler(
                mag_filter=filtering,
                min_filter=filtering,
                mip_filter=filtering,
                max_anisotropy=16 if filter_name == "anisotropic" else 1,
                address_mode_u=address,
                address_mode_v=address,
                address_mode_w=address,
            )
            self._resources.append(sampler)
            result[usage] = sampler
        return result

    def _bind_payload_data(
        self,
        result: dict[str, Any],
        payloads: Mapping[str, bytes],
        descriptors: Mapping[str, Mapping[str, Any]],
        *,
        aggregate: bool,
    ) -> None:
        for name, payload in payloads.items():
            descriptor = descriptors[name]
            kind = str(descriptor.get("kind", "structured-buffer"))
            if kind == "slang-module-source":
                continue
            usage = str(descriptor["usage"])
            if usage in result:
                raise ValueError(f"typed reference binding usage is duplicated: {usage}")
            if kind == "structured-buffer":
                resource = self._structured_payload(payload, descriptor, aggregate=aggregate)
            elif kind in {"texture2d", "texture3d"}:
                resource = _create_texture_payload(
                    self._falcor, self._device, name, payload, descriptor
                )
            else:
                raise ValueError(f"unsupported typed reference binding kind {kind!r}")
            self._resources.append(resource)
            result[usage] = resource

    def _structured_payload(
        self, payload: bytes, descriptor: Mapping[str, Any], *, aggregate: bool
    ):
        stride = int(descriptor["stride"])
        if stride < 1 or len(payload) % stride:
            raise ValueError("structured reference payload size disagrees with stride")
        resource = self._device.create_structured_buffer(
            struct_size=stride,
            element_count=len(payload) // stride,
            bind_flags=self._falcor.ResourceBindFlags.ShaderResource,
        )
        dtype = str(descriptor["dtype"])
        numpy_dtype = {
            "uint8": np.uint8,
            "uint32": np.uint32,
            "float32": np.float32,
        }.get(dtype)
        if numpy_dtype is None:
            raise ValueError(f"unsupported structured payload dtype {dtype!r}")
        values = np.frombuffer(payload, dtype=numpy_dtype).copy()
        if not aggregate:
            expected = int(np.prod(descriptor["shape"]))
            if values.size != expected:
                raise ValueError("structured reference payload shape mismatch")
        resource.from_numpy(values)
        return resource

    def _shared_buffer(self, *, writable: bool = False):
        flags = self._falcor.ResourceBindFlags.ShaderResource | self._falcor.ResourceBindFlags.Shared
        if writable:
            flags |= self._falcor.ResourceBindFlags.UnorderedAccess
        return self._device.create_structured_buffer(
            struct_size=16,
            element_count=self.query_capacity,
            bind_flags=flags,
        )

    def _create_slot(self) -> _QuerySlot:
        input_names = (
            "gNclsQueryPosition",
            "gNclsQueryGeometricNormal",
            "gNclsQueryShadingNormal",
            "gNclsQueryTangent",
            "gNclsQueryUv",
            "gNclsQueryUvDerivatives",
            "gNclsQueryWo",
            "gNclsQueryWi",
            "gNclsQueryMeta",
        )
        output_names = (
            "gNclsQueryFValid",
            "gNclsQueryPdf",
            "gNclsQueryEventValid",
            "gNclsQuerySampleDirectionPdf",
            "gNclsQuerySampleWeightEta",
        )
        inputs = {name: self._shared_buffer() for name in input_names}
        outputs = {name: self._shared_buffer(writable=True) for name in output_names}
        tensors = {
            name: value.to_torch([self.query_capacity, 4], self._falcor.float32)
            for name, value in outputs.items()
        }
        return _QuerySlot(inputs, outputs, tensors)

    def _acquire(self) -> tuple[int, ReferenceQueryLease]:
        if self._closed:
            raise RuntimeError("reference backend session is closed")
        for index in range(len(self._slots)):
            if index not in self._active:
                lease = ReferenceQueryLease(self, index)
                self._active[index] = lease
                return index, lease
        raise RuntimeError("all reference query slots have active leases")

    def _release(self, lease: ReferenceQueryLease) -> None:
        if self._active.get(lease.slot_index) is not lease:
            raise RuntimeError("reference query lease does not own its slot")
        del self._active[lease.slot_index]

    @property
    def active_lease_count(self) -> int:
        return len(self._active)

    def assert_idle(self) -> None:
        if self._active:
            raise RuntimeError("reference execution group has active query leases")

    @staticmethod
    def _float4(value: torch.Tensor, channels: int) -> torch.Tensor:
        result = torch.zeros(
            (len(value), 4), dtype=torch.float32, device=value.device
        )
        result[:, :channels].copy_(value.to(dtype=torch.float32))
        return result

    def _rows(
        self,
        query: ScatteringQuery,
        wi: torch.Tensor | None,
        seeds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if wi is not None:
            if wi.ndim != 3 or wi.shape[0] != query.batch_size or wi.shape[2] != 3:
                raise ValueError("reference query wi must be [batch,direction,3]")
            direction_count = int(wi.shape[1])
        else:
            direction_count = 1
        count = query.batch_size * direction_count
        if count > self.query_capacity:
            raise ValueError("reference query exceeds backend session capacity")

        def repeat(value: torch.Tensor) -> torch.Tensor:
            return value.repeat_interleave(direction_count, dim=0)

        default_position = torch.zeros_like(query.wo)
        default_normal = torch.zeros_like(query.wo)
        default_normal[:, 2] = 1.0
        default_tangent = torch.zeros_like(query.wo)
        default_tangent[:, 0] = 1.0
        default_uv = torch.zeros((query.batch_size, 2), device=query.device)
        uv = default_uv if query.uv is None else query.uv
        uv_dx = default_uv if query.uv_dx is None else query.uv_dx
        uv_dy = default_uv if query.uv_dy is None else query.uv_dy
        if seeds.shape != (query.batch_size, direction_count):
            raise ValueError("reference query seeds must match [batch,direction]")
        meta = torch.zeros((count, 4), dtype=torch.int32, device=query.device)
        meta[:, 0] = repeat(query.source_index).to(torch.int32)
        meta[:, 1] = seeds.reshape(-1).to(torch.int32)
        if query.filter_random is None:
            meta[:, 2] = 1056964608  # asuint(0.5f)
        else:
            meta[:, 2] = repeat(query.filter_random).to(torch.float32).contiguous().view(torch.int32)
        return {
            "gNclsQueryPosition": self._float4(
                repeat(default_position if query.position is None else query.position), 3
            ),
            "gNclsQueryGeometricNormal": self._float4(
                repeat(default_normal if query.geometric_normal is None else query.geometric_normal), 3
            ),
            "gNclsQueryShadingNormal": self._float4(
                repeat(default_normal if query.shading_normal is None else query.shading_normal), 3
            ),
            "gNclsQueryTangent": self._float4(
                repeat(default_tangent if query.tangent is None else query.tangent), 3
            ),
            "gNclsQueryUv": self._float4(repeat(uv), 2),
            "gNclsQueryUvDerivatives": torch.cat((repeat(uv_dx), repeat(uv_dy)), dim=1),
            "gNclsQueryWo": self._float4(repeat(query.wo), 3),
            "gNclsQueryWi": self._float4(
                torch.zeros((count, 3), device=query.device)
                if wi is None
                else wi.reshape(count, 3),
                3,
            ),
            "gNclsQueryMeta": meta,
        }

    def _dispatch(
        self,
        operation: str,
        query: ScatteringQuery,
        wi: torch.Tensor | None,
        seeds: torch.Tensor,
        *,
        evaluation_samples: int = 1,
        footprint_samples: int = 1,
        source_execution_mode: str = "authoritative@1",
    ) -> tuple[_QuerySlot, ReferenceQueryLease, int, int]:
        if operation not in self._passes:
            raise RuntimeError(
                f"reference operation {operation!r} was not requested when the session opened"
            )
        if not 1 <= evaluation_samples <= 256:
            raise ValueError("reference evaluation_samples must lie in [1,256]")
        if not 1 <= footprint_samples <= 64:
            raise ValueError("reference footprint_samples must lie in [1,64]")
        execution_modes = {
            "authoritative@1": 0,
            "prepare-hoisted-pdf-reuse@1": 1,
        }
        if source_execution_mode not in execution_modes:
            raise ValueError("reference source_execution_mode is unknown")
        rows = self._rows(query, wi, seeds)
        direction_count = 1 if wi is None else int(wi.shape[1])
        count = query.batch_size * direction_count
        slot_index, lease = self._acquire()
        slot = self._slots[slot_index]
        try:
            for name, value in rows.items():
                slot.inputs[name].from_torch(value)
            self._device.render_context.wait_for_cuda()
            compute = self._passes[operation]
            for name, value in slot.inputs.items():
                compute.globals[name] = value
            for name, value in slot.outputs.items():
                compute.globals[name] = value
            compute.globals.gNclsQueryCount = count
            compute.globals.gNclsEvaluationSamples = evaluation_samples
            compute.globals.gNclsFootprintSamples = footprint_samples
            compute.globals.gNclsSourceExecutionMode = execution_modes[source_execution_mode]
            compute.execute(threads_x=count)
            self._device.render_context.wait_for_falcor()
        except BaseException:
            lease.release()
            raise
        return slot, lease, query.batch_size, direction_count

    def evaluate(
        self,
        query: ScatteringQuery,
        wi: torch.Tensor,
        seeds: torch.Tensor,
        *,
        evaluation_samples: int = 1,
        footprint_samples: int = 1,
        source_execution_mode: str = "authoritative@1",
    ) -> ReferenceEvaluateResult:
        slot, lease, batch, directions = self._dispatch(
            "evaluate",
            query,
            wi,
            seeds,
            evaluation_samples=evaluation_samples,
            footprint_samples=footprint_samples,
            source_execution_mode=source_execution_mode,
        )
        f_valid = slot.tensors["gNclsQueryFValid"][: batch * directions].reshape(
            batch, directions, 4
        )
        pdf = slot.tensors["gNclsQueryPdf"][: batch * directions].reshape(
            batch, directions, 4
        )
        event = slot.tensors["gNclsQueryEventValid"][: batch * directions].reshape(
            batch, directions, 4
        )
        return ReferenceEvaluateResult(
            f_valid[..., :3],
            pdf[..., 0],
            pdf[..., 1],
            event[..., 0].to(torch.int64),
            event[..., 1] > 0.5,
            lease,
        )

    def sample(
        self, query: ScatteringQuery, seeds: torch.Tensor
    ) -> ReferenceSampleResult:
        slot, lease, batch, _ = self._dispatch("sample", query, None, seeds)
        direction = slot.tensors["gNclsQuerySampleDirectionPdf"][:batch]
        weight = slot.tensors["gNclsQuerySampleWeightEta"][:batch]
        pdf = slot.tensors["gNclsQueryPdf"][:batch]
        event = slot.tensors["gNclsQueryEventValid"][:batch]
        return ReferenceSampleResult(
            direction[:, :3],
            weight[:, :3],
            pdf[:, 0],
            pdf[:, 1],
            weight[:, 3],
            event[:, 0].to(torch.int64),
            event[:, 1] > 0.5,
            lease,
        )

    def pdf(
        self, query: ScatteringQuery, wi: torch.Tensor, seeds: torch.Tensor
    ) -> ReferencePdfResult:
        slot, lease, batch, directions = self._dispatch("pdf", query, wi, seeds)
        pdf = slot.tensors["gNclsQueryPdf"][: batch * directions].reshape(
            batch, directions, 4
        )
        event = slot.tensors["gNclsQueryEventValid"][: batch * directions].reshape(
            batch, directions, 4
        )
        return ReferencePdfResult(pdf[..., 0], pdf[..., 1], event[..., 1] > 0.5, lease)

    def close(self) -> None:
        if self._active:
            raise RuntimeError("cannot close reference backend session with active query leases")
        if self._closed:
            return
        self._slots = ()
        self._passes = {}
        self._static_bindings = {}
        self._resources = []
        self._device = None
        self._closed = True


class ReferenceBackendSession:
    """执行 `ReferenceExecutionPlan@1` 的 family-agnostic session pool。"""

    def __init__(
        self,
        plan: ReferenceExecutionPlan,
        *,
        backend_descriptor: ReferenceBackendDescriptor,
        falcor: Any,
        device_handle: Any,
        query_capacity: int,
        device: torch.device | str = "cuda:0",
        slot_count: int = 2,
        max_resident_groups: int = 8,
        requested_operations: Sequence[str] = ("evaluate", "sample", "pdf"),
    ) -> None:
        if not isinstance(plan, ReferenceExecutionPlan):
            raise TypeError("ReferenceBackendSession requires ReferenceExecutionPlan@1")
        self.plan = plan
        self.backend_descriptor = backend_descriptor
        self.backend_identity = backend_descriptor.identity
        self.snapshots = plan.snapshots
        self.query_capacity = int(query_capacity)
        self.requested_device = torch.device(device)
        if max_resident_groups < 1:
            raise ValueError("reference backend max_resident_groups must be positive")
        self.max_resident_groups = int(max_resident_groups)
        operations = tuple(str(value) for value in requested_operations)
        if (
            not operations
            or len(set(operations)) != len(operations)
            or not set(operations).issubset({"evaluate", "sample", "pdf"})
        ):
            raise ValueError("reference requested_operations are invalid")
        self.requested_operations = operations
        self.reference_program_identity = sha256_json(
            {
                "reference_execution_plan_identity": plan.identity,
                "reference_backend_identity": self.backend_identity,
            }
        )
        self._device_handle = device_handle
        self._falcor = falcor
        self._slot_count = int(slot_count)
        self._groups = {group.group_id: group for group in plan.groups}
        self._sessions: OrderedDict[str, _ReferenceExecutionGroupSession] = OrderedDict()
        self._profile: dict[str, int | float] = {
            "session_hits": 0,
            "session_misses": 0,
            "group_creations": 0,
            "group_evictions": 0,
            "group_build_seconds": 0.0,
            "group_build_seconds_max": 0.0,
            "group_runtime_compile_seconds": 0.0,
            "group_runtime_compile_seconds_max": 0.0,
            "group_pass_build_seconds": 0.0,
            "group_pass_build_seconds_max": 0.0,
            "group_resource_bind_seconds": 0.0,
            "group_resource_bind_seconds_max": 0.0,
            "group_slot_build_seconds": 0.0,
            "group_slot_build_seconds_max": 0.0,
            "evaluate_requests": 0,
            "evaluate_seconds": 0.0,
            "sample_requests": 0,
            "sample_seconds": 0.0,
            "pdf_requests": 0,
            "pdf_seconds": 0.0,
        }
        self.device = self.requested_device
        self._global_to_local: dict[str, torch.Tensor] = {}
        for group in plan.groups:
            mapping = torch.full(
                (len(plan.snapshots),),
                -1,
                dtype=torch.int64,
                device=self.device,
            )
            global_indices = torch.tensor(
                group.global_source_indices,
                dtype=torch.int64,
                device=self.device,
            )
            mapping[global_indices] = torch.arange(
                len(group.records), dtype=torch.int64, device=self.device
            )
            self._global_to_local[group.group_id] = mapping
        self._closed = False

    @property
    def resident_group_ids(self) -> tuple[str, ...]:
        return tuple(self._sessions)

    def _session(self, group_id: str) -> _ReferenceExecutionGroupSession:
        session = self._sessions.get(group_id)
        if session is not None:
            self._profile["session_hits"] += 1
            self._sessions.move_to_end(group_id)
            return session
        self._profile["session_misses"] += 1
        try:
            group = self._groups[group_id]
        except KeyError as error:
            raise ValueError(f"query references unknown execution group {group_id!r}") from error
        if len(self._sessions) >= self.max_resident_groups:
            evicted_id = next(
                (
                    candidate_id
                    for candidate_id, candidate in self._sessions.items()
                    if candidate.active_lease_count == 0
                ),
                None,
            )
            if evicted_id is None:
                raise RuntimeError("all resident reference execution groups have active leases")
            evicted = self._sessions.pop(evicted_id)
            evicted.close()
            self._profile["group_evictions"] += 1
        build_started = time.perf_counter()
        session = _ReferenceExecutionGroupSession(
            group,
            backend_descriptor=self.backend_descriptor,
            falcor=self._falcor,
            device_handle=self._device_handle,
            query_capacity=self.query_capacity,
            device=self.requested_device,
            slot_count=self._slot_count,
            requested_operations=self.requested_operations,
        )
        build_seconds = time.perf_counter() - build_started
        self._profile["group_creations"] += 1
        self._profile["group_build_seconds"] += build_seconds
        self._profile["group_build_seconds_max"] = max(
            self._profile["group_build_seconds_max"], build_seconds
        )
        for name, value in getattr(session, "build_profile", {}).items():
            key = f"group_{name}"
            maximum_key = f"{key}_max"
            self._profile[key] += float(value)
            self._profile[maximum_key] = max(
                self._profile[maximum_key], float(value)
            )
        if session.device != self.device:
            session.close()
            raise RuntimeError("reference execution groups mapped to different CUDA devices")
        self._sessions[group_id] = session
        return session

    @property
    def reference_execution_plan_identity(self) -> str:
        return self.plan.identity

    def _route(
        self, query: ScatteringQuery
    ) -> tuple[_ReferenceExecutionGroupSession, ScatteringQuery]:
        if self._closed:
            raise RuntimeError("reference backend session is closed")
        try:
            session = self._session(query.execution_group_id)
            mapping = self._global_to_local[query.execution_group_id]
        except KeyError as error:
            raise ValueError(
                f"query references unknown execution group {query.execution_group_id!r}"
            ) from error
        local_source_index = mapping.index_select(0, query.source_index)
        valid = torch.all(local_source_index >= 0)
        if valid.device.type == "cuda":
            torch._assert_async(valid)
        elif not bool(valid):
            raise ValueError("query source_index is not owned by its execution group")
        return session, replace(query, source_index=local_source_index)

    def evaluate(
        self,
        query: ScatteringQuery,
        wi: torch.Tensor,
        seeds: torch.Tensor,
        *,
        evaluation_samples: int = 1,
        footprint_samples: int = 1,
        source_execution_mode: str = "authoritative@1",
    ) -> ReferenceEvaluateResult:
        self._profile["evaluate_requests"] += 1
        session, local_query = self._route(query)
        started = time.perf_counter()
        try:
            return session.evaluate(
                local_query,
                wi,
                seeds,
                evaluation_samples=evaluation_samples,
                footprint_samples=footprint_samples,
                source_execution_mode=source_execution_mode,
            )
        finally:
            self._profile["evaluate_seconds"] += time.perf_counter() - started

    def sample(
        self, query: ScatteringQuery, seeds: torch.Tensor
    ) -> ReferenceSampleResult:
        self._profile["sample_requests"] += 1
        session, local_query = self._route(query)
        started = time.perf_counter()
        try:
            return session.sample(local_query, seeds)
        finally:
            self._profile["sample_seconds"] += time.perf_counter() - started

    def pdf(
        self, query: ScatteringQuery, wi: torch.Tensor, seeds: torch.Tensor
    ) -> ReferencePdfResult:
        self._profile["pdf_requests"] += 1
        session, local_query = self._route(query)
        started = time.perf_counter()
        try:
            return session.pdf(local_query, wi, seeds)
        finally:
            self._profile["pdf_seconds"] += time.perf_counter() - started

    def end_iteration(self) -> None:
        if self._closed:
            raise RuntimeError("reference backend session is closed")
        for session in self._sessions.values():
            session.assert_idle()
        self._device_handle.end_frame()

    def profile_snapshot(self, *, reset: bool = False) -> Mapping[str, float]:
        result = {str(name): float(value) for name, value in self._profile.items()}
        result["resident_groups"] = float(len(self._sessions))
        if reset:
            for name in self._profile:
                self._profile[name] = 0.0 if "seconds" in name else 0
        return result

    def close(self) -> None:
        if self._closed:
            return
        for session in self._sessions.values():
            session.assert_idle()
        for session in self._sessions.values():
            session.close()
        self._sessions = {}
        self._global_to_local = {}
        self._groups = {}
        self._falcor = None
        self._device_handle = None
        self._closed = True


__all__ = [
    "ReferenceEvaluateResult",
    "ReferencePdfResult",
    "ReferenceBackendSession",
    "ReferenceQueryLease",
    "ReferenceSampleResult",
    "ScatteringQuery",
]
