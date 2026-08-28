from __future__ import annotations

from dataclasses import dataclass
import io
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import pyexr
import torch

from ncls.core.identity import sha256_json
from ncls.core.scattering import MaterialPayload, ReferenceProgramDefinition, RuntimePayload
from ncls.core.source import SourceSnapshot
from ncls.references.falcor import create_falcor_device, import_falcor


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUERY_SHADER = PROJECT_ROOT / "shaders/ncls/reference_query/reference_query.cs.slang"


@dataclass(frozen=True)
class ScatteringQuery:
    source_index: torch.Tensor
    wo: torch.Tensor
    position: torch.Tensor | None = None
    geometric_normal: torch.Tensor | None = None
    shading_normal: torch.Tensor | None = None
    tangent: torch.Tensor | None = None
    uv: torch.Tensor | None = None
    uv_dx: torch.Tensor | None = None
    uv_dy: torch.Tensor | None = None

    def __post_init__(self) -> None:
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

    @property
    def batch_size(self) -> int:
        return int(self.source_index.shape[0])

    @property
    def device(self) -> torch.device:
        return self.wo.device


@dataclass
class ReferenceQueryLease:
    owner: "ReferenceQueryDispatcher"
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


class ReferenceQueryDispatcher:
    """通过一个 family-agnostic kernel 调用 canonical scattering backend。"""

    def __init__(
        self,
        definition: ReferenceProgramDefinition,
        snapshots: Sequence[SourceSnapshot],
        *,
        query_capacity: int,
        device: torch.device | str = "cuda:0",
        slot_count: int = 2,
    ) -> None:
        values = tuple(snapshots)
        if not values or query_capacity < 1 or slot_count < 2:
            raise ValueError("reference dispatcher requires snapshots, capacity and two slots")
        for snapshot in values:
            definition.validate_snapshot(snapshot)
        requested_device = torch.device(device)
        if requested_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("ReferenceQueryDispatcher requires a CUDA device")
        self.definition = definition
        self.snapshots = values
        self.query_capacity = int(query_capacity)
        self.requested_device = requested_device
        self.runtime = definition.compile_runtime()
        self.materials = tuple(definition.compile_material(value) for value in values)
        source_modules = [
            descriptor
            for material in self.materials
            for descriptor in material.blob_descriptors.values()
            if descriptor.get("kind") == "slang-module-source"
        ]
        if source_modules and len(self.materials) != 1:
            raise ValueError(
                "material-specific source modules currently require one source snapshot"
            )
        self.reference_program_identity = sha256_json(
            {
                "descriptor": definition.descriptor.to_dict(),
                "source_snapshot_ids": [value.snapshot_id for value in values],
            }
        )
        self._falcor = import_falcor()
        self._device = create_falcor_device(self._falcor)
        self._passes = {
            name: self._create_pass(entry)
            for name, entry in (
                ("evaluate", "evaluateReference"),
                ("sample", "sampleReference"),
                ("pdf", "pdfReference"),
            )
        }
        self._resources: list[Any] = []
        self._static_bindings = self._create_static_bindings()
        for compute in self._passes.values():
            for usage, resource in self._static_bindings.items():
                compute.globals[usage] = resource
        self._slots = tuple(self._create_slot() for _ in range(slot_count))
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
        for material in self.materials:
            for name, payload in material.blobs.items():
                descriptor = material.blob_descriptors[name]
                if descriptor.get("kind") == "slang-module-source":
                    desc.add_shader_module(str(descriptor["module_name"])).add_string(
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
            payloads = [material.blobs[name] for material in self.materials if name in material.blobs]
            descriptors = [
                material.blob_descriptors[name]
                for material in self.materials
                if name in material.blob_descriptors
            ]
            if descriptors and descriptors[0].get("kind") != "slang-module-source":
                self._bind_payload_data(
                    result,
                    {name: b"".join(payloads)},
                    {name: descriptors[0]},
                    aggregate=len(payloads) > 1,
                )
        for material in self.materials:
            self._bind_payload_data(
                result,
                material.resources,
                material.resource_descriptors,
                aggregate=False,
            )
        sampler_descriptors = {
            **self.runtime.sampler_descriptors,
            **{
                name: descriptor
                for material in self.materials
                for name, descriptor in material.sampler_descriptors.items()
            },
        }
        for descriptor in sampler_descriptors.values():
            usage = str(descriptor["usage"])
            if usage in result:
                continue
            mode = str(descriptor.get("address_mode", "clamp"))
            address = (
                self._falcor.TextureAddressingMode.Wrap
                if mode == "wrap"
                else self._falcor.TextureAddressingMode.Clamp
            )
            sampler = self._device.create_sampler(
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
                resource = self._texture_payload(name, payload, descriptor)
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

    def _texture_payload(
        self, name: str, payload: bytes, descriptor: Mapping[str, Any]
    ):
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
        scalar = values.ndim == (3 if kind == "texture3d" else 2)
        resource_format = (
            self._falcor.ResourceFormat.R8Unorm
            if values.dtype == np.uint8 and scalar
            else self._falcor.ResourceFormat.R32Float
            if values.dtype == np.float32 and scalar
            else
            self._falcor.ResourceFormat.RGBA8UnormSrgb
            if values.dtype == np.uint8 and descriptor.get("color_space") == "srgb"
            else self._falcor.ResourceFormat.RGBA8Unorm
            if values.dtype == np.uint8
            else self._falcor.ResourceFormat.RGBA32Float
        )
        kwargs = {
            "width": int(values.shape[-2] if kind == "texture3d" else values.shape[1]),
            "height": int(values.shape[-3] if kind == "texture3d" else values.shape[0]),
            "format": resource_format,
            "mip_levels": 1,
            "bind_flags": self._falcor.ResourceBindFlags.ShaderResource,
        }
        if kind == "texture3d":
            kwargs["depth"] = int(values.shape[0])
        texture = self._device.create_texture(**kwargs)
        texture.from_numpy(np.ascontiguousarray(values))
        return texture

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
            raise RuntimeError("reference dispatcher is closed")
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
            raise ValueError("reference query exceeds dispatcher capacity")

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
        meta[:, 2] = 1056964608  # asuint(0.5f)
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
    ) -> tuple[_QuerySlot, ReferenceQueryLease, int, int]:
        if not 1 <= evaluation_samples <= 256:
            raise ValueError("reference evaluation_samples must lie in [1,256]")
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
    ) -> ReferenceEvaluateResult:
        slot, lease, batch, directions = self._dispatch(
            "evaluate", query, wi, seeds, evaluation_samples=evaluation_samples
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

    def end_iteration(self) -> None:
        if self._active:
            raise RuntimeError("cannot end a reference frame with active query leases")
        self._device.end_frame()

    def close(self) -> None:
        if self._active:
            raise RuntimeError("cannot close reference dispatcher with active query leases")
        if self._closed:
            return
        self._slots = ()
        self._passes = {}
        self._static_bindings = {}
        self._resources = []
        self._device = None
        self._closed = True


__all__ = [
    "ReferenceEvaluateResult",
    "ReferencePdfResult",
    "ReferenceQueryDispatcher",
    "ReferenceQueryLease",
    "ReferenceSampleResult",
    "ScatteringQuery",
]
