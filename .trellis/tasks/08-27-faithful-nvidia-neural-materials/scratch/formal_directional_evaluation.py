from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch

from ncls.bundle import RGBA16F_DDS_DTYPE, ScatteringPackage, inspect_rgba16f_dds
from ncls.core.identity import sha256_file
from ncls.data import CollectionConfig
from ncls.data.native_features import MaterialXNativeFeaturePyramid
from ncls.data.providers import MaterialXGpuQueryRuntime, MaterialXProvider, MaterialXProviderConfig
from ncls.learning.methods import get_method
from ncls.learning.models.nvidia_neural_appearance import NvidiaNeuralAppearanceModel
from ncls.learning.training import TrainingConfig, load_checkpoint


class PackageGpuEvaluator:
    """在Falcor中直接执行最终ScatteringPackage，而不是近似FP16 master。"""

    def __init__(self, falcor, device, package_root: Path, capacity: int) -> None:
        self.falcor = falcor
        self.device = device
        self.package = ScatteringPackage.open(package_root)
        self.binding = self.package.create_binding()
        defines = {
            str(name): str(value)
            for name, value in self.binding.program["defines"].items()
        }
        defines["NCLS_PACKAGE_PROGRAM_HEADER"] = (
            '"' + self.binding.program_module.as_posix() + '"'
        )
        self.compute = falcor.ComputePass(
            device,
            file=Path(__file__).with_name("formal_package_evaluation.cs.slang"),
            cs_entry="main",
            defines=defines,
        )
        self.capacity = int(capacity)
        self.resources: list[object] = []
        self.views = self._shared_buffer()
        self.lights = self._shared_buffer()
        self.uv = self._shared_buffer()
        self.gradients = self._shared_buffer()
        self.output = self._shared_buffer(writable=True)
        self.output_tensor = self.output.to_torch(
            [self.capacity, 4], falcor.float32
        )
        self._bind_blob_group(self.binding.program["blobs"], runtime=True)
        self._bind_blob_group(self.binding.material["blobs"], runtime=False)
        self._bind_resources(self.binding.material["resources"])

    def _shared_buffer(self, *, writable: bool = False):
        flags = (
            self.falcor.ResourceBindFlags.ShaderResource
            | self.falcor.ResourceBindFlags.Shared
        )
        if writable:
            flags |= self.falcor.ResourceBindFlags.UnorderedAccess
        return self.device.create_structured_buffer(
            struct_size=16,
            element_count=self.capacity,
            bind_flags=flags,
        )

    def _bind_blob_group(self, descriptors: dict, *, runtime: bool) -> None:
        del runtime
        for logical_name, descriptor in descriptors.items():
            payload = self.binding.files[str(logical_name)].read_bytes()
            usage = str(descriptor["usage"])
            stride = int(descriptor["stride"])
            if usage == "gNclsRuntimeWeights":
                if len(payload) % 4:
                    raise ValueError("package FP16 weights do not contain complete uint words")
                stride = 4
            if stride < 1 or len(payload) % stride:
                raise ValueError(f"package blob {logical_name} has an invalid stride")
            resource = self.device.create_structured_buffer(
                struct_size=stride,
                element_count=len(payload) // stride,
                bind_flags=self.falcor.ResourceBindFlags.ShaderResource,
            )
            resource.from_numpy(np.frombuffer(payload, dtype=np.uint8).copy())
            self.compute.globals[usage] = resource
            self.resources.append(resource)

    @staticmethod
    def _dds_levels(payload: bytes) -> tuple[np.ndarray, ...]:
        width, height, mip_count = inspect_rgba16f_dds(payload)
        levels = []
        offset = 148
        for _ in range(mip_count):
            count = width * height * 4
            level = np.frombuffer(payload, dtype="<f2", count=count, offset=offset)
            # Falcor's NumPy uploader requires writable storage. frombuffer() over
            # immutable package bytes is contiguous but read-only, so force a copy.
            levels.append(level.reshape(height, width, 4).copy())
            offset += count * 2
            width, height = max(1, width // 2), max(1, height // 2)
        if offset != len(payload):
            raise ValueError("package DDS decoder did not consume the full mip chain")
        return tuple(levels)

    def _bind_resources(self, descriptors: dict) -> None:
        for logical_name, descriptor in descriptors.items():
            usage = str(descriptor["usage"])
            dtype = str(descriptor["dtype"])
            if dtype == RGBA16F_DDS_DTYPE:
                levels = self._dds_levels(
                    self.binding.files[str(logical_name)].read_bytes()
                )
                texture = self.device.create_texture(
                    width=levels[0].shape[1],
                    height=levels[0].shape[0],
                    format=self.falcor.ResourceFormat.RGBA16Float,
                    mip_levels=len(levels),
                    bind_flags=self.falcor.ResourceBindFlags.ShaderResource,
                )
                for mip_level, values in enumerate(levels):
                    texture.from_numpy(values, mip_level=mip_level)
                resource = texture
            elif dtype == "sampler-linear-wrap-explicit-lod@1":
                resource = self.device.create_sampler(
                    mag_filter=self.falcor.TextureFilteringMode.Linear,
                    min_filter=self.falcor.TextureFilteringMode.Linear,
                    mip_filter=self.falcor.TextureFilteringMode.Point,
                    address_mode_u=self.falcor.TextureAddressingMode.Wrap,
                    address_mode_v=self.falcor.TextureAddressingMode.Wrap,
                    address_mode_w=self.falcor.TextureAddressingMode.Wrap,
                )
            else:
                raise ValueError(f"formal package evaluator does not support {dtype}")
            self.compute.globals[usage] = resource
            self.resources.append(resource)

    @staticmethod
    def _input(values: torch.Tensor, channels: int) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != channels:
            raise ValueError("formal package evaluation input has an invalid shape")
        result = torch.zeros(
            (len(values), 4), dtype=torch.float32, device=values.device
        )
        result[:, :channels].copy_(values)
        return result

    def evaluate(
        self,
        views: torch.Tensor,
        lights: torch.Tensor,
        uv: torch.Tensor,
        gradients: torch.Tensor,
    ) -> torch.Tensor:
        count = len(views)
        if not 1 <= count <= self.capacity or any(
            len(values) != count for values in (lights, uv, gradients)
        ):
            raise ValueError("formal package evaluation queries are not aligned")
        self.views.from_torch(self._input(views, 3))
        self.lights.from_torch(self._input(lights, 3))
        self.uv.from_torch(self._input(uv, 2))
        self.gradients.from_torch(self._input(gradients, 4))
        self.device.render_context.wait_for_cuda()
        self.compute.globals.gViews = self.views
        self.compute.globals.gLights = self.lights
        self.compute.globals.gUv = self.uv
        self.compute.globals.gGradients = self.gradients
        self.compute.globals.gOutput = self.output
        self.compute.globals.gQueryCount = count
        self.compute.globals.gCompiledMaterialIndex = 0
        self.compute.execute(threads_x=count)
        self.device.render_context.wait_for_falcor()
        result = self.output_tensor[:count, :3].clone()
        torch.cuda.synchronize(result.device)
        return result

    def close(self) -> None:
        self.output_tensor = None
        self.resources.clear()


def hemisphere(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    random_values = torch.rand((*shape, 2), generator=generator, device=device)
    z = random_values[..., 0]
    phi = random_values[..., 1] * (2.0 * math.pi)
    radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    return torch.stack((radius * torch.cos(phi), radius * torch.sin(phi), z), dim=-1)


def distribution(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().flatten()
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        raise RuntimeError("formal evaluation metric distribution is empty or non-finite")
    return {
        "mean": float(flat.mean()),
        "median": float(flat.median()),
        "p90": float(torch.quantile(flat, 0.90)),
        "p95": float(torch.quantile(flat, 0.95)),
        "maximum": float(flat.max()),
    }


def directional_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    directions: torch.Tensor,
) -> dict[str, object]:
    target_integral = target.mean(dim=1) * (2.0 * math.pi)
    prediction_integral = prediction.mean(dim=1) * (2.0 * math.pi)
    floor = max(float(target_integral.max()) * 1e-5, 1e-8)
    integrated_absolute = torch.abs(prediction - target).mean(dim=1) * (2.0 * math.pi)
    normalized_l1 = integrated_absolute / torch.clamp(target_integral.abs(), min=floor)
    energy_error = torch.abs(prediction_integral - target_integral) / torch.clamp(
        target_integral.abs(), min=floor
    )
    log_error = torch.abs(
        torch.log1p(torch.clamp(prediction, min=0.0))
        - torch.log1p(torch.clamp(target, min=0.0))
    ).mean(dim=(1, 2))
    luminance_weights = torch.tensor(
        (0.2126, 0.7152, 0.0722), dtype=torch.float32, device=target.device
    )
    target_peak = torch.argmax(torch.sum(target * luminance_weights, dim=-1), dim=1)
    prediction_peak = torch.argmax(
        torch.sum(prediction * luminance_weights, dim=-1), dim=1
    )
    row = torch.arange(len(target), device=target.device)
    peak_cosine = torch.sum(
        directions[row, target_peak] * directions[row, prediction_peak], dim=1
    )
    peak_degrees = torch.rad2deg(torch.acos(torch.clamp(peak_cosine, -1.0, 1.0)))
    return {
        "solid_angle_normalized_l1": distribution(normalized_l1),
        "hemisphere_energy_relative_error": distribution(energy_error),
        "log1p_l1": distribution(log_error),
        "sampled_peak_direction_error_degrees": distribution(peak_degrees),
    }


def matched_sampler_metrics(
    model: NvidiaNeuralAppearanceModel,
    latent: torch.Tensor,
    views: torch.Tensor,
    uniform_directions: torch.Tensor,
    generator: torch.Generator,
) -> dict[str, object]:
    position_count, direction_count = uniform_directions.shape[:2]
    with torch.no_grad():
        uniform_pdf = model.sampler_pdf(
            latent, views, uniform_directions, "nvidia-diffuse-ggx9"
        )
        normalization = uniform_pdf.mean(dim=1) * (2.0 * math.pi)
        uniform_response = model.response(latent, views, uniform_directions)
        repeated_latent = torch.repeat_interleave(latent, direction_count, dim=0)
        repeated_views = torch.repeat_interleave(views, direction_count, dim=0)
        sample_u = torch.rand(
            (position_count * direction_count, 2),
            generator=generator,
            device=latent.device,
        )
        sampled_wi, sampled_pdf, _, valid = model.sampler_sample_with_head(
            repeated_latent,
            repeated_views,
            sample_u,
            "nvidia-diffuse-ggx9",
        )
        sampled_response = model.response(repeated_latent, repeated_views, sampled_wi)
        luminance_weights = torch.tensor(
            (0.2126, 0.7152, 0.0722), dtype=torch.float32, device=latent.device
        )
        uniform_weights = (
            torch.sum(uniform_response * luminance_weights, dim=-1) * (2.0 * math.pi)
        )
        learned_weights = torch.sum(
            sampled_response[:, 0] * luminance_weights, dim=-1
        ) / torch.clamp(sampled_pdf[:, 0], min=1e-12)
        learned_weights = learned_weights.reshape(position_count, direction_count)
        valid_rows = valid.reshape(position_count, direction_count)
        learned_weights = torch.where(valid_rows, learned_weights, 0.0)

        def relative_variation(values: torch.Tensor) -> torch.Tensor:
            return values.std(dim=1, correction=1) / torch.clamp(
                values.mean(dim=1).abs(), min=1e-8
            )

        uniform_cv = relative_variation(uniform_weights)
        learned_cv = relative_variation(learned_weights)
    return {
        "pdf_hemisphere_integral": distribution(normalization),
        "valid_fraction": float(valid.float().mean()),
        "uniform_estimator_relative_stddev": distribution(uniform_cv),
        "learned_estimator_relative_stddev": distribution(learned_cv),
        "learned_to_uniform_relative_stddev": distribution(
            learned_cv / torch.clamp(uniform_cv, min=1e-8)
        ),
    }


parser = argparse.ArgumentParser()
parser.add_argument("config", type=Path)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("package", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--positions", type=int, default=64)
parser.add_argument("--directions", type=int, default=4096)
parser.add_argument("--expected-step", type=int, required=True)
args = parser.parse_args()
if args.positions < 2 or args.directions < 64:
    raise ValueError("directional evaluation requires at least 2 positions and 64 directions")
if not 1 <= args.expected_step <= 300_000:
    raise ValueError("expected step must be within the frozen formal schedule")

config = TrainingConfig.load(args.config)
definition = get_method(config.method_key)
checkpoint = load_checkpoint(args.checkpoint, descriptor=definition.descriptor)
expected_phase = "complete" if args.expected_step == config.total_steps else "finetune"
if checkpoint.step != args.expected_step or checkpoint.phase != expected_phase:
    raise ValueError(
        "formal directional evaluation checkpoint disagrees with the explicitly "
        f"recorded step/phase ({args.expected_step}, {expected_phase})"
    )

device = torch.device(config.device)
generator = torch.Generator(device=device).manual_seed(config.seed ^ 0x4E564556)
provider = MaterialXProvider(
    CollectionConfig(
        name="nvidia-formal-directional-evaluation",
        view_count=1,
        light_count=1,
        spatial_sample_count=1,
        proposal="uniform",
        seed=config.seed,
    ),
    MaterialXProviderConfig(
        asset_ids=(str(config.batch_source["options"]["materialx_asset_id"]),)
    ),
)
state = tuple(provider.source_states())[0]
query_count = args.positions * args.directions
runtime = MaterialXGpuQueryRuntime(
    provider, state, query_capacity=query_count, slot_count=2
)
package_runtime: PackageGpuEvaluator | None = None
try:
    uv = torch.rand((args.positions, 2), generator=generator, device=device)
    views = hemisphere((args.positions,), generator, device)
    lights = hemisphere((args.positions, args.directions), generator, device)
    gradients = torch.zeros((args.positions, 4), dtype=torch.float32, device=device)
    gradients[:, 0] = 1.0 / float(config.model_context["latent_width"])
    gradients[:, 3] = 1.0 / float(config.model_context["latent_height"])
    target = runtime.evaluate_torch(
        0,
        torch.repeat_interleave(views, args.directions, dim=0),
        lights.reshape(query_count, 3),
        torch.repeat_interleave(uv, args.directions, dim=0),
        torch.repeat_interleave(gradients, args.directions, dim=0),
    ).clone().reshape(args.positions, args.directions, 3)
    torch.cuda.synchronize(device)

    package_runtime = PackageGpuEvaluator(
        runtime.falcor, runtime.device, args.package, query_count
    )
    if package_runtime.binding.source_snapshot_id != state.state_id:
        raise ValueError("formal package source identity disagrees with the reference")
    if (
        package_runtime.package.manifest.program_descriptor_sha256
        != definition.descriptor.descriptor_sha256
    ):
        raise ValueError("formal package method descriptor identity is stale")
    flat_views = torch.repeat_interleave(views, args.directions, dim=0)
    flat_lights = lights.reshape(query_count, 3)
    flat_uv = torch.repeat_interleave(uv, args.directions, dim=0)
    flat_gradients = torch.repeat_interleave(gradients, args.directions, dim=0)
    deployed_prediction = package_runtime.evaluate(
        flat_views, flat_lights, flat_uv, flat_gradients
    ).reshape(args.positions, args.directions, 3)
    deployed_metrics = directional_metrics(deployed_prediction, target, lights)

    model = definition.create_trainable(config.model_context).to(device)
    assert isinstance(model, NvidiaNeuralAppearanceModel)
    definition.restore_training_state(model, checkpoint.model_state)
    definition.configure_lifecycle(model, checkpoint.lifecycle_state)
    model.eval()
    mip_zero = torch.zeros(args.positions, dtype=torch.float32, device=device)
    with torch.no_grad():
        trained_latent = model.fetch_latent(uv, mip_zero)
        trained_prediction = model.response(trained_latent, views, lights)
        trained_metrics = directional_metrics(trained_prediction, target, lights)
        runtime_vs_master = directional_metrics(
            deployed_prediction, trained_prediction, lights
        )
        sampler_metrics = matched_sampler_metrics(
            model, trained_latent, views, lights, generator
        )
    del model, trained_latent, trained_prediction, deployed_prediction
    torch.cuda.empty_cache()

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    initial_model = definition.create_trainable(config.model_context).to(device)
    assert isinstance(initial_model, NvidiaNeuralAppearanceModel)
    definition.configure_lifecycle(initial_model, {"stage": "bootstrap"})
    source_runtime = state.runtime_state
    feature_pyramid = MaterialXNativeFeaturePyramid.from_textures(
        source_runtime.inputs,
        base_color=source_runtime.base_color,
        roughness=source_runtime.roughness,
        metalness=source_runtime.metalness,
        normal=source_runtime.normal,
    )
    with torch.no_grad():
        initial_features = feature_pyramid.sample_torch(uv, mip_zero)
        initial_latent = initial_model.encode(initial_features)
        initial_prediction = initial_model.response(initial_latent, views, lights)
        initial_metrics = directional_metrics(initial_prediction, target, lights)

    report = {
        "schema_name": "ncls.nvidia-formal-directional-evaluation",
        "schema_version": 1,
        "scope": "single-trained-material-mip0-matched-query-diagnostic",
        "claim_boundary": (
            "Observed single-snapshot quality; no author-asset image claim and no "
            "source-state bootstrap confidence interval."
        ),
        "training_config_sha256": config.sha256,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": checkpoint.step,
        "checkpoint_phase": checkpoint.phase,
        "configured_total_steps": config.total_steps,
        "package_id": package_runtime.package.manifest.package_id,
        "program_runtime_id": package_runtime.binding.program_runtime_id,
        "material_asset_id": package_runtime.binding.material_asset_id,
        "source_snapshot_id": state.state_id,
        "query": {
            "positions": args.positions,
            "directions_per_position": args.directions,
            "incident_proposal": "uniform-hemisphere@1",
            "mip_level": 0,
            "seed": config.seed ^ 0x4E564556,
        },
        "initial": initial_metrics,
        "fp32_training_master": trained_metrics,
        "packed_fp16_runtime": deployed_metrics,
        "packed_runtime_vs_fp32_master": runtime_vs_master,
        "fp32_master_matched_sampler": sampler_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
finally:
    falcor_device = runtime.device
    if package_runtime is not None:
        package_runtime.close()
    runtime.close()
    torch.cuda.synchronize(device)
    gc.collect()
    # Both MaterialX and package output buffers expose CUDA shared views. Advance
    # the Falcor frame only after every lease is released; doing it per dispatch
    # can retire a CopyContext while CUDA still owns the resource.
    falcor_device.end_frame()
    provider.close()
