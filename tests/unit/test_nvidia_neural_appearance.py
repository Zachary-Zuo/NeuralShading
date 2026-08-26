"""锁定 NVIDIA 方法形态的原规模结构、私有 ABI、方向顺序与 response 适配。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from ncls.core.scattering import BackendCapability, BackendDescriptor
from ncls.learning.models import NvidiaNeuralAppearanceModel
from ncls.learning.nvidia_neural_artifacts import (
    NVIDIA_PARAMETER_FIELDS,
    _runtime_adapter,
    pack_nvidia_neural_record,
    pack_nvidia_neural_shared_parameters,
)
from ncls.learning.pipelines import create_pipeline
from ncls.learning.slang import (
    NVIDIA_NEURAL_APPEARANCE_LAYOUT,
    nvidia_neural_appearance_layout_sha256,
    render_nvidia_neural_appearance_layout_slang,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _learned_frame(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = np.asarray([raw[0], raw[1], raw[2] + 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    tangent = np.asarray([raw[3] + 1.0, raw[4], raw[5]], dtype=np.float64)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    return tangent, bitangent, normal


def _paper_input(
    latent: np.ndarray,
    frame_raw: np.ndarray,
    project_wo: np.ndarray,
    project_wi: np.ndarray,
) -> np.ndarray:
    values: list[float] = []
    for offset in (0, 6):
        tangent, bitangent, normal = _learned_frame(frame_raw[offset : offset + 6])
        for direction in (project_wo, project_wi):
            values.extend(
                [
                    float(np.dot(direction, tangent)),
                    float(np.dot(direction, bitangent)),
                    float(np.dot(direction, normal)),
                ]
            )
    values.extend(latent.tolist())
    return np.asarray(values)


def test_nvidia_layout_is_private_z8_original_scale() -> None:
    layout = NVIDIA_NEURAL_APPEARANCE_LAYOUT
    assert layout["compiled_material"]["latent_count"] == 8
    assert layout["compiled_material"]["total_bytes"] == 32
    assert layout["state"]["stride_bytes"] == 96
    assert layout["evaluator"] == {
        "frame_input": 8,
        "frame_output": 12,
        "input": 20,
        "width": 64,
        "hidden_layers": 3,
        "output": 3,
        "frame_macs": 96,
        "evaluate_macs": 9664,
    }
    assert layout["sampler"]["input"] == 11
    assert layout["sampler"]["width"] == 32
    assert layout["sampler"]["hidden_layers"] == 3
    assert len(nvidia_neural_appearance_layout_sha256()) == 64
    generated = PROJECT_ROOT / (
        "shaders/ncls/backends/nvidia_neural_appearance/"
        "nvidia_neural_appearance_layout.slang"
    )
    assert generated.read_text(
        encoding="utf-8"
    ) == render_nvidia_neural_appearance_layout_slang()


def test_nvidia_model_contains_only_paper_scale_parameters() -> None:
    model = NvidiaNeuralAppearanceModel(state_count=2)
    shapes = {name: tuple(value.shape) for name, value in model.named_parameters()}
    assert shapes["latent"] == (2, 8)
    assert shapes["frame_w"] == (12, 8)
    assert shapes["evaluate_w0"] == (64, 20)
    assert shapes["evaluate_w1"] == (64, 64)
    assert shapes["evaluate_w2"] == (64, 64)
    assert shapes["evaluate_out_w"] == (3, 64)
    assert shapes["sampler_w0"] == (32, 11)
    assert shapes["sampler_w1"] == (32, 32)
    assert shapes["sampler_w2"] == (32, 32)
    assert shapes["sampler_out_w"] == (9, 32)
    forbidden = {"response_scale", "top_kind", "prepare_w0", "ltc_sampler_w"}
    assert forbidden.isdisjoint(dict(model.named_parameters()))
    assert forbidden.isdisjoint(dict(model.named_buffers()))


def test_nvidia_pipeline_identity_and_real_costs_are_not_unified_padding() -> None:
    pipeline = create_pipeline("nvidia-frame-two-lobe-layer-stack-budget-adapted-v1")
    assert pipeline.descriptor.model == {
        "representation": "nvidia-learned-frame-two-lobe-v1",
        "architecture": "nvidia-evaluator-3x64-sampler-3x32-v1",
        "latent": "direct-fit-z8-v1",
    }
    assert not pipeline.descriptor.deployment_candidate
    costs = pipeline.parameter_costs(None)
    assert costs["B_asset"] == 32
    assert costs["B_shared"] == 2 * 12_748
    assert costs["C_prepare_macs"] == 96 + 2_688
    assert costs["C_eval_macs"] == 9_664
    assert costs["state_bytes_per_pixel"] == 96


def test_nvidia_independent_oracle_locks_nonorthogonal_frames_and_input_order() -> None:
    latent = np.arange(8, dtype=np.float64) / 10.0
    raw = np.asarray(
        [0.2, -0.1, 0.3, -0.4, 0.5, 0.1, -0.2, 0.3, -0.1, 0.2, -0.4, 0.6]
    )
    project_wo = np.asarray([0.0, 0.0, 1.0])
    project_wi = np.asarray([0.6, 0.0, 0.8])
    values = _paper_input(latent, raw, project_wo, project_wi)
    assert values.shape == (20,)
    np.testing.assert_array_equal(values[12:], latent)
    tangent, bitangent, normal = _learned_frame(raw[:6])
    assert not np.isclose(np.dot(tangent, normal), 0.0)
    np.testing.assert_allclose(
        values[:6],
        [
            np.dot(project_wo, tangent),
            np.dot(project_wo, bitangent),
            np.dot(project_wo, normal),
            np.dot(project_wi, tangent),
            np.dot(project_wi, bitangent),
            np.dot(project_wi, normal),
        ],
    )


def test_nvidia_response_activation_and_log1p_loss_have_no_quality_threshold() -> None:
    np.testing.assert_allclose(np.exp(np.zeros(3) - 3.0), math.exp(-3.0))
    pipeline = create_pipeline("nvidia-frame-two-lobe-layer-stack-budget-adapted-v1")
    prediction_f = torch.tensor([[[0.5, 1.0, 2.0]]])
    batch = {
        "wi": torch.tensor([[[0.0, 0.0, 0.8]]]),
        "mean": torch.tensor([[[0.2, 0.8, 4.0]]]),
    }
    loss = pipeline.training_loss(prediction_f, batch)
    expected = torch.mean(
        torch.abs(
            torch.log1p(torch.tensor([0.4, 0.8, 1.6]))
            - torch.log1p(torch.tensor([0.2, 0.8, 4.0]))
        )
    )
    torch.testing.assert_close(loss, expected)


def test_nvidia_private_record_and_shared_parameter_pack_match_static_cost() -> None:
    model = NvidiaNeuralAppearanceModel(state_count=2)
    record = pack_nvidia_neural_record(np.arange(8, dtype=np.float32) / 10.0)
    assert len(record) == 32
    np.testing.assert_allclose(
        np.frombuffer(record, dtype="<f2", count=8).astype(np.float32),
        np.arange(8, dtype=np.float32) / 10.0,
        rtol=5e-4,
        atol=5e-4,
    )
    assert int.from_bytes(record[16:20], "little") == 1
    assert record[20:] == bytes(12)
    shared, layout, offsets = pack_nvidia_neural_shared_parameters(model)
    assert len(shared) == 2 * 12_748
    assert tuple(offsets) == tuple(field for field, _ in NVIDIA_PARAMETER_FIELDS)
    assert set(layout) == {name for _, name in NVIDIA_PARAMETER_FIELDS}
    assert layout["frame_w"]["offset_elements"] == 0
    assert layout["sampler_out_b"]["element_count"] == 9
    adapter = _runtime_adapter(
        offsets,
        record_stride=32,
        state_stride=96,
        cost=create_pipeline("nvidia-frame-two-lobe-layer-stack-budget-adapted-v1").parameter_costs(model),
    )
    descriptor = BackendDescriptor.from_dict(adapter["backend_descriptor"])
    assert descriptor.capabilities & BackendCapability.SAMPLE
    assert descriptor.capabilities & BackendCapability.PDF
    assert adapter["shader_defines"]["NCLS_NVIDIA_FRAME_WEIGHT_OFFSET"] == "0"
    assert adapter["compiled_material_stride"] == 32
    assert adapter["packed_state_stride"] == 96
