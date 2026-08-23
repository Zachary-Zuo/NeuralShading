from __future__ import annotations

from pathlib import Path

import pytest

from ncls.core.scattering import (
    REQUIRED_REALTIME_CAPABILITIES,
    BackendCapability,
    BackendCostModel,
    BackendDescriptor,
    ScatteringContext,
    ScatteringEvent,
    ShadingFrame,
    StateStorage,
    SurfaceInteraction,
    ScatteringEval,
    ScatteringPdf,
    response_cosine,
)
from ncls.core.scattering.abi_layout import render_slang_contract


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_scattering_event_values_are_stable() -> None:
    assert int(ScatteringEvent.REFLECTION) == 1
    assert int(ScatteringEvent.TRANSMISSION) == 2
    assert int(ScatteringEvent.VOLUME_BOUNDARY) == 128


def test_anisotropic_shading_frame_and_context() -> None:
    frame = ShadingFrame((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
    surface = SurfaceInteraction(
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 1.0),
        frame,
        material_instance_id=2,
        primitive_id=9,
    )
    context = ScatteringContext(surface, (0.0, 0.6, 0.8))
    assert context.surface.shading_frame.tangent == (0.0, 1.0, 0.0)
    assert context.component_mask & ScatteringEvent.REFLECTION


def test_shading_frame_rejects_nonorthogonal_axes() -> None:
    with pytest.raises(ValueError, match="orthogonal"):
        ShadingFrame((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))


def test_response_measure_multiplies_the_light_cosine_exactly_once() -> None:
    frame = ShadingFrame((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    evaluation = ScatteringEval(
        (0.2, 0.4, 0.8),
        ScatteringPdf(0.25),
        ScatteringEvent.REFLECTION | ScatteringEvent.DIFFUSE,
    )
    assert response_cosine(evaluation, frame, (0.6, 0.0, 0.8)) == pytest.approx((0.16, 0.32, 0.64))


def make_descriptor() -> BackendDescriptor:
    return BackendDescriptor(
        backend_id="test-analytic",
        backend_version=1,
        supported_ir_ids=("ncls.layer-stack-ir@1",),
        capabilities=REQUIRED_REALTIME_CAPABILITIES,
        state_storage=StateStorage.STRUCTURED,
        state_stride=32,
        state_alignment=16,
        deterministic_eval=True,
        bounded_execution=True,
        shader_entry_points={"prepare": "prepareTest", "lighting": "shadeTest"},
        cost_model=BackendCostModel(state_bytes_per_pixel=32),
    )


def test_backend_descriptor_round_trip_and_realtime_gate() -> None:
    descriptor = make_descriptor()
    restored = BackendDescriptor.from_dict(descriptor.to_dict())
    assert restored == descriptor
    assert restored.is_complete_realtime_backend


def test_backend_descriptor_does_not_confuse_optional_capability_with_required_interface() -> None:
    descriptor = make_descriptor()
    reduced = BackendDescriptor.from_dict(
        {**descriptor.to_dict(), "capabilities": int(descriptor.capabilities & ~BackendCapability.SAMPLE)}
    )
    assert not reduced.is_complete_realtime_backend


def test_backend_descriptor_rejects_layout_cost_disagreement() -> None:
    with pytest.raises(ValueError, match="state_bytes_per_pixel"):
        BackendDescriptor(
            backend_id="invalid",
            backend_version=1,
            supported_ir_ids=("ncls.layer-stack-ir@1",),
            capabilities=REQUIRED_REALTIME_CAPABILITIES,
            state_storage=StateStorage.STRUCTURED,
            state_stride=32,
            state_alignment=16,
            deterministic_eval=True,
            bounded_execution=True,
            shader_entry_points={},
            cost_model=BackendCostModel(state_bytes_per_pixel=16),
        )


def test_generated_scattering_contract_is_current() -> None:
    shader = PROJECT_ROOT / "shaders" / "ncls" / "contracts" / "scattering_contract.slang"
    assert shader.read_text(encoding="utf-8") == render_slang_contract()
