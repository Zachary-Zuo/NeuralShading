from __future__ import annotations

from pathlib import Path

from ncls.viewer import ComparisonSlot, SlotMode, SlotStatus, panel_extents


def test_fixed_panels_preserve_equal_extent_for_even_and_odd_widths():
    for width, divider in ((1280, 0), (1281, 1)):
        left, right, actual = panel_extents(width, 720)
        assert left[2:] == right[2:] == (width // 2, 720)
        assert actual == divider and right[0] - (left[0] + left[2]) == divider


def test_slot_failure_never_changes_peer_or_extent():
    peer = ComparisonSlot(mode=SlotMode.PATH_TRACING)
    failed = ComparisonSlot(mode=SlotMode.DEFERRED).activate(
        package_id="a" * 64, program_runtime_id="b" * 64, material_asset_id="c" * 64,
        source_snapshot_id="d" * 64, capabilities=1,
    )
    assert failed.status == SlotStatus.UNSUPPORTED and peer.status == SlotStatus.EMPTY
    assert panel_extents(801, 600)[0][2:] == panel_extents(801, 600)[1][2:]


def test_package_path_tracer_uses_package_prepare_evaluate_pdf_and_sample() -> None:
    source = Path("apps/viewer/shaders/PackagePathTracer.cs.slang").read_text(
        encoding="utf-8"
    )
    for call in (
        "backend.prepare(context, material)",
        "state.evaluate(wiWorld, sampleGenerator)",
        "state.pdf(lightWorld)",
        "state.sample(scatter, sampleGenerator)",
    ):
        assert call in source
    assert "ReferencePathTracer.cs.slang" not in source


def test_reference_and_package_pt_share_the_path_surface_contract() -> None:
    for shader in ("ReferencePathTracer.cs.slang", "PackagePathTracer.cs.slang"):
        source = Path("apps/viewer/shaders", shader).read_text(encoding="utf-8")
        assert '#include "PathSurface.slang"' in source
        assert "nclsViewerPrimaryRayConeSpreadAngle(" in source
        assert "nclsViewerPreparePathSurface(" in source
        assert "nclsViewerLoadPathVertexData(" in source
        assert "getVertexDataRayCones(" not in source
        assert "NCLS_PT_SURFACE_PROBE" not in source


def test_capture_uses_single_panel_difference_extent_and_fixed_spp() -> None:
    viewer = Path("apps/viewer/NclsViewer.cpp").read_text(encoding="utf-8")
    header = Path("apps/viewer/NclsViewer.h").read_text(encoding="utf-8")
    composite = Path("apps/viewer/shaders/Composite.cs.slang").read_text(
        encoding="utf-8"
    )

    assert "constexpr uint32_t kCapturePathTracingSpp = 1024;" in viewer
    assert "options.frameCount = (kCapturePathTracingSpp + samples - 1u) / samples;" in viewer
    assert '{"reference_spp", kCapturePathTracingSpp}' in viewer
    assert "uint32_t frameCount = 1024;" in header
    assert "slot.spp += samplesThisFrame;" in viewer
    assert "slot.spp != kCapturePathTracingSpp" in viewer
    assert "must reach exactly 1024 spp" in viewer
    assert "mpDifferenceLinear = viewTexture();" in viewer
    assert "mpDifferenceDisplay = viewTexture();" in viewer
    assert "mpDifferenceLinear->captureToFile(" in viewer
    assert "mpComparisonLinear->captureToFile(0, 0, differencePath" not in viewer
    assert '{"difference_resolution", {mViewWidth, mOutputHeight}}' in viewer
    assert "gDifferenceLinear[pixel]" in composite
    assert "gDifferenceDisplay[pixel]" in composite
    assert "gSlot0.SampleLevel(gLinearSampler, panelUv" in composite
    assert "gSlot1.SampleLevel(gLinearSampler, panelUv" in composite
    assert ": rightPanel ? pixel.x - panelWidth - dividerWidth : 0u;" in composite
