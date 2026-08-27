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


def test_path_tracers_share_unbiased_environment_mis_and_directional_origin() -> None:
    common = Path("apps/viewer/shaders/ViewerCommon.slang").read_text(
        encoding="utf-8"
    )
    assert "NCLS_VIEWER_ENVIRONMENT_NEE_SAMPLE_COUNT = 4u" in common

    for shader, environment_pdf, direct_origin in (
        (
            "ReferencePathTracer.cs.slang",
            "nclsEnvironmentPdfPath(directionWorld)",
            "nclsDirectRayOrigin(surface, directionWorld)",
        ),
        (
            "PackagePathTracer.cs.slang",
            "nclsPackageEnvironmentPdf(directionWorld)",
            "nclsPackageDirectOrigin(surface, directionWorld)",
        ),
    ):
        source = Path("apps/viewer/shaders", shader).read_text(encoding="utf-8")
        assert "environmentSample < NCLS_VIEWER_ENVIRONMENT_NEE_SAMPLE_COUNT" in source
        assert "const float scaledLightPdf = lightPdf" in source
        assert "* float(NCLS_VIEWER_ENVIRONMENT_NEE_SAMPLE_COUNT);" in source
        assert (
            "previousEnvironmentPdf = "
            "float(NCLS_VIEWER_ENVIRONMENT_NEE_SAMPLE_COUNT)" in source
        )
        assert environment_pdf in source
        assert direct_origin in source
        assert "scatter.eventFlags & (uint)NclsScatteringEvent::Transmission" not in source


def test_reference_path_uses_canonical_scene_backend_contract_only() -> None:
    source = Path("apps/viewer/shaders/ReferencePathTracer.cs.slang").read_text(
        encoding="utf-8"
    )
    scene = Path("apps/viewer/shaders/SceneReferenceProgram.slang").read_text(
        encoding="utf-8"
    )
    mdl = Path("shaders/ncls/reference_backends/mdl.slang").read_text(encoding="utf-8")
    assert '#include "SceneReferenceProgram.slang"' in source
    assert "backend.prepare(context, material)" in source
    assert "state.evaluate(wiWorld, sampleGenerator)" in source
    assert "state.pdf(lightWorld)" in source
    assert "state.sample(scatter, sampleGenerator)" in source
    assert "surface.family" not in source
    for legacy in (
        "nclsEvalReferencePath",
        "nclsSampleReferencePath",
        "nclsReferencePdfPath",
        "nclsReflectionProposalPdf",
        "nclsMdlEvaluateSurface",
        "nclsMdlSampleSurface",
        "nclsMdlPdfSurface",
    ):
        assert legacy not in source and legacy not in scene
    assert "import NclsMdlGenerated;" in mdl
    assert "struct NclsMdlReferenceState : INclsScatteringState" in mdl
    assert "surface_scattering_evaluate(data, targetState);" in mdl
    assert "surface_scattering_sample(data, targetState);" in mdl
    assert "surface_scattering_pdf(data, state);" in mdl
    assert not Path("apps/viewer/shaders/MdlViewerAdapter.slang").exists()
    assert "SampleLevel(gMdlTextureSampler, coordinate, 0.0)" in Path(
        "shaders/ncls/reference_backends/mdl_runtime.slangh"
    ).read_text(encoding="utf-8")


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
    assert "kLinearExrExportFlags = Bitmap::ExportFlags::Uncompressed" in viewer
    assert viewer.count("Bitmap::FileFormat::ExrFile, kLinearExrExportFlags") == 3
    assert "Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None" not in viewer
    assert '{"difference_resolution", {mViewWidth, mOutputHeight}}' in viewer
    assert "gDifferenceLinear[pixel]" in composite
    assert "gDifferenceDisplay[pixel]" in composite
    assert "gSlot0.SampleLevel(gLinearSampler, panelUv" in composite
    assert "gSlot1.SampleLevel(gLinearSampler, panelUv" in composite
    assert ": rightPanel ? pixel.x - panelWidth - dividerWidth : 0u;" in composite
