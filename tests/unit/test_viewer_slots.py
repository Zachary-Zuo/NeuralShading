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
        package_id="a" * 64, program_id="b" * 64, asset_id="c" * 64,
        instance_id="d" * 64, source_snapshot_id="e" * 64, capabilities=1,
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
    environment = Path("apps/viewer/shaders/PathEnvironmentMath.slang").read_text(
        encoding="utf-8"
    )
    assert "NCLS_VIEWER_ENVIRONMENT_NEE_SAMPLE_COUNT = 4u" in common
    assert "NCLS_VIEWER_ENVIRONMENT_BSDF_SAMPLE_COUNT = 4u" in common
    assert (
        "NCLS_VIEWER_PRIMARY_PATH_SAMPLE_COUNT =\n"
        "    NCLS_VIEWER_ENVIRONMENT_BSDF_SAMPLE_COUNT"
    ) in common
    assert "float(NCLS_VIEWER_ENVIRONMENT_NEE_SAMPLE_COUNT) * lightPdf" in environment
    assert "float(NCLS_VIEWER_ENVIRONMENT_BSDF_SAMPLE_COUNT) * bsdfPdf" in environment
    assert "nclsViewerEnvironmentPowerHeuristic" in environment

    for shader, direct_origin in (
        (
            "ReferencePathTracer.cs.slang",
            "nclsDirectRayOrigin(surface, directionWorld)",
        ),
        (
            "PackagePathTracer.cs.slang",
            "nclsPackageDirectOrigin(surface, directionWorld)",
        ),
    ):
        source = Path("apps/viewer/shaders", shader).read_text(encoding="utf-8")
        compact = "".join(source.split())
        assert '#include "PathEnvironment.slang"' in source
        assert "environmentSample < NCLS_VIEWER_ENVIRONMENT_NEE_SAMPLE_COUNT" in source
        assert "bsdfSample < NCLS_VIEWER_ENVIRONMENT_BSDF_SAMPLE_COUNT" in source
        assert "bsdfSample < NCLS_VIEWER_PRIMARY_PATH_SAMPLE_COUNT" in source
        assert "scatter.weight/float(NCLS_VIEWER_PRIMARY_PATH_SAMPLE_COUNT)" in compact
        assert "depth==1u&&gUseEnvironment!=0u" in compact
        assert "initialBsdfPdf" in source
        assert "0u,false,rng" in compact
        assert "depth,true,rng" in compact
        assert "nclsViewerEnvironmentLightTechniquePdf(lightPdf)" in compact
        assert "nclsViewerEnvironmentLightMisWeight(" in source
        assert "nclsViewerEnvironmentBsdfMisWeight(" in source
        assert "scatter.weight * nclsViewerEnvironmentRadiance(directionWorld)" in source
        assert "nclsViewerEnvironmentBsdfMisWeight(" in source
        assert "previousEnvironmentPdf" not in source
        assert "previousBsdfPdf" not in source
        assert direct_origin in source
        assert "scatter.eventFlags & (uint)NclsScatteringEvent::Transmission" not in source


def test_path_sample_generator_is_the_pinned_falcor_uniform_generator() -> None:
    source = Path("apps/viewer/shaders/PathSampleGenerator.slang").read_text(
        encoding="utf-8"
    )
    assert "import Utils.Sampling.UniformSampleGenerator;" in source
    assert "UniformSampleGenerator generator;" in source
    assert "UniformSampleGenerator(pixel, sequence)" in source
    assert "struct NclsViewerPathSampleGenerator : ISampleGenerator" in source
    assert "Cranley" not in source
    assert "lattice" not in source.lower()


def test_environment_cdf_matches_the_bilinear_radiance_reconstruction() -> None:
    viewer = Path("apps/viewer/NclsViewer.cpp").read_text(encoding="utf-8")
    assert "kBilinearCellKernel{0.125, 0.75, 0.125}" in viewer
    assert "filteredLuminance(x, y) * solidAngleFactor" in viewer
    assert "sourceX = (int64_t(x) + dx) % int64_t(width)" in viewer
    assert "std::clamp<int64_t>(" in viewer


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


def test_capture_uses_single_panel_difference_extent_and_headless_target_spp() -> None:
    viewer = Path("apps/viewer/NclsViewer.cpp").read_text(encoding="utf-8")
    header = Path("apps/viewer/NclsViewer.h").read_text(encoding="utf-8")
    composite = Path("apps/viewer/shaders/Composite.cs.slang").read_text(
        encoding="utf-8"
    )

    assert "constexpr uint32_t kDefaultCapturePathTracingSpp = 1024;" in viewer
    assert '"reference_spp", kDefaultCapturePathTracingSpp' in viewer
    assert "1u + (maximumTargetSpp - 1u) / options.captureSamplesPerDispatch" in viewer
    assert '{"reference_spp", capturedReferenceSpp}' in viewer
    assert "uint32_t frameCount = 1024;" in header
    assert "std::array<uint32_t, 2> captureTargetSpp{1024, 1024};" in header
    assert "uint32_t captureTargetSpp = 1024;" in header
    assert "uint32_t captureSamplesPerDispatch = 1;" in header
    assert "slot.spp += samplesThisFrame;" in viewer
    assert "slot.spp != slot.captureTargetSpp" in viewer
    assert '" must reach exactly " + std::to_string(slot.captureTargetSpp)' in viewer
    assert 'options.capturePurpose != "training-diagnostic"' in viewer
    assert '"comparison_purpose", mOptions.capturePurpose' in viewer
    assert '"target_spp", slot.contract.mode == ncls::SlotMode::PathTracing' in viewer
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


def test_interactive_path_tracing_is_one_unbounded_sample_sequence() -> None:
    viewer = Path("apps/viewer/NclsViewer.cpp").read_text(encoding="utf-8")
    header = Path("apps/viewer/NclsViewer.h").read_text(encoding="utf-8")
    source = Path("apps/viewer/shaders/ReferencePathTracer.cs.slang").read_text(
        encoding="utf-8"
    )
    package = Path("apps/viewer/shaders/PackagePathTracer.cs.slang").read_text(
        encoding="utf-8"
    )

    assert "if (!mOptions.headless) return 1u;" in viewer
    assert "std::min(mOptions.captureSamplesPerDispatch, remaining)" in viewer
    assert viewer.count(
        "const uint32_t samplesThisFrame = pathSamplesThisDispatch(slot);"
    ) == 2
    assert "mSamplesPerFrame" not in viewer
    assert "mSamplesPerFrame" not in header
    assert 'group.var("Samples per frame"' not in viewer
    assert '{"samples_per_frame"' not in viewer
    assert '{"format_name", "ncls.viewer-scene"}' in viewer
    assert '{"format_version", 2}' in viewer
    assert "sceneVersion != 1u && sceneVersion != 2u" in viewer
    assert "const bool accumulate = !mCameraDragging && !mPanDragging;" not in viewer
    assert "else slot.spp = 0" not in viewer

    for shader, spp_name in (
        (source, "gReferenceSpp"),
        (package, "gPackageSpp"),
    ):
        assert "gAccumulate" not in shader
        assert f"const uint globalSample = {spp_name} + sampleIndex;" in shader
        assert "gResetAccumulation != 0u" in shader


def test_release_viewer_uses_v2_studio_and_package_id_slot_cli() -> None:
    viewer = Path("apps/viewer/NclsViewer.cpp").read_text(encoding="utf-8")
    benchmark = Path("scripts/benchmark_viewer.ps1").read_text(encoding="utf-8")
    composite = Path("apps/viewer/shaders/Composite.cs.slang").read_text(
        encoding="utf-8"
    )

    assert 'data/ncls-viewer/studio-v2.json"' in viewer
    assert 'data/ncls-viewer/studio-v1.json"' not in viewer
    for option in (
        "--slot0-package", "--slot1-package", "--slot0-mode", "--slot1-mode"
    ):
        assert option in viewer
    assert "--bundle-root $packageRootPath" in benchmark
    assert "--slot0-package $Slot0PackageId" in benchmark
    assert "--slot1-package $Slot1PackageId" in benchmark
    assert "--slot0-package $packages[$Slot0PackageId]" not in benchmark
    assert 'constants["gDividerColor"] = mDividerColor;' in viewer
    assert "else { comparisonLinear = gDividerColor;" in composite


def test_package_source_identity_uses_canonical_mdl_snapshot() -> None:
    viewer = Path("apps/viewer/NclsViewer.cpp").read_text(encoding="utf-8")
    compatibility = viewer[
        viewer.index("bool NclsViewer::allMaterialsSupportedBy") :
        viewer.index("bool NclsViewer::hasActiveProgram")
    ]

    assert "source.familyId() != method.asset.sourceFamilyId" in compatibility
    assert "source.family == ncls::ReferenceFamily::Mdl" in compatibility
    assert "? method.asset.sourceSnapshotId" in compatibility
    assert ": method.asset.sourceAssetSha256" in compatibility
    assert "source.sourceSha256 == expectedIdentity" in compatibility


def test_layer_stack_upload_preserves_native_asset_identity() -> None:
    viewer = Path("apps/viewer/NclsViewer.cpp").read_text(encoding="utf-8")
    update = viewer[
        viewer.index("void NclsViewer::updateMaterialBuffer") :
        viewer.index("void NclsViewer::updateReferenceSourceBuffer")
    ]
    install = viewer[
        viewer.index("void NclsViewer::installReferenceSource") :
        viewer.index("void NclsViewer::saveMaterial")
    ]

    assert "void NclsViewer::updateMaterialBuffer(bool sourceStateChanged)" in update
    assert "if (sourceStateChanged && mReferenceSource.sourcePath.empty())" in update
    assert "updateMaterialBuffer(false);" in install
    assert "updateMaterialBuffer();" not in viewer


def test_package_program_runtime_is_shared_by_program_identity() -> None:
    viewer = Path("apps/viewer/NclsViewer.cpp").read_text(encoding="utf-8")
    runtime = viewer[
        viewer.index("std::shared_ptr<NclsViewer::ProgramGpuRuntime>") :
        viewer.index("void NclsViewer::compileMaterialInstance")
    ]

    assert "mProgramGpuRuntimes.find(method.program->programId)" in runtime
    assert "runtime->programId = method.program->programId" in runtime
    assert "mProgramGpuRuntimes.emplace(runtime->programId, runtime)" in runtime
    assert "mode == ncls::SlotMode::Deferred && !runtime->pDeferredPass" in runtime
    assert "mode == ncls::SlotMode::PathTracing && !runtime->pPathPass" in runtime
    assert "programGpuRuntime(method, candidate.contract.mode)" in viewer
    assert "method.asset.assetId" not in runtime
    assert "method.instance.instanceId" not in runtime


def test_typed_material_edit_compiles_candidate_before_atomic_replacement() -> None:
    viewer = Path("apps/viewer/NclsViewer.cpp").read_text(encoding="utf-8")
    editor = viewer[
        viewer.index("void NclsViewer::applyMaterialEditor") :
        viewer.index("bool NclsViewer::runParityProbe")
    ]
    activation = viewer[
        viewer.index("void NclsViewer::activateComparisonSlot") :
        viewer.index("ref<Texture> NclsViewer::slotOutput")
    ]

    assert "auto candidateBuffers = slot.buffers;" in editor
    assert "uploadMaterialEditorValues(method, editorView, candidateBuffers);" in editor
    assert "compileMaterialInstance(*slot.programRuntime, method, candidateBuffers);" in editor
    assert editor.index("compileMaterialInstance") < editor.index(
        "slot.buffers = std::move(candidateBuffers);"
    )
    assert "slot.editorView = std::move(editorView);" in editor
    assert "ComparisonSlotRuntime candidate;" in activation
    assert "mComparisonSlots[slotIndex] = std::move(candidate);" in activation
    assert "previous slot binding preserved" in activation


def test_linked_mdl_catalog_switches_reference_and_neural_from_one_typed_state() -> None:
    viewer = Path("apps/viewer/NclsViewer.cpp").read_text(encoding="utf-8")
    reference = Path("apps/viewer/ReferenceSource.cpp").read_text(encoding="utf-8")
    reference_header = Path("apps/viewer/ReferenceSource.h").read_text(
        encoding="utf-8"
    )
    catalog = Path("apps/viewer/MdlReference.cpp").read_text(encoding="utf-8")
    launcher = Path("scripts/launch_metal_viewer.ps1").read_text(encoding="utf-8")

    linked = viewer[
        viewer.index("void NclsViewer::applyLinkedMdlSource") :
        viewer.index("bool NclsViewer::runParityProbe")
    ]
    on_load = viewer[
        viewer.index("void NclsViewer::onLoad") : viewer.index(
            "void NclsViewer::createPasses"
        )
    ]
    assert 'schemaName == "ncls.viewer-material-catalog"' in reference
    assert "std::shared_ptr<const MdlViewerCatalog> mdlCatalog;" in reference_header
    assert "std::make_shared<const MdlViewerCatalog>" in reference
    assert '"ViewerMaterialCatalog entry"' in catalog
    assert "ensureLinkedMdlProgram(entry)" in linked
    assert "installReferenceSource(std::move(source), catalogPath);" in linked
    assert "mComparisonSlots[0].contract.mode = ncls::SlotMode::PathTracing;" in linked
    assert "mComparisonSlots[1].contract.mode = ncls::SlotMode::Deferred;" in linked
    assert "applyMaterialEditor(mComparisonSlots[1]" in linked
    assert linked.index("applyMaterialEditor(mComparisonSlots[1]") < linked.index(
        "mLinkedMdlMode = true;"
    )
    for restored in (
        "mReferenceSource = previousSource;",
        "mComparisonSlots = previousSlots;",
        "mLinkedMdlMode = previousLinkedMdlMode;",
        "mAccumulationSeconds = previousAccumulationSeconds;",
    ):
        assert restored in linked
    assert '"viewer_material_binding"' in viewer
    assert '"viewer_material_state"' in viewer
    assert "selected->packageId == entry.packageId" in on_load
    assert (
        "applyMaterialEditor(slot, *selected, mReferenceSource.mdlParameterView);"
        in on_load
    )
    assert '"--evaluator-preview-lighting"' in launcher
    assert '"--slot0-package"' in launcher
    assert '"--slot1-package"' in launcher
    assert '"HybridVsDirect"' in launcher
    assert '"exact-diagnostic-evaluator-preview"' in launcher
    assert "manual-packages" not in launcher
    assert "learn train" not in launcher.lower()


def test_powershell_viewer_entrypoints_do_not_reuse_stale_native_exit_codes() -> None:
    build = Path("scripts/build_viewer.ps1").read_text(encoding="utf-8")
    launch = Path("scripts/launch_metal_viewer.ps1").read_text(encoding="utf-8")

    assert '& (Join-Path $PSScriptRoot "fetch_viewer_assets.ps1")' in build
    assert 'if (-not $?) { throw "Failed to provision the fixed viewer scene" }' in build
    assert 'if (-not $?) { throw "Failed to build NclsViewer" }' in launch
    assert 'if ($LASTEXITCODE -ne 0) { throw "Failed to provision' not in build
    assert "$LASTEXITCODE" not in launch


def test_package_profile_and_diagnostic_identity_reach_ui_and_capture() -> None:
    package = Path("apps/viewer/ScatteringPackage.cpp").read_text(encoding="utf-8")
    viewer = Path("apps/viewer/NclsViewer.cpp").read_text(encoding="utf-8")
    exporter = Path("src/ncls/learning/evaluation_package.py").read_text(
        encoding="utf-8"
    )

    assert '"checkpoint_profile_id"' in exporter
    assert '"checkpoint_compatibility"' in exporter
    assert 'programProvenance.value("checkpoint_profile_id"' in package
    assert '"checkpoint_compatibility", std::string()' in package
    assert 'result.displayName += " / " + result.checkpointProfileId;' in package
    assert '{"checkpoint_profile_id", program ? program->checkpointProfileId' in viewer
    assert '{"checkpoint_compatibility", program ? program->checkpointCompatibility' in viewer


def test_package_rendering_time_slices_interactive_preview_without_model_shortcuts() -> None:
    viewer = Path("apps/viewer/NclsViewer.cpp").read_text(encoding="utf-8")
    header = Path("apps/viewer/NclsViewer.h").read_text(encoding="utf-8")
    deferred = Path("apps/viewer/shaders/DeferredRenderer.cs.slang").read_text(
        encoding="utf-8"
    )
    path = Path("apps/viewer/shaders/PackagePathTracer.cs.slang").read_text(
        encoding="utf-8"
    )
    full_dispatch = viewer[
        viewer.index("void NclsViewer::executePackageTiles") :
        viewer.index("void NclsViewer::executeInteractivePackageTile")
    ]
    interactive_dispatch = viewer[
        viewer.index("void NclsViewer::executeInteractivePackageTile") :
        viewer.index("void NclsViewer::renderPackagePath")
    ]
    deferred_render = viewer[
        viewer.index("void NclsViewer::renderApproximation") :
        viewer.index("void NclsViewer::executePackageTiles")
    ]

    assert "kPackageDispatchTileWidth = 8u" in viewer
    assert "kPackageDispatchTileRows = 8u" in viewer
    assert "kInteractiveDeferredInitialStride = 16u" in viewer
    assert "void executePackageTiles(" in header
    assert "void executeInteractivePackageTile(" in header
    assert 'root[constantBufferName]["gDispatchOffset"] = uint2(column, row);' in full_dispatch
    assert "pRenderContext->submit(true)" in full_dispatch
    assert "pRenderContext->submit(true)" not in interactive_dispatch
    assert "for (" not in interactive_dispatch
    assert "if (mOptions.headless)" in deferred_render
    assert "executeInteractivePackageTile(" in deferred_render
    assert "deferredPreviewStride /= 2u" in deferred_render
    assert "ReferenceFamily::Mdl" not in full_dispatch + interactive_dispatch
    assert "metal" not in (full_dispatch + interactive_dispatch).lower()
    assert viewer.count("executePackageTiles(") == 3
    for shader in (deferred, path):
        assert "uint2 gDispatchOffset;" in shader
        assert "[numthreads(8, 8, 1)]" in shader
    assert "dispatchThreadID.xy + gDispatchOffset" in path
    assert "previewPixel = dispatchThreadID.xy + gDispatchOffset" in deferred
    assert "uint gPreviewStride;" in deferred
    assert "blockOrigin + stride" in deferred
