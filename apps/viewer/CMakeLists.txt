add_falcor_executable(NclsViewer)

set(NCLS_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

target_sources(NclsViewer PRIVATE
    Hash.cpp
    Hash.h
    MaterialProgram.cpp
    MaterialProgram.h
    MethodBundle.cpp
    MethodBundle.h
    OpenPbrLuts.cpp
    OpenPbrLuts.h
    ReferenceSource.cpp
    ReferenceSource.h
    NclsViewer.cpp
    NclsViewer.h
    shaders/Approximation.cs.slang
    shaders/ClearVisibility.cs.slang
    shaders/Composite.cs.slang
    shaders/Denoise.cs.slang
    shaders/MethodBackend.slang
    shaders/SceneVisibility.3d.slang
    shaders/Parity.cs.slang
    shaders/Prepare.cs.slang
    shaders/ReferencePathTracer.cs.slang
    shaders/ViewerCommon.slang
    "${NCLS_PROJECT_ROOT}/shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_core.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_layout.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_mlp.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_pack.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/backends/unified_neural/unified_neural.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/backends/unified_neural/unified_neural_core.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/backends/unified_neural/unified_neural_layout.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/backends/unified_neural/unified_neural_pack.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/backends/ltc_k2_analytic_control/ltc_k2_analytic_control_core.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/contracts/layer_stack_ir.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/contracts/response_measure.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/contracts/scattering_backend.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/contracts/scattering_contract.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/scattering/common.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/scattering/frame.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/scattering/cosine.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/scattering/ltc.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/scattering/ggx.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/scattering/fresnel.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/scattering/mixture.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/scattering/nvidia_proposal.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/reference/interfaces.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/reference/merl_reference.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/reference/materialx_standard_surface_reference.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/reference/openpbr_reference.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/reference/random_walk_reference.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/reference/sampling.slang"
)

target_link_libraries(NclsViewer PRIVATE bcrypt pugixml OpenEXR)
target_include_directories(NclsViewer PRIVATE
    "${NCLS_PROJECT_ROOT}/external/openpbr-bsdf"
    "${NCLS_PROJECT_ROOT}/external/glm"
)
target_compile_definitions(NclsViewer PRIVATE NCLS_WINDOWS_VIEWER=1)
target_compile_options(NclsViewer PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
target_copy_shaders(NclsViewer NclsViewer)

add_custom_command(TARGET NclsViewer POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${NCLS_PROJECT_ROOT}/external/openpbr-bsdf"
        "$<TARGET_FILE_DIR:NclsViewer>/external/openpbr-bsdf"
    COMMENT "NclsViewer: Copying pinned OpenPBR Slang reference headers"
)

add_custom_target(NclsViewerStudioAssets ALL
    COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${FALCOR_OUTPUT_DIRECTORY}/data/ncls-viewer"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${NCLS_PROJECT_ROOT}/configs/viewer-studio-v1.json"
        "${FALCOR_OUTPUT_DIRECTORY}/data/ncls-viewer/studio-v1.json"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${NCLS_PROJECT_ROOT}/configs/viewer-studio-material-v1.json"
        "${FALCOR_OUTPUT_DIRECTORY}/data/ncls-viewer/viewer-studio-material-v1.json"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${NCLS_PROJECT_ROOT}/assets/viewer/scenes/studio-v1/shaderball.glb"
        "${FALCOR_OUTPUT_DIRECTORY}/data/ncls-viewer/shaderball.glb"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${NCLS_PROJECT_ROOT}/assets/viewer/environments/polyhaven-1k/studio_small_03_1k.exr"
        "${FALCOR_OUTPUT_DIRECTORY}/data/ncls-viewer/studio_small_03_1k.exr"
    DEPENDS
        "${NCLS_PROJECT_ROOT}/configs/viewer-studio-v1.json"
        "${NCLS_PROJECT_ROOT}/configs/viewer-studio-material-v1.json"
        "${NCLS_PROJECT_ROOT}/assets/viewer/scenes/studio-v1/shaderball.glb"
        "${NCLS_PROJECT_ROOT}/assets/viewer/environments/polyhaven-1k/studio_small_03_1k.exr"
    COMMENT "NclsViewer: Copying the fixed studio-v1 scene"
)
add_dependencies(NclsViewer NclsViewerStudioAssets)
