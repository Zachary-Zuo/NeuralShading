add_falcor_executable(NclsViewer)

set(NCLS_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

target_sources(NclsViewer PRIVATE
    Hash.cpp
    Hash.h
    MaterialProgram.cpp
    MaterialProgram.h
    MethodBundle.cpp
    MethodBundle.h
    NclsViewer.cpp
    NclsViewer.h
    shaders/Approximation.cs.slang
    shaders/Composite.cs.slang
    shaders/Parity.cs.slang
    shaders/Prepare.cs.slang
    shaders/Reference.cs.slang
    shaders/ViewerCommon.slang
    shaders/Visibility.cs.slang
    "${NCLS_PROJECT_ROOT}/shaders/ncls/backends/legacy_ltc_k2/legacy_ltc_k2.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/backends/legacy_ltc_k2/p1_compiler.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/contracts/layer_stack_ir.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/contracts/response_measure.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/contracts/scattering_backend.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/contracts/scattering_contract.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/reference/interfaces.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/reference/random_walk_reference.slang"
    "${NCLS_PROJECT_ROOT}/shaders/ncls/reference/sampling.slang"
)

target_link_libraries(NclsViewer PRIVATE bcrypt)
target_compile_definitions(NclsViewer PRIVATE NCLS_WINDOWS_VIEWER=1)
target_compile_options(NclsViewer PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
target_copy_shaders(NclsViewer NclsViewer)
