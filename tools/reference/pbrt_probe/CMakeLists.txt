cmake_minimum_required(VERSION 3.20)
project(ncls_pbrt_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)
set(PBRT_SOURCE_CODE "${CMAKE_CURRENT_LIST_DIR}/../../../external/pbrt-v4")
add_subdirectory(
    "${PBRT_SOURCE_CODE}"
    "${CMAKE_CURRENT_BINARY_DIR}/pbrt"
    EXCLUDE_FROM_ALL
)

add_executable(ncls_pbrt_probe main.cpp)
target_link_libraries(
    ncls_pbrt_probe
    PRIVATE
        pbrt::pbrt_lib
        OpenEXR::OpenEXR
        Ptex_static
        zlibstatic
        deflate::deflate
        double-conversion
        utf8proc
        glfw
        glad
        OpenGL::GL
        dbghelp
        wsock32
        ws2_32
)
target_compile_definitions(
    ncls_pbrt_probe
    PRIVATE
        PBRT_IS_WINDOWS
        PBRT_IS_MSVC
        PBRT_HAS_INTRIN_H
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
        _ENABLE_EXTENDED_ALIGNED_STORAGE
)
