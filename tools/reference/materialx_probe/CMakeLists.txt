cmake_minimum_required(VERSION 3.25)

project(ncls_materialx_probe LANGUAGES CXX)

find_package(MaterialX CONFIG REQUIRED)

add_executable(ncls_materialx_probe main.cpp)
target_compile_features(ncls_materialx_probe PRIVATE cxx_std_17)
target_link_libraries(ncls_materialx_probe PRIVATE
    MaterialXFormat
    MaterialXGenGlsl
    MaterialXRenderGlsl
)

if(MSVC)
    target_compile_options(ncls_materialx_probe PRIVATE /W4 /permissive-)
endif()
