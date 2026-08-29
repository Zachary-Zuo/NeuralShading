cmake_minimum_required(VERSION 3.20)
project(ncls_openpbr_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPENPBR_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../../../external/openpbr-bsdf")
set(GLM_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../../../external/glm")
foreach(REQUIRED_PATH IN ITEMS "${OPENPBR_SOURCE}/openpbr.h" "${GLM_SOURCE}/glm/glm.hpp")
    if(NOT EXISTS "${REQUIRED_PATH}")
        message(FATAL_ERROR "Missing pinned OpenPBR dependency: ${REQUIRED_PATH}")
    endif()
endforeach()

add_executable(ncls_openpbr_probe main.cpp)
target_include_directories(ncls_openpbr_probe PRIVATE "${OPENPBR_SOURCE}" "${GLM_SOURCE}")
set_target_properties(ncls_openpbr_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
)
if(NOT WIN32)
    # Keep the validation-only probe path identical across Windows and Linux.
    set_target_properties(ncls_openpbr_probe PROPERTIES SUFFIX ".exe")
endif()
if(MSVC)
    target_compile_options(ncls_openpbr_probe PRIVATE /utf-8)
endif()
