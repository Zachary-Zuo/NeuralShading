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
if(MSVC)
    target_compile_options(ncls_openpbr_probe PRIVATE /utf-8)
endif()
