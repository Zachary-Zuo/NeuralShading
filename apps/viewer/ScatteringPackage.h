#pragma once

#include <cstddef>
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace ncls
{
struct ParityProbe
{
    std::vector<std::array<float, 3>> lights;
    std::vector<std::array<float, 3>> expectedResponseCos;
    std::array<float, 3> view{0.f, 0.f, 1.f};
    float relativeTolerance = 4e-5f;
    float absoluteTolerance = 4e-6f;
};

struct ViewerTypedResource
{
    std::filesystem::path path;
    std::string dtype;
    std::vector<uint32_t> shape;
    uint32_t stride = 1;
    uint32_t alignment = 1;
    std::string usage;
};

struct ViewerProgram
{
    std::filesystem::path root;
    std::string packageId;
    std::string programRuntimeId;
    std::string materialAssetId;
    std::string sourceSnapshotId;
    std::string sourceFamilyId;
    std::string sourceAssetSha256;
    std::string displayName;
    std::string sourceGitCommit;
    std::string backendId;
    uint32_t backendVersion = 1;
    std::string runtimeClass;
    std::string architectureId;
    std::string compiledStateId;
    std::string compiledMaterialIrSha256;
    std::filesystem::path previewMaterial;
    std::vector<std::string> supportedIrIds;
    std::string shaderModule;
    std::map<std::string, std::string> shaderDefines;
    uint32_t parameterCount = 0;
    uint32_t capabilities = 0;
    uint32_t stateBytesPerPixel = 16;
    uint32_t compiledMaterialBytes = 16;
    uint32_t compiledMaterialCount = 1;
    uint32_t compiledMaterialIndex = 0;
    uint32_t environmentQueryBudget = 1;
    uint32_t rectangleQueryBudget = 1;
    std::vector<uint32_t> sharedWeightWords;
    std::vector<std::byte> compiledMaterials;
    std::vector<ViewerTypedResource> resources;
    ParityProbe parity;
};

struct PackageFailure { std::filesystem::path path; std::string reason; };
struct PackageScanResult { std::vector<ViewerProgram> programs; std::vector<PackageFailure> failures; };

PackageScanResult scanScatteringPackages(const std::filesystem::path& root, const std::filesystem::path& runtimeShaderDirectory);
} // namespace ncls
