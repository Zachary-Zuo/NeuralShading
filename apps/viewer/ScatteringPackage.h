#pragma once

#include <cstddef>
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ncls
{
struct ParityProbe
{
    std::vector<std::array<float, 3>> lights;
    std::vector<std::array<float, 3>> expectedF;
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

struct ViewerTypedBlob
{
    std::vector<std::byte> data;
    std::string dtype;
    std::vector<uint32_t> shape;
    uint32_t stride = 1;
    uint32_t alignment = 1;
    std::string usage;
};

struct ViewerSamplerDescriptor
{
    std::string usage;
    std::string filter;
    std::string addressMode;
};

struct ProgramRuntimeCache
{
    std::string programId;
    std::string backendId;
    uint32_t backendVersion = 1;
    std::string runtimeClass;
    std::string architectureId;
    std::vector<std::string> supportedIrIds;
    std::string shaderModule;
    std::map<std::string, std::string> shaderDefines;
    uint32_t capabilities = 0;
    uint32_t stateBytesPerPixel = 16;
    uint32_t environmentQueryBudget = 1;
    uint32_t rectangleQueryBudget = 1;
    std::vector<ViewerTypedBlob> blobs;
    std::vector<ViewerSamplerDescriptor> samplers;
};

struct AssetBinding
{
    std::string assetId;
    std::string sourceSnapshotId;
    std::string sourceFamilyId;
    std::string sourceAssetSha256;
    std::string compiledStateId;
    std::string compiledMaterialIrSha256;
    std::filesystem::path previewMaterial;
    std::vector<ViewerTypedBlob> blobs;
    std::vector<ViewerTypedResource> resources;
    std::vector<ViewerSamplerDescriptor> samplers;
};

struct InstanceBinding
{
    std::string instanceId;
    std::string programId;
    std::string assetId;
    uint32_t compiledMaterialIndex = 0;
};

struct ViewerProgram
{
    std::filesystem::path root;
    std::string packageId;
    std::string displayName;
    std::shared_ptr<const ProgramRuntimeCache> program;
    AssetBinding asset;
    InstanceBinding instance;
    ParityProbe parity;
};

struct PackageFailure { std::filesystem::path path; std::string reason; };
struct PackageScanResult { std::vector<ViewerProgram> programs; std::vector<PackageFailure> failures; };

PackageScanResult scanScatteringPackages(const std::filesystem::path& root, const std::filesystem::path& runtimeShaderDirectory);
} // namespace ncls
