#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ncls
{
struct ParityProbe
{
    std::array<std::byte, 752> material{};
    std::array<float, 3> view{};
    std::vector<std::array<float, 3>> lights;
    std::vector<std::array<float, 3>> expectedResponseCos;
    float relativeTolerance = 4e-5f;
    float absoluteTolerance = 4e-6f;
};

struct ViewerMethod
{
    std::filesystem::path root;
    std::string methodId;
    std::string displayName;
    std::string sourceGitCommit;
    std::string backendId;
    uint32_t backendVersion = 0;
    std::string architectureId;
    uint32_t width = 0;
    uint32_t parameterCount = 0;
    uint32_t stateBytesPerPixel = 0;
    uint32_t compiledMaterialBytes = 0;
    std::vector<float> weights;
    ParityProbe parity;
};

struct BundleFailure
{
    std::filesystem::path path;
    std::string reason;
};

struct BundleScanResult
{
    std::vector<ViewerMethod> methods;
    std::vector<BundleFailure> failures;
};

BundleScanResult scanMethodBundles(
    const std::filesystem::path& root,
    const std::filesystem::path& runtimeShaderDirectory
);
} // namespace ncls
