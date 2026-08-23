#pragma once

#include "MaterialProgram.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ncls
{
enum class ReferenceFamily : uint32_t
{
    LayerStack = 0,
    Merl = 1,
    OpenPbr = 2,
    MaterialX = 3,
};

struct ReferenceSource
{
    ReferenceFamily family = ReferenceFamily::LayerStack;
    LayerStackIR layerStack = makeDefaultMaterial();
    std::vector<std::array<float, 3>> merlBrdf;
    std::array<float, 77> openPbrInputs{};
    uint32_t openPbrColorSpace = 0; // 0: linear-sRGB, 1: ACEScg
    // Family-specific MaterialX standard_surface inputs. The layout is private to
    // the MaterialX reference runtime and is not a public ScatteringState ABI.
    std::array<float, 24> materialXInputs{};
    std::filesystem::path materialXBaseColorTexture;
    std::filesystem::path materialXRoughnessTexture;
    std::filesystem::path materialXMetalnessTexture;
    std::filesystem::path materialXNormalTexture;
    std::filesystem::path materialXDisplacementTexture;
    std::filesystem::path sourcePath;
    std::string displayName = "默认多层材质";
    std::string sourceSha256;

    bool supportsCurrentCompiler() const { return family == ReferenceFamily::LayerStack; }
    const char* familyId() const;
};

ReferenceSource makeDefaultReferenceSource();
ReferenceSource loadReferenceSource(const std::filesystem::path& path);
} // namespace ncls
