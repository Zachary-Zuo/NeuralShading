#pragma once

#include "MaterialProgram.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

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
    std::string displayName = "Default layered material";
    std::string sourceSha256;

    const char* familyId() const;
};

ReferenceSource makeDefaultReferenceSource();
ReferenceSource makeDefaultReferenceSource(ReferenceFamily family);
ReferenceSource loadReferenceSource(const std::filesystem::path& path);
nlohmann::json serializeReferenceSourceState(
    const ReferenceSource& source,
    const std::filesystem::path& manifestDirectory);
ReferenceSource deserializeReferenceSourceState(
    const nlohmann::json& document,
    const std::filesystem::path& manifestDirectory);
std::string referenceSourceStateHash(const ReferenceSource& source);
void saveOpenPbrReferenceSource(const std::filesystem::path& path, const ReferenceSource& source);
} // namespace ncls
