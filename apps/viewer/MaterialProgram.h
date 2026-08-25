#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

#include <nlohmann/json_fwd.hpp>

namespace ncls
{
// 与 src/ncls/core/material/abi/layer_stack_ir_v1.json 和生成的 Slang ABI 一致。
constexpr uint32_t kLayerStackMagic = 0x52494C4Eu;
constexpr uint32_t kLayerStackVersion = 1u;
constexpr uint32_t kMaximumInterfaces = 8u;
constexpr uint32_t kMaximumMedia = 7u;

enum class InterfaceKind : uint32_t
{
    RoughDielectric = 0,
    RoughConductor = 1,
    Diffuse = 2,
    Sheen = 3,
};

struct LayerInterface
{
    uint32_t kind = static_cast<uint32_t>(InterfaceKind::RoughDielectric);
    uint32_t flags = 0;
    float alphaX = 0.f;
    float alphaY = 0.f;
    float relativeIor = 0.f;
    float etaR = 0.f;
    float etaG = 0.f;
    float etaB = 0.f;
    float kR = 0.f;
    float kG = 0.f;
    float kB = 0.f;
    float colorR = 0.f;
    float colorG = 0.f;
    float colorB = 0.f;
    float tangentRotation = 0.f;
    float reserved = 0.f;
};

struct HomogeneousMedium
{
    float sigmaAR = 0.f;
    float sigmaAG = 0.f;
    float sigmaAB = 0.f;
    float sigmaSR = 0.f;
    float sigmaSG = 0.f;
    float sigmaSB = 0.f;
    float g = 0.f;
    // ABI padding record must remain all-zero. Call sites creating a real medium
    // set the physical default thickness explicitly.
    float thickness = 0.f;
};

struct LayerStackIR
{
    uint32_t magic = kLayerStackMagic;
    uint32_t abiVersion = kLayerStackVersion;
    uint32_t interfaceCount = 0;
    uint32_t mediumCount = 0;
    std::array<LayerInterface, kMaximumInterfaces> interfaces{};
    std::array<HomogeneousMedium, kMaximumMedia> media{};
};

static_assert(sizeof(LayerInterface) == 64);
static_assert(sizeof(HomogeneousMedium) == 32);
static_assert(sizeof(LayerStackIR) == 752);

LayerStackIR makeDefaultMaterial();
LayerStackIR loadMaterialProgram(const std::filesystem::path& path, std::string* displayName = nullptr);
LayerStackIR loadMaterialProgramDocument(
    const nlohmann::json& document,
    std::string* displayName = nullptr,
    const std::string& fallbackDisplayName = "Embedded LayerStack material");
nlohmann::json makeMaterialProgramDocument(
    const LayerStackIR& stack,
    const std::string& displayName);
void saveMaterialProgram(const std::filesystem::path& path, const LayerStackIR& stack, const std::string& displayName);
void validateLayerStack(const LayerStackIR& stack);
std::string layerStackHash(const LayerStackIR& stack);

bool addDielectricCoat(LayerStackIR& stack);
bool removeCoat(LayerStackIR& stack, uint32_t index);
bool moveCoat(LayerStackIR& stack, uint32_t index, int direction);
} // namespace ncls
