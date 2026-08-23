#include "OpenPbrLuts.h"

#include <glm/glm.hpp>
#include <openpbr.h>

#include <vector>

namespace ncls
{
namespace
{
template<typename T>
std::vector<Falcor::float4> scalarRgba(const T* source, size_t count)
{
    std::vector<Falcor::float4> result(count);
    for (size_t index = 0; index < count; ++index)
        result[index] = Falcor::float4(float(source[index]) * (1.f / 65535.f), 0.f, 0.f, 0.f);
    return result;
}

Falcor::ref<Falcor::Texture> scalar2D(
    const Falcor::ref<Falcor::Device>& device,
    const OpenPBR_EnergyTableElement* source,
    uint32_t width,
    uint32_t height)
{
    const auto values = scalarRgba(source, size_t(width) * height);
    return device->createTexture2D(width, height, Falcor::ResourceFormat::RGBA32Float, 1, 1, values.data());
}

Falcor::ref<Falcor::Texture> scalar3D(
    const Falcor::ref<Falcor::Device>& device,
    const OpenPBR_EnergyTableElement* source)
{
    constexpr uint32_t size = OpenPBR_EnergyTableSize;
    const auto values = scalarRgba(source, size_t(size) * size * size);
    return device->createTexture3D(size, size, size, Falcor::ResourceFormat::RGBA32Float, 1, values.data());
}
} // namespace

OpenPbrLuts OpenPbrLuts::create(const Falcor::ref<Falcor::Device>& device)
{
    OpenPbrLuts result;
    result.idealDielectricEnergy = scalar3D(device, OpenPBR_IdealDielectricEnergyComplement_Array);
    result.idealDielectricAverage = scalar2D(
        device, OpenPBR_IdealDielectricAverageEnergyComplement_Array, OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize);
    result.idealDielectricRatio = scalar2D(
        device, OpenPBR_IdealDielectricReflectionRatio_Array, OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize);
    result.opaqueDielectricEnergy = scalar3D(device, OpenPBR_OpaqueDielectricEnergyComplement_Array);
    result.opaqueDielectricAverage = scalar2D(
        device, OpenPBR_OpaqueDielectricAverageEnergyComplement_Array, OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize);
    result.idealMetalEnergy = scalar2D(
        device, OpenPBR_IdealMetalEnergyComplement_Array, OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize);
    result.idealMetalAverage = scalar2D(
        device, OpenPBR_IdealMetalAverageEnergyComplement_Array, OpenPBR_EnergyTableSize, 1);
    std::vector<Falcor::float4> ltcValues(OpenPBR_LTCTableSize * OpenPBR_LTCTableSize);
    for (size_t index = 0; index < ltcValues.size(); ++index)
    {
        const auto& value = OpenPBR_LTC_Array[index];
        ltcValues[index] = Falcor::float4(value.x, value.y, value.z, 0.f);
    }
    result.ltc = device->createTexture2D(
        OpenPBR_LTCTableSize, OpenPBR_LTCTableSize, Falcor::ResourceFormat::RGBA32Float, 1, 1, ltcValues.data());
    Falcor::Sampler::Desc sampler;
    sampler.setFilterMode(
        Falcor::TextureFilteringMode::Linear,
        Falcor::TextureFilteringMode::Linear,
        Falcor::TextureFilteringMode::Linear);
    sampler.setAddressingMode(
        Falcor::TextureAddressingMode::Clamp,
        Falcor::TextureAddressingMode::Clamp,
        Falcor::TextureAddressingMode::Clamp);
    result.sampler = device->createSampler(sampler);
    return result;
}

void OpenPbrLuts::bind(Falcor::ShaderVar root) const
{
    root["gOpenPbrIdealDielectricEnergy"] = idealDielectricEnergy;
    root["gOpenPbrIdealDielectricAverage"] = idealDielectricAverage;
    root["gOpenPbrIdealDielectricRatio"] = idealDielectricRatio;
    root["gOpenPbrOpaqueDielectricEnergy"] = opaqueDielectricEnergy;
    root["gOpenPbrOpaqueDielectricAverage"] = opaqueDielectricAverage;
    root["gOpenPbrIdealMetalEnergy"] = idealMetalEnergy;
    root["gOpenPbrIdealMetalAverage"] = idealMetalAverage;
    root["gOpenPbrLtc"] = ltc;
    root["gOpenPbrLutSampler"] = sampler;
}
} // namespace ncls
