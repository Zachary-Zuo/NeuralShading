#pragma once

#include "Falcor.h"

namespace ncls
{
struct OpenPbrLuts
{
    Falcor::ref<Falcor::Texture> idealDielectricEnergy;
    Falcor::ref<Falcor::Texture> idealDielectricAverage;
    Falcor::ref<Falcor::Texture> idealDielectricRatio;
    Falcor::ref<Falcor::Texture> opaqueDielectricEnergy;
    Falcor::ref<Falcor::Texture> opaqueDielectricAverage;
    Falcor::ref<Falcor::Texture> idealMetalEnergy;
    Falcor::ref<Falcor::Texture> idealMetalAverage;
    Falcor::ref<Falcor::Texture> ltc;
    Falcor::ref<Falcor::Sampler> sampler;

    static OpenPbrLuts create(const Falcor::ref<Falcor::Device>& device);
    void bind(Falcor::ShaderVar root) const;
};
} // namespace ncls
