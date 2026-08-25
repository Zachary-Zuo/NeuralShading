#ifndef NCLS_SCATTERING_FRESNEL_SLANG
#define NCLS_SCATTERING_FRESNEL_SLANG

#include "common.slang"

float nclsFresnelDielectric(float etaIOverEtaT, float cosThetaI)
{
    cosThetaI = clamp(abs(cosThetaI), 0.0f, 1.0f);
    const float sinThetaTSquared = etaIOverEtaT * etaIOverEtaT
        * max(0.0f, 1.0f - cosThetaI * cosThetaI);
    if (sinThetaTSquared >= 1.0f) return 1.0f;
    const float cosThetaT = sqrt(max(0.0f, 1.0f - sinThetaTSquared));
    const float rs = (etaIOverEtaT * cosThetaI - cosThetaT)
        / (etaIOverEtaT * cosThetaI + cosThetaT);
    const float rp = (etaIOverEtaT * cosThetaT - cosThetaI)
        / (etaIOverEtaT * cosThetaT + cosThetaI);
    return 0.5f * (rs * rs + rp * rp);
}

float nclsFresnelConductorChannel(float eta, float k, float cosThetaI)
{
    cosThetaI = clamp(abs(cosThetaI), 0.0f, 1.0f);
    const float cosSquared = cosThetaI * cosThetaI;
    const float sinSquared = max(1.0f - cosSquared, 0.0f);
    const float sinFourth = sinSquared * sinSquared;
    const float inner = eta * eta - k * k - sinSquared;
    const float aSquaredPlusBSquared = sqrt(max(inner * inner + 4.0f * eta * eta * k * k, 0.0f));
    const float a = sqrt(max(0.5f * (aSquaredPlusBSquared + inner), 0.0f));
    const float rs = (aSquaredPlusBSquared + cosSquared - 2.0f * a * cosThetaI)
        / (aSquaredPlusBSquared + cosSquared + 2.0f * a * cosThetaI);
    const float rp = (cosSquared * aSquaredPlusBSquared + sinFourth
        - 2.0f * a * cosThetaI * sinSquared)
        / (cosSquared * aSquaredPlusBSquared + sinFourth
            + 2.0f * a * cosThetaI * sinSquared);
    return 0.5f * (rs + rs * rp);
}

float3 nclsFresnelConductor(float3 eta, float3 k, float cosThetaI)
{
    return float3(
        nclsFresnelConductorChannel(eta.x, k.x, cosThetaI),
        nclsFresnelConductorChannel(eta.y, k.y, cosThetaI),
        nclsFresnelConductorChannel(eta.z, k.z, cosThetaI));
}

#endif
