#ifndef NCLS_SCATTERING_MIXTURE_SLANG
#define NCLS_SCATTERING_MIXTURE_SLANG

#include "common.slang"

struct NclsMixture3Selection
{
    uint component;
    float remapped;
    uint valid;
};

float3 nclsNormalizeMixture3(float3 weights)
{
    if (!nclsIsFiniteFloat3(weights) || any(weights < 0.0f)) return float3(0.0f);
    const float sum = weights.x + weights.y + weights.z;
    return isfinite(sum) && sum > 0.0f ? weights / sum : float3(0.0f);
}

float2 nclsStableSoftmax2(float2 logits)
{
    if (!nclsIsFiniteFloat2(logits)) return float2(0.0f);
    const float maximum = max(logits.x, logits.y);
    const float2 exponentials = exp(logits - maximum);
    const float sum = exponentials.x + exponentials.y;
    return isfinite(sum) && sum > 0.0f ? exponentials / sum : float2(0.0f);
}

NclsMixture3Selection nclsSelectMixture3(float3 weights, float u)
{
    NclsMixture3Selection result = {};
    weights = nclsNormalizeMixture3(weights);
    if (!nclsIsUnitRandom(u) || weights.x + weights.y + weights.z <= 0.0f)
        return result;
    if (u < weights.x)
    {
        if (weights.x <= 0.0f) return result;
        result.component = 0u;
        result.remapped = nclsClampUnitRandom(u / weights.x);
    }
    else if (u < weights.x + weights.y)
    {
        if (weights.y <= 0.0f) return result;
        result.component = 1u;
        result.remapped = nclsClampUnitRandom((u - weights.x) / weights.y);
    }
    else
    {
        if (weights.z <= 0.0f) return result;
        result.component = 2u;
        result.remapped = nclsClampUnitRandom((u - weights.x - weights.y) / weights.z);
    }
    result.valid = 1u;
    return result;
}

float nclsMixture3Pdf(float3 weights, float3 componentPdfs)
{
    weights = nclsNormalizeMixture3(weights);
    if (weights.x + weights.y + weights.z <= 0.0f
        || !nclsIsFiniteFloat3(componentPdfs) || any(componentPdfs < 0.0f))
        return 0.0f;
    const float value = dot(weights, componentPdfs);
    return isfinite(value) && value >= 0.0f ? value : 0.0f;
}

#endif
