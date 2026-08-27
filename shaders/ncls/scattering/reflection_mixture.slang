#ifndef NCLS_SCATTERING_REFLECTION_MIXTURE_SLANG
#define NCLS_SCATTERING_REFLECTION_MIXTURE_SLANG

#include "cosine.slang"
#include "ggx.slang"

// 有界的反射 proposal：一个 cosine support component 与最多四个旋转的
// anisotropic GGX VNDF component。它只拥有采样数学，不拥有材质语义。
static const uint NCLS_REFLECTION_GGX_COMPONENT_COUNT = 4u;

struct NclsReflectionMixture
{
    float cosineWeight;
    float4 ggxWeights;
    float4 alphaX;
    float4 alphaY;
    float4 rotations;
    uint ggxCount;
};

float nclsReflectionMixtureTotalWeight(NclsReflectionMixture proposal)
{
    float total = proposal.cosineWeight;
    [unroll]
    for (uint i = 0u; i < NCLS_REFLECTION_GGX_COMPONENT_COUNT; ++i)
        if (i < proposal.ggxCount) total += proposal.ggxWeights[i];
    return total;
}

bool nclsReflectionMixtureValid(NclsReflectionMixture proposal)
{
    if (proposal.ggxCount > NCLS_REFLECTION_GGX_COMPONENT_COUNT
        || !isfinite(proposal.cosineWeight) || proposal.cosineWeight < 0.0f
        || !all(isfinite(proposal.ggxWeights)) || any(proposal.ggxWeights < 0.0f)
        || !all(isfinite(proposal.alphaX)) || !all(isfinite(proposal.alphaY))
        || !all(isfinite(proposal.rotations)))
        return false;
    [unroll]
    for (uint i = 0u; i < NCLS_REFLECTION_GGX_COMPONENT_COUNT; ++i)
        if (i < proposal.ggxCount
            && (proposal.alphaX[i] < 1e-4f || proposal.alphaY[i] < 1e-4f))
            return false;
    const float total = nclsReflectionMixtureTotalWeight(proposal);
    return isfinite(total) && total > 0.0f;
}

float nclsRotatedGgxReflectionPdf(
    float3 wo,
    float3 wi,
    float2 alpha,
    float rotation)
{
    if (wo.z <= NCLS_MIN_COS || wi.z <= NCLS_MIN_COS) return 0.0f;
    const float3 rotatedWo = nclsRotateTangentToLocal(wo, rotation);
    const float3 rotatedWi = nclsRotateTangentToLocal(wi, rotation);
    const float3 sum = rotatedWo + rotatedWi;
    if (dot(sum, sum) <= 1e-20f) return 0.0f;
    const float3 halfVector = nclsSafeNormalize(sum);
    const float woDotHalf = dot(rotatedWo, halfVector);
    if (woDotHalf <= 1e-8f) return 0.0f;
    const float value = nclsGgxVisibleNormalPdf(rotatedWo, halfVector, alpha)
        / (4.0f * woDotHalf);
    return isfinite(value) && value >= 0.0f ? value : 0.0f;
}

float nclsReflectionMixturePdf(
    NclsReflectionMixture proposal,
    float3 wo,
    float3 wi)
{
    if (!nclsReflectionMixtureValid(proposal)
        || !nclsIsFiniteFloat3(wo) || !nclsIsFiniteFloat3(wi)
        || wo.z <= NCLS_MIN_COS || wi.z <= NCLS_MIN_COS)
        return 0.0f;
    float weightedPdf = proposal.cosineWeight * nclsCosineHemispherePdf(wi);
    [unroll]
    for (uint i = 0u; i < NCLS_REFLECTION_GGX_COMPONENT_COUNT; ++i)
    {
        if (i >= proposal.ggxCount) break;
        weightedPdf += proposal.ggxWeights[i] * nclsRotatedGgxReflectionPdf(
            wo, wi, float2(proposal.alphaX[i], proposal.alphaY[i]), proposal.rotations[i]);
    }
    const float value = weightedPdf / nclsReflectionMixtureTotalWeight(proposal);
    return isfinite(value) && value >= 0.0f ? value : 0.0f;
}

NclsDirectionalSample nclsSampleReflectionMixture(
    NclsReflectionMixture proposal,
    float3 wo,
    float2 u,
    out uint component)
{
    component = 0u;
    if (!nclsReflectionMixtureValid(proposal)
        || !nclsIsFiniteFloat3(wo) || wo.z <= NCLS_MIN_COS
        || !nclsIsUnitRandom(u.x) || !nclsIsUnitRandom(u.y))
        return nclsInvalidDirectionalSample();

    const float total = nclsReflectionMixtureTotalWeight(proposal);
    const float selection = u.x * total;
    float lower = 0.0f;
    float weight = proposal.cosineWeight;
    NclsDirectionalSample result;
    if (selection < weight)
    {
        component = 0u;
        result = nclsSampleCosineDirection(float2(
            nclsClampUnitRandom((selection - lower) / weight), u.y));
    }
    else
    {
        lower = weight;
        uint selected = NCLS_REFLECTION_GGX_COMPONENT_COUNT;
        [unroll]
        for (uint i = 0u; i < NCLS_REFLECTION_GGX_COMPONENT_COUNT; ++i)
        {
            if (i >= proposal.ggxCount) break;
            weight = proposal.ggxWeights[i];
            if (selection < lower + weight)
            {
                selected = i;
                break;
            }
            lower += weight;
        }
        if (selected >= proposal.ggxCount || weight <= 0.0f)
            return nclsInvalidDirectionalSample();
        component = selected + 1u;
        const float2 remapped = float2(
            nclsClampUnitRandom((selection - lower) / weight), u.y);
        const float rotation = proposal.rotations[selected];
        const float3 rotatedWo = nclsRotateTangentToLocal(wo, rotation);
        const float3 halfVector = nclsSampleGgxVisibleNormal(
            rotatedWo,
            float2(proposal.alphaX[selected], proposal.alphaY[selected]),
            remapped);
        const float3 rotatedWi = reflect(-rotatedWo, halfVector);
        const float3 wi = nclsRotateTangentFromLocal(rotatedWi, rotation);
        if (!nclsIsFiniteFloat3(wi)) return nclsInvalidDirectionalSample();
        if (wi.z <= NCLS_MIN_COS) return nclsNullDirectionalSample(wi);
        result = nclsContinuousDirectionalSample(wi, 1.0f);
    }

    if (result.valid == 0u || result.nullEvent != 0u) return result;
    result.pdf = nclsReflectionMixturePdf(proposal, wo, result.direction);
    return result.pdf > 0.0f ? result : nclsInvalidDirectionalSample();
}

#endif
