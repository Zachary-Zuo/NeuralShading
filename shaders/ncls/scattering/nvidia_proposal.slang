#ifndef NCLS_SCATTERING_NVIDIA_PROPOSAL_SLANG
#define NCLS_SCATTERING_NVIDIA_PROPOSAL_SLANG

#include "cosine.slang"
#include "ggx.slang"
#include "mixture.slang"

static const uint NCLS_NVIDIA_PROPOSAL_RAW_COUNT = 9u;
static const float NCLS_NVIDIA_PROPOSAL_SAFETY_WEIGHT = 0.03125f;
static const float NCLS_NVIDIA_PROPOSAL_MAX_ABS_RHO = 0.999999f;

struct NclsNvidiaProposalParams
{
    float2 alpha;
    float rho;
    float2 specularSlope;
    float2 diffuseSlope;
    float2 learnedWeights;
    uint valid;
};

float nclsNvidiaTanhApprox(float value)
{
    const float magnitude = abs(value);
    if (magnitude <= 1.0f) return value / sqrt(1.0f + value * value);
    const float inverseMagnitude = 1.0f / magnitude;
    return (value < 0.0f ? -1.0f : 1.0f)
        / sqrt(1.0f + inverseMagnitude * inverseMagnitude);
}

float nclsNvidiaSinhApprox(float value)
{
    return value * sqrt(1.0f + value * value);
}

NclsNvidiaProposalParams nclsDecodeNvidiaProposal(
    float raw[NCLS_NVIDIA_PROPOSAL_RAW_COUNT])
{
    NclsNvidiaProposalParams params = {};
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_PROPOSAL_RAW_COUNT; ++index)
        if (!isfinite(raw[index])) return params;
    params.alpha = 1e-4f + 0.5f * float2(
        1.0f + nclsNvidiaTanhApprox(raw[0]),
        1.0f + nclsNvidiaTanhApprox(raw[1]));
    params.rho = clamp(
        nclsNvidiaTanhApprox(raw[2]),
        -NCLS_NVIDIA_PROPOSAL_MAX_ABS_RHO,
        NCLS_NVIDIA_PROPOSAL_MAX_ABS_RHO);
    params.specularSlope = float2(
        nclsNvidiaSinhApprox(raw[3]),
        nclsNvidiaSinhApprox(raw[4]));
    params.diffuseSlope = float2(
        nclsNvidiaSinhApprox(raw[5]),
        nclsNvidiaSinhApprox(raw[6]));
    params.learnedWeights = nclsStableSoftmax2(float2(raw[7], raw[8]));
    const NclsNoncenteredGgxParams specular = {
        params.alpha, params.rho, params.specularSlope};
    params.valid = nclsNoncenteredGgxParamsValid(specular)
        && nclsIsFiniteFloat2(params.diffuseSlope)
        && nclsIsFiniteFloat2(params.learnedWeights)
        && params.learnedWeights.x + params.learnedWeights.y > 0.0f
        ? 1u
        : 0u;
    return params;
}

float3 nclsNvidiaProposalMixtureWeights(NclsNvidiaProposalParams params)
{
    const float learnedMass = 1.0f - NCLS_NVIDIA_PROPOSAL_SAFETY_WEIGHT;
    return float3(
        NCLS_NVIDIA_PROPOSAL_SAFETY_WEIGHT,
        learnedMass * params.learnedWeights.x,
        learnedMass * params.learnedWeights.y);
}

float nclsNvidiaProposalPdf(
    NclsNvidiaProposalParams params,
    float3 wo,
    float3 wi)
{
    if (params.valid == 0u || !nclsIsFiniteFloat3(wo) || wo.z <= NCLS_MIN_COS
        || !nclsIsFiniteFloat3(wi) || wi.z <= NCLS_MIN_COS)
        return 0.0f;
    const NclsNoncenteredGgxParams specular = {
        params.alpha, params.rho, params.specularSlope};
    const float3 componentPdfs = float3(
        nclsCosineHemispherePdf(wi),
        nclsNoncenteredGgxPdf(specular, wo, wi),
        nclsTiltedCosinePdf(wi, params.diffuseSlope));
    return nclsMixture3Pdf(nclsNvidiaProposalMixtureWeights(params), componentPdfs);
}

NclsDirectionalSample nclsSampleNvidiaProposal(
    NclsNvidiaProposalParams params,
    float3 wo,
    float3 u,
    out uint component)
{
    component = 0u;
    if (params.valid == 0u || !nclsIsFiniteFloat3(wo) || wo.z <= NCLS_MIN_COS
        || !nclsIsUnitRandom(u.x) || !nclsIsUnitRandom(u.y) || !nclsIsUnitRandom(u.z))
        return nclsInvalidDirectionalSample();
    const float3 weights = nclsNvidiaProposalMixtureWeights(params);
    const NclsMixture3Selection selection = nclsSelectMixture3(weights, u.x);
    if (selection.valid == 0u) return nclsInvalidDirectionalSample();
    component = selection.component;

    // u.x 只选择分量，独立的 u.yz 驱动二维方向采样；remapped 供复用同一维随机数的调用者使用。

    NclsDirectionalSample result;
    if (component == 0u)
        result = nclsSampleCosineDirection(u.yz);
    else if (component == 1u)
    {
        const NclsNoncenteredGgxParams specular = {
            params.alpha, params.rho, params.specularSlope};
        result = nclsSampleNoncenteredGgx(specular, wo, u.yz);
    }
    else
        result = nclsSampleTiltedCosine(params.diffuseSlope, u.yz);

    if (result.valid == 0u || result.nullEvent != 0u) return result;
    result.pdf = nclsNvidiaProposalPdf(params, wo, result.direction);
    if (!isfinite(result.pdf) || result.pdf <= 0.0f)
        return nclsInvalidDirectionalSample();
    return result;
}

#endif
