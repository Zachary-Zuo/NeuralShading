#ifndef NCLS_SCATTERING_METAL_FUSED_PROPOSAL_SLANG
#define NCLS_SCATTERING_METAL_FUSED_PROPOSAL_SLANG

#include "common.slang"
#include "../backends/metal_fused/metal_fused_layout.generated.slang"

struct NclsMetalFusedProposal
{
    float state[NCLS_METAL_PROPOSAL_COMPONENT_COUNT][NCLS_METAL_PROPOSAL_STATE_WIDTH];
    float3 frames[NCLS_METAL_PROPOSAL_FRAME_COUNT][3];
    uint valid;
};

struct NclsMetalFusedProposalBasis
{
    float3 tangent;
    float3 bitangent;
    float3 axis;
    uint valid;
};

struct NclsMetalFusedProposalSample
{
    float3 direction;
    float forwardPdf;
    float reversePdf;
    uint valid;
    uint component;
};

float3 nclsMetalProposalNormalize(float3 value, float3 fallback, out uint valid)
{
    const float lengthSquared = dot(value, value);
    valid = isfinite(lengthSquared) && lengthSquared > 1e-12f ? 1u : 0u;
    return valid != 0u ? value * rsqrt(lengthSquared) : fallback;
}

float3 nclsMetalProposalFallbackTangent(float3 axis)
{
    const float3 helper = abs(axis.z) < 0.9f
        ? float3(0.0f, 0.0f, 1.0f)
        : float3(1.0f, 0.0f, 0.0f);
    return nclsSafeNormalize(cross(helper, axis));
}

bool nclsMetalFusedProposalValid(NclsMetalFusedProposal proposal)
{
    if (proposal.valid == 0u) return false;
    float totalWeight = 0.0f;
    float totalEnergyClue = 0.0f;
    [loop]
    for (uint component = 0u; component < NCLS_METAL_PROPOSAL_COMPONENT_COUNT; ++component)
    {
        [loop]
        for (uint field = 0u; field < NCLS_METAL_PROPOSAL_STATE_WIDTH; ++field)
            if (!isfinite(proposal.state[component][field])) return false;
        const float weight = proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_NORMALIZED_WEIGHT];
        const float active = proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_ACTIVE];
        const float alphaX = proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_ALPHA_X];
        const float alphaY = proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_ALPHA_Y];
        const float energyClue = proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_ENERGY_CLUE];
        if (weight < 0.0f || (weight > 0.0f && active <= 0.5f)
            || energyClue < 0.0f
            || alphaX < 0.01f || alphaX > 1.0f
            || alphaY < 0.01f || alphaY > 1.0f
            || abs(proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_FRAME_INDEX]
                - float(nclsMetalProposalFrameIndex(component))) > 0.25f
            || abs(proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_DISTRIBUTION_ID]
                - float(nclsMetalProposalDistribution(component))) > 0.25f)
            return false;
        totalWeight += weight;
        totalEnergyClue += energyClue;
    }
    if (!isfinite(totalWeight) || totalWeight <= 0.0f || totalEnergyClue <= 0.0f
        || proposal.state[NCLS_METAL_PROPOSAL_COMPONENT_COUNT - 1u]
            [NCLS_METAL_PROPOSAL_FIELD_NORMALIZED_WEIGHT] <= 0.0f
        || proposal.state[NCLS_METAL_PROPOSAL_COMPONENT_COUNT - 1u]
            [NCLS_METAL_PROPOSAL_FIELD_ACTIVE] <= 0.5f)
        return false;
    [loop]
    for (uint frame = 0u; frame < NCLS_METAL_PROPOSAL_FRAME_COUNT; ++frame)
    {
        [loop]
        for (uint axis = 0u; axis < 3u; ++axis)
            if (!nclsIsFiniteFloat3(proposal.frames[frame][axis])) return false;
        if (dot(proposal.frames[frame][0], proposal.frames[frame][0]) <= 1e-12f
            || dot(proposal.frames[frame][2], proposal.frames[frame][2]) <= 1e-12f)
            return false;
    }
    return true;
}

float nclsMetalProposalTotalWeight(NclsMetalFusedProposal proposal)
{
    float total = 0.0f;
    [loop]
    for (uint component = 0u; component < NCLS_METAL_PROPOSAL_COMPONENT_COUNT; ++component)
        total += proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_NORMALIZED_WEIGHT];
    return total;
}

NclsMetalFusedProposalBasis nclsMetalProposalBasis(
    NclsMetalFusedProposal proposal,
    uint component,
    float3 wo)
{
    NclsMetalFusedProposalBasis result = {};
    const uint frameIndex = nclsMetalProposalFrameIndex(component);
    float3 frameTangent = proposal.frames[frameIndex][0];
    float3 frameNormal = proposal.frames[frameIndex][2];
    uint normalValid;
    frameNormal = nclsMetalProposalNormalize(
        frameNormal, float3(0.0f, 0.0f, 1.0f), normalValid);
    if (frameNormal.z < 0.0f) frameNormal = -frameNormal;
    uint tangentValid;
    frameTangent = nclsMetalProposalNormalize(
        frameTangent - dot(frameTangent, frameNormal) * frameNormal,
        nclsMetalProposalFallbackTangent(frameNormal),
        tangentValid);
    const float3 frameBitangent = cross(frameNormal, frameTangent);
    const float rotation = proposal.state[component]
        [NCLS_METAL_PROPOSAL_FIELD_ROTATION_RADIANS];
    const float3 rotatedTangent =
        cos(rotation) * frameTangent + sin(rotation) * frameBitangent;
    const float3 reflected = 2.0f * dot(wo, frameNormal) * frameNormal - wo;
    const float3 axisSeed = nclsMetalProposalIsSpecular(component)
        ? reflected
        : frameNormal;
    uint axisValid;
    result.axis = nclsMetalProposalNormalize(
        axisSeed, float3(0.0f, 0.0f, 1.0f), axisValid);
    uint basisTangentValid;
    result.tangent = nclsMetalProposalNormalize(
        rotatedTangent - dot(rotatedTangent, result.axis) * result.axis,
        nclsMetalProposalFallbackTangent(result.axis),
        basisTangentValid);
    result.bitangent = cross(result.axis, result.tangent);
    // A valid reflected axis can align with the authored tangent at isolated
    // outgoing directions.  The deterministic fallback still defines a full
    // orthonormal basis there, so this is not a malformed proposal state.
    result.valid = normalValid & tangentValid & axisValid;
    return result;
}

float nclsMetalProposalLocalPdf(
    NclsMetalFusedProposal proposal,
    uint component,
    float3 localDirection)
{
    if (localDirection.z <= 0.0f || !nclsIsFiniteFloat3(localDirection)) return 0.0f;
    const float z = max(localDirection.z, 1e-8f);
    const float alphaX = clamp(
        proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_ALPHA_X], 0.01f, 1.0f);
    const float alphaY = clamp(
        proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_ALPHA_Y], 0.01f, 1.0f);
    const uint distribution = nclsMetalProposalDistribution(component);
    float value = 0.0f;
    if (distribution == NCLS_METAL_DISTRIBUTION_GGX)
    {
        const float denominator =
            (localDirection.x / alphaX) * (localDirection.x / alphaX)
            + (localDirection.y / alphaY) * (localDirection.y / alphaY)
            + z * z;
        value = z / max(
            NCLS_PI * alphaX * alphaY * denominator * denominator, 1e-20f);
    }
    else if (distribution == NCLS_METAL_DISTRIBUTION_BECKMANN)
    {
        const float slopeRadius =
            (localDirection.x / (alphaX * z)) * (localDirection.x / (alphaX * z))
            + (localDirection.y / (alphaY * z)) * (localDirection.y / (alphaY * z));
        const float logPdf = -slopeRadius - log(NCLS_PI) - log(alphaX)
            - log(alphaY) - 3.0f * log(z);
        value = exp(min(logPdf, 80.0f));
    }
    else if (distribution == NCLS_METAL_DISTRIBUTION_COSINE)
    {
        value = z * NCLS_INV_PI;
    }
    else
    {
        value = 0.5f * NCLS_INV_PI;
    }
    return isfinite(value) && value >= 0.0f ? value : 0.0f;
}

float nclsMetalProposalComponentPdf(
    NclsMetalFusedProposal proposal,
    uint component,
    float3 wo,
    float3 wi)
{
    const NclsMetalFusedProposalBasis basis = nclsMetalProposalBasis(
        proposal, component, wo);
    if (basis.valid == 0u) return 0.0f;
    const float3 local = float3(
        dot(wi, basis.tangent),
        dot(wi, basis.bitangent),
        dot(wi, basis.axis));
    const float3 mirroredWi = float3(wi.x, wi.y, -wi.z);
    const float3 mirroredLocal = float3(
        dot(mirroredWi, basis.tangent),
        dot(mirroredWi, basis.bitangent),
        dot(mirroredWi, basis.axis));
    return nclsMetalProposalLocalPdf(proposal, component, local)
        + nclsMetalProposalLocalPdf(proposal, component, mirroredLocal);
}

float nclsMetalFusedProposalPdf(
    NclsMetalFusedProposal proposal,
    float3 wo,
    float3 wi)
{
    if (!nclsMetalFusedProposalValid(proposal)
        || !nclsIsFiniteFloat3(wo) || wo.z <= NCLS_MIN_COS
        || !nclsIsFiniteFloat3(wi) || wi.z <= NCLS_MIN_COS)
        return 0.0f;
    const float totalWeight = nclsMetalProposalTotalWeight(proposal);
    float weightedPdf = 0.0f;
    [loop]
    for (uint component = 0u; component < NCLS_METAL_PROPOSAL_COMPONENT_COUNT; ++component)
    {
        weightedPdf += proposal.state[component]
            [NCLS_METAL_PROPOSAL_FIELD_NORMALIZED_WEIGHT]
            * nclsMetalProposalComponentPdf(proposal, component, wo, wi);
    }
    const float result = weightedPdf / totalWeight;
    return isfinite(result) && result > 0.0f ? result : 0.0f;
}

float3 nclsMetalProposalSampleLocal(
    NclsMetalFusedProposal proposal,
    uint component,
    float2 u)
{
    const float u0 = clamp(u.x, 5.960464477539063e-8f, NCLS_UNIT_RANDOM_MAX);
    const float phi = 2.0f * NCLS_PI * u.y;
    const float2 azimuth = float2(cos(phi), sin(phi));
    const uint distribution = nclsMetalProposalDistribution(component);
    if (distribution == NCLS_METAL_DISTRIBUTION_COSINE)
    {
        const float radius = sqrt(u0);
        return float3(
            radius * azimuth,
            sqrt(max(0.0f, 1.0f - u0)));
    }
    if (distribution == NCLS_METAL_DISTRIBUTION_UNIFORM)
    {
        return float3(
            sqrt(max(0.0f, 1.0f - u0 * u0)) * azimuth,
            u0);
    }
    const float radius = distribution == NCLS_METAL_DISTRIBUTION_BECKMANN
        ? sqrt(-log(max(1.0f - u0, 5.960464477539063e-8f)))
        : sqrt(u0 / max(1.0f - u0, 5.960464477539063e-8f));
    const float alphaX = clamp(
        proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_ALPHA_X], 0.01f, 1.0f);
    const float alphaY = clamp(
        proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_ALPHA_Y], 0.01f, 1.0f);
    return nclsSafeNormalize(float3(
        alphaX * radius * azimuth.x,
        alphaY * radius * azimuth.y,
        1.0f));
}

NclsMetalFusedProposalSample nclsSampleMetalFusedProposal(
    NclsMetalFusedProposal proposal,
    float3 wo,
    float2 u)
{
    NclsMetalFusedProposalSample result = {};
    result.direction = float3(0.0f, 0.0f, 1.0f);
    if (!nclsMetalFusedProposalValid(proposal)
        || !nclsIsFiniteFloat3(wo) || wo.z <= NCLS_MIN_COS
        || !nclsIsUnitRandom(u.x) || !nclsIsUnitRandom(u.y))
        return result;
    const float totalWeight = nclsMetalProposalTotalWeight(proposal);
    const float target = u.x * totalWeight;
    float previous = 0.0f;
    uint selected = NCLS_METAL_PROPOSAL_COMPONENT_COUNT - 1u;
    [loop]
    for (uint component = 0u; component < NCLS_METAL_PROPOSAL_COMPONENT_COUNT; ++component)
    {
        const float current = previous + proposal.state[component]
            [NCLS_METAL_PROPOSAL_FIELD_NORMALIZED_WEIGHT];
        if (target < current)
        {
            selected = component;
            break;
        }
        previous = current;
    }
    const float selectedWeight = proposal.state[selected]
        [NCLS_METAL_PROPOSAL_FIELD_NORMALIZED_WEIGHT];
    if (selectedWeight <= 0.0f) return result;
    const float2 remapped = float2(
        nclsClampUnitRandom((target - previous) / selectedWeight), u.y);
    const NclsMetalFusedProposalBasis basis = nclsMetalProposalBasis(
        proposal, selected, wo);
    if (basis.valid == 0u) return result;
    const float3 local = nclsMetalProposalSampleLocal(proposal, selected, remapped);
    float3 wi = local.x * basis.tangent
        + local.y * basis.bitangent
        + local.z * basis.axis;
    wi.z = abs(wi.z);
    wi = nclsSafeNormalize(wi);
    if (!nclsIsFiniteFloat3(wi) || wi.z <= NCLS_MIN_COS) return result;
    const float forwardPdf = nclsMetalFusedProposalPdf(proposal, wo, wi);
    const float reversePdf = nclsMetalFusedProposalPdf(proposal, wi, wo);
    if (forwardPdf <= 0.0f || reversePdf <= 0.0f
        || !isfinite(forwardPdf) || !isfinite(reversePdf))
        return result;
    result.direction = wi;
    result.forwardPdf = forwardPdf;
    result.reversePdf = reversePdf;
    result.valid = 1u;
    result.component = selected;
    return result;
}

#endif
