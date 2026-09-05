#ifndef NCLS_METAL_BUDGETED_SAMPLER_SLANG
#define NCLS_METAL_BUDGETED_SAMPLER_SLANG

static const float NCLS_METAL_PROPOSAL_PI = 3.14159265358979323846f;
static const float NCLS_METAL_PROPOSAL_RANDOM_EPSILON = 0x1p-24f;

bool nclsMetalBudgetedProposalWeights(NclsMetalBudgetedPrepared state, out float3 weights)
{
    weights = float3(state.proposal[0], state.proposal[4], state.proposal[8]);
    bool valid = state.valid != 0u && all(isfinite(weights))
        && all(weights >= 0.0f) && weights.z > 0.0f;
    [unroll]
    for (uint i = 0u; i < 3u; ++i)
    {
        const float2 alpha = float2(state.proposal[4u*i+1u], state.proposal[4u*i+2u]);
        const float distribution = state.proposal[4u*i+3u];
        valid = valid && all(isfinite(alpha)) && all(alpha >= 0.015f)
            && all(alpha <= 1.0f) && isfinite(distribution)
            && round(distribution) >= 0.0f && round(distribution) <= 2.0f
            && abs(distribution - round(distribution)) <= 0.25f;
    }
    const float total = weights.x + weights.y + weights.z;
    valid = valid && isfinite(total) && total > 0.0f;
    weights = valid ? weights / max(total, 1e-12f) : float3(0.0f);
    return valid;
}

bool nclsMetalBudgetedProposalBasis(
    NclsMetalBudgetedPrepared state, uint component, float3 wo,
    out NclsMetalBudgetedFrame basis)
{
    const NclsMetalBudgetedFrame frame = state.frames[component == 1u ? 1u : 0u];
    uint normalValid, axisValid, unused;
    float3 normal = nclsMetalBudgetedSafeNormalize(frame.normal, float3(0, 0, 1), normalValid);
    normal = normal.z >= 0.0f ? normal : -normal;
    basis.normal = nclsMetalBudgetedSafeNormalize(
        component < 2u ? 2.0f * dot(wo, normal) * normal - wo : float3(0, 0, 1),
        float3(0, 0, 1), axisValid);
    const float3 helper = abs(basis.normal.z) < 0.9f ? float3(0, 0, 1) : float3(1, 0, 0);
    const float3 fallback = nclsMetalBudgetedSafeNormalize(
        cross(helper, basis.normal), float3(1, 0, 0), unused);
    basis.tangent = nclsMetalBudgetedSafeNormalize(
        frame.tangent - dot(frame.tangent, basis.normal) * basis.normal, fallback, unused);
    basis.bitangent = cross(basis.normal, basis.tangent);
    return normalValid != 0u && axisValid != 0u && all(isfinite(frame.tangent))
        && all(isfinite(frame.bitangent)) && all(isfinite(frame.normal));
}

float nclsMetalBudgetedProposalLocalPdf(float3 local, float2 alpha, uint distribution)
{
    if (local.z <= 0.0f) return 0.0f;
    const float z = max(local.z, 1e-8f);
    alpha = clamp(alpha, 0.015f, 1.0f);
    const float2 scaled = local.xy / alpha;
    float value;
    if (distribution == 0u)
    {
        const float denominator = dot(scaled, scaled) + z * z;
        value = z / (NCLS_METAL_PROPOSAL_PI * alpha.x * alpha.y
            * max(denominator * denominator, 1e-20f));
    }
    else if (distribution == 1u)
        value = exp(min(-dot(scaled / z, scaled / z) - log(NCLS_METAL_PROPOSAL_PI)
            - log(alpha.x) - log(alpha.y) - 3.0f * log(z), 80.0f));
    else value = 0.5f / NCLS_METAL_PROPOSAL_PI;
    return isfinite(value) ? value : 0.0f;
}

float nclsMetalBudgetedProposalDensity(NclsMetalBudgetedPrepared state, float3 wo, float3 wi)
{
    float3 weights;
    if (!nclsMetalBudgetedProposalWeights(state, weights)
        || !all(isfinite(wo)) || !all(isfinite(wi)) || wo.z <= 1e-6f || wi.z <= 1e-6f)
        return 0.0f;
    const float3 mirrored = float3(wi.xy, -wi.z);
    float density = 0.0f;
    [unroll]
    for (uint i = 0u; i < 3u; ++i)
    {
        NclsMetalBudgetedFrame basis;
        if (!nclsMetalBudgetedProposalBasis(state, i, wo, basis)) return 0.0f;
        const float2 alpha = float2(state.proposal[4u*i+1u], state.proposal[4u*i+2u]);
        const uint distribution = uint(round(state.proposal[4u*i+3u]));
        density += weights[i] * (
            nclsMetalBudgetedProposalLocalPdf(nclsMetalBudgetedToFrame(basis, wi), alpha, distribution)
            + nclsMetalBudgetedProposalLocalPdf(nclsMetalBudgetedToFrame(basis, mirrored), alpha, distribution));
    }
    return isfinite(density) && density > 0.0f ? density : 0.0f;
}

NclsScatteringPdf nclsMetalBudgetedProposalPdf(NclsMetalBudgetedPrepared state, float3 wo, float3 wi)
{
    return {nclsMetalBudgetedProposalDensity(state, wo, wi),
        nclsMetalBudgetedProposalDensity(state, wi, wo)};
}

bool nclsMetalBudgetedSampleProposal(
    NclsMetalBudgetedPrepared state, float3 wo, float2 u,
    out float3 wi, out NclsScatteringPdf density)
{
    wi = float3(0.0f);
    density = {};
    float3 weights;
    if (!nclsMetalBudgetedProposalWeights(state, weights)
        || !all(isfinite(u)) || any(u < 0.0f) || any(u >= 1.0f)) return false;
    const uint component = u.x < weights.x ? 0u : u.x < weights.x + weights.y ? 1u : 2u;
    const float previous = component == 0u ? 0.0f : component == 1u ? weights.x : weights.x + weights.y;
    const float first = clamp((u.x - previous) / max(weights[component], 1e-12f),
        NCLS_METAL_PROPOSAL_RANDOM_EPSILON, 1.0f - NCLS_METAL_PROPOSAL_RANDOM_EPSILON);
    const float phi = 2.0f * NCLS_METAL_PROPOSAL_PI * u.y;
    const uint distribution = uint(round(state.proposal[4u*component+3u]));
    float3 local;
    if (distribution == 2u)
        local = float3(sqrt(max(1.0f-first*first, 0.0f)) * float2(cos(phi), sin(phi)), first);
    else
    {
        const float radius = distribution == 1u ? sqrt(-log(1.0f-first)) : sqrt(first/(1.0f-first));
        const float2 alpha = float2(state.proposal[4u*component+1u], state.proposal[4u*component+2u]);
        local = normalize(float3(alpha * radius * float2(cos(phi), sin(phi)), 1.0f));
    }
    NclsMetalBudgetedFrame basis;
    if (!nclsMetalBudgetedProposalBasis(state, component, wo, basis)) return false;
    wi = local.x*basis.tangent + local.y*basis.bitangent + local.z*basis.normal;
    wi.z = abs(wi.z);
    uint valid;
    wi = nclsMetalBudgetedSafeNormalize(wi, float3(0, 0, 1), valid);
    density = nclsMetalBudgetedProposalPdf(state, wo, wi);
    return valid != 0u && density.forward > 0.0f && density.reverse > 0.0f;
}

#endif
