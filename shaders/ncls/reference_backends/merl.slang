#ifndef NCLS_REFERENCE_BACKENDS_MERL_SLANG
#define NCLS_REFERENCE_BACKENDS_MERL_SLANG

// MERL measured BRDF reference 的统一 scattering contract 包装。
#include "../reference/merl_reference.slang"
#include "../scattering/reflection_mixture.slang"
#include "../contracts/scattering_backend.slang"

float nclsMerlResponseProxy(StructuredBuffer<float3> material, float3 wo, float3 wi)
{
    const float3 response = nclsEvalMerlReference(material, wo, wi);
    return max(max(response.x, response.y), response.z);
}

NclsReflectionMixture nclsMerlReflectionProposal(
    StructuredBuffer<float3> material,
    float3 wo)
{
    NclsReflectionMixture proposal = {};
    const float3 mirror = float3(-wo.x, -wo.y, wo.z);
    const float3 shoulder = nclsSafeNormalize(float3(mirror.xy + float2(0.10f, 0.0f), mirror.z));
    const float3 broad = nclsSafeNormalize(float3(mirror.xy + float2(0.35f, 0.17f), mirror.z));
    const float peakValue = nclsMerlResponseProxy(material, wo, mirror);
    const float shoulderValue = nclsMerlResponseProxy(material, wo, shoulder);
    const float broadValue = nclsMerlResponseProxy(material, wo, broad);

    proposal.cosineWeight = max(0.05f * max(broadValue, 1e-4f), 1e-5f);
    proposal.ggxWeights = float4(
        max(peakValue - shoulderValue, 0.0f),
        max(shoulderValue - broadValue, 0.0f),
        max(broadValue, 0.0f),
        max(0.25f * broadValue, 0.0f));
    proposal.alphaX = float4(0.01f, 0.05f, 0.20f, 0.55f);
    proposal.alphaY = proposal.alphaX;
    proposal.rotations = float4(0.0f);
    proposal.ggxCount = 4u;
    return proposal;
}

struct NclsMerlReferenceState : INclsScatteringState
{
    NclsScatteringContext context;
    StructuredBuffer<float3> material;
    NclsReflectionMixture proposal;

    NclsScatteringEval evaluate<S : ISampleGenerator>(float3 wiWorld, inout S sg)
    {
        NclsScatteringEval result = {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        if (wo.z <= NCLS_MIN_COS || wi.z <= NCLS_MIN_COS) return result;
        result.f = nclsEvalMerlReference(material, wo, wi) / wi.z;
        result.pdf = {
            nclsReflectionMixturePdf(proposal, wo, wi),
            nclsReflectionMixturePdf(nclsMerlReflectionProposal(material, wi), wi, wo)};
        result.eventFlags = (uint)NclsScatteringEvent::Reflection | (uint)NclsScatteringEvent::Glossy;
        result.valid = all(isfinite(result.f)) && all(result.f >= 0.0f) ? 1u : 0u;
        return result;
    }

    bool sample<S : ISampleGenerator>(out NclsScatteringSample result, inout S sg)
    {
        result = {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        uint component;
        const NclsDirectionalSample direction = nclsSampleReflectionMixture(
            proposal, wo, sampleNext2D(sg), component);
        if (direction.valid == 0u || direction.nullEvent != 0u) return false;
        result.wiWorld = nclsFrameToWorld(context.surface.shadingFrame, direction.direction);
        const NclsScatteringEval value = evaluate(result.wiWorld, sg);
        if (value.valid == 0u || value.pdf.forward <= 0.0f) return false;
        result.weight = value.f * abs(direction.direction.z) / value.pdf.forward;
        result.pdf = value.pdf; result.eta = 1.0f; result.eventFlags = value.eventFlags; result.valid = 1u;
        return true;
    }

    NclsScatteringPdf pdf(float3 wiWorld)
    {
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        return {
            nclsReflectionMixturePdf(proposal, wo, wi),
            nclsReflectionMixturePdf(nclsMerlReflectionProposal(material, wi), wi, wo)};
    }
};

struct NclsMerlReferenceBackend : INclsScatteringBackend
{
    typedef StructuredBuffer<float3> CompiledMaterial;
    typedef NclsMerlReferenceState State;
    State prepare(NclsScatteringContext context, CompiledMaterial material)
    {
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        return {context, material, nclsMerlReflectionProposal(material, wo)};
    }
};

#ifndef NCLS_REFERENCE_BACKEND_NO_PACKAGE_BINDING
typedef NclsMerlReferenceBackend NclsPackageBackend;
typedef StructuredBuffer<float3> NclsPackageCompiledMaterial;
typedef NclsMerlReferenceState NclsPackageState;
StructuredBuffer<float3> gNclsMeasuredBrdfTable;
NclsPackageBackend nclsCreatePackageBackend() { return {}; }
NclsPackageCompiledMaterial nclsLoadPackageMaterial(uint index) { return gNclsMeasuredBrdfTable; }
#endif

#endif
