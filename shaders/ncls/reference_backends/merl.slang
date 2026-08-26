#ifndef NCLS_REFERENCE_BACKENDS_MERL_SLANG
#define NCLS_REFERENCE_BACKENDS_MERL_SLANG

// MERL measured BRDF reference 的统一 scattering contract 包装。
#include "../reference/merl_reference.slang"
#include "../scattering/cosine.slang"
#include "../contracts/scattering_backend.slang"

struct NclsMerlReferenceState : INclsScatteringState
{
    NclsScatteringContext context;
    StructuredBuffer<float3> material;

    NclsScatteringEval evaluate<S : ISampleGenerator>(float3 wiWorld, inout S sg)
    {
        NclsScatteringEval result = {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        if (wo.z <= NCLS_MIN_COS || wi.z <= NCLS_MIN_COS) return result;
        result.f = nclsEvalMerlReference(material, wo, wi) / wi.z;
        result.pdf = {nclsCosineHemispherePdf(wi), nclsCosineHemispherePdf(wo)};
        result.eventFlags = (uint)NclsScatteringEvent::Reflection | (uint)NclsScatteringEvent::Glossy;
        result.valid = all(isfinite(result.f)) && all(result.f >= 0.0f) ? 1u : 0u;
        return result;
    }

    bool sample<S : ISampleGenerator>(out NclsScatteringSample result, inout S sg)
    {
        result = {};
        const NclsDirectionalSample direction = nclsSampleCosineDirection(sampleNext2D(sg));
        if (direction.valid == 0u || direction.nullEvent != 0u) return false;
        result.wiWorld = nclsFrameToWorld(context.surface.shadingFrame, direction.direction);
        const NclsScatteringEval value = evaluate(result.wiWorld, sg);
        if (value.valid == 0u || value.pdf.forward <= 0.0f) return false;
        result.weight = value.f * direction.direction.z / value.pdf.forward;
        result.pdf = value.pdf; result.eta = 1.0f; result.eventFlags = value.eventFlags; result.valid = 1u;
        return true;
    }

    NclsScatteringPdf pdf(float3 wiWorld)
    {
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        return {nclsCosineHemispherePdf(wi), nclsCosineHemispherePdf(wo)};
    }
};

struct NclsMerlReferenceBackend : INclsScatteringBackend
{
    typedef StructuredBuffer<float3> CompiledMaterial;
    typedef NclsMerlReferenceState State;
    State prepare(NclsScatteringContext context, CompiledMaterial material) { return {context, material}; }
};

typedef NclsMerlReferenceBackend NclsPackageBackend;
typedef StructuredBuffer<float3> NclsPackageCompiledMaterial;
typedef NclsMerlReferenceState NclsPackageState;
StructuredBuffer<float3> gNclsMeasuredBrdfTable;
NclsPackageBackend nclsCreatePackageBackend() { return {}; }
NclsPackageCompiledMaterial nclsLoadPackageMaterial(uint index) { return gNclsMeasuredBrdfTable; }

#endif
