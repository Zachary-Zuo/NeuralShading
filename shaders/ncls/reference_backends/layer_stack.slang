#ifndef NCLS_REFERENCE_BACKENDS_LAYER_STACK_SLANG
#define NCLS_REFERENCE_BACKENDS_LAYER_STACK_SLANG

// LayerStack random-walk reference 的统一 scattering contract 包装。
#include "../reference/random_walk_reference.slang"
#include "../scattering/cosine.slang"
#include "../contracts/scattering_backend.slang"

struct NclsLayerStackReferenceState : INclsScatteringState
{
    NclsScatteringContext context;
    NclsLayerStackIR material;

    NclsScatteringEval evaluate<S : ISampleGenerator>(float3 wiWorld, inout S sg)
    {
        NclsScatteringEval result = {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        if (wo.z <= NCLS_MIN_COS || wi.z <= NCLS_MIN_COS) return result;
        const float2 u = sampleNext2D(sg);
        const float3 response = nclsEvalRandomWalkReference(
            material, wo, wi, uint2(asuint(u.x), asuint(u.y)), 64u);
        result.f = response / wi.z;
        result.pdf.forward = nclsCosineHemispherePdf(wi);
        result.pdf.reverse = nclsCosineHemispherePdf(wo);
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
        result.pdf = value.pdf;
        result.eta = 1.0f;
        result.eventFlags = value.eventFlags;
        result.valid = 1u;
        return true;
    }

    NclsScatteringPdf pdf(float3 wiWorld)
    {
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        return {nclsCosineHemispherePdf(wi), nclsCosineHemispherePdf(wo)};
    }
};

struct NclsLayerStackReferenceBackend : INclsScatteringBackend
{
    typedef NclsLayerStackIR CompiledMaterial;
    typedef NclsLayerStackReferenceState State;
    State prepare(NclsScatteringContext context, CompiledMaterial material) { return {context, material}; }
};

typedef NclsLayerStackReferenceBackend NclsPackageBackend;
typedef NclsLayerStackIR NclsPackageCompiledMaterial;
typedef NclsLayerStackReferenceState NclsPackageState;
StructuredBuffer<NclsLayerStackIR> gNclsCompiledMaterials;
NclsPackageBackend nclsCreatePackageBackend() { return {}; }
NclsPackageCompiledMaterial nclsLoadPackageMaterial(uint index) { return gNclsCompiledMaterials[index]; }

#endif
