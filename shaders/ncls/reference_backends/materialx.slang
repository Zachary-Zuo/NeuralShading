#ifndef NCLS_REFERENCE_BACKENDS_MATERIALX_SLANG
#define NCLS_REFERENCE_BACKENDS_MATERIALX_SLANG

// MaterialX standard_surface reference 的统一 scattering contract 包装。
#include "../reference/materialx_standard_surface_reference.slang"
#include "../scattering/cosine.slang"
#include "../contracts/scattering_backend.slang"

struct NclsMaterialXCompiledMaterial
{
    StructuredBuffer<float> inputs;
    Texture2D<float4> baseColor;
    Texture2D<float4> roughness;
    Texture2D<float4> metalness;
};

struct NclsMaterialXReferenceState : INclsScatteringState
{
    NclsScatteringContext context;
    NclsMaterialXCompiledMaterial material;
    NclsMaterialXResolvedSurface surface;

    NclsScatteringEval evaluate<S : ISampleGenerator>(float3 wiWorld, inout S sg)
    {
        NclsScatteringEval result = {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        if (wo.z <= NCLS_MIN_COS || wi.z <= NCLS_MIN_COS) return result;
        result.f = nclsEvalMaterialXStandardSurface(material.inputs, surface, wo, wi) / wi.z;
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

struct NclsMaterialXReferenceBackend : INclsScatteringBackend
{
    typedef NclsMaterialXCompiledMaterial CompiledMaterial;
    typedef NclsMaterialXReferenceState State;
    SamplerState materialSampler;
    State prepare(NclsScatteringContext context, CompiledMaterial material)
    {
        const NclsMaterialXResolvedSurface surface = nclsResolveMaterialXSurface(
            material.inputs, material.baseColor, material.roughness, material.metalness,
            materialSampler, context.surface.uv, context.surface.uvDx, context.surface.uvDy);
        return {context, material, surface};
    }
};

typedef NclsMaterialXReferenceBackend NclsPackageBackend;
typedef NclsMaterialXCompiledMaterial NclsPackageCompiledMaterial;
typedef NclsMaterialXReferenceState NclsPackageState;
StructuredBuffer<float> gNclsMaterialXInputs;
Texture2D<float4> gNclsMaterialXBaseColor;
Texture2D<float4> gNclsMaterialXRoughness;
Texture2D<float4> gNclsMaterialXMetalness;
SamplerState gNclsMaterialXSampler;
NclsPackageBackend nclsCreatePackageBackend() { return {gNclsMaterialXSampler}; }
NclsPackageCompiledMaterial nclsLoadPackageMaterial(uint index)
{
    return {gNclsMaterialXInputs, gNclsMaterialXBaseColor, gNclsMaterialXRoughness, gNclsMaterialXMetalness};
}

#endif
