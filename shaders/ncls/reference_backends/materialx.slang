#ifndef NCLS_REFERENCE_BACKENDS_MATERIALX_SLANG
#define NCLS_REFERENCE_BACKENDS_MATERIALX_SLANG

// MaterialX standard_surface reference 的统一 scattering contract 包装。
#include "../reference/materialx_standard_surface_reference.slang"
#include "../scattering/reflection_mixture.slang"
#include "../contracts/scattering_backend.slang"

struct NclsMaterialXCompiledMaterial
{
    StructuredBuffer<float> inputs;
    Texture2D<float4> baseColor;
    Texture2D<float4> roughness;
    Texture2D<float4> metalness;
    Texture2D<float4> normalMap;
};

struct NclsMaterialXReferenceState : INclsScatteringState
{
    NclsScatteringContext context;
    NclsMaterialXResolvedSurface surface;
    NclsReflectionMixture proposal;
    // Resource handles stay last. Slang/DXC lowers them out of value aggregates;
    // placing prepared values first keeps the state layout stable when a scene
    // dispatcher copies only the resource-free portion.
    NclsMaterialXCompiledMaterial material;

    NclsScatteringEval evaluate<S : ISampleGenerator>(float3 wiWorld, inout S sg)
    {
        NclsScatteringEval result = {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        if (wo.z <= NCLS_MIN_COS || wi.z <= NCLS_MIN_COS) return result;
        result.f = nclsEvalMaterialXStandardSurface(material.inputs, surface, wo, wi) / wi.z;
        result.pdf = {
            nclsReflectionMixturePdf(proposal, wo, wi),
            nclsReflectionMixturePdf(proposal, wi, wo)};
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
            nclsReflectionMixturePdf(proposal, wi, wo)};
    }
};

struct NclsMaterialXReferenceBackend : INclsScatteringBackend
{
    typedef NclsMaterialXCompiledMaterial CompiledMaterial;
    typedef NclsMaterialXReferenceState State;
    SamplerState materialSampler;
    State prepare(NclsScatteringContext context, CompiledMaterial material)
    {
        context.surface.shadingFrame.normal = nclsMxForwardFacingNormal(
            nclsMaterialXShadingNormal(
                material.inputs, material.normalMap, materialSampler,
                context.surface.uv, context.surface.uvDx, context.surface.uvDy,
                context.surface.shadingFrame.normal, context.surface.shadingFrame.tangent),
            context.woWorld);
        context.surface.shadingFrame.tangent -= context.surface.shadingFrame.normal
            * dot(context.surface.shadingFrame.normal, context.surface.shadingFrame.tangent);
        context.surface.shadingFrame.tangent = dot(
            context.surface.shadingFrame.tangent, context.surface.shadingFrame.tangent) > 1e-10f
            ? normalize(context.surface.shadingFrame.tangent)
            : nclsDirectionalFrameFromNormal(context.surface.shadingFrame.normal).tangent;
        context.surface.shadingFrame.bitangent = cross(
            context.surface.shadingFrame.normal, context.surface.shadingFrame.tangent);
        const NclsMaterialXResolvedSurface surface = nclsResolveMaterialXSurface(
            material.inputs, material.baseColor, material.roughness, material.metalness,
            materialSampler, context.surface.uv, context.surface.uvDx, context.surface.uvDy);
        NclsReflectionMixture proposal = {};
        const float diffuseWeight = max(material.inputs[0] * (1.0f - surface.metalness), 0.0f);
        const float specularWeight = max(material.inputs[8] + surface.metalness, 0.0f);
        const float2 alpha = nclsMxRoughnessAnisotropy(
            surface.specularRoughness, material.inputs[15]);
        proposal.cosineWeight = max(diffuseWeight, 0.05f);
        proposal.ggxWeights[0] = max(specularWeight, 0.05f);
        proposal.alphaX[0] = max(alpha.x, 1e-4f);
        proposal.alphaY[0] = max(alpha.y, 1e-4f);
        proposal.rotations[0] = material.inputs[16] * (2.0f * NCLS_PI);
        proposal.ggxCount = 1u;
        return {context, surface, proposal, material};
    }
};

#ifndef NCLS_REFERENCE_BACKEND_NO_PACKAGE_BINDING
typedef NclsMaterialXReferenceBackend NclsPackageBackend;
typedef NclsMaterialXCompiledMaterial NclsPackageCompiledMaterial;
typedef NclsMaterialXReferenceState NclsPackageState;
StructuredBuffer<float> gNclsMaterialXInputs;
Texture2D<float4> gNclsMaterialXBaseColor;
Texture2D<float4> gNclsMaterialXRoughness;
Texture2D<float4> gNclsMaterialXMetalness;
Texture2D<float4> gNclsMaterialXNormalMap;
SamplerState gNclsMaterialXSampler;
NclsPackageBackend nclsCreatePackageBackend() { return {gNclsMaterialXSampler}; }
NclsPackageCompiledMaterial nclsLoadPackageMaterial(uint index)
{
    return {
        gNclsMaterialXInputs,
        gNclsMaterialXBaseColor,
        gNclsMaterialXRoughness,
        gNclsMaterialXMetalness,
        gNclsMaterialXNormalMap};
}
#endif

#endif
