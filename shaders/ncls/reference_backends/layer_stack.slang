#ifndef NCLS_REFERENCE_BACKENDS_LAYER_STACK_SLANG
#define NCLS_REFERENCE_BACKENDS_LAYER_STACK_SLANG

// LayerStack random-walk reference 的统一 scattering contract 包装。
#include "../reference/random_walk_reference.slang"
#include "../scattering/reflection_mixture.slang"
#include "../contracts/scattering_backend.slang"

NclsReflectionMixture nclsLayerStackReflectionProposal(NclsLayerStackIR material)
{
    NclsReflectionMixture proposal = {};
    proposal.cosineWeight = 0.125f;
    proposal.ggxCount = min(material.interfaceCount, NCLS_REFLECTION_GGX_COMPONENT_COUNT);
    [unroll]
    for (uint i = 0u; i < NCLS_REFLECTION_GGX_COMPONENT_COUNT; ++i)
    {
        if (i >= proposal.ggxCount) break;
        const NclsLayerInterfaceIR layer = material.interfaces[i];
        proposal.alphaX[i] = max(layer.alphaX, 1e-3f);
        proposal.alphaY[i] = max(layer.alphaY, 1e-3f);
        proposal.rotations[i] = layer.tangentRotation;
        const float3 color = float3(layer.colorR, layer.colorG, layer.colorB);
        const float colorProxy = max(max(color.x, color.y), color.z);
        proposal.ggxWeights[i] = layer.kind == (uint)NclsInterfaceKind::Diffuse
            ? 0.0f
            : max(colorProxy, 0.25f);
    }
    return proposal;
}

struct NclsLayerStackReferenceState : INclsScatteringState
{
    NclsScatteringContext context;
    NclsLayerStackIR material;
    NclsReflectionMixture proposal;
    uint maximumWalkDepth;

    NclsScatteringEval evaluate<S : ISampleGenerator>(float3 wiWorld, inout S sg)
    {
        NclsScatteringEval result = {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        if (wo.z <= NCLS_MIN_COS || wi.z <= NCLS_MIN_COS) return result;
        const float2 u = sampleNext2D(sg);
        const float3 response = nclsEvalRandomWalkReference(
            material, wo, wi, uint2(asuint(u.x), asuint(u.y)), maximumWalkDepth);
        result.f = response / wi.z;
        if (material.interfaceCount == 1u)
        {
            const NclsLayerInterfaceIR layer = material.interfaces[0];
            const float etaT = layer.kind == (uint)NclsInterfaceKind::RoughDielectric
                ? layer.relativeIor : 1.0f;
            result.pdf.forward = nclsInterfacePdf(
                layer, wo, wi, 1.0f, etaT, NCLS_SAMPLE_REFLECTION);
            result.pdf.reverse = nclsInterfacePdf(
                layer, wi, wo, 1.0f, etaT, NCLS_SAMPLE_REFLECTION);
        }
        else
        {
            result.pdf.forward = nclsReflectionMixturePdf(proposal, wo, wi);
            result.pdf.reverse = nclsReflectionMixturePdf(proposal, wi, wo);
        }
        result.eventFlags = (uint)NclsScatteringEvent::Reflection | (uint)NclsScatteringEvent::Glossy;
        result.valid = all(isfinite(result.f)) && all(result.f >= 0.0f) ? 1u : 0u;
        return result;
    }

    bool sample<S : ISampleGenerator>(out NclsScatteringSample result, inout S sg)
    {
        result = {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        NclsDirectionalSample direction = {};
        if (material.interfaceCount == 1u)
        {
            const float2 u = sampleNext2D(sg);
            NclsRng rng = NclsRng(uint2(asuint(u.x), asuint(u.y)));
            const NclsLayerInterfaceIR layer = material.interfaces[0];
            const float etaT = layer.kind == (uint)NclsInterfaceKind::RoughDielectric
                ? layer.relativeIor : 1.0f;
            const LayerBsdfSample sampled = nclsSampleInterfaceTransport(
                layer, wo, 1.0f, etaT, NCLS_SAMPLE_REFLECTION,
                context.transportMode == (uint)NclsTransportMode::Radiance, rng);
            if (sampled.valid == 0u) return false;
            direction = nclsContinuousDirectionalSample(sampled.direction, sampled.pdf);
        }
        else
        {
            uint component;
            direction = nclsSampleReflectionMixture(proposal, wo, sampleNext2D(sg), component);
        }
        if (direction.valid == 0u || direction.nullEvent != 0u) return false;
        result.wiWorld = nclsFrameToWorld(context.surface.shadingFrame, direction.direction);
        const NclsScatteringEval value = evaluate(result.wiWorld, sg);
        if (value.valid == 0u || value.pdf.forward <= 0.0f) return false;
        result.weight = value.f * abs(direction.direction.z) / value.pdf.forward;
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
        if (material.interfaceCount == 1u)
        {
            const NclsLayerInterfaceIR layer = material.interfaces[0];
            const float etaT = layer.kind == (uint)NclsInterfaceKind::RoughDielectric
                ? layer.relativeIor : 1.0f;
            return {
                nclsInterfacePdf(layer, wo, wi, 1.0f, etaT, NCLS_SAMPLE_REFLECTION),
                nclsInterfacePdf(layer, wi, wo, 1.0f, etaT, NCLS_SAMPLE_REFLECTION)};
        }
        return {
            nclsReflectionMixturePdf(proposal, wo, wi),
            nclsReflectionMixturePdf(proposal, wi, wo)};
    }
};

struct NclsLayerStackReferenceBackend : INclsScatteringBackend
{
    typedef NclsLayerStackIR CompiledMaterial;
    typedef NclsLayerStackReferenceState State;
    uint maximumWalkDepth;
    State prepare(NclsScatteringContext context, CompiledMaterial material)
    {
        return {context, material, nclsLayerStackReflectionProposal(material), maximumWalkDepth};
    }
};

#ifndef NCLS_REFERENCE_BACKEND_NO_PACKAGE_BINDING
typedef NclsLayerStackReferenceBackend NclsPackageBackend;
typedef NclsLayerStackIR NclsPackageCompiledMaterial;
typedef NclsLayerStackReferenceState NclsPackageState;
StructuredBuffer<NclsLayerStackIR> gNclsCompiledMaterials;
NclsPackageBackend nclsCreatePackageBackend() { return {64u}; }
NclsPackageCompiledMaterial nclsLoadPackageMaterial(uint index) { return gNclsCompiledMaterials[index]; }
#endif

#endif
