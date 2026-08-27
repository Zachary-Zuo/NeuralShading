#ifndef NCLS_VIEWER_SCENE_REFERENCE_PROGRAM_SLANG
#define NCLS_VIEWER_SCENE_REFERENCE_PROGRAM_SLANG

#ifndef NCLS_REFERENCE_FAMILY_MASK
#define NCLS_REFERENCE_FAMILY_MASK 31
#endif
#ifndef NCLS_MAX_SCENE_MATERIALS
#define NCLS_MAX_SCENE_MATERIALS 64
#endif

#include "../../../shaders/ncls/contracts/layer_stack_ir.slang"

#define NCLS_REFERENCE_BACKEND_NO_PACKAGE_BINDING 1
#if (NCLS_REFERENCE_FAMILY_MASK & 1) != 0
#include "../../../shaders/ncls/reference_backends/layer_stack.slang"
#endif
#if (NCLS_REFERENCE_FAMILY_MASK & 2) != 0
#include "../../../shaders/ncls/reference_backends/merl.slang"
#endif
#if (NCLS_REFERENCE_FAMILY_MASK & 4) != 0
#include "../../../shaders/ncls/reference_backends/openpbr.slang"
#endif
#if (NCLS_REFERENCE_FAMILY_MASK & 8) != 0
#include "../../../shaders/ncls/reference_backends/materialx.slang"
#endif
#if (NCLS_REFERENCE_FAMILY_MASK & 16) != 0
#include "../../../shaders/ncls/reference_backends/mdl.slang"
#endif
#undef NCLS_REFERENCE_BACKEND_NO_PACKAGE_BINDING

StructuredBuffer<uint4> gMaterialMetadata;
StructuredBuffer<NclsLayerStackIR> gLayerStacks[NCLS_MAX_SCENE_MATERIALS];
StructuredBuffer<float3> gMerlBrdfs[NCLS_MAX_SCENE_MATERIALS];
StructuredBuffer<float> gOpenPbrInputs[NCLS_MAX_SCENE_MATERIALS];
StructuredBuffer<float> gMaterialXInputs[NCLS_MAX_SCENE_MATERIALS];
Texture2D<float4> gMaterialXBaseColors[NCLS_MAX_SCENE_MATERIALS];
Texture2D<float4> gMaterialXRoughnesses[NCLS_MAX_SCENE_MATERIALS];
Texture2D<float4> gMaterialXMetalnesses[NCLS_MAX_SCENE_MATERIALS];
Texture2D<float4> gMaterialXNormalMaps[NCLS_MAX_SCENE_MATERIALS];

float3 nclsSceneOpenPbrAcesCgToLinearSrgb(float3 value)
{
    return float3(
        1.70505099f * value.x - 0.62179212f * value.y - 0.08325887f * value.z,
        -0.13025642f * value.x + 1.14080474f * value.y - 0.01054832f * value.z,
        -0.02400336f * value.x - 0.12896898f * value.y + 1.15297233f * value.z);
}

float3 nclsSceneOpenPbrLinearSrgbToAcesCg(float3 value)
{
    return float3(
        0.61309740f * value.x + 0.33952315f * value.y + 0.04737945f * value.z,
        0.07019372f * value.x + 0.91635388f * value.y + 0.01345240f * value.z,
        0.02061560f * value.x + 0.10956978f * value.y + 0.86981464f * value.z);
}

struct NclsSceneReferenceCompiledMaterial
{
    uint materialId;
    uint family;
    uint openPbrColorSpace;
};

struct NclsSceneReferenceState : INclsScatteringState
{
    uint family;
    uint openPbrColorSpace;
    uint materialId;
    NclsScatteringContext context;
    #if (NCLS_REFERENCE_FAMILY_MASK & 1) != 0
    NclsLayerStackReferenceState layerStack;
    #endif
    #if (NCLS_REFERENCE_FAMILY_MASK & 2) != 0
    NclsReflectionMixture merlProposal;
    #endif
    #if (NCLS_REFERENCE_FAMILY_MASK & 4) != 0
    OpenPBR_PreparedBsdf openPbrPrepared;
    #endif
    #if (NCLS_REFERENCE_FAMILY_MASK & 8) != 0
    NclsMaterialXResolvedSurface materialXSurface;
    NclsReflectionMixture materialXProposal;
    #endif

    NclsScatteringEval evaluate<S : ISampleGenerator>(float3 wiWorld, inout S sg)
    {
        #if (NCLS_REFERENCE_FAMILY_MASK & 1) != 0
        if (family == 0u) return layerStack.evaluate(wiWorld, sg);
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 2) != 0
        if (family == 1u)
        {
            NclsMerlReferenceState concrete = {
                context, gMerlBrdfs[materialId], merlProposal};
            return concrete.evaluate(wiWorld, sg);
        }
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 4) != 0
        if (family == 2u)
        {
            NclsOpenPbrReferenceState concrete = {
                context, gOpenPbrInputs[materialId], openPbrPrepared};
            return concrete.evaluate(wiWorld, sg);
        }
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 8) != 0
        if (family == 3u)
        {
            const NclsMaterialXCompiledMaterial compiled = {
                gMaterialXInputs[materialId], gMaterialXBaseColors[materialId],
                gMaterialXRoughnesses[materialId], gMaterialXMetalnesses[materialId],
                gMaterialXNormalMaps[materialId]};
            NclsMaterialXReferenceState concrete = {
                context, materialXSurface, materialXProposal, compiled};
            return concrete.evaluate(wiWorld, sg);
        }
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 16) != 0
        if (family == 4u)
        {
            NclsMdlReferenceState concrete = {context};
            return concrete.evaluate(wiWorld, sg);
        }
        #endif
        NclsScatteringEval result = {};
        return result;
    }

    bool sample<S : ISampleGenerator>(out NclsScatteringSample result, inout S sg)
    {
        result = {};
        bool valid = false;
        #if (NCLS_REFERENCE_FAMILY_MASK & 1) != 0
        if (family == 0u) valid = layerStack.sample(result, sg);
        else
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 2) != 0
        if (family == 1u)
        {
            NclsMerlReferenceState concrete = {
                context, gMerlBrdfs[materialId], merlProposal};
            valid = concrete.sample(result, sg);
        }
        else
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 4) != 0
        if (family == 2u)
        {
            NclsOpenPbrReferenceState concrete = {
                context, gOpenPbrInputs[materialId], openPbrPrepared};
            valid = concrete.sample(result, sg);
        }
        else
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 8) != 0
        if (family == 3u)
        {
            const NclsMaterialXCompiledMaterial compiled = {
                gMaterialXInputs[materialId], gMaterialXBaseColors[materialId],
                gMaterialXRoughnesses[materialId], gMaterialXMetalnesses[materialId],
                gMaterialXNormalMaps[materialId]};
            NclsMaterialXReferenceState concrete = {
                context, materialXSurface, materialXProposal, compiled};
            valid = concrete.sample(result, sg);
        }
        else
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 16) != 0
        if (family == 4u)
        {
            NclsMdlReferenceState concrete = {context};
            valid = concrete.sample(result, sg);
        }
        else
        #endif
        { return false; }

        #if (NCLS_REFERENCE_FAMILY_MASK & 4) != 0
        if (valid && family == 2u && openPbrColorSpace != 0u)
            result.weight = nclsSceneOpenPbrAcesCgToLinearSrgb(result.weight);
        #endif
        return valid;
    }

    NclsScatteringPdf pdf(float3 wiWorld)
    {
        #if (NCLS_REFERENCE_FAMILY_MASK & 1) != 0
        if (family == 0u) return layerStack.pdf(wiWorld);
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 2) != 0
        if (family == 1u)
        {
            NclsMerlReferenceState concrete = {
                context, gMerlBrdfs[materialId], merlProposal};
            return concrete.pdf(wiWorld);
        }
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 4) != 0
        if (family == 2u)
        {
            NclsOpenPbrReferenceState concrete = {
                context, gOpenPbrInputs[materialId], openPbrPrepared};
            return concrete.pdf(wiWorld);
        }
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 8) != 0
        if (family == 3u)
        {
            const NclsMaterialXCompiledMaterial compiled = {
                gMaterialXInputs[materialId], gMaterialXBaseColors[materialId],
                gMaterialXRoughnesses[materialId], gMaterialXMetalnesses[materialId],
                gMaterialXNormalMaps[materialId]};
            NclsMaterialXReferenceState concrete = {
                context, materialXSurface, materialXProposal, compiled};
            return concrete.pdf(wiWorld);
        }
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 16) != 0
        if (family == 4u)
        {
            NclsMdlReferenceState concrete = {context};
            return concrete.pdf(wiWorld);
        }
        #endif
        return {0.0f, 0.0f};
    }

    NclsSurfaceInteraction surfaceInteraction()
    {
        return context.surface;
    }

    float3 emission()
    {
        #if (NCLS_REFERENCE_FAMILY_MASK & 4) != 0
        if (family == 2u)
        {
            const float3 wo = nclsFrameToLocal(
                context.surface.shadingFrame, context.woWorld);
            const float3 value = nclsOpenPbrEmission(gOpenPbrInputs[materialId], wo);
            return openPbrColorSpace == 0u
                ? value : nclsSceneOpenPbrAcesCgToLinearSrgb(value);
        }
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 8) != 0
        if (family == 3u) return nclsMaterialXEmission(gMaterialXInputs[materialId]);
        #endif
        return float3(0.0f);
    }

    float3 illuminate(float3 response, float3 linearSrgbRadiance)
    {
        #if (NCLS_REFERENCE_FAMILY_MASK & 4) != 0
        if (family == 2u && openPbrColorSpace != 0u)
            return nclsSceneOpenPbrAcesCgToLinearSrgb(
                response * nclsSceneOpenPbrLinearSrgbToAcesCg(linearSrgbRadiance));
        #endif
        return response * linearSrgbRadiance;
    }
};

struct NclsSceneReferenceBackend : INclsScatteringBackend
{
    typedef NclsSceneReferenceCompiledMaterial CompiledMaterial;
    typedef NclsSceneReferenceState State;
    uint maximumLayerWalkDepth;
    SamplerState materialXSampler;

    State prepare(NclsScatteringContext context, CompiledMaterial material)
    {
        State result = {};
        result.family = material.family;
        result.openPbrColorSpace = material.openPbrColorSpace;
        result.materialId = material.materialId;
        result.context = context;
        #if (NCLS_REFERENCE_FAMILY_MASK & 1) != 0
        if (material.family == 0u)
        {
            const NclsLayerStackReferenceBackend backend = {maximumLayerWalkDepth};
            result.layerStack = backend.prepare(context, gLayerStacks[material.materialId][0]);
            result.context = result.layerStack.context;
            return result;
        }
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 2) != 0
        if (material.family == 1u)
        {
            const NclsMerlReferenceBackend backend = {};
            const NclsMerlReferenceState concrete = backend.prepare(
                context, gMerlBrdfs[material.materialId]);
            result.context = concrete.context;
            result.merlProposal = concrete.proposal;
            return result;
        }
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 4) != 0
        if (material.family == 2u)
        {
            const NclsOpenPbrReferenceBackend backend = {};
            const NclsOpenPbrReferenceState concrete = backend.prepare(
                context, gOpenPbrInputs[material.materialId]);
            result.context = concrete.context;
            result.openPbrPrepared = concrete.prepared;
            return result;
        }
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 8) != 0
        if (material.family == 3u)
        {
            const NclsMaterialXReferenceBackend backend = {materialXSampler};
            const NclsMaterialXCompiledMaterial compiled = {
                gMaterialXInputs[material.materialId],
                gMaterialXBaseColors[material.materialId],
                gMaterialXRoughnesses[material.materialId],
                gMaterialXMetalnesses[material.materialId],
                gMaterialXNormalMaps[material.materialId]};
            const NclsMaterialXReferenceState concrete = backend.prepare(context, compiled);
            result.context = concrete.context;
            result.materialXSurface = concrete.surface;
            result.materialXProposal = concrete.proposal;
            return result;
        }
        #endif
        #if (NCLS_REFERENCE_FAMILY_MASK & 16) != 0
        if (material.family == 4u)
        {
            const NclsMdlReferenceBackend backend = {};
            const NclsMdlReferenceState concrete = backend.prepare(context, {0u});
            result.context = concrete.context;
            return result;
        }
        #endif
        return result;
    }
};

NclsSceneReferenceCompiledMaterial nclsLoadSceneReferenceMaterial(
    uint materialId,
    uint materialCount)
{
    const uint boundedId = min(materialId, materialCount - 1u);
    const uint4 metadata = gMaterialMetadata[boundedId];
    return {boundedId, metadata.x, metadata.y};
}

#endif
