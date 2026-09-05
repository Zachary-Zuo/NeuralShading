#ifndef NCLS_METAL_BUDGETED_SLANG
#define NCLS_METAL_BUDGETED_SLANG

#include "../../contracts/scattering_backend.slang"
#include "../../scattering/frame.slang"

StructuredBuffer<uint> gNclsRuntimeWeights;
StructuredBuffer<uint> gNclsCompiledMaterials;
Texture2D<float4> gNclsMetalBudgetedDetail;
Texture2D<float4> gNclsMetalBudgetedContext;
SamplerState gNclsMetalBudgetedSampler;

#include "metal_budgeted_asset.slang"
#include "metal_budgeted_evaluator.slang"
#include "metal_budgeted_sampler.slang"

struct NclsMetalBudgetedContractState : INclsScatteringState
{
    NclsScatteringContext context;
    NclsMetalBudgetedCompiledMaterial material;
    NclsMetalBudgetedPrepared prepared;

    NclsScatteringEval evaluate<S : ISampleGenerator>(
        float3 wiWorld, inout S sampleGenerator)
    {
        NclsScatteringEval result = {};
        const float3 wo = nclsFrameToLocal(
            context.surface.shadingFrame, context.woWorld);
        const float3 wi = nclsFrameToLocal(
            context.surface.shadingFrame, wiWorld);
        result.f = nclsMetalBudgetedEvaluateF(prepared, wo, wi);
        result.eventFlags = (uint)NclsScatteringEvent::Reflection
            | (uint)NclsScatteringEvent::Glossy;
        result.valid = prepared.valid != 0u && all(isfinite(result.f)) ? 1u : 0u;
        return result;
    }

    bool sample<S : ISampleGenerator>(
        out NclsScatteringSample result, inout S sampleGenerator)
    {
        result = {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const uint u0 = sampleGenerator.next();
        const uint u1 = sampleGenerator.next();
        const float2 u = float2(u0 >> 8u, u1 >> 8u) * 0x1p-24f;
        float3 wi;
        if (!nclsMetalBudgetedSampleProposal(prepared, wo, u, wi, result.pdf)) return false;
        result.wiWorld = nclsFrameToWorld(context.surface.shadingFrame, wi);
        result.weight = nclsMetalBudgetedEvaluateF(prepared, wo, wi) * max(wi.z, 0.0f)
            / max(result.pdf.forward, 1e-12f);
        result.eta = 1.0f;
        result.eventFlags = (uint)NclsScatteringEvent::Reflection | (uint)NclsScatteringEvent::Glossy;
        result.valid = all(isfinite(result.weight)) ? 1u : 0u;
        return result.valid != 0u;
    }

    NclsScatteringPdf pdf(float3 wiWorld)
    {
        return nclsMetalBudgetedProposalPdf(prepared,
            nclsFrameToLocal(context.surface.shadingFrame, context.woWorld),
            nclsFrameToLocal(context.surface.shadingFrame, wiWorld));
    }
};

struct NclsMetalBudgetedBackend : INclsScatteringBackend
{
    typedef NclsMetalBudgetedCompiledMaterial CompiledMaterial;
    typedef NclsMetalBudgetedContractState State;

    State prepare(
        NclsScatteringContext context,
        NclsMetalBudgetedCompiledMaterial material)
    {
        return {context, material, nclsMetalBudgetedPrepare(context, material)};
    }
};

NclsMetalBudgetedPackedState nclsPackMethodState(
    NclsMetalBudgetedContractState state)
{
    NclsMetalBudgetedPackedState packed = {};
    float values[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    [unroll]
    for (uint index = 0u; index < 24u; ++index)
        values[index] = state.prepared.semantic[index];
    nclsMetalBudgetedPackHalfArray(packed, 0u, 24u, values);
    [unroll]
    for (uint index = 0u; index < 8u; ++index)
        values[index] = state.prepared.viewState[index];
    nclsMetalBudgetedPackHalfArray(packed, 12u, 8u, values);
    [unroll]
    for (uint index = 0u; index < 8u; ++index)
        values[index] = state.prepared.compactFrame[index];
    nclsMetalBudgetedPackHalfArray(packed, 16u, 8u, values);
    [unroll]
    for (uint index = 0u; index < 16u; ++index)
        values[index] = state.prepared.lobes[index];
    nclsMetalBudgetedPackHalfArray(packed, 20u, 16u, values);
    [unroll]
    for (uint index = 0u; index < 12u; ++index)
        values[index] = state.prepared.proposal[index];
    nclsMetalBudgetedPackHalfArray(packed, 28u, 12u, values);
    [unroll]
    for (uint index = 0u; index < 4u; ++index)
        values[index] = state.prepared.accessState[index];
    nclsMetalBudgetedPackHalfArray(packed, 34u, 4u, values);
    [unroll]
    for (uint index = 0u; index < 4u; ++index)
        packed.words[36u + index] = state.prepared.identityAndFlags[index];
    packed.words[39] = (packed.words[39] & 0xffff0000u)
        | (state.prepared.valid != 0u ? 1u : 0u);
    return packed;
}

NclsMetalBudgetedContractState nclsUnpackMethodState(
    NclsScatteringContext context,
    NclsMetalBudgetedCompiledMaterial material,
    NclsMetalBudgetedPackedState packed,
    StructuredBuffer<uint> weights)
{
    NclsMetalBudgetedContractState state = {};
    state.context = context;
    state.material = material;
    [unroll]
    for (uint index = 0u; index < 24u; ++index)
        state.prepared.semantic[index] = nclsMetalBudgetedUnpackHalf(packed, index);
    [unroll]
    for (uint index = 0u; index < 8u; ++index)
        state.prepared.viewState[index] = nclsMetalBudgetedUnpackHalf(packed, 24u + index);
    [unroll]
    for (uint index = 0u; index < 8u; ++index)
        state.prepared.compactFrame[index] = nclsMetalBudgetedUnpackHalf(packed, 32u + index);
    [unroll]
    for (uint index = 0u; index < 16u; ++index)
        state.prepared.lobes[index] = nclsMetalBudgetedUnpackHalf(packed, 40u + index);
    [unroll]
    for (uint index = 0u; index < 12u; ++index)
        state.prepared.proposal[index] = nclsMetalBudgetedUnpackHalf(packed, 56u + index);
    // The legal minimum 0.015 rounds down in binary16. Restore that boundary
    // after storage quantization, before validating the proposal contract.
    [unroll]
    for (uint component = 0u; component < 3u; ++component)
    {
        [unroll]
        for (uint axis = 1u; axis <= 2u; ++axis)
            if (state.prepared.proposal[4u * component + axis] == f16tof32(f32tof16(0.015f)))
                state.prepared.proposal[4u * component + axis] = 0.015f;
    }
    [unroll]
    for (uint index = 0u; index < 4u; ++index)
        state.prepared.accessState[index] = nclsMetalBudgetedUnpackHalf(packed, 68u + index);
    [unroll]
    for (uint index = 0u; index < 4u; ++index)
        state.prepared.identityAndFlags[index] = packed.words[36u + index];
    state.prepared.valid = packed.words[39] & 0xffffu;
    const float2 primarySlope = float2(
        state.prepared.compactFrame[0], state.prepared.compactFrame[1]);
    const float2 secondarySlope = float2(
        state.prepared.compactFrame[2], state.prepared.compactFrame[3]);
    state.prepared.frames[0] = nclsMetalBudgetedLocalFrame(
        primarySlope, state.prepared.lobes[6]);
    state.prepared.frames[1] = nclsMetalBudgetedLocalFrame(
        secondarySlope, state.prepared.lobes[14]);
    return state;
}

typedef NclsMetalBudgetedBackend NclsPackageBackend;
typedef NclsMetalBudgetedCompiledMaterial NclsPackageCompiledMaterial;
typedef NclsMetalBudgetedContractState NclsPackageState;
typedef NclsMetalBudgetedCompiledMaterial NclsMethodCompiledMaterial;
typedef NclsMetalBudgetedContractState NclsMethodState;
typedef NclsMetalBudgetedPackedState NclsMethodPackedState;

NclsPackageBackend nclsCreatePackageBackend() { return {}; }
NclsPackageCompiledMaterial nclsLoadPackageMaterial(uint index)
{
    NclsPackageCompiledMaterial result;
    result.baseWord = index * NCLS_METAL_BUDGETED_COMPILED_WORD_COUNT;
    return result;
}

#endif
