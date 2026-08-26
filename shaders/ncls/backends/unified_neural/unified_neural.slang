#ifndef NCLS_UNIFIED_NEURAL_SLANG
#define NCLS_UNIFIED_NEURAL_SLANG

// 03 candidate 的公共散射合同包装；所有 offset 由 MethodBundle reflection 注入。
#include "unified_neural_pack.slang"
#include "../../contracts/scattering_backend.slang"

#ifndef NCLS_UNIFIED_PREPARE_W0_OFFSET
#error NCLS_UNIFIED_PREPARE_W0_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_UNIFIED_PREPARE_B0_OFFSET
#error NCLS_UNIFIED_PREPARE_B0_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_UNIFIED_PREPARE_W1_OFFSET
#error NCLS_UNIFIED_PREPARE_W1_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_UNIFIED_PREPARE_B1_OFFSET
#error NCLS_UNIFIED_PREPARE_B1_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_UNIFIED_EVALUATOR_STATE_W_OFFSET
#error NCLS_UNIFIED_EVALUATOR_STATE_W_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_UNIFIED_EVALUATOR_STATE_B_OFFSET
#error NCLS_UNIFIED_EVALUATOR_STATE_B_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_UNIFIED_SAMPLER_W_OFFSET
#error NCLS_UNIFIED_SAMPLER_W_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_UNIFIED_SAMPLER_B_OFFSET
#error NCLS_UNIFIED_SAMPLER_B_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_UNIFIED_EVALUATE_W0_OFFSET
#error NCLS_UNIFIED_EVALUATE_W0_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_UNIFIED_EVALUATE_B0_OFFSET
#error NCLS_UNIFIED_EVALUATE_B0_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_UNIFIED_EVALUATE_W1_OFFSET
#error NCLS_UNIFIED_EVALUATE_W1_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_UNIFIED_EVALUATE_B1_OFFSET
#error NCLS_UNIFIED_EVALUATE_B1_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_UNIFIED_EVALUATE_OUT_W_OFFSET
#error NCLS_UNIFIED_EVALUATE_OUT_W_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_UNIFIED_EVALUATE_OUT_B_OFFSET
#error NCLS_UNIFIED_EVALUATE_OUT_B_OFFSET must come from MethodBundle reflection
#endif

float nclsUnifiedPackedWeight(StructuredBuffer<uint> weights, uint index)
{
    return nclsUnifiedReadHalf(weights, index);
}

void nclsUnifiedPrepareDeployment(
    NclsUnifiedCompiledMaterial material,
    float3 wo,
    StructuredBuffer<uint> weights,
    out float prepared[NCLS_UNIFIED_PREPARE_OUTPUT])
{
    float latent[NCLS_UNIFIED_LATENT_COUNT];
    nclsUnifiedCompiledLatent(material, latent);
    const float input[NCLS_UNIFIED_PREPARE_INPUT] = nclsUnifiedPrepareInputInference(latent, wo);
    float hidden0[NCLS_UNIFIED_PREPARE_WIDTH];
    for (uint row = 0u; row < NCLS_UNIFIED_PREPARE_WIDTH; ++row)
    {
        float rowWeights[23];
        [unroll] for (uint column = 0u; column < 23u; ++column)
            rowWeights[column] = nclsUnifiedPackedWeight(
                weights, NCLS_UNIFIED_PREPARE_W0_OFFSET + row * 23u + column);
        hidden0[row] = nclsUnifiedLeakyRelu(nclsUnifiedDot23(
            rowWeights,
            nclsUnifiedPackedWeight(weights, NCLS_UNIFIED_PREPARE_B0_OFFSET + row),
            input));
    }
    float hidden1[NCLS_UNIFIED_PREPARE_WIDTH];
    for (uint row = 0u; row < NCLS_UNIFIED_PREPARE_WIDTH; ++row)
    {
        float rowWeights[64];
        [unroll] for (uint column = 0u; column < 64u; ++column)
            rowWeights[column] = nclsUnifiedPackedWeight(
                weights, NCLS_UNIFIED_PREPARE_W1_OFFSET + row * 64u + column);
        hidden1[row] = nclsUnifiedLeakyRelu(nclsUnifiedDot64(
            rowWeights,
            nclsUnifiedPackedWeight(weights, NCLS_UNIFIED_PREPARE_B1_OFFSET + row),
            hidden0));
    }
    float evaluator[14];
    for (uint row = 0u; row < 14u; ++row)
    {
        float rowWeights[64];
        [unroll] for (uint column = 0u; column < 64u; ++column)
            rowWeights[column] = nclsUnifiedPackedWeight(
                weights, NCLS_UNIFIED_EVALUATOR_STATE_W_OFFSET + row * 64u + column);
        evaluator[row] = nclsUnifiedDot64(
            rowWeights,
            nclsUnifiedPackedWeight(weights, NCLS_UNIFIED_EVALUATOR_STATE_B_OFFSET + row),
            hidden1);
    }
    if ((material.flags & 2u) == 0u)
    {
        float sampler[9];
        for (uint row = 0u; row < 9u; ++row)
        {
            float rowWeights[64];
            [unroll] for (uint column = 0u; column < 64u; ++column)
                rowWeights[column] = nclsUnifiedPackedWeight(
                    weights, NCLS_UNIFIED_SAMPLER_W_OFFSET + row * 64u + column);
            sampler[row] = nclsUnifiedDot64(
                rowWeights,
                nclsUnifiedPackedWeight(weights, NCLS_UNIFIED_SAMPLER_B_OFFSET + row),
                hidden1);
        }
        prepared = nclsUnifiedJoinNvidiaState(evaluator, sampler);
    }
    else
    {
        float sampler[13];
        for (uint row = 0u; row < 13u; ++row)
        {
            float rowWeights[64];
            [unroll] for (uint column = 0u; column < 64u; ++column)
                rowWeights[column] = nclsUnifiedPackedWeight(
                    weights, NCLS_UNIFIED_SAMPLER_W_OFFSET + row * 64u + column);
            sampler[row] = nclsUnifiedDot64(
                rowWeights,
                nclsUnifiedPackedWeight(weights, NCLS_UNIFIED_SAMPLER_B_OFFSET + row),
                hidden1);
        }
        prepared = nclsUnifiedJoinLtcState(evaluator, sampler);
    }
}

float3 nclsUnifiedEvaluateDeployment(
    NclsUnifiedCompiledMaterial material,
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT],
    float3 wo,
    float3 wi,
    StructuredBuffer<uint> weights)
{
    const float3 core = (material.flags & 1u) != 0u
        ? nclsUnifiedTopInterface(material.topInterface, wo, wi)
        : float3(0.0f);
    const float features[NCLS_UNIFIED_EVALUATE_INPUT] = (material.flags & 1u) != 0u
        ? nclsUnifiedResidualFeaturesInference(
            prepared, material.responseScale, core, wo, wi)
        : nclsUnifiedDirectFeatures(prepared, wo, wi);
    float hidden0[32];
    for (uint row = 0u; row < 32u; ++row)
    {
        float rowWeights[17];
        [unroll] for (uint column = 0u; column < 17u; ++column)
            rowWeights[column] = nclsUnifiedPackedWeight(
                weights, NCLS_UNIFIED_EVALUATE_W0_OFFSET + row * 17u + column);
        hidden0[row] = nclsUnifiedLeakyRelu(nclsUnifiedDot17(
            rowWeights,
            nclsUnifiedPackedWeight(weights, NCLS_UNIFIED_EVALUATE_B0_OFFSET + row),
            features));
    }
    float hidden1[32];
    for (uint row = 0u; row < 32u; ++row)
    {
        float rowWeights[32];
        [unroll] for (uint column = 0u; column < 32u; ++column)
            rowWeights[column] = nclsUnifiedPackedWeight(
                weights, NCLS_UNIFIED_EVALUATE_W1_OFFSET + row * 32u + column);
        hidden1[row] = nclsUnifiedLeakyRelu(nclsUnifiedDot32(
            rowWeights,
            nclsUnifiedPackedWeight(weights, NCLS_UNIFIED_EVALUATE_B1_OFFSET + row),
            hidden0));
    }
    float3 raw;
    for (uint row = 0u; row < 3u; ++row)
    {
        float rowWeights[32];
        [unroll] for (uint column = 0u; column < 32u; ++column)
            rowWeights[column] = nclsUnifiedPackedWeight(
                weights, NCLS_UNIFIED_EVALUATE_OUT_W_OFFSET + row * 32u + column);
        raw[row] = nclsUnifiedDot32(
            rowWeights,
            nclsUnifiedPackedWeight(weights, NCLS_UNIFIED_EVALUATE_OUT_B_OFFSET + row),
            hidden1);
    }
    return (material.flags & 1u) != 0u
        ? nclsUnifiedDecodeResidualInference(raw, prepared, material.responseScale, core)
        : nclsUnifiedDecodeDirect(raw, material.responseScale);
}

struct NclsUnifiedContractState : INclsScatteringState
{
    NclsScatteringContext context;
    NclsUnifiedCompiledMaterial material;
    NclsUnifiedPrepared prepared;
    StructuredBuffer<uint> weights;

    NclsScatteringEval evaluate<S : ISampleGenerator>(float3 wiWorld, inout S sampleGenerator)
    {
        NclsScatteringEval result = {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        if (wo.z <= NCLS_MIN_COS || wi.z <= NCLS_MIN_COS) return result;
        result.f = nclsUnifiedEvaluateDeployment(material, prepared.payload, wo, wi, weights);
        result.pdf.forward = (material.flags & 2u) == 0u
            ? nclsUnifiedNvidiaPreparedPdf(prepared.payload, wo, wi)
            : nclsUnifiedLtcPreparedPdf(prepared.payload, wi);
        float reversePrepared[NCLS_UNIFIED_PREPARE_OUTPUT];
        nclsUnifiedPrepareDeployment(material, wi, weights, reversePrepared);
        result.pdf.reverse = (material.flags & 2u) == 0u
            ? nclsUnifiedNvidiaPreparedPdf(reversePrepared, wi, wo)
            : nclsUnifiedLtcPreparedPdf(reversePrepared, wo);
        result.eventFlags = (uint)NclsScatteringEvent::Reflection | (uint)NclsScatteringEvent::Glossy;
        result.valid = all(isfinite(result.f)) && result.pdf.forward >= 0.0f
            && result.pdf.reverse >= 0.0f ? 1u : 0u;
        return result;
    }

    bool sample<S : ISampleGenerator>(out NclsScatteringSample result, inout S sampleGenerator)
    {
        result = {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        uint component = 0u;
        const float3 u = sampleNext3D(sampleGenerator);
        const NclsDirectionalSample directional = (material.flags & 2u) == 0u
            ? nclsSampleUnifiedNvidiaPrepared(prepared.payload, wo, u, component)
            : nclsSampleUnifiedLtcPrepared(prepared.payload, u, component);
        if (directional.valid == 0u) return false;
        if (directional.nullEvent != 0u)
        {
            result.valid = 1u;
            return true;
        }
        result.wiWorld = nclsFrameToWorld(context.surface.shadingFrame, directional.direction);
        const NclsScatteringEval evaluation = evaluate(result.wiWorld, sampleGenerator);
        if (evaluation.valid == 0u || evaluation.pdf.forward <= 0.0f) return false;
        result.weight = evaluation.f * directional.direction.z / evaluation.pdf.forward;
        result.pdf = evaluation.pdf;
        result.eta = 1.0f;
        result.eventFlags = evaluation.eventFlags;
        result.valid = 1u;
        return true;
    }

    NclsScatteringPdf pdf(float3 wiWorld)
    {
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        NclsScatteringPdf result = {};
        if (wo.z <= NCLS_MIN_COS || wi.z <= NCLS_MIN_COS) return result;
        result.forward = (material.flags & 2u) == 0u
            ? nclsUnifiedNvidiaPreparedPdf(prepared.payload, wo, wi)
            : nclsUnifiedLtcPreparedPdf(prepared.payload, wi);
        float reversePrepared[NCLS_UNIFIED_PREPARE_OUTPUT];
        nclsUnifiedPrepareDeployment(material, wi, weights, reversePrepared);
        result.reverse = (material.flags & 2u) == 0u
            ? nclsUnifiedNvidiaPreparedPdf(reversePrepared, wi, wo)
            : nclsUnifiedLtcPreparedPdf(reversePrepared, wo);
        return result;
    }
};

struct NclsUnifiedNeuralBackend : INclsScatteringBackend
{
    typedef NclsUnifiedCompiledMaterial CompiledMaterial;
    typedef NclsUnifiedContractState State;
    StructuredBuffer<uint> weights;

    State prepare(NclsScatteringContext context, CompiledMaterial material)
    {
        State result;
        result.context = context;
        result.material = material;
        result.weights = weights;
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        nclsUnifiedPrepareDeployment(material, wo, weights, result.prepared.payload);
        return result;
    }
};

typedef NclsUnifiedNeuralBackend NclsMethodBackend;
typedef NclsUnifiedCompiledMaterial NclsMethodCompiledMaterial;
typedef NclsUnifiedPackedState NclsMethodPackedState;
typedef NclsUnifiedContractState NclsMethodState;

NclsMethodBackend nclsCreateMethodBackend(StructuredBuffer<uint> weights) { return {weights}; }
NclsMethodPackedState nclsPackMethodState(NclsMethodState state)
{
    return nclsUnifiedPackPrepared(state.prepared.payload);
}
NclsMethodState nclsUnpackMethodState(
    NclsScatteringContext context,
    NclsMethodCompiledMaterial material,
    NclsMethodPackedState packed,
    StructuredBuffer<uint> weights)
{
    NclsMethodState result;
    result.context = context;
    result.material = material;
    result.weights = weights;
    nclsUnifiedUnpackPrepared(packed, result.prepared.payload);
    return result;
}

#endif
