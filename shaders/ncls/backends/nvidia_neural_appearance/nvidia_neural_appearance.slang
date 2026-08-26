#ifndef NCLS_NVIDIA_NEURAL_APPEARANCE_SLANG
#define NCLS_NVIDIA_NEURAL_APPEARANCE_SLANG

// NVIDIA 论文 baseline 的公共散射合同包装；训练、测试与 viewer 共用同一 core。
#define NCLS_NVIDIA_NEURAL_WEIGHTS_T StructuredBuffer<uint>
#define NCLS_NVIDIA_NEURAL_WEIGHT_READ(weights, index) (no_diff nclsNvidiaNeuralReadPackedWeight(weights, index))

float nclsNvidiaNeuralReadPackedWeight(StructuredBuffer<uint> weights, uint index)
{
    const uint bits = weights[index >> 1u];
    return f16tof32((index & 1u) == 0u ? (bits & 0xffffu) : (bits >> 16u));
}

#include "nvidia_neural_appearance_pack.slang"
#include "../../contracts/scattering_backend.slang"

#ifndef NCLS_NVIDIA_FRAME_WEIGHT_OFFSET
#error NCLS_NVIDIA_FRAME_WEIGHT_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_EVALUATE_WEIGHT0_OFFSET
#error NCLS_NVIDIA_EVALUATE_WEIGHT0_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_EVALUATE_BIAS0_OFFSET
#error NCLS_NVIDIA_EVALUATE_BIAS0_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_EVALUATE_WEIGHT1_OFFSET
#error NCLS_NVIDIA_EVALUATE_WEIGHT1_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_EVALUATE_BIAS1_OFFSET
#error NCLS_NVIDIA_EVALUATE_BIAS1_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_EVALUATE_WEIGHT2_OFFSET
#error NCLS_NVIDIA_EVALUATE_WEIGHT2_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_EVALUATE_BIAS2_OFFSET
#error NCLS_NVIDIA_EVALUATE_BIAS2_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_EVALUATE_OUT_WEIGHT_OFFSET
#error NCLS_NVIDIA_EVALUATE_OUT_WEIGHT_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_EVALUATE_OUT_BIAS_OFFSET
#error NCLS_NVIDIA_EVALUATE_OUT_BIAS_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_SAMPLER_WEIGHT0_OFFSET
#error NCLS_NVIDIA_SAMPLER_WEIGHT0_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_SAMPLER_BIAS0_OFFSET
#error NCLS_NVIDIA_SAMPLER_BIAS0_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_SAMPLER_WEIGHT1_OFFSET
#error NCLS_NVIDIA_SAMPLER_WEIGHT1_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_SAMPLER_BIAS1_OFFSET
#error NCLS_NVIDIA_SAMPLER_BIAS1_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_SAMPLER_WEIGHT2_OFFSET
#error NCLS_NVIDIA_SAMPLER_WEIGHT2_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_SAMPLER_BIAS2_OFFSET
#error NCLS_NVIDIA_SAMPLER_BIAS2_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_SAMPLER_OUT_WEIGHT_OFFSET
#error NCLS_NVIDIA_SAMPLER_OUT_WEIGHT_OFFSET must come from MethodBundle reflection
#endif
#ifndef NCLS_NVIDIA_SAMPLER_OUT_BIAS_OFFSET
#error NCLS_NVIDIA_SAMPLER_OUT_BIAS_OFFSET must come from MethodBundle reflection
#endif

NclsNvidiaNeuralParams nclsNvidiaNeuralParams(StructuredBuffer<uint> weights)
{
    NclsNvidiaNeuralParams result;
    result.weights = weights;
    result.frameWeight = NCLS_NVIDIA_FRAME_WEIGHT_OFFSET;
    result.evaluateWeight0 = NCLS_NVIDIA_EVALUATE_WEIGHT0_OFFSET;
    result.evaluateBias0 = NCLS_NVIDIA_EVALUATE_BIAS0_OFFSET;
    result.evaluateWeight1 = NCLS_NVIDIA_EVALUATE_WEIGHT1_OFFSET;
    result.evaluateBias1 = NCLS_NVIDIA_EVALUATE_BIAS1_OFFSET;
    result.evaluateWeight2 = NCLS_NVIDIA_EVALUATE_WEIGHT2_OFFSET;
    result.evaluateBias2 = NCLS_NVIDIA_EVALUATE_BIAS2_OFFSET;
    result.evaluateOutWeight = NCLS_NVIDIA_EVALUATE_OUT_WEIGHT_OFFSET;
    result.evaluateOutBias = NCLS_NVIDIA_EVALUATE_OUT_BIAS_OFFSET;
    result.samplerWeight0 = NCLS_NVIDIA_SAMPLER_WEIGHT0_OFFSET;
    result.samplerBias0 = NCLS_NVIDIA_SAMPLER_BIAS0_OFFSET;
    result.samplerWeight1 = NCLS_NVIDIA_SAMPLER_WEIGHT1_OFFSET;
    result.samplerBias1 = NCLS_NVIDIA_SAMPLER_BIAS1_OFFSET;
    result.samplerWeight2 = NCLS_NVIDIA_SAMPLER_WEIGHT2_OFFSET;
    result.samplerBias2 = NCLS_NVIDIA_SAMPLER_BIAS2_OFFSET;
    result.samplerOutWeight = NCLS_NVIDIA_SAMPLER_OUT_WEIGHT_OFFSET;
    result.samplerOutBias = NCLS_NVIDIA_SAMPLER_OUT_BIAS_OFFSET;
    return result;
}

struct NclsNvidiaNeuralContractState : INclsScatteringState
{
    NclsScatteringContext context;
    NclsNvidiaNeuralCompiledMaterial material;
    NclsNvidiaNeuralPrepared prepared;
    NclsNvidiaNeuralParams params;

    NclsScatteringEval evaluate<S : ISampleGenerator>(float3 wiWorld, inout S sampleGenerator)
    {
        NclsScatteringEval result = {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        if (wo.z <= NCLS_MIN_COS || wi.z <= NCLS_MIN_COS) return result;
        result.f = nclsNvidiaNeuralEvaluateF(prepared, wi, params);
        result.pdf.forward = nclsNvidiaNeuralPreparedPdf(prepared, wo, wi);
        const NclsNvidiaNeuralPrepared reverse = nclsNvidiaNeuralUnpackPrepared(
            nclsNvidiaNeuralPreparePacked(material, wi, params));
        result.pdf.reverse = nclsNvidiaNeuralPreparedPdf(reverse, wi, wo);
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
        const NclsDirectionalSample directional = nclsSampleNvidiaNeuralPrepared(
            prepared, wo, sampleNext3D(sampleGenerator), component);
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
        result.forward = nclsNvidiaNeuralPreparedPdf(prepared, wo, wi);
        const NclsNvidiaNeuralPrepared reverse = nclsNvidiaNeuralUnpackPrepared(
            nclsNvidiaNeuralPreparePacked(material, wi, params));
        result.reverse = nclsNvidiaNeuralPreparedPdf(reverse, wi, wo);
        return result;
    }
};

struct NclsNvidiaNeuralBackend : INclsScatteringBackend
{
    typedef NclsNvidiaNeuralCompiledMaterial CompiledMaterial;
    typedef NclsNvidiaNeuralContractState State;
    StructuredBuffer<uint> weights;

    State prepare(NclsScatteringContext context, CompiledMaterial material)
    {
        const NclsNvidiaNeuralParams params = nclsNvidiaNeuralParams(weights);
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        return {context, material, nclsNvidiaNeuralUnpackPrepared(
            nclsNvidiaNeuralPreparePacked(material, wo, params)), params};
    }
};

typedef NclsNvidiaNeuralBackend NclsMethodBackend;
typedef NclsNvidiaNeuralCompiledMaterial NclsMethodCompiledMaterial;
typedef NclsNvidiaNeuralState NclsMethodPackedState;
typedef NclsNvidiaNeuralContractState NclsMethodState;

NclsMethodBackend nclsCreateMethodBackend(StructuredBuffer<uint> weights) { return {weights}; }
NclsMethodPackedState nclsPackMethodState(NclsMethodState state)
{
    return nclsNvidiaNeuralPackPrepared(state.prepared);
}
NclsMethodState nclsUnpackMethodState(
    NclsScatteringContext context,
    NclsMethodCompiledMaterial material,
    NclsMethodPackedState packed,
    StructuredBuffer<uint> weights)
{
    return {context, material, nclsNvidiaNeuralUnpackPrepared(packed),
        nclsNvidiaNeuralParams(weights)};
}

#endif
