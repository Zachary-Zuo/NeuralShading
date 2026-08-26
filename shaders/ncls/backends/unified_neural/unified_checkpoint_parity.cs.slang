// 03 checkpoint/FP16 packed parity kernel；全部网络算术转调唯一production core。
#include "unified_neural_core.slang"

StructuredBuffer<float> gWeights;
StructuredBuffer<float> gLatent;
StructuredBuffer<float> gResponseScale;
StructuredBuffer<uint> gTopKind;
StructuredBuffer<float> gTopFields;
StructuredBuffer<uint> gStateIndex;
StructuredBuffer<float4> gWo;
StructuredBuffer<float4> gWi;
RWStructuredBuffer<float4> gOutput;

uniform uint gCount;
uniform uint gPaper;
uniform uint gCore;
uniform uint gSampler;
uniform uint gPrepareW0Offset;
uniform uint gPrepareB0Offset;
uniform uint gPrepareW1Offset;
uniform uint gPrepareB1Offset;
uniform uint gEvaluatorStateWOffset;
uniform uint gEvaluatorStateBOffset;
uniform uint gSamplerWOffset;
uniform uint gSamplerBOffset;
uniform uint gEvaluateW0Offset;
uniform uint gEvaluateB0Offset;
uniform uint gEvaluateW1Offset;
uniform uint gEvaluateB1Offset;
uniform uint gEvaluateW2Offset;
uniform uint gEvaluateB2Offset;
uniform uint gEvaluateOutWOffset;
uniform uint gEvaluateOutBOffset;

NclsLayerInterfaceIR nclsParityTop(uint stateIndex)
{
    const uint base = stateIndex * 14u;
    NclsLayerInterfaceIR result = {};
    result.kind = gTopKind[stateIndex];
    result.alphaX = gTopFields[base + 0u];
    result.alphaY = gTopFields[base + 1u];
    result.relativeIor = gTopFields[base + 2u];
    result.etaR = gTopFields[base + 3u];
    result.etaG = gTopFields[base + 4u];
    result.etaB = gTopFields[base + 5u];
    result.kR = gTopFields[base + 6u];
    result.kG = gTopFields[base + 7u];
    result.kB = gTopFields[base + 8u];
    result.colorR = gTopFields[base + 9u];
    result.colorG = gTopFields[base + 10u];
    result.colorB = gTopFields[base + 11u];
    result.tangentRotation = gTopFields[base + 12u];
    result.reserved = gTopFields[base + 13u];
    return result;
}

void nclsParityPrepare(
    uint stateIndex,
    float3 wo,
    out float prepared[NCLS_UNIFIED_PREPARE_OUTPUT])
{
    float latent[NCLS_UNIFIED_LATENT_COUNT];
    [unroll]
    for (uint index = 0u; index < NCLS_UNIFIED_LATENT_COUNT; ++index)
        latent[index] = gLatent[stateIndex * NCLS_UNIFIED_LATENT_COUNT + index];
    const float input[NCLS_UNIFIED_PREPARE_INPUT] = nclsUnifiedPrepareInput(latent, wo);
    float hidden0[NCLS_UNIFIED_PREPARE_WIDTH];
    for (uint row = 0u; row < NCLS_UNIFIED_PREPARE_WIDTH; ++row)
    {
        float weights[23];
        [unroll] for (uint column = 0u; column < 23u; ++column)
            weights[column] = gWeights[gPrepareW0Offset + row * 23u + column];
        hidden0[row] = nclsUnifiedLeakyRelu(
            nclsUnifiedDot23(weights, gWeights[gPrepareB0Offset + row], input));
    }
    float hidden1[NCLS_UNIFIED_PREPARE_WIDTH];
    for (uint row = 0u; row < NCLS_UNIFIED_PREPARE_WIDTH; ++row)
    {
        float weights[64];
        [unroll] for (uint column = 0u; column < 64u; ++column)
            weights[column] = gWeights[gPrepareW1Offset + row * 64u + column];
        hidden1[row] = nclsUnifiedLeakyRelu(nclsUnifiedPrepareDot64(
            weights, gWeights[gPrepareB1Offset + row], hidden0));
    }
    float evaluator[14];
    for (uint row = 0u; row < 14u; ++row)
    {
        float weights[64];
        [unroll] for (uint column = 0u; column < 64u; ++column)
            weights[column] = gWeights[gEvaluatorStateWOffset + row * 64u + column];
        evaluator[row] = nclsUnifiedEvaluatorStateDot64(
            weights, gWeights[gEvaluatorStateBOffset + row], hidden1);
    }
    if (gSampler == 0u)
    {
        float sampler[9];
        for (uint row = 0u; row < 9u; ++row)
        {
            float weights[64];
            [unroll] for (uint column = 0u; column < 64u; ++column)
                weights[column] = gWeights[gSamplerWOffset + row * 64u + column];
            sampler[row] = nclsUnifiedNvidiaSamplerHeadDot64(
                weights, gWeights[gSamplerBOffset + row], hidden1);
        }
        prepared = nclsUnifiedJoinNvidiaSamplerState(evaluator, sampler);
    }
    else
    {
        float sampler[13];
        for (uint row = 0u; row < 13u; ++row)
        {
            float weights[64];
            [unroll] for (uint column = 0u; column < 64u; ++column)
                weights[column] = gWeights[gSamplerWOffset + row * 64u + column];
            sampler[row] = nclsUnifiedLtcSamplerHeadDot64(
                weights, gWeights[gSamplerBOffset + row], hidden1);
        }
        prepared = nclsUnifiedJoinLtcSamplerState(evaluator, sampler);
    }
}

float3 nclsParityEvaluate(
    uint stateIndex,
    float3 wo,
    float3 wi,
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT])
{
    const uint scaleBase = stateIndex * 3u;
    const float3 scale = float3(
        gResponseScale[scaleBase],
        gResponseScale[scaleBase + 1u],
        gResponseScale[scaleBase + 2u]);
    const float3 core = gCore != 0u
        ? nclsUnifiedTopInterface(nclsParityTop(stateIndex), wo, wi)
        : float3(0.0f);
    const float features[NCLS_UNIFIED_EVALUATE_INPUT] = gCore != 0u
        ? nclsUnifiedResidualFeatures(prepared, scale, core, wo, wi)
        : nclsUnifiedDirectFeatures(prepared, wo, wi);
    float3 raw = float3(0.0f);
    if (gPaper != 0u)
    {
        float hidden0[64];
        for (uint row = 0u; row < 64u; ++row)
        {
            float weights[17];
            [unroll] for (uint column = 0u; column < 17u; ++column)
                weights[column] = gWeights[gEvaluateW0Offset + row * 17u + column];
            hidden0[row] = nclsUnifiedLeakyRelu(nclsUnifiedDot17(
                weights, gWeights[gEvaluateB0Offset + row], features));
        }
        float hidden1[64];
        for (uint row = 0u; row < 64u; ++row)
        {
            float weights[64];
            [unroll] for (uint column = 0u; column < 64u; ++column)
                weights[column] = gWeights[gEvaluateW1Offset + row * 64u + column];
            hidden1[row] = nclsUnifiedLeakyRelu(nclsUnifiedPaperDot64A(
                weights, gWeights[gEvaluateB1Offset + row], hidden0));
        }
        float hidden2[64];
        for (uint row = 0u; row < 64u; ++row)
        {
            float weights[64];
            [unroll] for (uint column = 0u; column < 64u; ++column)
                weights[column] = gWeights[gEvaluateW2Offset + row * 64u + column];
            hidden2[row] = nclsUnifiedLeakyRelu(nclsUnifiedPaperDot64B(
                weights, gWeights[gEvaluateB2Offset + row], hidden1));
        }
        for (uint row = 0u; row < 3u; ++row)
        {
            float weights[64];
            [unroll] for (uint column = 0u; column < 64u; ++column)
                weights[column] = gWeights[gEvaluateOutWOffset + row * 64u + column];
            raw[row] = nclsUnifiedPaperDot64Out(
                weights, gWeights[gEvaluateOutBOffset + row], hidden2);
        }
    }
    else
    {
        float hidden0[32];
        for (uint row = 0u; row < 32u; ++row)
        {
            float weights[17];
            [unroll] for (uint column = 0u; column < 17u; ++column)
                weights[column] = gWeights[gEvaluateW0Offset + row * 17u + column];
            hidden0[row] = nclsUnifiedLeakyRelu(nclsUnifiedDot17(
                weights, gWeights[gEvaluateB0Offset + row], features));
        }
        float hidden1[32];
        for (uint row = 0u; row < 32u; ++row)
        {
            float weights[32];
            [unroll] for (uint column = 0u; column < 32u; ++column)
                weights[column] = gWeights[gEvaluateW1Offset + row * 32u + column];
            hidden1[row] = nclsUnifiedLeakyRelu(nclsUnifiedDot32(
                weights, gWeights[gEvaluateB1Offset + row], hidden0));
        }
        for (uint row = 0u; row < 3u; ++row)
        {
            float weights[32];
            [unroll] for (uint column = 0u; column < 32u; ++column)
                weights[column] = gWeights[gEvaluateOutWOffset + row * 32u + column];
            raw[row] = nclsUnifiedDot32(
                weights, gWeights[gEvaluateOutBOffset + row], hidden1);
        }
    }
    return gCore != 0u
        ? nclsUnifiedDecodeResidual(raw, prepared, scale, core)
        : nclsUnifiedDecodeDirect(raw, scale);
}

[numthreads(64, 1, 1)]
void evaluateUnifiedCheckpointParity(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    const uint stateIndex = gStateIndex[index];
    const float3 wo = gWo[index].xyz;
    const float3 wi = gWi[index].xyz;
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT];
    nclsParityPrepare(stateIndex, wo, prepared);
    const float3 prediction = nclsParityEvaluate(stateIndex, wo, wi, prepared);
    const float pdf = gSampler == 0u
        ? nclsUnifiedNvidiaPreparedPdf(prepared, wo, wi)
        : nclsUnifiedLtcPreparedPdf(prepared, wi);
    gOutput[index] = float4(prediction, pdf);
}
