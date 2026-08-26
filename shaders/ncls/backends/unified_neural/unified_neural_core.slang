// 03 定义的 learned-frame evaluator 与 matched sampler 唯一生产前向；本文件不依赖 Falcor。
#ifndef NCLS_UNIFIED_NEURAL_CORE_SLANG
#define NCLS_UNIFIED_NEURAL_CORE_SLANG

#include "unified_neural_layout.slang"
#include "../../reference/interfaces.slang"
#include "../../scattering/frame.slang"
#include "../../scattering/nvidia_proposal.slang"
#include "../../scattering/ltc.slang"

struct NclsUnifiedPrepared
{
    float payload[NCLS_UNIFIED_PREPARE_OUTPUT];
};

struct NclsUnifiedLearnedFrame : IDifferentiable
{
    float3 tangent;
    float3 bitangent;
    float3 normal;
};

struct NclsUnifiedLtcProposal
{
    NclsLtcTransform lobe0;
    NclsLtcTransform lobe1;
    float learnedWeight;
    uint valid;
};

[Differentiable]
float nclsUnifiedSoftplus(float value)
{
    return log(1.0f + exp(-abs(value))) + max(value, 0.0f);
}

[Differentiable]
float nclsUnifiedLeakyRelu(float value)
{
    return value >= 0.0f ? value : 0.1f * value;
}

[Differentiable]
float nclsUnifiedDot23(float weights[23], float bias, float input[23])
{
    float result = bias;
    for (uint index = 0u; index < 23u; ++index) result += weights[index] * input[index];
    return result;
}

[Differentiable]
float nclsUnifiedDot64(float weights[64], float bias, float input[64])
{
    float result = bias;
    for (uint index = 0u; index < 64u; ++index) result += weights[index] * input[index];
    return result;
}

// 训练角色拥有独立的公开callable identity；算术仍只由nclsUnifiedDot64定义。
[Differentiable]
float nclsUnifiedPrepareDot64(float weights[64], float bias, float input[64])
{
    return nclsUnifiedDot64(weights, bias, input);
}

[Differentiable]
float nclsUnifiedEvaluatorStateDot64(float weights[64], float bias, float input[64])
{
    return nclsUnifiedDot64(weights, bias, input);
}

[Differentiable]
float nclsUnifiedNvidiaSamplerHeadDot64(float weights[64], float bias, float input[64])
{
    return nclsUnifiedDot64(weights, bias, input);
}

[Differentiable]
float nclsUnifiedLtcSamplerHeadDot64(float weights[64], float bias, float input[64])
{
    return nclsUnifiedDot64(weights, bias, input);
}

// SlangPy 0.43.1 generates one backward wrapper per callable identity. Give
// the diagnostic's repeated 64-wide layers distinct callable identities.
[Differentiable]
float nclsUnifiedPaperDot64A(float weights[64], float bias, float input[64])
{
    float result = bias;
    for (uint index = 0u; index < 64u; ++index) result += weights[index] * input[index];
    return result;
}

[Differentiable]
float nclsUnifiedPaperDot64B(float weights[64], float bias, float input[64])
{
    float result = bias;
    for (uint index = 0u; index < 64u; ++index) result += weights[index] * input[index];
    return result;
}

[Differentiable]
float nclsUnifiedPaperDot64Out(float weights[64], float bias, float input[64])
{
    float result = bias;
    for (uint index = 0u; index < 64u; ++index) result += weights[index] * input[index];
    return result;
}

[Differentiable]
float nclsUnifiedDot17(float weights[17], float bias, float input[17])
{
    float result = bias;
    for (uint index = 0u; index < 17u; ++index) result += weights[index] * input[index];
    return result;
}

[Differentiable]
float nclsUnifiedDot32(float weights[32], float bias, float input[32])
{
    float result = bias;
    for (uint index = 0u; index < 32u; ++index) result += weights[index] * input[index];
    return result;
}

[Differentiable]
float[NCLS_UNIFIED_PREPARE_INPUT] nclsUnifiedPrepareInput(
    float latent[NCLS_UNIFIED_LATENT_COUNT],
    no_diff float3 wo)
{
    float result[NCLS_UNIFIED_PREPARE_INPUT] = {};
    [unroll]
    for (uint index = 0u; index < NCLS_UNIFIED_LATENT_COUNT; ++index)
        result[index] = latent[index];
    const float inverseZ = 1.0f / max(wo.z, 0.03125f);
    result[16] = wo.x;
    result[17] = wo.y;
    result[18] = wo.z;
    result[19] = wo.x * inverseZ;
    result[20] = wo.y * inverseZ;
    result[21] = sqrt(max(0.0f, 1.0f - wo.z * wo.z));
    result[22] = 1.0f;
    return result;
}

[Differentiable]
float[NCLS_UNIFIED_PREPARE_WIDTH] nclsUnifiedLinear23x64(
    float weights[NCLS_UNIFIED_PREPARE_WIDTH][NCLS_UNIFIED_PREPARE_INPUT],
    float bias[NCLS_UNIFIED_PREPARE_WIDTH],
    float input[NCLS_UNIFIED_PREPARE_INPUT])
{
    float result[NCLS_UNIFIED_PREPARE_WIDTH] = {};
    for (uint row = 0u; row < NCLS_UNIFIED_PREPARE_WIDTH; ++row)
    {
        float value = bias[row];
        for (uint column = 0u; column < NCLS_UNIFIED_PREPARE_INPUT; ++column)
            value += weights[row][column] * input[column];
        result[row] = value;
    }
    return result;
}

[Differentiable]
float[NCLS_UNIFIED_PREPARE_WIDTH] nclsUnifiedLinear64x64(
    float weights[NCLS_UNIFIED_PREPARE_WIDTH][NCLS_UNIFIED_PREPARE_WIDTH],
    float bias[NCLS_UNIFIED_PREPARE_WIDTH],
    float input[NCLS_UNIFIED_PREPARE_WIDTH])
{
    float result[NCLS_UNIFIED_PREPARE_WIDTH] = {};
    for (uint row = 0u; row < NCLS_UNIFIED_PREPARE_WIDTH; ++row)
    {
        float value = bias[row];
        for (uint column = 0u; column < NCLS_UNIFIED_PREPARE_WIDTH; ++column)
            value += weights[row][column] * input[column];
        result[row] = value;
    }
    return result;
}

[Differentiable]
float[NCLS_UNIFIED_PREPARE_WIDTH] nclsUnifiedActivate64(
    float input[NCLS_UNIFIED_PREPARE_WIDTH])
{
    float result[NCLS_UNIFIED_PREPARE_WIDTH] = {};
    for (uint index = 0u; index < NCLS_UNIFIED_PREPARE_WIDTH; ++index)
        result[index] = nclsUnifiedLeakyRelu(input[index]);
    return result;
}

[Differentiable]
float[NCLS_UNIFIED_PREPARE_OUTPUT] nclsUnifiedLinear64x27(
    float weights[NCLS_UNIFIED_PREPARE_OUTPUT][NCLS_UNIFIED_PREPARE_WIDTH],
    float bias[NCLS_UNIFIED_PREPARE_OUTPUT],
    float input[NCLS_UNIFIED_PREPARE_WIDTH])
{
    float result[NCLS_UNIFIED_PREPARE_OUTPUT] = {};
    for (uint row = 0u; row < NCLS_UNIFIED_PREPARE_OUTPUT; ++row)
    {
        float value = bias[row];
        for (uint column = 0u; column < NCLS_UNIFIED_PREPARE_WIDTH; ++column)
            value += weights[row][column] * input[column];
        result[row] = value;
    }
    return result;
}

[Differentiable]
float[14] nclsUnifiedLinear64x14(
    float weights[14][NCLS_UNIFIED_PREPARE_WIDTH],
    float bias[14],
    float input[NCLS_UNIFIED_PREPARE_WIDTH])
{
    float result[14] = {};
    for (uint row = 0u; row < 14u; ++row)
    {
        float value = bias[row];
        for (uint column = 0u; column < NCLS_UNIFIED_PREPARE_WIDTH; ++column)
            value += weights[row][column] * input[column];
        result[row] = value;
    }
    return result;
}

[Differentiable]
float[9] nclsUnifiedLinear64x9(
    float weights[9][NCLS_UNIFIED_PREPARE_WIDTH],
    float bias[9],
    float input[NCLS_UNIFIED_PREPARE_WIDTH])
{
    float result[9] = {};
    for (uint row = 0u; row < 9u; ++row)
    {
        float value = bias[row];
        for (uint column = 0u; column < NCLS_UNIFIED_PREPARE_WIDTH; ++column)
            value += weights[row][column] * input[column];
        result[row] = value;
    }
    return result;
}

[Differentiable]
float[13] nclsUnifiedLinear64x13(
    float weights[13][NCLS_UNIFIED_PREPARE_WIDTH],
    float bias[13],
    float input[NCLS_UNIFIED_PREPARE_WIDTH])
{
    float result[13] = {};
    for (uint row = 0u; row < 13u; ++row)
    {
        float value = bias[row];
        for (uint column = 0u; column < NCLS_UNIFIED_PREPARE_WIDTH; ++column)
            value += weights[row][column] * input[column];
        result[row] = value;
    }
    return result;
}

[Differentiable]
float[NCLS_UNIFIED_PREPARE_OUTPUT] nclsUnifiedJoinNvidiaState(float evaluator[14], float sampler[9])
{
    float result[NCLS_UNIFIED_PREPARE_OUTPUT] = {};
    [unroll] for (uint index = 0u; index < 14u; ++index) result[index] = evaluator[index];
    [unroll] for (uint index = 0u; index < 9u; ++index) result[14u + index] = sampler[index];
    [unroll] for (uint index = 23u; index < NCLS_UNIFIED_PREPARE_OUTPUT; ++index) result[index] = 0.0f;
    return result;
}

[Differentiable]
float[NCLS_UNIFIED_PREPARE_OUTPUT] nclsUnifiedJoinEvaluatorState(float evaluator[14])
{
    float result[NCLS_UNIFIED_PREPARE_OUTPUT] = {};
    [unroll] for (uint index = 0u; index < 14u; ++index) result[index] = evaluator[index];
    return result;
}

[Differentiable]
float[NCLS_UNIFIED_PREPARE_OUTPUT] nclsUnifiedJoinNvidiaSamplerState(
    float evaluator[14], float sampler[9])
{
    return nclsUnifiedJoinNvidiaState(evaluator, sampler);
}

[Differentiable]
float[NCLS_UNIFIED_PREPARE_OUTPUT] nclsUnifiedJoinLtcState(float evaluator[14], float sampler[13])
{
    float result[NCLS_UNIFIED_PREPARE_OUTPUT] = {};
    [unroll] for (uint index = 0u; index < 14u; ++index) result[index] = evaluator[index];
    [unroll] for (uint index = 0u; index < 13u; ++index) result[14u + index] = sampler[index];
    return result;
}

[Differentiable]
float[NCLS_UNIFIED_PREPARE_OUTPUT] nclsUnifiedJoinLtcSamplerState(
    float evaluator[14], float sampler[13])
{
    return nclsUnifiedJoinLtcState(evaluator, sampler);
}

[Differentiable]
float nclsUnifiedTanhApprox(float value)
{
    return value / sqrt(1.0f + value * value);
}

[Differentiable]
NclsUnifiedLearnedFrame nclsUnifiedLearnedFrame(float3 raw)
{
    const float2 slope = 2.0f * float2(
        nclsUnifiedTanhApprox(raw.x), nclsUnifiedTanhApprox(raw.y));
    NclsUnifiedLearnedFrame frame;
    frame.normal = normalize(float3(-slope.x, -slope.y, 1.0f));
    const float3 axis = abs(frame.normal.x) < 0.999f
        ? float3(1.0f, 0.0f, 0.0f)
        : float3(0.0f, 1.0f, 0.0f);
    frame.tangent = normalize(axis - frame.normal * dot(frame.normal, axis));
    frame.bitangent = cross(frame.normal, frame.tangent);
    const float angle = NCLS_PI * nclsUnifiedTanhApprox(raw.z);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    const float3 tangent = cosine * frame.tangent + sine * frame.bitangent;
    frame.bitangent = -sine * frame.tangent + cosine * frame.bitangent;
    frame.tangent = tangent;
    return frame;
}

float2 nclsUnifiedHalfSlope(float3 wo, float3 wi)
{
    const float3 sum = wo + wi;
    if (dot(sum, sum) <= 1e-12f) return float2(0.0f);
    const float3 halfVector = nclsSafeNormalize(sum);
    return halfVector.xy / max(abs(halfVector.z), 0.03125f);
}

[Differentiable]
float[NCLS_UNIFIED_EVALUATE_INPUT] nclsUnifiedDirectFeatures(
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT],
    no_diff float3 wo,
    no_diff float3 wi)
{
    float result[NCLS_UNIFIED_EVALUATE_INPUT] = {};
    [unroll]
    for (uint index = 0u; index < 5u; ++index) result[index] = prepared[index];
    const NclsUnifiedLearnedFrame frame0 = nclsUnifiedLearnedFrame(float3(prepared[5], prepared[6], prepared[7]));
    const NclsUnifiedLearnedFrame frame1 = nclsUnifiedLearnedFrame(float3(prepared[8], prepared[9], prepared[10]));
    const float3 local0 = float3(dot(wi, frame0.tangent), dot(wi, frame0.bitangent), dot(wi, frame0.normal));
    const float3 local1 = float3(dot(wi, frame1.tangent), dot(wi, frame1.bitangent), dot(wi, frame1.normal));
    result[5] = local0.x; result[6] = local0.y; result[7] = local0.z;
    result[8] = local1.x; result[9] = local1.y; result[10] = local1.z;
    const float2 slope = no_diff(nclsUnifiedHalfSlope(wo, wi));
    result[11] = slope.x; result[12] = slope.y; result[13] = wi.z;
    const float3 halfVector = no_diff(nclsSafeNormalize(wo + wi));
    result[14] = dot(wo, wi); result[15] = halfVector.z; result[16] = 1.0f;
    return result;
}

float3 nclsUnifiedTopInterface(
    NclsLayerInterfaceIR top,
    float3 wo,
    float3 wi)
{
    const float etaT = top.kind == (uint)NclsInterfaceKind::RoughDielectric
        ? top.relativeIor : 1.0f;
    return nclsEvalInterface(top, wo, wi, 1.0f, etaT);
}

[Differentiable]
float[NCLS_UNIFIED_EVALUATE_INPUT] nclsUnifiedResidualFeatures(
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT],
    no_diff float3 responseScale,
    no_diff float3 core,
    no_diff float3 wo,
    no_diff float3 wi)
{
    float result[NCLS_UNIFIED_EVALUATE_INPUT] = nclsUnifiedDirectFeatures(prepared, wo, wi);
    result[14] = log(1.0f + core.x / max(responseScale.x, 1e-6f));
    result[15] = log(1.0f + core.y / max(responseScale.y, 1e-6f));
    result[16] = log(1.0f + core.z / max(responseScale.z, 1e-6f));
    return result;
}

[Differentiable]
float[NCLS_UNIFIED_EVALUATE_WIDTH] nclsUnifiedLinear17x32(
    float weights[NCLS_UNIFIED_EVALUATE_WIDTH][NCLS_UNIFIED_EVALUATE_INPUT],
    float bias[NCLS_UNIFIED_EVALUATE_WIDTH],
    float input[NCLS_UNIFIED_EVALUATE_INPUT])
{
    float result[NCLS_UNIFIED_EVALUATE_WIDTH] = {};
    for (uint row = 0u; row < NCLS_UNIFIED_EVALUATE_WIDTH; ++row)
    {
        float value = bias[row];
        for (uint column = 0u; column < NCLS_UNIFIED_EVALUATE_INPUT; ++column)
            value += weights[row][column] * input[column];
        result[row] = value;
    }
    return result;
}

[Differentiable]
float[NCLS_UNIFIED_EVALUATE_WIDTH] nclsUnifiedLinear32x32(
    float weights[NCLS_UNIFIED_EVALUATE_WIDTH][NCLS_UNIFIED_EVALUATE_WIDTH],
    float bias[NCLS_UNIFIED_EVALUATE_WIDTH],
    float input[NCLS_UNIFIED_EVALUATE_WIDTH])
{
    float result[NCLS_UNIFIED_EVALUATE_WIDTH] = {};
    for (uint row = 0u; row < NCLS_UNIFIED_EVALUATE_WIDTH; ++row)
    {
        float value = bias[row];
        for (uint column = 0u; column < NCLS_UNIFIED_EVALUATE_WIDTH; ++column)
            value += weights[row][column] * input[column];
        result[row] = value;
    }
    return result;
}

[Differentiable]
float[NCLS_UNIFIED_EVALUATE_WIDTH] nclsUnifiedActivate32(
    float input[NCLS_UNIFIED_EVALUATE_WIDTH])
{
    float result[NCLS_UNIFIED_EVALUATE_WIDTH] = {};
    for (uint index = 0u; index < NCLS_UNIFIED_EVALUATE_WIDTH; ++index)
        result[index] = nclsUnifiedLeakyRelu(input[index]);
    return result;
}

[Differentiable]
float3 nclsUnifiedLinear32x3(
    float weights[NCLS_UNIFIED_RGB_COUNT][NCLS_UNIFIED_EVALUATE_WIDTH],
    float bias[NCLS_UNIFIED_RGB_COUNT],
    float input[NCLS_UNIFIED_EVALUATE_WIDTH])
{
    float3 result = float3(0.0f);
    for (uint row = 0u; row < NCLS_UNIFIED_RGB_COUNT; ++row)
    {
        float value = bias[row];
        for (uint column = 0u; column < NCLS_UNIFIED_EVALUATE_WIDTH; ++column)
            value += weights[row][column] * input[column];
        result[row] = value;
    }
    return result;
}

[Differentiable]
float[NCLS_UNIFIED_PAPER_WIDTH] nclsUnifiedLinear17x64(
    float weights[NCLS_UNIFIED_PAPER_WIDTH][NCLS_UNIFIED_EVALUATE_INPUT],
    float bias[NCLS_UNIFIED_PAPER_WIDTH],
    float input[NCLS_UNIFIED_EVALUATE_INPUT])
{
    float result[NCLS_UNIFIED_PAPER_WIDTH] = {};
    for (uint row = 0u; row < NCLS_UNIFIED_PAPER_WIDTH; ++row)
    {
        float value = bias[row];
        for (uint column = 0u; column < NCLS_UNIFIED_EVALUATE_INPUT; ++column)
            value += weights[row][column] * input[column];
        result[row] = value;
    }
    return result;
}

[Differentiable]
float3 nclsUnifiedLinear64x3(
    float weights[NCLS_UNIFIED_RGB_COUNT][NCLS_UNIFIED_PAPER_WIDTH],
    float bias[NCLS_UNIFIED_RGB_COUNT],
    float input[NCLS_UNIFIED_PAPER_WIDTH])
{
    float3 result = float3(0.0f);
    for (uint row = 0u; row < NCLS_UNIFIED_RGB_COUNT; ++row)
    {
        float value = bias[row];
        for (uint column = 0u; column < NCLS_UNIFIED_PAPER_WIDTH; ++column)
            value += weights[row][column] * input[column];
        result[row] = value;
    }
    return result;
}

[Differentiable]
float3 nclsUnifiedDecodeDirect(float3 raw, no_diff float3 responseScale)
{
    return responseScale * float3(
        nclsUnifiedSoftplus(raw.x),
        nclsUnifiedSoftplus(raw.y),
        nclsUnifiedSoftplus(raw.z));
}

[Differentiable]
float3 nclsUnifiedDecodeResidual(
    float3 raw,
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT],
    no_diff float3 responseScale,
    no_diff float3 core)
{
    const float3 scale = float3(
        nclsUnifiedSoftplus(prepared[11]),
        nclsUnifiedSoftplus(prepared[12]),
        nclsUnifiedSoftplus(prepared[13]));
    const float3 residual = scale * responseScale * float3(
        nclsUnifiedSoftplus(raw.x),
        nclsUnifiedSoftplus(raw.y),
        nclsUnifiedSoftplus(raw.z));
    return core + residual;
}

NclsUnifiedLtcProposal nclsDecodeUnifiedLtc(float raw[NCLS_UNIFIED_LTC_RAW_COUNT])
{
    NclsUnifiedLtcProposal result = {};
    result.lobe0.inverseScale = exp(clamp(float2(raw[0], raw[1]), -4.0f, 4.0f));
    result.lobe0.shear = 2.0f * float3(
        nclsNvidiaTanhApprox(raw[2]), nclsNvidiaTanhApprox(raw[3]), nclsNvidiaTanhApprox(raw[4]));
    result.lobe0.angle = NCLS_PI * nclsNvidiaTanhApprox(raw[5]);
    result.lobe1.inverseScale = exp(clamp(float2(raw[6], raw[7]), -4.0f, 4.0f));
    result.lobe1.shear = 2.0f * float3(
        nclsNvidiaTanhApprox(raw[8]), nclsNvidiaTanhApprox(raw[9]), nclsNvidiaTanhApprox(raw[10]));
    result.lobe1.angle = NCLS_PI * nclsNvidiaTanhApprox(raw[11]);
    result.learnedWeight = 0.5f + 0.5f * nclsNvidiaTanhApprox(raw[12]);
    result.valid = nclsLtcTransformValid(result.lobe0) && nclsLtcTransformValid(result.lobe1) ? 1u : 0u;
    return result;
}

float nclsUnifiedLtcPdf(NclsUnifiedLtcProposal params, float3 wi)
{
    if (params.valid == 0u || wi.z <= NCLS_MIN_COS) return 0.0f;
    const float learnedMass = 1.0f - NCLS_UNIFIED_SAFETY_WEIGHT;
    return NCLS_UNIFIED_SAFETY_WEIGHT * nclsCosineHemispherePdf(wi)
        + learnedMass * (params.learnedWeight * nclsLtcPdf(params.lobe0, wi)
        + (1.0f - params.learnedWeight) * nclsLtcPdf(params.lobe1, wi));
}

float nclsUnifiedNvidiaPreparedPdf(
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT],
    float3 wo,
    float3 wi)
{
    float raw[NCLS_UNIFIED_NVIDIA_RAW_COUNT] = {};
    [unroll] for (uint index = 0u; index < NCLS_UNIFIED_NVIDIA_RAW_COUNT; ++index)
        raw[index] = prepared[14u + index];
    return nclsNvidiaProposalPdf(nclsDecodeNvidiaProposal(raw), wo, wi);
}

[Differentiable]
float nclsUnifiedNvidiaPreparedPdfTraining(
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT],
    no_diff float3 wo,
    no_diff float3 wi)
{
    float raw[NCLS_UNIFIED_NVIDIA_RAW_COUNT] = {};
    [unroll] for (uint index = 0u; index < NCLS_UNIFIED_NVIDIA_RAW_COUNT; ++index)
        raw[index] = prepared[14u + index];
    const NclsNvidiaDifferentiableParams params = nclsDecodeNvidiaProposalUnchecked(raw);
    return nclsNvidiaProposalPdfUnchecked(params, wo, wi);
}

float nclsUnifiedLtcPreparedPdf(
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT],
    float3 wi)
{
    float raw[NCLS_UNIFIED_LTC_RAW_COUNT] = {};
    [unroll] for (uint index = 0u; index < NCLS_UNIFIED_LTC_RAW_COUNT; ++index)
        raw[index] = prepared[14u + index];
    return nclsUnifiedLtcPdf(nclsDecodeUnifiedLtc(raw), wi);
}

[Differentiable]
float nclsUnifiedLtcPreparedPdfTraining(
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT],
    no_diff float3 wi)
{
    NclsLtcTransform lobe0;
    lobe0.inverseScale = exp(clamp(float2(prepared[14], prepared[15]), -4.0f, 4.0f));
    lobe0.shear = 2.0f * float3(
        nclsNvidiaTanhApprox(prepared[16]),
        nclsNvidiaTanhApprox(prepared[17]),
        nclsNvidiaTanhApprox(prepared[18]));
    lobe0.angle = NCLS_PI * nclsNvidiaTanhApprox(prepared[19]);
    NclsLtcTransform lobe1;
    lobe1.inverseScale = exp(clamp(float2(prepared[20], prepared[21]), -4.0f, 4.0f));
    lobe1.shear = 2.0f * float3(
        nclsNvidiaTanhApprox(prepared[22]),
        nclsNvidiaTanhApprox(prepared[23]),
        nclsNvidiaTanhApprox(prepared[24]));
    lobe1.angle = NCLS_PI * nclsNvidiaTanhApprox(prepared[25]);
    const float learnedWeight = 0.5f + 0.5f * nclsNvidiaTanhApprox(prepared[26]);
    const float learnedMass = 1.0f - NCLS_UNIFIED_SAFETY_WEIGHT;
    return NCLS_UNIFIED_SAFETY_WEIGHT * no_diff(nclsCosineHemispherePdf(wi))
        + learnedMass * (
            learnedWeight * nclsLtcBasisUnchecked(lobe0, wi)
            + (1.0f - learnedWeight) * nclsLtcBasisUnchecked(lobe1, wi));
}

NclsDirectionalSample nclsSampleUnifiedNvidiaPrepared(
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT],
    float3 wo,
    float3 u,
    out uint component)
{
    float raw[NCLS_UNIFIED_NVIDIA_RAW_COUNT] = {};
    [unroll] for (uint index = 0u; index < NCLS_UNIFIED_NVIDIA_RAW_COUNT; ++index)
        raw[index] = prepared[14u + index];
    return nclsSampleNvidiaProposal(nclsDecodeNvidiaProposal(raw), wo, u, component);
}

NclsDirectionalSample nclsSampleUnifiedLtc(
    NclsUnifiedLtcProposal params,
    float3 u,
    out uint component)
{
    component = 0u;
    if (params.valid == 0u) return nclsInvalidDirectionalSample();
    const float learnedMass = 1.0f - NCLS_UNIFIED_SAFETY_WEIGHT;
    const float3 weights = float3(
        NCLS_UNIFIED_SAFETY_WEIGHT,
        learnedMass * params.learnedWeight,
        learnedMass * (1.0f - params.learnedWeight));
    const NclsMixture3Selection selection = nclsSelectMixture3(weights, u.x);
    if (selection.valid == 0u) return nclsInvalidDirectionalSample();
    component = selection.component;
    NclsDirectionalSample sample = component == 0u
        ? nclsSampleCosineDirection(u.yz)
        : (component == 1u ? nclsSampleLtc(params.lobe0, u.yz) : nclsSampleLtc(params.lobe1, u.yz));
    if (sample.valid != 0u && sample.nullEvent == 0u)
        sample.pdf = nclsUnifiedLtcPdf(params, sample.direction);
    return sample;
}


NclsDirectionalSample nclsSampleUnifiedLtcPrepared(
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT],
    float3 u,
    out uint component)
{
    float raw[NCLS_UNIFIED_LTC_RAW_COUNT] = {};
    [unroll] for (uint index = 0u; index < NCLS_UNIFIED_LTC_RAW_COUNT; ++index)
        raw[index] = prepared[14u + index];
    return nclsSampleUnifiedLtc(nclsDecodeUnifiedLtc(raw), u, component);
}

// SlangPy 0.43.1 会按公开 callable 缓存首次 active-gradient mask；
// evaluator 的 validation warm 与训练使用不同身份，但共享上面的唯一算术。
[Differentiable]
float[NCLS_UNIFIED_PREPARE_INPUT] nclsUnifiedPrepareInputTraining(
    float latent[NCLS_UNIFIED_LATENT_COUNT], no_diff float3 wo)
{
    return nclsUnifiedPrepareInput(latent, wo);
}

float[NCLS_UNIFIED_PREPARE_INPUT] nclsUnifiedPrepareInputInference(
    float latent[NCLS_UNIFIED_LATENT_COUNT], float3 wo)
{
    return nclsUnifiedPrepareInput(latent, wo);
}

[Differentiable]
float nclsUnifiedPrepareDot23Training(float weights[23], float bias, float input[23])
{
    return nclsUnifiedDot23(weights, bias, input);
}

float nclsUnifiedPrepareDot23Inference(float weights[23], float bias, float input[23])
{
    return nclsUnifiedDot23(weights, bias, input);
}

[Differentiable]
float nclsUnifiedPrepareReluTraining(float value)
{
    return nclsUnifiedLeakyRelu(value);
}

float nclsUnifiedPrepareReluInference(float value)
{
    return nclsUnifiedLeakyRelu(value);
}

[Differentiable]
float nclsUnifiedPrepareDot64Training(float weights[64], float bias, float input[64])
{
    return nclsUnifiedDot64(weights, bias, input);
}

float nclsUnifiedPrepareDot64Inference(float weights[64], float bias, float input[64])
{
    return nclsUnifiedDot64(weights, bias, input);
}

[Differentiable]
float nclsUnifiedEvaluatorStateDot64Training(float weights[64], float bias, float input[64])
{
    return nclsUnifiedDot64(weights, bias, input);
}

float nclsUnifiedEvaluatorStateDot64Inference(float weights[64], float bias, float input[64])
{
    return nclsUnifiedDot64(weights, bias, input);
}

[Differentiable]
float[NCLS_UNIFIED_PREPARE_OUTPUT] nclsUnifiedJoinEvaluatorStateTraining(
    float evaluator[14])
{
    return nclsUnifiedJoinEvaluatorState(evaluator);
}

float[NCLS_UNIFIED_PREPARE_OUTPUT] nclsUnifiedJoinEvaluatorStateInference(
    float evaluator[14])
{
    return nclsUnifiedJoinEvaluatorState(evaluator);
}

[Differentiable]
float[NCLS_UNIFIED_EVALUATE_INPUT] nclsUnifiedResidualFeaturesTraining(
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT],
    no_diff float3 responseScale,
    no_diff float3 core,
    no_diff float3 wo,
    no_diff float3 wi)
{
    return nclsUnifiedResidualFeatures(prepared, responseScale, core, wo, wi);
}

float[NCLS_UNIFIED_EVALUATE_INPUT] nclsUnifiedResidualFeaturesInference(
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT],
    float3 responseScale,
    float3 core,
    float3 wo,
    float3 wi)
{
    return nclsUnifiedResidualFeatures(prepared, responseScale, core, wo, wi);
}

[Differentiable]
float nclsUnifiedEvaluateDot17Training(float weights[17], float bias, float input[17])
{
    return nclsUnifiedDot17(weights, bias, input);
}

float nclsUnifiedEvaluateDot17Inference(float weights[17], float bias, float input[17])
{
    return nclsUnifiedDot17(weights, bias, input);
}

[Differentiable]
float nclsUnifiedEvaluateReluTraining(float value)
{
    return nclsUnifiedLeakyRelu(value);
}

float nclsUnifiedEvaluateReluInference(float value)
{
    return nclsUnifiedLeakyRelu(value);
}

[Differentiable]
float nclsUnifiedEvaluateDot32ATraining(float weights[32], float bias, float input[32])
{
    return nclsUnifiedDot32(weights, bias, input);
}

float nclsUnifiedEvaluateDot32AInference(float weights[32], float bias, float input[32])
{
    return nclsUnifiedDot32(weights, bias, input);
}

[Differentiable]
float nclsUnifiedEvaluateDot32OutTraining(float weights[32], float bias, float input[32])
{
    return nclsUnifiedDot32(weights, bias, input);
}

float nclsUnifiedEvaluateDot32OutInference(float weights[32], float bias, float input[32])
{
    return nclsUnifiedDot32(weights, bias, input);
}

[Differentiable]
float3 nclsUnifiedDecodeResidualTraining(
    float3 raw,
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT],
    no_diff float3 responseScale,
    no_diff float3 core)
{
    return nclsUnifiedDecodeResidual(raw, prepared, responseScale, core);
}

float3 nclsUnifiedDecodeResidualInference(
    float3 raw,
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT],
    float3 responseScale,
    float3 core)
{
    return nclsUnifiedDecodeResidual(raw, prepared, responseScale, core);
}

[Differentiable]
float nclsUnifiedNvidiaSamplerHeadDot64Training(
    float weights[64], float bias, float input[64])
{
    return nclsUnifiedDot64(weights, bias, input);
}

float nclsUnifiedNvidiaSamplerHeadDot64Inference(
    float weights[64], float bias, float input[64])
{
    return nclsUnifiedDot64(weights, bias, input);
}

[Differentiable]
float nclsUnifiedLtcSamplerHeadDot64Training(
    float weights[64], float bias, float input[64])
{
    return nclsUnifiedDot64(weights, bias, input);
}

float nclsUnifiedLtcSamplerHeadDot64Inference(
    float weights[64], float bias, float input[64])
{
    return nclsUnifiedDot64(weights, bias, input);
}

[Differentiable]
float[NCLS_UNIFIED_PREPARE_OUTPUT] nclsUnifiedJoinNvidiaSamplerStateTraining(
    float evaluator[14], float sampler[9])
{
    return nclsUnifiedJoinNvidiaState(evaluator, sampler);
}

float[NCLS_UNIFIED_PREPARE_OUTPUT] nclsUnifiedJoinNvidiaSamplerStateInference(
    float evaluator[14], float sampler[9])
{
    return nclsUnifiedJoinNvidiaState(evaluator, sampler);
}

[Differentiable]
float[NCLS_UNIFIED_PREPARE_OUTPUT] nclsUnifiedJoinLtcSamplerStateTraining(
    float evaluator[14], float sampler[13])
{
    return nclsUnifiedJoinLtcState(evaluator, sampler);
}

float[NCLS_UNIFIED_PREPARE_OUTPUT] nclsUnifiedJoinLtcSamplerStateInference(
    float evaluator[14], float sampler[13])
{
    return nclsUnifiedJoinLtcState(evaluator, sampler);
}

#endif
