// NVIDIA Real-Time Neural Appearance Models supplemental §2 定义的原规模 evaluator/sampler 唯一前向。
#ifndef NCLS_NVIDIA_NEURAL_APPEARANCE_CORE_SLANG
#define NCLS_NVIDIA_NEURAL_APPEARANCE_CORE_SLANG

#include "nvidia_neural_appearance_layout.slang"
#include "nvidia_neural_appearance_mlp.slang"
#include "../../scattering/nvidia_proposal.slang"

struct NclsNvidiaNeuralLearnedFrame : IDifferentiable
{
    float3 tangent;
    float3 bitangent;
    float3 normal;
};

struct NclsNvidiaNeuralPrepared : IDifferentiable
{
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT];
    NclsNvidiaNeuralLearnedFrame frames[2];
    float projectedView[6];
    float samplerRaw[NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT];
};

[Differentiable]
float nclsNvidiaNeuralRelu(float value)
{
    return max(value, 0.0f);
}

[Differentiable]
float nclsNvidiaNeuralFrameDot8(
    float weights[NCLS_NVIDIA_NEURAL_LATENT_COUNT],
    float input[NCLS_NVIDIA_NEURAL_LATENT_COUNT])
{
    float result = 0.0f;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++index)
        result += weights[index] * input[index];
    return result;
}

[Differentiable]
float nclsNvidiaNeuralEvaluatorDot20(
    float weights[NCLS_NVIDIA_NEURAL_EVALUATE_INPUT],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_EVALUATE_INPUT])
{
    float result = bias;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_EVALUATE_INPUT; ++index)
        result += weights[index] * input[index];
    return result;
}

[Differentiable]
float nclsNvidiaNeuralEvaluatorDot64A(
    float weights[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH])
{
    float result = bias;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH; ++index)
        result += weights[index] * input[index];
    return result;
}

[Differentiable]
float nclsNvidiaNeuralEvaluatorDot64B(
    float weights[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH])
{
    float result = bias;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH; ++index)
        result += weights[index] * input[index];
    return result;
}

[Differentiable]
float nclsNvidiaNeuralEvaluatorDot64Out(
    float weights[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH])
{
    float result = bias;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH; ++index)
        result += weights[index] * input[index];
    return result;
}

[Differentiable]
float nclsNvidiaNeuralSamplerDot11(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT])
{
    float result = bias;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_SAMPLER_INPUT; ++index)
        result += weights[index] * input[index];
    return result;
}

[Differentiable]
float nclsNvidiaNeuralSamplerDot32A(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    float result = bias;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH; ++index)
        result += weights[index] * input[index];
    return result;
}

[Differentiable]
float nclsNvidiaNeuralSamplerDot32B(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    float result = bias;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH; ++index)
        result += weights[index] * input[index];
    return result;
}

[Differentiable]
float nclsNvidiaNeuralSamplerDot32Out(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    float result = bias;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH; ++index)
        result += weights[index] * input[index];
    return result;
}

[Differentiable]
NclsNvidiaNeuralLearnedFrame nclsNvidiaNeuralLearnedFrame(
    float raw[NCLS_NVIDIA_NEURAL_FRAME_RAW_COUNT],
    no_diff uint offset)
{
    NclsNvidiaNeuralLearnedFrame frame;
    frame.normal = normalize(float3(
        raw[offset + 0u], raw[offset + 1u], raw[offset + 2u] + 1.0f));
    frame.tangent = normalize(float3(
        raw[offset + 3u] + 1.0f, raw[offset + 4u], raw[offset + 5u]));
    frame.bitangent = cross(frame.normal, frame.tangent);
    return frame;
}

[Differentiable]
float3 nclsNvidiaNeuralProject(
    NclsNvidiaNeuralLearnedFrame frame,
    no_diff float3 direction)
{
    return float3(
        dot(direction, frame.tangent),
        dot(direction, frame.bitangent),
        dot(direction, frame.normal));
}

[Differentiable]
float[NCLS_NVIDIA_NEURAL_EVALUATE_INPUT] nclsNvidiaNeuralEvaluationInput(
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT],
    float frameRaw[NCLS_NVIDIA_NEURAL_FRAME_RAW_COUNT],
    no_diff float3 projectWo,
    no_diff float3 projectWi)
{
    float result[NCLS_NVIDIA_NEURAL_EVALUATE_INPUT] = {};
    [unroll]
    for (uint frameIndex = 0u; frameIndex < 2u; ++frameIndex)
    {
        const uint rawOffset = 6u * frameIndex;
        const uint inputOffset = 6u * frameIndex;
        const NclsNvidiaNeuralLearnedFrame frame =
            nclsNvidiaNeuralLearnedFrame(frameRaw, rawOffset);
        // 论文 wi 对应项目固定 wo；论文 wo 对应项目 queried wi。
        const float3 paperWi = nclsNvidiaNeuralProject(frame, projectWo);
        const float3 paperWo = nclsNvidiaNeuralProject(frame, projectWi);
        result[inputOffset + 0u] = paperWi.x;
        result[inputOffset + 1u] = paperWi.y;
        result[inputOffset + 2u] = paperWi.z;
        result[inputOffset + 3u] = paperWo.x;
        result[inputOffset + 4u] = paperWo.y;
        result[inputOffset + 5u] = paperWo.z;
    }
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++index)
        result[12u + index] = latent[index];
    return result;
}

[Differentiable]
float[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT] nclsNvidiaNeuralSamplerInput(
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT],
    no_diff float3 projectWo)
{
    float result[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT] = {};
    result[0] = projectWo.x;
    result[1] = projectWo.y;
    result[2] = projectWo.z;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++index)
        result[3u + index] = latent[index];
    return result;
}

[Differentiable]
float3 nclsNvidiaNeuralDecodeF(float3 raw)
{
    return exp(raw - 3.0f);
}

[Differentiable]
float nclsNvidiaNeuralSamplerPdfTraining(
    float raw[NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT],
    no_diff float3 projectWo,
    no_diff float3 projectWi)
{
    const NclsNvidiaDifferentiableParams params =
        nclsDecodeNvidiaProposalUnchecked(raw);
    return nclsNvidiaProposalPdfUnchecked(params, projectWo, projectWi);
}

float nclsNvidiaNeuralSamplerPdf(
    float raw[NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT],
    float3 projectWo,
    float3 projectWi)
{
    return nclsNvidiaProposalPdf(
        nclsDecodeNvidiaProposal(raw), projectWo, projectWi);
}

NclsDirectionalSample nclsSampleNvidiaNeuralSampler(
    float raw[NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT],
    float3 projectWo,
    float2 u,
    out uint component)
{
    return nclsSampleNvidiaProposal(
        nclsDecodeNvidiaProposal(raw), projectWo, u, component);
}

float4 nclsNvidiaNeuralSamplerSample(
    float raw[NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT],
    float3 projectWo,
    float2 u)
{
    uint component;
    const NclsDirectionalSample sample = nclsSampleNvidiaNeuralSampler(
        raw, projectWo, u, component);
    return sample.valid != 0u && sample.nullEvent == 0u
        ? float4(sample.direction, 1.0f)
        : float4(0.0f, 0.0f, 1.0f, 0.0f);
}

[Differentiable]
NclsNvidiaNeuralPrepared nclsNvidiaNeuralPrepare(
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT],
    no_diff float3 projectWo,
    NclsNvidiaNeuralParams params)
{
    NclsNvidiaNeuralPrepared prepared;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++index)
        prepared.latent[index] = latent[index];
    float frameRaw[NCLS_NVIDIA_NEURAL_FRAME_RAW_COUNT];
    nclsNvidiaNeuralFrameRaw(latent, params, frameRaw);
    [unroll]
    for (uint frameIndex = 0u; frameIndex < 2u; ++frameIndex)
    {
        prepared.frames[frameIndex] = nclsNvidiaNeuralLearnedFrame(
            frameRaw, 6u * frameIndex);
        const float3 projected = nclsNvidiaNeuralProject(
            prepared.frames[frameIndex], projectWo);
        prepared.projectedView[3u * frameIndex + 0u] = projected.x;
        prepared.projectedView[3u * frameIndex + 1u] = projected.y;
        prepared.projectedView[3u * frameIndex + 2u] = projected.z;
    }
    const float[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT] samplerInput =
        nclsNvidiaNeuralSamplerInput(latent, projectWo);
    nclsNvidiaNeuralSamplerMlp(samplerInput, params, prepared.samplerRaw);
    return prepared;
}

[Differentiable]
float[NCLS_NVIDIA_NEURAL_EVALUATE_INPUT] nclsNvidiaNeuralPreparedEvaluationInput(
    NclsNvidiaNeuralPrepared prepared,
    no_diff float3 projectWi)
{
    float input[NCLS_NVIDIA_NEURAL_EVALUATE_INPUT] = {};
    [unroll]
    for (uint frameIndex = 0u; frameIndex < 2u; ++frameIndex)
    {
        const uint offset = 6u * frameIndex;
        input[offset + 0u] = prepared.projectedView[3u * frameIndex + 0u];
        input[offset + 1u] = prepared.projectedView[3u * frameIndex + 1u];
        input[offset + 2u] = prepared.projectedView[3u * frameIndex + 2u];
        const float3 projected = nclsNvidiaNeuralProject(
            prepared.frames[frameIndex], projectWi);
        input[offset + 3u] = projected.x;
        input[offset + 4u] = projected.y;
        input[offset + 5u] = projected.z;
    }
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++index)
        input[12u + index] = prepared.latent[index];
    return input;
}

[Differentiable]
float3 nclsNvidiaNeuralEvaluateF(
    NclsNvidiaNeuralPrepared prepared,
    no_diff float3 projectWi,
    NclsNvidiaNeuralParams params)
{
    const float[NCLS_NVIDIA_NEURAL_EVALUATE_INPUT] input =
        nclsNvidiaNeuralPreparedEvaluationInput(prepared, projectWi);
    float3 raw;
    nclsNvidiaNeuralEvaluateMlp(input, params, raw);
    return nclsNvidiaNeuralDecodeF(raw);
}

float nclsNvidiaNeuralPreparedPdf(
    NclsNvidiaNeuralPrepared prepared,
    float3 projectWo,
    float3 projectWi)
{
    return nclsNvidiaNeuralSamplerPdf(prepared.samplerRaw, projectWo, projectWi);
}

NclsDirectionalSample nclsSampleNvidiaNeuralPrepared(
    NclsNvidiaNeuralPrepared prepared,
    float3 projectWo,
    float2 u,
    out uint component)
{
    return nclsSampleNvidiaNeuralSampler(
        prepared.samplerRaw, projectWo, u, component);
}

// SlangPy 0.43.1 按公开 callable 缓存 active-gradient mask；训练与部署身份必须分离。
[Differentiable]
float nclsNvidiaNeuralReluTraining(float value)
{
    return nclsNvidiaNeuralRelu(value);
}

[Differentiable]
float nclsNvidiaNeuralReluInference(float value)
{
    return nclsNvidiaNeuralRelu(value);
}

[Differentiable]
float nclsNvidiaNeuralFrameDot8Training(
    float weights[NCLS_NVIDIA_NEURAL_LATENT_COUNT],
    float input[NCLS_NVIDIA_NEURAL_LATENT_COUNT])
{
    return nclsNvidiaNeuralFrameDot8(weights, input);
}

[Differentiable]
float nclsNvidiaNeuralFrameDot8Inference(
    float weights[NCLS_NVIDIA_NEURAL_LATENT_COUNT],
    float input[NCLS_NVIDIA_NEURAL_LATENT_COUNT])
{
    return nclsNvidiaNeuralFrameDot8(weights, input);
}

[Differentiable]
float nclsNvidiaNeuralEvaluatorDot20Training(
    float weights[NCLS_NVIDIA_NEURAL_EVALUATE_INPUT], float bias,
    float input[NCLS_NVIDIA_NEURAL_EVALUATE_INPUT])
{
    return nclsNvidiaNeuralEvaluatorDot20(weights, bias, input);
}

[Differentiable]
float nclsNvidiaNeuralEvaluatorDot20Inference(
    float weights[NCLS_NVIDIA_NEURAL_EVALUATE_INPUT], float bias,
    float input[NCLS_NVIDIA_NEURAL_EVALUATE_INPUT])
{
    return nclsNvidiaNeuralEvaluatorDot20(weights, bias, input);
}

[Differentiable]
float nclsNvidiaNeuralEvaluatorDot64ATraining(
    float weights[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH], float bias,
    float input[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH])
{
    return nclsNvidiaNeuralEvaluatorDot64A(weights, bias, input);
}

[Differentiable]
float nclsNvidiaNeuralEvaluatorDot64AInference(
    float weights[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH], float bias,
    float input[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH])
{
    return nclsNvidiaNeuralEvaluatorDot64A(weights, bias, input);
}

[Differentiable]
float nclsNvidiaNeuralEvaluatorDot64BTraining(
    float weights[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH], float bias,
    float input[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH])
{
    return nclsNvidiaNeuralEvaluatorDot64B(weights, bias, input);
}

[Differentiable]
float nclsNvidiaNeuralEvaluatorDot64BInference(
    float weights[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH], float bias,
    float input[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH])
{
    return nclsNvidiaNeuralEvaluatorDot64B(weights, bias, input);
}

[Differentiable]
float nclsNvidiaNeuralEvaluatorDot64OutTraining(
    float weights[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH], float bias,
    float input[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH])
{
    return nclsNvidiaNeuralEvaluatorDot64Out(weights, bias, input);
}

[Differentiable]
float nclsNvidiaNeuralEvaluatorDot64OutInference(
    float weights[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH], float bias,
    float input[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH])
{
    return nclsNvidiaNeuralEvaluatorDot64Out(weights, bias, input);
}

[Differentiable]
float nclsNvidiaNeuralSamplerDot11Training(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT], float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT])
{
    return nclsNvidiaNeuralSamplerDot11(weights, bias, input);
}

[Differentiable]
float nclsNvidiaNeuralSamplerDot11Inference(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT], float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT])
{
    return nclsNvidiaNeuralSamplerDot11(weights, bias, input);
}

[Differentiable]
float nclsNvidiaNeuralSamplerDot32ATraining(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH], float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    return nclsNvidiaNeuralSamplerDot32A(weights, bias, input);
}

[Differentiable]
float nclsNvidiaNeuralSamplerDot32AInference(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH], float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    return nclsNvidiaNeuralSamplerDot32A(weights, bias, input);
}

[Differentiable]
float nclsNvidiaNeuralSamplerDot32BTraining(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH], float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    return nclsNvidiaNeuralSamplerDot32B(weights, bias, input);
}

[Differentiable]
float nclsNvidiaNeuralSamplerDot32BInference(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH], float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    return nclsNvidiaNeuralSamplerDot32B(weights, bias, input);
}

[Differentiable]
float nclsNvidiaNeuralSamplerDot32OutTraining(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH], float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    return nclsNvidiaNeuralSamplerDot32Out(weights, bias, input);
}

[Differentiable]
float nclsNvidiaNeuralSamplerDot32OutInference(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH], float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    return nclsNvidiaNeuralSamplerDot32Out(weights, bias, input);
}

[Differentiable]
float[NCLS_NVIDIA_NEURAL_EVALUATE_INPUT] nclsNvidiaNeuralEvaluationInputTraining(
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT],
    float frameRaw[NCLS_NVIDIA_NEURAL_FRAME_RAW_COUNT],
    no_diff float3 projectWo, no_diff float3 projectWi)
{
    return nclsNvidiaNeuralEvaluationInput(latent, frameRaw, projectWo, projectWi);
}

[Differentiable]
float[NCLS_NVIDIA_NEURAL_EVALUATE_INPUT] nclsNvidiaNeuralEvaluationInputInference(
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT],
    float frameRaw[NCLS_NVIDIA_NEURAL_FRAME_RAW_COUNT],
    no_diff float3 projectWo, no_diff float3 projectWi)
{
    return nclsNvidiaNeuralEvaluationInput(latent, frameRaw, projectWo, projectWi);
}

[Differentiable]
float[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT] nclsNvidiaNeuralSamplerInputTraining(
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT], no_diff float3 projectWo)
{
    return nclsNvidiaNeuralSamplerInput(latent, projectWo);
}

[Differentiable]
float[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT] nclsNvidiaNeuralSamplerInputInference(
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT], no_diff float3 projectWo)
{
    return nclsNvidiaNeuralSamplerInput(latent, projectWo);
}

[Differentiable]
float3 nclsNvidiaNeuralDecodeFTraining(float3 raw)
{
    return nclsNvidiaNeuralDecodeF(raw);
}

[Differentiable]
float3 nclsNvidiaNeuralDecodeFInference(float3 raw)
{
    return nclsNvidiaNeuralDecodeF(raw);
}

#endif
