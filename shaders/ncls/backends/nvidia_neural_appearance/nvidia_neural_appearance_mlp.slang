#ifndef NCLS_NVIDIA_NEURAL_APPEARANCE_MLP_SLANG
#define NCLS_NVIDIA_NEURAL_APPEARANCE_MLP_SLANG

// 原规模网络的部署参数与 dense 原语；offset 由导出侧按模型反射结果写入。
#include "nvidia_neural_appearance_layout.slang"

#ifndef NCLS_NVIDIA_NEURAL_WEIGHTS_T
#define NCLS_NVIDIA_NEURAL_WEIGHTS_T StructuredBuffer<float>
#endif
#ifndef NCLS_NVIDIA_NEURAL_WEIGHT_READ
#define NCLS_NVIDIA_NEURAL_WEIGHT_READ(weights, index) (no_diff weights[index])
#endif

typedef NCLS_NVIDIA_NEURAL_WEIGHTS_T NclsNvidiaNeuralWeights;

struct NclsNvidiaNeuralParams
{
    NclsNvidiaNeuralWeights weights;
    uint frameWeight;
    uint evaluateWeight0;
    uint evaluateBias0;
    uint evaluateWeight1;
    uint evaluateBias1;
    uint evaluateWeight2;
    uint evaluateBias2;
    uint evaluateOutWeight;
    uint evaluateOutBias;
    uint samplerWeight0;
    uint samplerBias0;
    uint samplerWeight1;
    uint samplerBias1;
    uint samplerWeight2;
    uint samplerBias2;
    uint samplerOutWeight;
    uint samplerOutBias;
};

[Differentiable]
float nclsNvidiaNeuralWeight(
    NclsNvidiaNeuralParams params,
    no_diff uint index)
{
    return NCLS_NVIDIA_NEURAL_WEIGHT_READ(params.weights, index);
}

[Differentiable]
void nclsNvidiaNeuralDense64(
    NclsNvidiaNeuralParams params,
    no_diff uint weightOffset,
    no_diff uint biasOffset,
    no_diff uint inputCount,
    no_diff uint outputCount,
    no_diff bool activate,
    float input[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH],
    out float output[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH])
{
    [MaxIters(NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH)]
    for (uint row = 0u; row < outputCount; ++row)
    {
        float value = nclsNvidiaNeuralWeight(params, biasOffset + row);
        [MaxIters(NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH)]
        for (uint column = 0u; column < inputCount; ++column)
            value += nclsNvidiaNeuralWeight(
                params, weightOffset + row * inputCount + column) * input[column];
        output[row] = activate ? max(value, 0.0f) : value;
    }
}

[Differentiable]
void nclsNvidiaNeuralFrameRaw(
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT],
    NclsNvidiaNeuralParams params,
    out float raw[NCLS_NVIDIA_NEURAL_FRAME_RAW_COUNT])
{
    [unroll]
    for (uint row = 0u; row < NCLS_NVIDIA_NEURAL_FRAME_RAW_COUNT; ++row)
    {
        float value = 0.0f;
        [unroll]
        for (uint column = 0u; column < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++column)
            value += nclsNvidiaNeuralWeight(
                params,
                params.frameWeight + row * NCLS_NVIDIA_NEURAL_LATENT_COUNT + column
            ) * latent[column];
        raw[row] = value;
    }
}

[Differentiable]
void nclsNvidiaNeuralEvaluateMlp(
    float input[NCLS_NVIDIA_NEURAL_EVALUATE_INPUT],
    NclsNvidiaNeuralParams params,
    out float3 raw)
{
    float expanded[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    float hidden0[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    float hidden1[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    float hidden2[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    float output[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_EVALUATE_INPUT; ++index)
        expanded[index] = input[index];
    nclsNvidiaNeuralDense64(
        params, params.evaluateWeight0, params.evaluateBias0,
        NCLS_NVIDIA_NEURAL_EVALUATE_INPUT, NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH,
        true, expanded, hidden0);
    nclsNvidiaNeuralDense64(
        params, params.evaluateWeight1, params.evaluateBias1,
        NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH, NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH,
        true, hidden0, hidden1);
    nclsNvidiaNeuralDense64(
        params, params.evaluateWeight2, params.evaluateBias2,
        NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH, NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH,
        true, hidden1, hidden2);
    nclsNvidiaNeuralDense64(
        params, params.evaluateOutWeight, params.evaluateOutBias,
        NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH, NCLS_NVIDIA_NEURAL_RGB_COUNT,
        false, hidden2, output);
    raw = float3(output[0], output[1], output[2]);
}

[Differentiable]
void nclsNvidiaNeuralSamplerMlp(
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT],
    NclsNvidiaNeuralParams params,
    out float raw[NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT])
{
    float expanded[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    float hidden0[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    float hidden1[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    float hidden2[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    float output[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_SAMPLER_INPUT; ++index)
        expanded[index] = input[index];
    nclsNvidiaNeuralDense64(
        params, params.samplerWeight0, params.samplerBias0,
        NCLS_NVIDIA_NEURAL_SAMPLER_INPUT, NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH,
        true, expanded, hidden0);
    nclsNvidiaNeuralDense64(
        params, params.samplerWeight1, params.samplerBias1,
        NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH, NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH,
        true, hidden0, hidden1);
    nclsNvidiaNeuralDense64(
        params, params.samplerWeight2, params.samplerBias2,
        NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH, NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH,
        true, hidden1, hidden2);
    nclsNvidiaNeuralDense64(
        params, params.samplerOutWeight, params.samplerOutBias,
        NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH, NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT,
        false, hidden2, output);
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT; ++index)
        raw[index] = output[index];
}

#endif
