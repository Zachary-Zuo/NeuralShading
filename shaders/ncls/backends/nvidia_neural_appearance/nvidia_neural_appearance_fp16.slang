#ifndef NCLS_NVIDIA_NEURAL_APPEARANCE_FP16_SLANG
#define NCLS_NVIDIA_NEURAL_APPEARANCE_FP16_SLANG

// Supplemental §2.1 regular functional path: MLP inputs, weights, biases,
// accumulators, activations and outputs are half. Frame normalization,
// analytic proposal math and the public scattering ABI remain float.
half nclsNvidiaNeuralWeightFp16(
    NclsNvidiaNeuralParams params,
    uint index)
{
    return half(nclsNvidiaNeuralWeight(params, index));
}

void nclsNvidiaNeuralDenseFp16(
    NclsNvidiaNeuralParams params,
    uint weightOffset,
    uint biasOffset,
    uint inputCount,
    uint outputCount,
    bool activate,
    half input[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH],
    out half output[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH])
{
    [MaxIters(NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH)]
    for (uint row = 0u; row < outputCount; ++row)
    {
        half value = nclsNvidiaNeuralWeightFp16(params, biasOffset + row);
        [MaxIters(NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH)]
        for (uint column = 0u; column < inputCount; ++column)
            value += nclsNvidiaNeuralWeightFp16(
                params, weightOffset + row * inputCount + column) * input[column];
        output[row] = activate ? max(value, half(0.0f)) : value;
    }
}

void nclsNvidiaNeuralFrameRawFp16(
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT],
    NclsNvidiaNeuralParams params,
    out float raw[NCLS_NVIDIA_NEURAL_FRAME_RAW_COUNT])
{
    half input[NCLS_NVIDIA_NEURAL_LATENT_COUNT];
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++index)
        input[index] = half(latent[index]);
    [unroll]
    for (uint row = 0u; row < NCLS_NVIDIA_NEURAL_FRAME_RAW_COUNT; ++row)
    {
        half value = half(0.0f);
        [unroll]
        for (uint column = 0u; column < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++column)
            value += nclsNvidiaNeuralWeightFp16(
                params,
                params.frameWeight + row * NCLS_NVIDIA_NEURAL_LATENT_COUNT + column
            ) * input[column];
        raw[row] = float(value);
    }
}

void nclsNvidiaNeuralSamplerMlpFp16(
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT],
    float3 projectWo,
    NclsNvidiaNeuralParams params,
    out float raw[NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT])
{
    half input[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    half hidden0[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    half hidden1[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    half hidden2[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    half output[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    input[0] = half(projectWo.x);
    input[1] = half(projectWo.y);
    input[2] = half(projectWo.z);
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++index)
        input[3u + index] = half(latent[index]);
    nclsNvidiaNeuralDenseFp16(
        params, params.samplerWeight0, params.samplerBias0,
        NCLS_NVIDIA_NEURAL_SAMPLER_INPUT, NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH,
        true, input, hidden0);
    nclsNvidiaNeuralDenseFp16(
        params, params.samplerWeight1, params.samplerBias1,
        NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH, NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH,
        true, hidden0, hidden1);
    nclsNvidiaNeuralDenseFp16(
        params, params.samplerWeight2, params.samplerBias2,
        NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH, NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH,
        true, hidden1, hidden2);
    nclsNvidiaNeuralDenseFp16(
        params, params.samplerOutWeight, params.samplerOutBias,
        NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH, NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT,
        false, hidden2, output);
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT; ++index)
        raw[index] = float(output[index]);
}

NclsNvidiaNeuralPrepared nclsNvidiaNeuralPrepareFp16(
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT],
    float3 projectWo,
    NclsNvidiaNeuralParams params)
{
    NclsNvidiaNeuralPrepared prepared;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++index)
        prepared.latent[index] = float(half(latent[index]));
    float frameRaw[NCLS_NVIDIA_NEURAL_FRAME_RAW_COUNT];
    nclsNvidiaNeuralFrameRawFp16(prepared.latent, params, frameRaw);
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
    nclsNvidiaNeuralSamplerMlpFp16(
        prepared.latent, projectWo, params, prepared.samplerRaw);
    return prepared;
}

float3 nclsNvidiaNeuralEvaluateFFp16(
    NclsNvidiaNeuralPrepared prepared,
    float3 projectWi,
    NclsNvidiaNeuralParams params)
{
    half input[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    half hidden0[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    half hidden1[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    half hidden2[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    half output[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH] = {};
    [unroll]
    for (uint frameIndex = 0u; frameIndex < 2u; ++frameIndex)
    {
        const uint offset = 6u * frameIndex;
        input[offset + 0u] = half(prepared.projectedView[3u * frameIndex + 0u]);
        input[offset + 1u] = half(prepared.projectedView[3u * frameIndex + 1u]);
        input[offset + 2u] = half(prepared.projectedView[3u * frameIndex + 2u]);
        const float3 projected = nclsNvidiaNeuralProject(
            prepared.frames[frameIndex], projectWi);
        input[offset + 3u] = half(projected.x);
        input[offset + 4u] = half(projected.y);
        input[offset + 5u] = half(projected.z);
    }
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++index)
        input[12u + index] = half(prepared.latent[index]);
    nclsNvidiaNeuralDenseFp16(
        params, params.evaluateWeight0, params.evaluateBias0,
        NCLS_NVIDIA_NEURAL_EVALUATE_INPUT, NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH,
        true, input, hidden0);
    nclsNvidiaNeuralDenseFp16(
        params, params.evaluateWeight1, params.evaluateBias1,
        NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH, NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH,
        true, hidden0, hidden1);
    nclsNvidiaNeuralDenseFp16(
        params, params.evaluateWeight2, params.evaluateBias2,
        NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH, NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH,
        true, hidden1, hidden2);
    nclsNvidiaNeuralDenseFp16(
        params, params.evaluateOutWeight, params.evaluateOutBias,
        NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH, NCLS_NVIDIA_NEURAL_RGB_COUNT,
        false, hidden2, output);
    return float3(
        half(exp(float(output[0]) - 3.0f)),
        half(exp(float(output[1]) - 3.0f)),
        half(exp(float(output[2]) - 3.0f)));
}

#endif
