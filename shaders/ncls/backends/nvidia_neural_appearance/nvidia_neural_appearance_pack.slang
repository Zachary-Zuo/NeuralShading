#ifndef NCLS_NVIDIA_NEURAL_APPEARANCE_PACK_SLANG
#define NCLS_NVIDIA_NEURAL_APPEARANCE_PACK_SLANG

// NVIDIA baseline 的 32 B CompiledMaterial 与 96 B prepare state；数值以 half 成对打包。
#include "nvidia_neural_appearance_core.slang"

struct NclsNvidiaNeuralCompiledMaterial
{
    uint latentBits[4];
    uint layoutVersion;
    uint flags;
    uint reserved[2];
};

struct NclsNvidiaNeuralState
{
    uint payloadBits[24];
};

uint nclsNvidiaNeuralPackHalf2(float low, float high)
{
    return f32tof16(low) | (f32tof16(high) << 16u);
}

float2 nclsNvidiaNeuralUnpackHalf2(uint bits)
{
    return float2(f16tof32(bits & 0xffffu), f16tof32(bits >> 16u));
}

void nclsNvidiaNeuralCompiledLatent(
    NclsNvidiaNeuralCompiledMaterial material,
    out float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT])
{
    [unroll]
    for (uint word = 0u; word < 4u; ++word)
    {
        const float2 pair = nclsNvidiaNeuralUnpackHalf2(material.latentBits[word]);
        latent[2u * word + 0u] = pair.x;
        latent[2u * word + 1u] = pair.y;
    }
}

NclsNvidiaNeuralState nclsNvidiaNeuralPackPrepared(
    NclsNvidiaNeuralPrepared prepared)
{
    float values[48] = {};
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++index)
        values[index] = prepared.latent[index];
    [unroll]
    for (uint frameIndex = 0u; frameIndex < 2u; ++frameIndex)
    {
        const uint offset = 8u + 9u * frameIndex;
        values[offset + 0u] = prepared.frames[frameIndex].tangent.x;
        values[offset + 1u] = prepared.frames[frameIndex].tangent.y;
        values[offset + 2u] = prepared.frames[frameIndex].tangent.z;
        values[offset + 3u] = prepared.frames[frameIndex].bitangent.x;
        values[offset + 4u] = prepared.frames[frameIndex].bitangent.y;
        values[offset + 5u] = prepared.frames[frameIndex].bitangent.z;
        values[offset + 6u] = prepared.frames[frameIndex].normal.x;
        values[offset + 7u] = prepared.frames[frameIndex].normal.y;
        values[offset + 8u] = prepared.frames[frameIndex].normal.z;
    }
    [unroll]
    for (uint index = 0u; index < 6u; ++index)
        values[26u + index] = prepared.projectedView[index];
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT; ++index)
        values[32u + index] = prepared.samplerRaw[index];
    NclsNvidiaNeuralState state;
    [unroll]
    for (uint word = 0u; word < 24u; ++word)
        state.payloadBits[word] = nclsNvidiaNeuralPackHalf2(
            values[2u * word], values[2u * word + 1u]);
    return state;
}

NclsNvidiaNeuralPrepared nclsNvidiaNeuralUnpackPrepared(
    NclsNvidiaNeuralState state)
{
    float values[48];
    [unroll]
    for (uint word = 0u; word < 24u; ++word)
    {
        const float2 pair = nclsNvidiaNeuralUnpackHalf2(state.payloadBits[word]);
        values[2u * word + 0u] = pair.x;
        values[2u * word + 1u] = pair.y;
    }
    NclsNvidiaNeuralPrepared prepared;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++index)
        prepared.latent[index] = values[index];
    [unroll]
    for (uint frameIndex = 0u; frameIndex < 2u; ++frameIndex)
    {
        const uint offset = 8u + 9u * frameIndex;
        prepared.frames[frameIndex].tangent = float3(
            values[offset + 0u], values[offset + 1u], values[offset + 2u]);
        prepared.frames[frameIndex].bitangent = float3(
            values[offset + 3u], values[offset + 4u], values[offset + 5u]);
        prepared.frames[frameIndex].normal = float3(
            values[offset + 6u], values[offset + 7u], values[offset + 8u]);
    }
    [unroll]
    for (uint index = 0u; index < 6u; ++index)
        prepared.projectedView[index] = values[26u + index];
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT; ++index)
        prepared.samplerRaw[index] = values[32u + index];
    return prepared;
}

NclsNvidiaNeuralState nclsNvidiaNeuralPreparePacked(
    NclsNvidiaNeuralCompiledMaterial material,
    float3 projectWo,
    NclsNvidiaNeuralParams params)
{
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT];
    nclsNvidiaNeuralCompiledLatent(material, latent);
    return nclsNvidiaNeuralPackPrepared(
        nclsNvidiaNeuralPrepare(latent, projectWo, params));
}

float3 nclsNvidiaNeuralEvaluatePackedF(
    NclsNvidiaNeuralState state,
    float3 projectWi,
    NclsNvidiaNeuralParams params)
{
    return nclsNvidiaNeuralEvaluateF(
        nclsNvidiaNeuralUnpackPrepared(state), projectWi, params);
}

#endif
