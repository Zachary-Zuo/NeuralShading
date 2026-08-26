#ifndef NCLS_UNIFIED_NEURAL_PACK_SLANG
#define NCLS_UNIFIED_NEURAL_PACK_SLANG

// 03 candidate 的 128 B CompiledMaterial 与 64 B prepare state；数值以 half 成对打包。
#include "unified_neural_core.slang"

struct NclsUnifiedCompiledMaterial
{
    NclsLayerInterfaceIR topInterface;
    uint latentBits[8];
    float3 responseScale;
    uint layoutVersion;
    uint flags;
    uint reserved[3];
};

struct NclsUnifiedPackedState
{
    uint payloadBits[16];
};

float nclsUnifiedReadHalf(StructuredBuffer<uint> values, uint index)
{
    const uint bits = values[index >> 1u];
    return f16tof32((index & 1u) == 0u ? (bits & 0xffffu) : (bits >> 16u));
}

float2 nclsUnifiedUnpackHalf2(uint bits)
{
    return float2(f16tof32(bits & 0xffffu), f16tof32(bits >> 16u));
}

uint nclsUnifiedPackHalf2(float low, float high)
{
    return f32tof16(low) | (f32tof16(high) << 16u);
}

void nclsUnifiedCompiledLatent(
    NclsUnifiedCompiledMaterial material,
    out float latent[NCLS_UNIFIED_LATENT_COUNT])
{
    [unroll]
    for (uint word = 0u; word < 8u; ++word)
    {
        const float2 pair = nclsUnifiedUnpackHalf2(material.latentBits[word]);
        latent[2u * word + 0u] = pair.x;
        latent[2u * word + 1u] = pair.y;
    }
}

NclsUnifiedPackedState nclsUnifiedPackPrepared(
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT])
{
    NclsUnifiedPackedState result = {};
    [unroll]
    for (uint word = 0u; word < 13u; ++word)
        result.payloadBits[word] = nclsUnifiedPackHalf2(
            prepared[2u * word], prepared[2u * word + 1u]);
    result.payloadBits[13] = nclsUnifiedPackHalf2(prepared[26], 0.0f);
    return result;
}

void nclsUnifiedUnpackPrepared(
    NclsUnifiedPackedState packed,
    out float prepared[NCLS_UNIFIED_PREPARE_OUTPUT])
{
    [unroll]
    for (uint word = 0u; word < 13u; ++word)
    {
        const float2 pair = nclsUnifiedUnpackHalf2(packed.payloadBits[word]);
        prepared[2u * word + 0u] = pair.x;
        prepared[2u * word + 1u] = pair.y;
    }
    prepared[26] = nclsUnifiedUnpackHalf2(packed.payloadBits[13]).x;
}

#endif
