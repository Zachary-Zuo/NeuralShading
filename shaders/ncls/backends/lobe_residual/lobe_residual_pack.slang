#ifndef NCLS_LOBE_RESIDUAL_PACK_SLANG
#define NCLS_LOBE_RESIDUAL_PACK_SLANG

// 部署侧 64 B state 的 half 打包/解包与按 state 求值的包装；训练路径不需要本文件。
#include "lobe_residual_core.slang"

uint nclsLobeResidualPackHalf2(float low, float high)
{
    return f32tof16(low) | (f32tof16(high) << 16u);
}

float2 nclsLobeResidualUnpackHalf2(uint bits)
{
    return float2(f16tof32(bits & 0xffffu), f16tof32(bits >> 16u));
}

NclsLobeResidualState nclsLobeResidualPack(uint materialSlot, NclsLobeResidualDecoded decoded)
{
    float values[NCLS_LOBE_RESIDUAL_LOBE_WORDS * 2u];
    [unroll]
    for (uint index = 0u; index < NCLS_LOBE_RESIDUAL_LOBE_WORDS * 2u; ++index)
        values[index] = 0.0f;
    [unroll]
    for (uint lobe = 0u; lobe < NCLS_LOBE_RESIDUAL_K; ++lobe)
    {
        const uint base = NCLS_LOBE_RESIDUAL_LOBE_PARAMS * lobe;
        values[base + 0u] = decoded.lobes[lobe].amplitude.x;
        values[base + 1u] = decoded.lobes[lobe].amplitude.y;
        values[base + 2u] = decoded.lobes[lobe].amplitude.z;
        values[base + 3u] = decoded.lobes[lobe].inverseScale.x;
        values[base + 4u] = decoded.lobes[lobe].inverseScale.y;
        values[base + 5u] = decoded.lobes[lobe].shear.x;
        values[base + 6u] = decoded.lobes[lobe].shear.y;
        values[base + 7u] = decoded.lobes[lobe].shear.z;
        values[base + 8u] = decoded.lobes[lobe].angle;
    }
    NclsLobeResidualState state;
    state.materialSlot = materialSlot;
    [unroll]
    for (uint word = 0u; word < NCLS_LOBE_RESIDUAL_LOBE_WORDS; ++word)
        state.lobeBits[word] = nclsLobeResidualPackHalf2(values[2u * word], values[2u * word + 1u]);
#if NCLS_LOBE_RESIDUAL_CORRECTION
    [unroll]
    for (uint word = 0u; word < NCLS_LOBE_RESIDUAL_HIDDEN / 2u; ++word)
        state.hiddenBits[word] = nclsLobeResidualPackHalf2(decoded.hidden[2u * word], decoded.hidden[2u * word + 1u]);
#endif
    state.flags = NCLS_LOBE_RESIDUAL_K | ((uint)NCLS_LOBE_RESIDUAL_CORRECTION << 8u);
    state.reserved = 0u;
    return state;
}

NclsLobeResidualDecoded nclsLobeResidualUnpack(NclsLobeResidualState state)
{
    float values[NCLS_LOBE_RESIDUAL_LOBE_WORDS * 2u];
    [unroll]
    for (uint word = 0u; word < NCLS_LOBE_RESIDUAL_LOBE_WORDS; ++word)
    {
        const float2 pair = nclsLobeResidualUnpackHalf2(state.lobeBits[word]);
        values[2u * word] = pair.x;
        values[2u * word + 1u] = pair.y;
    }
    NclsLobeResidualDecoded decoded;
    [unroll]
    for (uint lobe = 0u; lobe < NCLS_LOBE_RESIDUAL_K; ++lobe)
    {
        const uint base = NCLS_LOBE_RESIDUAL_LOBE_PARAMS * lobe;
        decoded.lobes[lobe].amplitude = float3(values[base + 0u], values[base + 1u], values[base + 2u]);
        decoded.lobes[lobe].inverseScale = float2(values[base + 3u], values[base + 4u]);
        decoded.lobes[lobe].shear = float3(values[base + 5u], values[base + 6u], values[base + 7u]);
        decoded.lobes[lobe].angle = values[base + 8u];
    }
    [unroll]
    for (uint word = 0u; word < NCLS_LOBE_RESIDUAL_HIDDEN / 2u; ++word)
    {
#if NCLS_LOBE_RESIDUAL_CORRECTION
        const float2 pair = nclsLobeResidualUnpackHalf2(state.hiddenBits[word]);
#else
        const float2 pair = float2(0.0f);
#endif
        decoded.hidden[2u * word] = pair.x;
        decoded.hidden[2u * word + 1u] = pair.y;
    }
    return decoded;
}

// 部署路径：解包 64 B state 后调用可微 evaluate；权重此时为常量缓冲。
float3 nclsLobeResidualEvaluateF(
    NclsLobeResidualState state,
    NclsLayerInterfaceIR top,
    float3 woLocal,
    float3 wiLocal,
    NclsLobeResidualParams params)
{
    return nclsLobeResidualEvaluateF(nclsLobeResidualUnpack(state), top, woLocal, wiLocal, params);
}

#endif
