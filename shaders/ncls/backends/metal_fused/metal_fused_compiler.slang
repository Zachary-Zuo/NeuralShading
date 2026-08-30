#ifndef NCLS_METAL_FUSED_COMPILER_SLANG
#define NCLS_METAL_FUSED_COMPILER_SLANG

#include "metal_fused_common.slang"

groupshared float gNclsMetalCompilerHidden[32 * 64];
groupshared float gNclsMetalCompilerQkv[32 * 192];

float nclsMetalRawFloat(uint offset) { return asfloat(gNclsMetalRawParameters[offset]); }
uint nclsMetalRawUint(uint offset) { return gNclsMetalRawParameters[offset]; }

bool nclsMetalCompilerAnyActive()
{
    bool result = false;
    [unroll] for (uint token = 0u; token < 32u; ++token)
        result = result || nclsMetalRawUint(NCLS_METAL_RAW_PRESENCE + token) != 0u;
    return result;
}

bool nclsMetalCompilerTokenActive(uint token, bool anyActive)
{
    return nclsMetalRawUint(NCLS_METAL_RAW_PRESENCE + token) != 0u
        || (!anyActive && token == 0u);
}

void nclsMetalCompilerLayerNorm(
    uint token, uint weightOffset, uint biasOffset, out float output[64])
{
    const uint base = token * 64u;
    float mean = 0.0f;
    [unroll] for (uint index = 0u; index < 64u; ++index)
        mean += gNclsMetalCompilerHidden[base + index];
    mean *= 1.0f / 64.0f;
    float variance = 0.0f;
    [unroll] for (uint index = 0u; index < 64u; ++index)
    {
        const float centered = gNclsMetalCompilerHidden[base + index] - mean;
        variance += centered * centered;
    }
    const float inverse = rsqrt(variance * (1.0f / 64.0f) + 1e-5f);
    [unroll] for (uint index = 0u; index < 64u; ++index)
        output[index] = (gNclsMetalCompilerHidden[base + index] - mean) * inverse
            * nclsMetalWeight(weightOffset + index) + nclsMetalWeight(biasOffset + index);
}

void nclsMetalCompilerBlock(
    uint token, bool active, bool anyActive,
    uint inProjectionWeight, uint inProjectionBias,
    uint outProjectionWeight, uint outProjectionBias,
    uint linear1Weight, uint linear1Bias,
    uint linear2Weight, uint linear2Bias,
    uint norm1Weight, uint norm1Bias,
    uint norm2Weight, uint norm2Bias)
{
    float normalized[64];
    if (active)
    {
        nclsMetalCompilerLayerNorm(token, norm1Weight, norm1Bias, normalized);
        [loop]
        for (uint row = 0u; row < 192u; ++row)
        {
            float value = nclsMetalWeight(inProjectionBias + row);
            [unroll] for (uint column = 0u; column < 64u; ++column)
                value += nclsMetalWeight(inProjectionWeight + row * 64u + column)
                    * normalized[column];
            gNclsMetalCompilerQkv[token * 192u + row] = value;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (active)
    {
        float attention[64];
        [unroll]
        for (uint head = 0u; head < 4u; ++head)
        {
            float maximum = -3.402823466e38f;
            [unroll]
            for (uint key = 0u; key < 32u; ++key)
            {
                if (!nclsMetalCompilerTokenActive(key, anyActive)) continue;
                float score = 0.0f;
                [unroll] for (uint dimension = 0u; dimension < 16u; ++dimension)
                    score += gNclsMetalCompilerQkv[token * 192u + head * 16u + dimension]
                        * gNclsMetalCompilerQkv[key * 192u + 64u + head * 16u + dimension];
                maximum = max(maximum, 0.25f * score);
            }
            float denominator = 0.0f;
            float numerator[16] = {};
            [unroll]
            for (uint key = 0u; key < 32u; ++key)
            {
                if (!nclsMetalCompilerTokenActive(key, anyActive)) continue;
                float score = 0.0f;
                [unroll] for (uint dimension = 0u; dimension < 16u; ++dimension)
                    score += gNclsMetalCompilerQkv[token * 192u + head * 16u + dimension]
                        * gNclsMetalCompilerQkv[key * 192u + 64u + head * 16u + dimension];
                const float probability = exp(0.25f * score - maximum);
                denominator += probability;
                [unroll] for (uint dimension = 0u; dimension < 16u; ++dimension)
                    numerator[dimension] += probability
                        * gNclsMetalCompilerQkv[key * 192u + 128u + head * 16u + dimension];
            }
            [unroll] for (uint dimension = 0u; dimension < 16u; ++dimension)
                attention[head * 16u + dimension] = numerator[dimension] / max(denominator, 1e-20f);
        }
        float projected[64];
        [unroll]
        for (uint row = 0u; row < 64u; ++row)
        {
            float value = nclsMetalWeight(outProjectionBias + row);
            [unroll] for (uint column = 0u; column < 64u; ++column)
                value += nclsMetalWeight(outProjectionWeight + row * 64u + column) * attention[column];
            projected[row] = value;
        }
        [unroll] for (uint index = 0u; index < 64u; ++index)
            gNclsMetalCompilerHidden[token * 64u + index] += projected[index];
    }
    GroupMemoryBarrierWithGroupSync();
    if (active)
    {
        nclsMetalCompilerLayerNorm(token, norm2Weight, norm2Bias, normalized);
        float feedForward[256];
        [loop]
        for (uint row = 0u; row < 256u; ++row)
        {
            float value = nclsMetalWeight(linear1Bias + row);
            [unroll] for (uint column = 0u; column < 64u; ++column)
                value += nclsMetalWeight(linear1Weight + row * 64u + column) * normalized[column];
            feedForward[row] = nclsMetalGelu(value);
        }
        [unroll]
        for (uint row = 0u; row < 64u; ++row)
        {
            float value = nclsMetalWeight(linear2Bias + row);
            [loop] for (uint column = 0u; column < 256u; ++column)
                value += nclsMetalWeight(linear2Weight + row * 256u + column) * feedForward[column];
            gNclsMetalCompilerHidden[token * 64u + row] += value;
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

#include "metal_fused_compiler_heads.slang"

void nclsMetalCompilerGlobalToken(out float globalToken[NCLS_METAL_MAX_WIDTH])
{
    float input[NCLS_METAL_MAX_WIDTH] = {};
    [unroll] for (uint index = 0u; index < 16u; ++index)
        input[index] = nclsMetalRawFloat(NCLS_METAL_RAW_OPTICAL + index);
    float hidden[NCLS_METAL_MAX_WIDTH] = {};
    nclsMetalDense(NCLS_METAL_W_TYPED_COMPILER_OPTICAL_PROJECTION_0_WEIGHT,
        NCLS_METAL_W_TYPED_COMPILER_OPTICAL_PROJECTION_0_BIAS,
        16u, 64u, 1u, input, hidden);
    nclsMetalDense(NCLS_METAL_W_TYPED_COMPILER_OPTICAL_PROJECTION_2_WEIGHT,
        NCLS_METAL_W_TYPED_COMPILER_OPTICAL_PROJECTION_2_BIAS,
        64u, 64u, 0u, hidden, globalToken);
    const uint graph = nclsMetalRawUint(NCLS_METAL_RAW_GLOBAL);
    const uint schema = nclsMetalRawUint(NCLS_METAL_RAW_GLOBAL + 1u);
    const uint recipe = nclsMetalRawUint(NCLS_METAL_RAW_GLOBAL + 2u);
    const uint metal = nclsMetalRawUint(NCLS_METAL_RAW_GLOBAL + 3u);
    const uint finish = nclsMetalRawUint(NCLS_METAL_RAW_GLOBAL + 4u);
    [unroll]
    for (uint index = 0u; index < 64u; ++index)
        globalToken[index] += nclsMetalWeight(NCLS_METAL_W_TYPED_COMPILER_GRAPH_EMBEDDING_WEIGHT + graph * 64u + index)
            + nclsMetalWeight(NCLS_METAL_W_TYPED_COMPILER_SCHEMA_EMBEDDING_WEIGHT + schema * 64u + index)
            + nclsMetalWeight(NCLS_METAL_W_TYPED_COMPILER_RECIPE_EMBEDDING_WEIGHT + recipe * 64u + index)
            + nclsMetalWeight(NCLS_METAL_W_TYPED_COMPILER_METAL_EMBEDDING_WEIGHT + metal * 64u + index)
            + nclsMetalWeight(NCLS_METAL_W_TYPED_COMPILER_FINISH_EMBEDDING_WEIGHT + finish * 64u + index);
}

void nclsMetalCompilerFinish()
{
    const bool anyActive = nclsMetalCompilerAnyActive();
    float pooled[NCLS_METAL_MAX_WIDTH] = {};
    float count = 0.0f;
    [unroll]
    for (uint token = 0u; token < 32u; ++token)
    {
        if (!nclsMetalCompilerTokenActive(token, anyActive)) continue;
        count += 1.0f;
        [unroll] for (uint index = 0u; index < 64u; ++index)
            pooled[index] += gNclsMetalCompilerHidden[token * 64u + index];
    }
    float globalToken[NCLS_METAL_MAX_WIDTH] = {};
    nclsMetalCompilerGlobalToken(globalToken);
    [unroll] for (uint index = 0u; index < 64u; ++index)
        pooled[index] = pooled[index] / max(count, 1.0f) + globalToken[index];
    float latent[NCLS_METAL_MAX_WIDTH] = {};
    nclsMetalLayerNorm(NCLS_METAL_W_TYPED_COMPILER_OUTPUT_NORM_WEIGHT,
        NCLS_METAL_W_TYPED_COMPILER_OUTPUT_NORM_BIAS, 64u, pooled, latent);
    nclsMetalWriteCompilerHeads(latent);
}

void nclsMetalCompileMaterialGroup(uint3 groupThreadId, uint3 groupId)
{
    if (any(groupId != uint3(0u))) return;
    const uint token = groupThreadId.x;
    const bool anyActive = nclsMetalCompilerAnyActive();
    const bool active = nclsMetalCompilerTokenActive(token, anyActive);
    if (active)
    {
        float globalToken[NCLS_METAL_MAX_WIDTH] = {};
        nclsMetalCompilerGlobalToken(globalToken);
        if (!anyActive)
        {
            [unroll] for (uint index = 0u; index < 64u; ++index)
                gNclsMetalCompilerHidden[index] = globalToken[index];
        }
        else
        {
            float input[NCLS_METAL_MAX_WIDTH] = {};
            [unroll] for (uint component = 0u; component < 4u; ++component)
                input[component] = nclsMetalRawFloat(NCLS_METAL_RAW_CONTINUOUS + 4u * token + component);
            float hidden[NCLS_METAL_MAX_WIDTH] = {};
            float projection[NCLS_METAL_MAX_WIDTH] = {};
            nclsMetalDense(NCLS_METAL_W_TYPED_COMPILER_VALUE_PROJECTION_0_WEIGHT,
                NCLS_METAL_W_TYPED_COMPILER_VALUE_PROJECTION_0_BIAS,
                4u, 64u, 1u, input, hidden);
            nclsMetalDense(NCLS_METAL_W_TYPED_COMPILER_VALUE_PROJECTION_2_WEIGHT,
                NCLS_METAL_W_TYPED_COMPILER_VALUE_PROJECTION_2_BIAS,
                64u, 64u, 0u, hidden, projection);
            const uint semantic = nclsMetalRawUint(NCLS_METAL_RAW_SEMANTIC + token);
            const uint typeId = nclsMetalRawUint(NCLS_METAL_RAW_TYPE + token);
            const uint responsibility = nclsMetalRawUint(NCLS_METAL_RAW_RESPONSIBILITY + token);
            const uint discrete = nclsMetalRawUint(NCLS_METAL_RAW_DISCRETE + token) & 63u;
            const uint presence = nclsMetalRawUint(NCLS_METAL_RAW_PRESENCE + token);
            [unroll]
            for (uint index = 0u; index < 64u; ++index)
                gNclsMetalCompilerHidden[token * 64u + index] = globalToken[index] + projection[index]
                    + nclsMetalWeight(NCLS_METAL_W_TYPED_COMPILER_SEMANTIC_EMBEDDING_WEIGHT + semantic * 64u + index)
                    + nclsMetalWeight(NCLS_METAL_W_TYPED_COMPILER_TYPE_EMBEDDING_WEIGHT + typeId * 64u + index)
                    + nclsMetalWeight(NCLS_METAL_W_TYPED_COMPILER_RESPONSIBILITY_EMBEDDING_WEIGHT + responsibility * 64u + index)
                    + nclsMetalWeight(NCLS_METAL_W_TYPED_COMPILER_DISCRETE_EMBEDDING_WEIGHT + discrete * 64u + index)
                    + nclsMetalWeight(NCLS_METAL_W_TYPED_COMPILER_PRESENCE_EMBEDDING_WEIGHT + presence * 64u + index);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    nclsMetalCompilerBlock(token, active, anyActive,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_0_SELF_ATTN_IN_PROJ_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_0_SELF_ATTN_IN_PROJ_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_0_SELF_ATTN_OUT_PROJ_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_0_SELF_ATTN_OUT_PROJ_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_0_LINEAR1_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_0_LINEAR1_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_0_LINEAR2_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_0_LINEAR2_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_0_NORM1_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_0_NORM1_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_0_NORM2_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_0_NORM2_BIAS);
    nclsMetalCompilerBlock(token, active, anyActive,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_1_SELF_ATTN_IN_PROJ_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_1_SELF_ATTN_IN_PROJ_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_1_SELF_ATTN_OUT_PROJ_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_1_SELF_ATTN_OUT_PROJ_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_1_LINEAR1_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_1_LINEAR1_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_1_LINEAR2_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_1_LINEAR2_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_1_NORM1_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_1_NORM1_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_1_NORM2_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_1_NORM2_BIAS);
    nclsMetalCompilerBlock(token, active, anyActive,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_2_SELF_ATTN_IN_PROJ_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_2_SELF_ATTN_IN_PROJ_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_2_SELF_ATTN_OUT_PROJ_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_2_SELF_ATTN_OUT_PROJ_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_2_LINEAR1_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_2_LINEAR1_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_2_LINEAR2_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_2_LINEAR2_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_2_NORM1_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_2_NORM1_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_2_NORM2_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_2_NORM2_BIAS);
    nclsMetalCompilerBlock(token, active, anyActive,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_3_SELF_ATTN_IN_PROJ_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_3_SELF_ATTN_IN_PROJ_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_3_SELF_ATTN_OUT_PROJ_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_3_SELF_ATTN_OUT_PROJ_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_3_LINEAR1_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_3_LINEAR1_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_3_LINEAR2_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_3_LINEAR2_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_3_NORM1_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_3_NORM1_BIAS,
        NCLS_METAL_W_TYPED_COMPILER_BLOCKS_3_NORM2_WEIGHT, NCLS_METAL_W_TYPED_COMPILER_BLOCKS_3_NORM2_BIAS);
    if (token == 0u) nclsMetalCompilerFinish();
}

[shader("compute")]
[numthreads(32, 1, 1)]
void nclsCompileMaterial(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    nclsMetalCompileMaterialGroup(groupThreadId, groupId);
}

#endif
