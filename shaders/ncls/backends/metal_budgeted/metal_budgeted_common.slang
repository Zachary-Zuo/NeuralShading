#ifndef NCLS_METAL_BUDGETED_COMMON_SLANG
#define NCLS_METAL_BUDGETED_COMMON_SLANG

#include "metal_budgeted_layout.generated.slang"

#ifdef NCLS_METAL_SPATIAL
static const uint NCLS_METAL_BUDGETED_MAX_WIDTH = 137u;
static const uint NCLS_METAL_BUDGETED_COMPILED_WORD_COUNT = 192u;
static const uint NCLS_METAL_BUDGETED_COMPILED_LAYOUT_VERSION = 2u;
#else
static const uint NCLS_METAL_BUDGETED_MAX_WIDTH = 64u;
static const uint NCLS_METAL_BUDGETED_COMPILED_WORD_COUNT = 48u;
static const uint NCLS_METAL_BUDGETED_COMPILED_LAYOUT_VERSION = 1u;
#endif
static const uint NCLS_METAL_BUDGETED_PROGRAM_FLOAT_WORDS = 32u;
static const uint NCLS_METAL_BUDGETED_PROGRAM_FLAGS = 32u;
static const uint NCLS_METAL_BUDGETED_METADATA = 40u;

float nclsMetalBudgetedWeight(uint index)
{
    const uint bits = gNclsRuntimeWeights[index >> 1u];
    return f16tof32((index & 1u) == 0u ? bits & 0xffffu : bits >> 16u);
}

float nclsMetalBudgetedCompiledHalf(uint baseWord, uint halfIndex)
{
    const uint bits = gNclsCompiledMaterials[baseWord + (halfIndex >> 1u)];
    return f16tof32((halfIndex & 1u) == 0u ? bits & 0xffffu : bits >> 16u);
}

uint nclsMetalBudgetedCompiledFlag(uint baseWord, uint index)
{
    return gNclsCompiledMaterials[
        baseWord + NCLS_METAL_BUDGETED_PROGRAM_FLAGS + index];
}

uint nclsMetalBudgetedMetadata(uint baseWord, uint index)
{
    return gNclsCompiledMaterials[baseWord + NCLS_METAL_BUDGETED_METADATA + index];
}

float nclsMetalBudgetedSilu(float value)
{
    return value / (1.0f + exp(-value));
}

float nclsMetalBudgetedSoftplus(float value)
{
    const float y = exp(-abs(value));
    const float t = y / (2.0f + y);
    const float t2 = t * t;
    float power = t;
    float sum = 0.0f;
    [unroll]
    for (uint k = 0u; k < 8u; ++k)
    {
        sum += power / float(2u * k + 1u);
        power *= t2;
    }
    return max(value, 0.0f) + 2.0f * sum;
}

void nclsMetalBudgetedDense(
    uint weightOffset,
    uint biasOffset,
    uint inputCount,
    uint outputCount,
    uint activation,
    float input[NCLS_METAL_BUDGETED_MAX_WIDTH],
    out float output[NCLS_METAL_BUDGETED_MAX_WIDTH])
{
    [unroll]
    for (uint row = 0u; row < outputCount; ++row)
    {
        float value = nclsMetalBudgetedWeight(biasOffset + row);
        [unroll]
        for (uint column = 0u; column < inputCount; ++column)
        {
            value += nclsMetalBudgetedWeight(
                weightOffset + row * inputCount + column) * input[column];
        }
        output[row] = activation == 1u ? nclsMetalBudgetedSilu(value) : value;
    }
}

float3 nclsMetalBudgetedSafeNormalize(
    float3 value, float3 fallback, out uint valid)
{
    const float lengthSquared = dot(value, value);
    valid = isfinite(lengthSquared) && lengthSquared > 1e-12f ? 1u : 0u;
    return valid != 0u ? value * rsqrt(max(lengthSquared, 1e-12f)) : fallback;
}

struct NclsMetalBudgetedFrame
{
    float3 tangent;
    float3 bitangent;
    float3 normal;
};

NclsMetalBudgetedFrame nclsMetalBudgetedOrthonormalFrame(float3 normal)
{
    const float a = 1.0f / (1.0f + normal.z);
    NclsMetalBudgetedFrame result;
    result.tangent = float3(1.0f - normal.x * normal.x * a,
        -normal.x * normal.y * a, -normal.x);
    result.bitangent = float3(-normal.x * normal.y * a,
        1.0f - normal.y * normal.y * a, -normal.y);
    result.normal = normal;
    return result;
}

NclsMetalBudgetedFrame nclsMetalBudgetedLocalFrame(
    float2 slope, float angle)
{
    uint valid;
    const float3 normal = nclsMetalBudgetedSafeNormalize(
        float3(slope, 1.0f), float3(0.0f, 0.0f, 1.0f), valid);
    const NclsMetalBudgetedFrame base = nclsMetalBudgetedOrthonormalFrame(normal);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    NclsMetalBudgetedFrame result;
    result.tangent = cosine * base.tangent + sine * base.bitangent;
    result.bitangent = -sine * base.tangent + cosine * base.bitangent;
    result.normal = normal;
    return result;
}

float3 nclsMetalBudgetedToFrame(
    NclsMetalBudgetedFrame frame, float3 direction)
{
    return float3(
        dot(frame.tangent, direction),
        dot(frame.bitangent, direction),
        dot(frame.normal, direction));
}

struct NclsMetalBudgetedCompiledMaterial
{
    uint baseWord;
};

struct NclsMetalBudgetedPrepared
{
    float semantic[24];
    float viewState[8];
    float compactFrame[8];
    NclsMetalBudgetedFrame frames[2];
    float lobes[16];
    float proposal[12];
    float accessState[4];
#ifdef NCLS_METAL_SPATIAL
    float compactProposalFrame[8];
    NclsMetalBudgetedFrame proposalFrames[2];
#endif
    uint identityAndFlags[4];
    uint valid;
};

struct NclsMetalBudgetedPackedState
{
#ifdef NCLS_METAL_SPATIAL
    uint words[44];
#else
    uint words[40];
#endif
};

uint nclsMetalBudgetedPackHalf2(float first, float second)
{
    return f32tof16(first) | (f32tof16(second) << 16u);
}

void nclsMetalBudgetedPackHalfArray(
    inout NclsMetalBudgetedPackedState packed,
    uint wordOffset,
    uint count,
    float values[NCLS_METAL_BUDGETED_MAX_WIDTH])
{
    [loop]
    for (uint index = 0u; index < count; index += 2u)
    {
        packed.words[wordOffset + index / 2u] = nclsMetalBudgetedPackHalf2(
            values[index], values[index + 1u]);
    }
}

float nclsMetalBudgetedUnpackHalf(
    NclsMetalBudgetedPackedState packed, uint halfIndex)
{
    const uint bits = packed.words[halfIndex >> 1u];
    return f16tof32((halfIndex & 1u) == 0u ? bits & 0xffffu : bits >> 16u);
}

#endif
