#ifndef NCLS_METAL_FUSED_COMMON_SLANG
#define NCLS_METAL_FUSED_COMMON_SLANG

#include "metal_fused_layout.generated.slang"

static const uint NCLS_METAL_MAX_WIDTH = 276u;
static const uint NCLS_METAL_HIDDEN_WIDTH = 128u;
static const uint NCLS_METAL_STRUCTURED_WIDTH = 64u;
static const uint NCLS_METAL_RAW_WORD_COUNT = 336u;
static const uint NCLS_METAL_COMPILED_WORD_COUNT = 384u;
static const uint NCLS_METAL_COMPILED_LAYOUT_VERSION = 1u;

static const uint NCLS_METAL_RAW_GLOBAL = 0u;
static const uint NCLS_METAL_RAW_SEMANTIC = 8u;
static const uint NCLS_METAL_RAW_TYPE = 40u;
static const uint NCLS_METAL_RAW_RESPONSIBILITY = 72u;
static const uint NCLS_METAL_RAW_DISCRETE = 104u;
static const uint NCLS_METAL_RAW_PRESENCE = 136u;
static const uint NCLS_METAL_RAW_CONTINUOUS = 168u;
static const uint NCLS_METAL_RAW_OPTICAL = 296u;
static const uint NCLS_METAL_RAW_ACCESS = 312u;
static const uint NCLS_METAL_RAW_FRAME = 328u;

static const uint NCLS_METAL_COMPILED_COMPILER = 0u;
static const uint NCLS_METAL_COMPILED_SPATIAL = 64u;
static const uint NCLS_METAL_COMPILED_CORE = 128u;
static const uint NCLS_METAL_COMPILED_RESIDUAL = 184u;
static const uint NCLS_METAL_COMPILED_BLOCK = 212u;
static const uint NCLS_METAL_COMPILED_PROPOSAL_LOGITS = 244u;
static const uint NCLS_METAL_COMPILED_PROPOSAL_MODULATION = 256u;
static const uint NCLS_METAL_COMPILED_TAIL_BOUNDS = 292u;
static const uint NCLS_METAL_COMPILED_FLAGS = 300u;
static const uint NCLS_METAL_COMPILED_ACCESS = 316u;
static const uint NCLS_METAL_COMPILED_FRAME = 332u;

float nclsMetalWeight(uint index)
{
    const uint bits = gNclsRuntimeWeights[index >> 1u];
    return f16tof32((index & 1u) == 0u ? bits & 0xffffu : bits >> 16u);
}

float nclsMetalCompiledFloat(uint baseWord, uint offset)
{
    return asfloat(gNclsCompiledMaterials[baseWord + offset]);
}

uint nclsMetalCompiledUint(uint baseWord, uint offset)
{
    return gNclsCompiledMaterials[baseWord + offset];
}

float nclsMetalSilu(float value)
{
    return value / (1.0f + exp(-value));
}

float nclsMetalGelu(float value)
{
    // PyTorch's exact GELU uses erf. Slang's target profiles do not expose
    // erf uniformly, so use the bounded 1.5e-7 Hastings approximation.
    const float x = value * 0.7071067811865475f;
    const float signValue = x < 0.0f ? -1.0f : 1.0f;
    const float absolute = abs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * absolute);
    const float polynomial = (((((1.061405429f * t - 1.453152027f) * t)
        + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    const float erfValue = signValue * (1.0f - polynomial * exp(-absolute * absolute));
    return 0.5f * value * (1.0f + erfValue);
}

float nclsMetalSoftplus(float value)
{
    return value > 20.0f ? value : log(1.0f + exp(value));
}

void nclsMetalDense(
    uint weightOffset,
    uint biasOffset,
    uint inputCount,
    uint outputCount,
    uint activation,
    float input[NCLS_METAL_MAX_WIDTH],
    out float output[NCLS_METAL_MAX_WIDTH])
{
    [loop]
    [MaxIters(NCLS_METAL_MAX_WIDTH)]
    for (uint row = 0u; row < outputCount; ++row)
    {
        float value = nclsMetalWeight(biasOffset + row);
        [loop]
        [MaxIters(NCLS_METAL_MAX_WIDTH)]
        for (uint column = 0u; column < inputCount; ++column)
            value += nclsMetalWeight(weightOffset + row * inputCount + column) * input[column];
        output[row] = activation == 1u ? nclsMetalSilu(value)
            : activation == 2u ? nclsMetalGelu(value) : value;
    }
}

void nclsMetalLayerNorm(
    uint weightOffset,
    uint biasOffset,
    uint count,
    float input[NCLS_METAL_MAX_WIDTH],
    out float output[NCLS_METAL_MAX_WIDTH])
{
    float mean = 0.0f;
    [loop]
    [MaxIters(NCLS_METAL_MAX_WIDTH)]
    for (uint index = 0u; index < count; ++index) mean += input[index];
    mean /= float(count);
    float variance = 0.0f;
    [loop]
    [MaxIters(NCLS_METAL_MAX_WIDTH)]
    for (uint index = 0u; index < count; ++index)
        variance += (input[index] - mean) * (input[index] - mean);
    const float inverse = rsqrt(variance / float(count) + 1e-5f);
    [loop]
    [MaxIters(NCLS_METAL_MAX_WIDTH)]
    for (uint index = 0u; index < count; ++index)
        output[index] = (input[index] - mean) * inverse
            * nclsMetalWeight(weightOffset + index) + nclsMetalWeight(biasOffset + index);
}

float3 nclsMetalSafeNormalize(float3 value, float3 fallback, out uint valid)
{
    const float lengthSquared = dot(value, value);
    valid = isfinite(lengthSquared) && lengthSquared > 1e-14f ? 1u : 0u;
    return valid != 0u ? value * rsqrt(lengthSquared) : fallback;
}

struct NclsMetalFrame
{
    float3 tangent;
    float3 bitangent;
    float3 normal;
};

float3 nclsMetalToFrame(NclsMetalFrame frame, float3 direction)
{
    return float3(dot(frame.tangent, direction), dot(frame.bitangent, direction), dot(frame.normal, direction));
}

float3 nclsMetalFromFrame(NclsMetalFrame frame, float3 direction)
{
    return direction.x * frame.tangent + direction.y * frame.bitangent + direction.z * frame.normal;
}

NclsMetalFrame nclsMetalBaseFrame()
{
    NclsMetalFrame result;
    result.tangent = float3(1.0f, 0.0f, 0.0f);
    result.bitangent = float3(0.0f, 1.0f, 0.0f);
    result.normal = float3(0.0f, 0.0f, 1.0f);
    return result;
}

struct NclsMetalCompiledMaterial
{
    uint baseWord;
};

struct NclsMetalPrepared
{
    float structured[64];
    float compiler[64];
    float viewToken[64];
    NclsMetalFrame frames[4];
    float core[54];
    float residual[28];
    float blockCondition[32];
    NclsMetalFusedProposal proposal;
    float correctionBound;
    float tailScale;
    uint valid;
};

#endif
