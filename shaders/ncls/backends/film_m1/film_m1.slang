#ifndef NCLS_FILM_M1_SLANG
#define NCLS_FILM_M1_SLANG

// P1 M1-M 的固定 float32 推理布局。权重偏移由
// ncls.bundle.film_m1._serialize_runtime_weights 生成并写入 bundle layout。
static const uint NCLS_FILM_M1_WIDTH = 256u;
static const uint NCLS_FILM_M1_MAX_INPUT = 294u;
static const uint NCLS_FILM_M1_PREPARE_INPUT_COUNT = 263u;
static const uint NCLS_FILM_M1_DIRECTION_FEATURE_COUNT = 38u;
static const uint NCLS_FILM_M1_PREPARE_BLOCKS = 3u;
static const uint NCLS_FILM_M1_EVALUATE_BLOCKS = 6u;
static const uint NCLS_FILM_M1_FOURIER_BANDS = 5u;
static const float NCLS_FILM_M1_INV_SQRT_TWO = 0.7071067811865475244f;
static const float NCLS_FILM_M1_PI = 3.14159265358979323846f;

static const uint NCLS_FILM_M1_PREPARE_INPUT_WEIGHT = 0u;
static const uint NCLS_FILM_M1_PREPARE_INPUT_BIAS = 67328u;
static const uint NCLS_FILM_M1_PREPARE_BLOCK_BASE = 67584u;
static const uint NCLS_FILM_M1_EVALUATE_INPUT_WEIGHT = 463872u;
static const uint NCLS_FILM_M1_EVALUATE_INPUT_BIAS = 539136u;
static const uint NCLS_FILM_M1_EVALUATE_BLOCK_BASE = 539392u;
static const uint NCLS_FILM_M1_BLOCK_STRIDE = 132096u;
static const uint NCLS_FILM_M1_HEAD_NORM_WEIGHT = 1331968u;
static const uint NCLS_FILM_M1_HEAD_NORM_BIAS = 1332224u;
static const uint NCLS_FILM_M1_HEAD_WEIGHT = 1332480u;
static const uint NCLS_FILM_M1_HEAD_BIAS = 1333248u;
static const uint NCLS_FILM_M1_CONDITION = 1333251u;
static const uint NCLS_FILM_M1_OUTPUT_SCALE = 1338115u;
static const uint NCLS_FILM_M1_TOTAL_FLOATS = 1338118u;

struct NclsFilmM1State
{
    float prepared[NCLS_FILM_M1_WIDTH];
};

float nclsFilmM1Erf(float value)
{
    // Numerical Recipes 近似；与 Falcor MathHelpers 相同，最大误差约 1.2e-7。
    const float t = 1.0f / (1.0f + 0.5f * abs(value));
    float p = 0.17087277f;
    p = -0.82215223f + p * t;
    p = 1.48851587f + p * t;
    p = -1.13520398f + p * t;
    p = 0.27886807f + p * t;
    p = -0.18628806f + p * t;
    p = 0.09678418f + p * t;
    p = 0.37409196f + p * t;
    p = 1.00002368f + p * t;
    p = -1.26551223f + p * t;
    const float tau = t * exp(-value * value + p);
    return value >= 0.0f ? 1.0f - tau : tau - 1.0f;
}

float nclsFilmM1Gelu(float value)
{
    // torch.nn.functional.gelu 的默认 approximate="none" 路径。
    return 0.5f * value * (1.0f + nclsFilmM1Erf(value * NCLS_FILM_M1_INV_SQRT_TWO));
}

float nclsFilmM1Softplus(float value)
{
    return value > 20.0f ? value : log(1.0f + exp(value));
}

void nclsFilmM1LinearInput(
    StructuredBuffer<float> weights,
    uint weightOffset,
    uint biasOffset,
    uint inputCount,
    float input[NCLS_FILM_M1_MAX_INPUT],
    out float output[NCLS_FILM_M1_WIDTH])
{
    [loop]
    for (uint row = 0u; row < NCLS_FILM_M1_WIDTH; ++row)
    {
        float value = weights[biasOffset + row];
        [loop]
        for (uint column = 0u; column < inputCount; ++column)
            value += weights[weightOffset + row * inputCount + column] * input[column];
        output[row] = value;
    }
}

void nclsFilmM1Linear256(
    StructuredBuffer<float> weights,
    uint weightOffset,
    uint biasOffset,
    float input[NCLS_FILM_M1_WIDTH],
    out float output[NCLS_FILM_M1_WIDTH])
{
    [loop]
    for (uint row = 0u; row < NCLS_FILM_M1_WIDTH; ++row)
    {
        float value = weights[biasOffset + row];
        [loop]
        for (uint column = 0u; column < NCLS_FILM_M1_WIDTH; ++column)
            value += weights[weightOffset + row * NCLS_FILM_M1_WIDTH + column] * input[column];
        output[row] = value;
    }
}

void nclsFilmM1LayerNorm(
    StructuredBuffer<float> weights,
    uint weightOffset,
    uint biasOffset,
    float input[NCLS_FILM_M1_WIDTH],
    out float output[NCLS_FILM_M1_WIDTH])
{
    float mean = 0.0f;
    [loop]
    for (uint index = 0u; index < NCLS_FILM_M1_WIDTH; ++index) mean += input[index];
    mean *= 1.0f / float(NCLS_FILM_M1_WIDTH);
    float variance = 0.0f;
    [loop]
    for (uint index = 0u; index < NCLS_FILM_M1_WIDTH; ++index)
    {
        const float centered = input[index] - mean;
        variance += centered * centered;
    }
    variance *= 1.0f / float(NCLS_FILM_M1_WIDTH);
    const float inverseStandardDeviation = rsqrt(variance + 1e-5f);
    [loop]
    for (uint index = 0u; index < NCLS_FILM_M1_WIDTH; ++index)
        output[index] = (input[index] - mean) * inverseStandardDeviation * weights[weightOffset + index]
            + weights[biasOffset + index];
}

void nclsFilmM1ResidualBlock(
    StructuredBuffer<float> weights,
    uint blockBase,
    uint filmIndex,
    inout float value[NCLS_FILM_M1_WIDTH])
{
    const uint normWeight = blockBase;
    const uint normBias = blockBase + 256u;
    const uint linear1Weight = blockBase + 512u;
    const uint linear1Bias = blockBase + 66048u;
    const uint linear2Weight = blockBase + 66304u;
    const uint linear2Bias = blockBase + 131840u;
    const uint filmBase = NCLS_FILM_M1_CONDITION + NCLS_FILM_M1_WIDTH
        + filmIndex * 2u * NCLS_FILM_M1_WIDTH;
    float normalized[NCLS_FILM_M1_WIDTH];
    float hidden[NCLS_FILM_M1_WIDTH];
    float residual[NCLS_FILM_M1_WIDTH];
    nclsFilmM1LayerNorm(weights, normWeight, normBias, value, normalized);
    nclsFilmM1Linear256(weights, linear1Weight, linear1Bias, normalized, hidden);
    [loop]
    for (uint index = 0u; index < NCLS_FILM_M1_WIDTH; ++index)
        hidden[index] = nclsFilmM1Gelu(hidden[index]);
    nclsFilmM1Linear256(weights, linear2Weight, linear2Bias, hidden, residual);
    [loop]
    for (uint index = 0u; index < NCLS_FILM_M1_WIDTH; ++index)
    {
        const float gamma = weights[filmBase + index];
        const float beta = weights[filmBase + NCLS_FILM_M1_WIDTH + index];
        residual[index] = (1.0f + 0.1f * tanh(gamma)) * residual[index] + 0.1f * beta;
        value[index] = (value[index] + residual[index]) * NCLS_FILM_M1_INV_SQRT_TWO;
    }
}

NclsFilmM1State nclsFilmM1Prepare(float3 woLocal, StructuredBuffer<float> weights)
{
    NclsFilmM1State state;
    float input[NCLS_FILM_M1_MAX_INPUT];
    const float radial = sqrt(max(woLocal.x * woLocal.x + woLocal.y * woLocal.y, 0.0f));
    input[0] = woLocal.x;
    input[1] = woLocal.y;
    input[2] = woLocal.z;
    input[3] = woLocal.z;
    input[4] = radial;
    input[5] = -log(max(woLocal.z, 1e-4f)) / 10.0f;
    input[6] = woLocal.z * woLocal.z;
    [loop]
    for (uint index = 0u; index < NCLS_FILM_M1_WIDTH; ++index)
        input[7u + index] = weights[NCLS_FILM_M1_CONDITION + index];
    float prepared[NCLS_FILM_M1_WIDTH];
    nclsFilmM1LinearInput(
        weights,
        NCLS_FILM_M1_PREPARE_INPUT_WEIGHT,
        NCLS_FILM_M1_PREPARE_INPUT_BIAS,
        NCLS_FILM_M1_PREPARE_INPUT_COUNT,
        input,
        prepared);
    [loop]
    for (uint block = 0u; block < NCLS_FILM_M1_PREPARE_BLOCKS; ++block)
        nclsFilmM1ResidualBlock(
            weights,
            NCLS_FILM_M1_PREPARE_BLOCK_BASE + block * NCLS_FILM_M1_BLOCK_STRIDE,
            block,
            prepared);
    [loop]
    for (uint index = 0u; index < NCLS_FILM_M1_WIDTH; ++index)
        state.prepared[index] = prepared[index];
    return state;
}

void nclsFilmM1DirectionFeatures(
    float3 woLocal,
    float3 wiLocal,
    out float features[NCLS_FILM_M1_DIRECTION_FEATURE_COUNT])
{
    features[0] = woLocal.x;
    features[1] = woLocal.y;
    features[2] = woLocal.z;
    features[3] = wiLocal.x;
    features[4] = wiLocal.y;
    features[5] = wiLocal.z;
    features[6] = woLocal.z;
    features[7] = wiLocal.z;
    features[8] = dot(woLocal, wiLocal);

    const float3 halfSum = woLocal + wiLocal;
    const float halfLengthSquared = dot(halfSum, halfSum);
    const bool halfValid = halfLengthSquared > 1e-16f;
    const float3 halfVector = halfValid
        ? halfSum * rsqrt(max(halfLengthSquared, 1e-16f))
        : float3(0.0f, 0.0f, 1.0f);
    features[9] = halfVector.x;
    features[10] = halfVector.y;
    features[11] = halfVector.z;
    const float2 slope = halfVector.xy / max(halfVector.z, 1e-5f);
    const float2 logSlope = sign(slope) * log(1.0f + abs(slope));
    features[12] = logSlope.x;
    features[13] = logSlope.y;

    const float radial = sqrt(max(halfVector.x * halfVector.x + halfVector.y * halfVector.y, 0.0f));
    const float3 tangent = radial > 1e-6f
        ? float3(-halfVector.y, halfVector.x, 0.0f) / max(radial, 1e-6f)
        : float3(1.0f, 0.0f, 0.0f);
    const float3 bitangent = cross(halfVector, tangent);
    features[14] = dot(wiLocal, tangent);
    features[15] = dot(wiLocal, bitangent);
    features[16] = dot(wiLocal, halfVector);
    features[17] = halfValid ? 1.0f : 0.0f;

    [unroll]
    for (uint component = 0u; component < 2u; ++component)
    {
        const uint base = 18u + component * 2u * NCLS_FILM_M1_FOURIER_BANDS;
        [unroll]
        for (uint band = 0u; band < NCLS_FILM_M1_FOURIER_BANDS; ++band)
        {
            const float phase = NCLS_FILM_M1_PI * logSlope[component] * float(1u << band);
            features[base + band] = sin(phase);
            features[base + NCLS_FILM_M1_FOURIER_BANDS + band] = cos(phase);
        }
    }
}

float3 nclsFilmM1EvaluateF(
    NclsFilmM1State state,
    float3 woLocal,
    float3 wiLocal,
    StructuredBuffer<float> weights)
{
    float direction[NCLS_FILM_M1_DIRECTION_FEATURE_COUNT];
    nclsFilmM1DirectionFeatures(woLocal, wiLocal, direction);
    float input[NCLS_FILM_M1_MAX_INPUT];
    [loop]
    for (uint index = 0u; index < NCLS_FILM_M1_DIRECTION_FEATURE_COUNT; ++index)
        input[index] = direction[index];
    [loop]
    for (uint index = 0u; index < NCLS_FILM_M1_WIDTH; ++index)
        input[NCLS_FILM_M1_DIRECTION_FEATURE_COUNT + index] = state.prepared[index];
    float hidden[NCLS_FILM_M1_WIDTH];
    nclsFilmM1LinearInput(
        weights,
        NCLS_FILM_M1_EVALUATE_INPUT_WEIGHT,
        NCLS_FILM_M1_EVALUATE_INPUT_BIAS,
        NCLS_FILM_M1_MAX_INPUT,
        input,
        hidden);
    [loop]
    for (uint block = 0u; block < NCLS_FILM_M1_EVALUATE_BLOCKS; ++block)
        nclsFilmM1ResidualBlock(
            weights,
            NCLS_FILM_M1_EVALUATE_BLOCK_BASE + block * NCLS_FILM_M1_BLOCK_STRIDE,
            NCLS_FILM_M1_PREPARE_BLOCKS + block,
            hidden);

    float normalized[NCLS_FILM_M1_WIDTH];
    nclsFilmM1LayerNorm(
        weights,
        NCLS_FILM_M1_HEAD_NORM_WEIGHT,
        NCLS_FILM_M1_HEAD_NORM_BIAS,
        hidden,
        normalized);
    float3 raw = float3(
        weights[NCLS_FILM_M1_HEAD_BIAS + 0u],
        weights[NCLS_FILM_M1_HEAD_BIAS + 1u],
        weights[NCLS_FILM_M1_HEAD_BIAS + 2u]);
    [loop]
    for (uint column = 0u; column < NCLS_FILM_M1_WIDTH; ++column)
    {
        raw.x += weights[NCLS_FILM_M1_HEAD_WEIGHT + 0u * NCLS_FILM_M1_WIDTH + column] * normalized[column];
        raw.y += weights[NCLS_FILM_M1_HEAD_WEIGHT + 1u * NCLS_FILM_M1_WIDTH + column] * normalized[column];
        raw.z += weights[NCLS_FILM_M1_HEAD_WEIGHT + 2u * NCLS_FILM_M1_WIDTH + column] * normalized[column];
    }
    const float3 scale = float3(
        weights[NCLS_FILM_M1_OUTPUT_SCALE + 0u],
        weights[NCLS_FILM_M1_OUTPUT_SCALE + 1u],
        weights[NCLS_FILM_M1_OUTPUT_SCALE + 2u]);
    return float3(
        nclsFilmM1Softplus(raw.x),
        nclsFilmM1Softplus(raw.y),
        nclsFilmM1Softplus(raw.z)) * scale;
}

float3 nclsFilmM1ResponseCos(
    NclsFilmM1State state,
    float3 woLocal,
    float3 wiLocal,
    StructuredBuffer<float> weights)
{
    return nclsFilmM1EvaluateF(state, woLocal, wiLocal, weights) * max(wiLocal.z, 0.0f);
}

#endif
