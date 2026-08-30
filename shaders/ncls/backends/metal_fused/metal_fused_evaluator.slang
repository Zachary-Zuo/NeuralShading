#ifndef NCLS_METAL_FUSED_EVALUATOR_SLANG
#define NCLS_METAL_FUSED_EVALUATOR_SLANG

#include "metal_fused_prepare.slang"

float nclsMetalAngularTexel(uint offset, uint size, uint channel, int x, int y)
{
    const uint clampedX = uint(clamp(x, 0, int(size) - 1));
    const uint clampedY = uint(clamp(y, 0, int(size) - 1));
    return nclsMetalWeight(offset + channel * size * size + clampedY * size + clampedX);
}

float nclsMetalAngularSample(uint offset, uint size, uint channel, float2 coordinate)
{
    const float2 position = clamp(0.5f * (coordinate + 1.0f), 0.0f, 1.0f) * float(size - 1u);
    const int2 low = int2(floor(position));
    const float2 weight = position - float2(low);
    const float top = lerp(
        nclsMetalAngularTexel(offset, size, channel, low.x, low.y),
        nclsMetalAngularTexel(offset, size, channel, low.x + 1, low.y), weight.x);
    const float bottom = lerp(
        nclsMetalAngularTexel(offset, size, channel, low.x, low.y + 1),
        nclsMetalAngularTexel(offset, size, channel, low.x + 1, low.y + 1), weight.x);
    return lerp(top, bottom, weight.y);
}

float nclsMetalDifferenceSample(uint offset, uint rank, float coordinate)
{
    const float position = clamp(0.5f * (coordinate + 1.0f), 0.0f, 1.0f) * 63.0f;
    const uint low = uint(floor(position));
    const uint high = min(low + 1u, 63u);
    return lerp(nclsMetalWeight(offset + rank * 64u + low),
        nclsMetalWeight(offset + rank * 64u + high), position - float(low));
}

float3 nclsMetalRotateXY(float3 value, float angle)
{
    const float cosine = cos(angle);
    const float sine = sin(angle);
    return float3(cosine * value.x + sine * value.y,
        -sine * value.x + cosine * value.y, value.z);
}

float nclsMetalGgx(float3 wo, float3 wi, float3 halfVector, float2 alpha)
{
    const float ax = max(alpha.x, 0.01f);
    const float ay = max(alpha.y, 0.01f);
    const float denominator = (halfVector.x / ax) * (halfVector.x / ax)
        + (halfVector.y / ay) * (halfVector.y / ay) + halfVector.z * halfVector.z;
    const float distribution = 1.0f / (3.141592653589793f * ax * ay * max(denominator * denominator, 1e-8f));
    const float cosO = max(wo.z, 1e-5f);
    const float cosI = max(wi.z, 1e-5f);
    const float lambdaO = 0.5f * (sqrt(1.0f
        + ((ax * wo.x) * (ax * wo.x) + (ay * wo.y) * (ay * wo.y)) / (cosO * cosO)) - 1.0f);
    const float lambdaI = 0.5f * (sqrt(1.0f
        + ((ax * wi.x) * (ax * wi.x) + (ay * wi.y) * (ay * wi.y)) / (cosI * cosI)) - 1.0f);
    return distribution / (1.0f + lambdaO + lambdaI) / max(4.0f * cosO * cosI, 1e-6f);
}

float nclsMetalBeckmann(float3 wo, float3 wi, float3 halfVector, float2 alpha)
{
    const float ax = max(alpha.x, 0.01f);
    const float ay = max(alpha.y, 0.01f);
    const float hz = max(halfVector.z, 1e-4f);
    const float exponent = -((halfVector.x / ax) * (halfVector.x / ax)
        + (halfVector.y / ay) * (halfVector.y / ay)) / (hz * hz);
    const float distribution = exp(max(exponent, -80.0f))
        / (3.141592653589793f * ax * ay * hz * hz * hz * hz);
    return distribution / max(4.0f * max(wo.z, 1e-5f) * max(wi.z, 1e-5f), 1e-6f);
}

float3 nclsMetalEvaluateF(NclsMetalPrepared prepared, float3 wo, float3 wi)
{
    if (prepared.valid == 0u || wo.z <= 0.0f || wi.z <= 0.0f) return float3(0.0f);
    uint halfValid;
    const float3 halfVector = nclsMetalSafeNormalize(wo + wi, float3(0.0f, 0.0f, 1.0f), halfValid);
    if (halfValid == 0u) return float3(0.0f);
    const float3 helper = abs(halfVector.z) < 0.999f
        ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    uint tangentValid;
    const float3 halfTangent = nclsMetalSafeNormalize(cross(helper, halfVector),
        float3(1.0f, 0.0f, 0.0f), tangentValid);
    const float3 difference = float3(
        dot(halfTangent, wi), dot(cross(halfVector, halfTangent), wi), dot(halfVector, wi));
    const float2 slope = halfVector.xy / max(halfVector.z, 0.02f);
    const float2 warped = slope / (1.0f + abs(slope));
    float features[NCLS_METAL_MAX_WIDTH] = {};
    uint cursor = 0u;
    features[cursor++] = wo.x; features[cursor++] = wo.y; features[cursor++] = wo.z;
    features[cursor++] = wi.x; features[cursor++] = wi.y; features[cursor++] = wi.z;
    features[cursor++] = dot(wo, wi); features[cursor++] = wo.z; features[cursor++] = wi.z;
    features[cursor++] = float(halfValid);
    features[cursor++] = halfVector.x; features[cursor++] = halfVector.y; features[cursor++] = halfVector.z;
    features[cursor++] = difference.x; features[cursor++] = difference.y; features[cursor++] = difference.z;
    features[cursor++] = warped.x; features[cursor++] = warped.y;
    [loop]
    for (uint frame = 1u; frame < 4u; ++frame)
    {
        const float3 projectedWo = nclsMetalToFrame(prepared.frames[frame], wo);
        const float3 projectedWi = nclsMetalToFrame(prepared.frames[frame], wi);
        features[cursor++] = projectedWo.x; features[cursor++] = projectedWo.y; features[cursor++] = projectedWo.z;
        features[cursor++] = projectedWi.x; features[cursor++] = projectedWi.y; features[cursor++] = projectedWi.z;
    }
    [loop]
    for (uint channel = 0u; channel < 8u; ++channel)
        features[cursor++] = nclsMetalAngularSample(
            NCLS_METAL_W_DIRECTIONAL_ANGULAR_BANK_LEVELS_0, 16u, channel, warped);
    [loop]
    for (uint channel = 0u; channel < 8u; ++channel)
        features[cursor++] = nclsMetalAngularSample(
            NCLS_METAL_W_DIRECTIONAL_ANGULAR_BANK_LEVELS_1, 32u, channel, warped);
    [loop]
    for (uint channel = 0u; channel < 8u; ++channel)
        features[cursor++] = nclsMetalAngularSample(
            NCLS_METAL_W_DIRECTIONAL_ANGULAR_BANK_LEVELS_2, 64u, channel, warped);
    [loop]
    for (uint channel = 0u; channel < 8u; ++channel)
        features[cursor++] = nclsMetalAngularSample(
            NCLS_METAL_W_DIRECTIONAL_ANGULAR_BANK_LEVELS_3, 128u, channel, warped);
    [loop]
    for (uint rank = 0u; rank < 16u; ++rank)
        features[cursor++] = nclsMetalDifferenceSample(
            NCLS_METAL_W_DIRECTIONAL_ANGULAR_BANK_DIFFERENCE_X, rank, difference.x)
            * nclsMetalDifferenceSample(
                NCLS_METAL_W_DIRECTIONAL_ANGULAR_BANK_DIFFERENCE_Y, rank, difference.y);
    [loop]
    for (uint index = 0u; index < 64u; ++index) features[cursor++] = prepared.structured[index];
    [loop]
    for (uint index = 0u; index < 64u; ++index) features[cursor++] = prepared.compiler[index];
    [loop]
    for (uint index = 0u; index < 64u; ++index) features[cursor++] = prepared.viewToken[index];
    if (cursor != 276u) return float3(0.0f);
    float hidden[NCLS_METAL_MAX_WIDTH] = {};
    nclsMetalDense(NCLS_METAL_W_HYBRID_INPUT_WEIGHT, NCLS_METAL_W_HYBRID_INPUT_BIAS,
        276u, 128u, 1u, features, hidden);
    float condition[8];
    [loop] for (uint index = 0u; index < 8u; ++index) condition[index] = prepared.blockCondition[index];
    nclsMetalConditionedBlock(
        NCLS_METAL_W_HYBRID_BLOCKS_0_LINEAR0_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_0_LINEAR0_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_0_LINEAR1_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_0_LINEAR1_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_0_FILM_SCALE_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_0_FILM_SCALE_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_0_FILM_BIAS_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_0_FILM_BIAS_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_0_LORA_DOWN_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_0_LORA_UP_WEIGHT,
        condition, hidden);
    [loop] for (uint index = 0u; index < 8u; ++index) condition[index] = prepared.blockCondition[8u + index];
    nclsMetalConditionedBlock(
        NCLS_METAL_W_HYBRID_BLOCKS_1_LINEAR0_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_1_LINEAR0_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_1_LINEAR1_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_1_LINEAR1_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_1_FILM_SCALE_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_1_FILM_SCALE_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_1_FILM_BIAS_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_1_FILM_BIAS_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_1_LORA_DOWN_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_1_LORA_UP_WEIGHT,
        condition, hidden);
    [loop] for (uint index = 0u; index < 8u; ++index) condition[index] = prepared.blockCondition[16u + index];
    nclsMetalConditionedBlock(
        NCLS_METAL_W_HYBRID_BLOCKS_2_LINEAR0_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_2_LINEAR0_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_2_LINEAR1_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_2_LINEAR1_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_2_FILM_SCALE_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_2_FILM_SCALE_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_2_FILM_BIAS_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_2_FILM_BIAS_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_2_LORA_DOWN_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_2_LORA_UP_WEIGHT,
        condition, hidden);
    [loop] for (uint index = 0u; index < 8u; ++index) condition[index] = prepared.blockCondition[24u + index];
    nclsMetalConditionedBlock(
        NCLS_METAL_W_HYBRID_BLOCKS_3_LINEAR0_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_3_LINEAR0_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_3_LINEAR1_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_3_LINEAR1_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_3_FILM_SCALE_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_3_FILM_SCALE_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_3_FILM_BIAS_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_3_FILM_BIAS_BIAS,
        NCLS_METAL_W_HYBRID_BLOCKS_3_LORA_DOWN_WEIGHT, NCLS_METAL_W_HYBRID_BLOCKS_3_LORA_UP_WEIGHT,
        condition, hidden);
    float coreF[3] = {0.0f, 0.0f, 0.0f};
    [loop]
    for (uint lobe = 0u; lobe < 6u; ++lobe)
    {
        const uint offset = 9u * lobe;
        const float3 color = float3(prepared.core[offset], prepared.core[offset + 1u], prepared.core[offset + 2u]);
        const float2 alpha = float2(prepared.core[offset + 3u], prepared.core[offset + 4u]);
        const float rotation = 3.141592653589793f * prepared.core[offset + 5u];
        const float energy = prepared.core[offset + 6u];
        const float active = prepared.core[offset + 7u];
        const float fresnelPower = prepared.core[offset + 8u];
        const float voh = clamp(dot(wo, halfVector), 0.0f, 1.0f);
        const float3 fresnel = color + (1.0f - color) * pow(1.0f - voh, 5.0f + 3.0f * fresnelPower);
        const float3 rotatedWo = nclsMetalRotateXY(wo, rotation);
        const float3 rotatedWi = nclsMetalRotateXY(wi, rotation);
        const float3 rotatedHalf = nclsMetalRotateXY(halfVector, rotation);
        const float ggx = nclsMetalGgx(rotatedWo, rotatedWi, rotatedHalf, alpha);
        const float beckmann = nclsMetalBeckmann(rotatedWo, rotatedWi, rotatedHalf, alpha);
        float3 candidate = lobe == 0u ? fresnel * ggx
            : lobe == 1u ? fresnel * beckmann
            : lobe == 2u ? (0.04f + 0.24f * color
                + (1.0f - (0.04f + 0.24f * color)) * pow(1.0f - voh, 5.0f)) * ggx
            : lobe == 3u ? color / 3.141592653589793f
            : lobe == 4u ? color * (0.2f + 0.8f * max(wi.z, 0.0f)) / 3.141592653589793f
            : fresnel * sqrt(max(ggx, 0.0f));
        candidate *= energy * active;
        [loop]
        for (uint channel = 0u; channel < 3u; ++channel)
            coreF[channel] += candidate[channel] * exp(nclsMetalWeight(
                NCLS_METAL_W_HYBRID_ANALYTIC_GAIN_LOG + 3u * lobe + channel));
    }
    float head[NCLS_METAL_MAX_WIDTH] = {};
    float correction[NCLS_METAL_MAX_WIDTH] = {};
    nclsMetalDense(NCLS_METAL_W_HYBRID_CORRECTION_HEAD_WEIGHT,
        NCLS_METAL_W_HYBRID_CORRECTION_HEAD_BIAS, 128u, 3u, 0u, hidden, correction);
    float amplitude[NCLS_METAL_MAX_WIDTH] = {};
    nclsMetalDense(NCLS_METAL_W_HYBRID_RESIDUAL_AMPLITUDE_HEAD_WEIGHT,
        NCLS_METAL_W_HYBRID_RESIDUAL_AMPLITUDE_HEAD_BIAS, 128u, 12u, 0u, hidden, amplitude);
    float tail[NCLS_METAL_MAX_WIDTH] = {};
    nclsMetalDense(NCLS_METAL_W_HYBRID_TAIL_HEAD_WEIGHT,
        NCLS_METAL_W_HYBRID_TAIL_HEAD_BIAS, 128u, 3u, 0u, hidden, tail);
    float3 result = float3(coreF[0], coreF[1], coreF[2])
        * exp(prepared.correctionBound * float3(tanh(correction[0]), tanh(correction[1]), tanh(correction[2])));
    [loop]
    for (uint lobe = 0u; lobe < 4u; ++lobe)
    {
        const uint offset = 7u * lobe;
        const float skew = prepared.residual[offset + 6u];
        const float2 alpha = clamp(float2(
            prepared.residual[offset + 3u] * exp(0.35f * skew),
            prepared.residual[offset + 4u] * exp(-0.35f * skew)), 0.01f, 1.0f);
        const float rotation = 1.570796326794897f * skew;
        const float shape = nclsMetalGgx(nclsMetalRotateXY(wo, rotation),
            nclsMetalRotateXY(wi, rotation), nclsMetalRotateXY(halfVector, rotation), alpha);
        const float3 color = float3(prepared.residual[offset],
            prepared.residual[offset + 1u], prepared.residual[offset + 2u]);
        result += float3(nclsMetalSoftplus(amplitude[3u * lobe]),
            nclsMetalSoftplus(amplitude[3u * lobe + 1u]),
            nclsMetalSoftplus(amplitude[3u * lobe + 2u]))
            * color * prepared.residual[offset + 5u] * shape;
    }
    result += prepared.tailScale * float3(
        nclsMetalSoftplus(tail[0]), nclsMetalSoftplus(tail[1]), nclsMetalSoftplus(tail[2]));
    return all(isfinite(result)) ? result : float3(0.0f);
}

#endif
