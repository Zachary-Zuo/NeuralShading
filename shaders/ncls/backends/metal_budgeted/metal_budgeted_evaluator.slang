#ifndef NCLS_METAL_BUDGETED_EVALUATOR_SLANG
#define NCLS_METAL_BUDGETED_EVALUATOR_SLANG

#include "metal_budgeted_common.slang"

float nclsMetalBudgetedGgx(
    float3 wo, float3 wi, float3 halfVector, float2 alpha)
{
    const float ax = max(alpha.x, 0.015f);
    const float ay = max(alpha.y, 0.015f);
    const float denominator =
        (halfVector.x / ax) * (halfVector.x / ax)
        + (halfVector.y / ay) * (halfVector.y / ay)
        + halfVector.z * halfVector.z;
    const float distribution = 1.0f / (
        3.14159265358979323846f * ax * ay
        * max(denominator * denominator, 1e-8f));
    const float cosO = max(wo.z, 1e-5f);
    const float cosI = max(wi.z, 1e-5f);
    const float lambdaO = 0.5f * (
        sqrt(1.0f + ((ax * wo.x) * (ax * wo.x)
            + (ay * wo.y) * (ay * wo.y)) / (cosO * cosO)) - 1.0f);
    const float lambdaI = 0.5f * (
        sqrt(1.0f + ((ax * wi.x) * (ax * wi.x)
            + (ay * wi.y) * (ay * wi.y)) / (cosI * cosI)) - 1.0f);
    const float geometry = 1.0f / (1.0f + lambdaO + lambdaI);
    return distribution * geometry / max(4.0f * cosO * cosI, 1e-6f);
}

float nclsMetalBudgetedBeckmann(
    float3 wo, float3 wi, float3 halfVector, float2 alpha)
{
    const float ax = max(alpha.x, 0.015f);
    const float ay = max(alpha.y, 0.015f);
    const float hz = max(halfVector.z, 1e-4f);
    const float exponent = -(
        (halfVector.x / ax) * (halfVector.x / ax)
        + (halfVector.y / ay) * (halfVector.y / ay)) / (hz * hz);
    const float distribution = exp(max(exponent, -80.0f)) / (
        3.14159265358979323846f * ax * ay * hz * hz * hz * hz);
    return distribution / max(4.0f * max(wo.z, 1e-5f)
        * max(wi.z, 1e-5f), 1e-6f);
}

float3 nclsMetalBudgetedAnalyticLobe(
    NclsMetalBudgetedPrepared prepared,
    uint lobeIndex,
    float3 wo,
    float3 wi,
    float3 halfVector)
{
    const NclsMetalBudgetedFrame frame = prepared.frames[lobeIndex];
    const float3 localWo = nclsMetalBudgetedToFrame(frame, wo);
    const float3 localWi = nclsMetalBudgetedToFrame(frame, wi);
    const float3 localHalf = nclsMetalBudgetedToFrame(frame, halfVector);
    const uint base = 8u * lobeIndex;
    const float2 alpha = float2(
        prepared.lobes[base + 3u], prepared.lobes[base + 4u]);
    const bool beckmann = lobeIndex == 0u
        && prepared.identityAndFlags[2] == 1u;
    const float shape = beckmann
        ? nclsMetalBudgetedBeckmann(localWo, localWi, localHalf, alpha)
        : nclsMetalBudgetedGgx(localWo, localWi, localHalf, alpha);
    const float voh = clamp(dot(localWo, localHalf), 0.0f, 1.0f);
    const float fresnelWeight = pow(1.0f - voh, 5.0f);
    const float3 color = float3(
        prepared.lobes[base],
        prepared.lobes[base + 1u],
        prepared.lobes[base + 2u]);
    const float3 fresnel = color + (1.0f - color) * fresnelWeight;
    const float active = localWo.z > 0.0f && localWi.z > 0.0f ? 1.0f : 0.0f;
    return fresnel * shape * prepared.lobes[base + 5u]
        * prepared.lobes[base + 7u] * active;
}

float3 nclsMetalBudgetedEvaluateF(
    NclsMetalBudgetedPrepared prepared, float3 wo, float3 wi)
{
    if (prepared.valid == 0u || wo.z <= 0.0f || wi.z <= 0.0f)
        return float3(0.0f);
    uint halfValid;
    const float3 halfVector = nclsMetalBudgetedSafeNormalize(
        wo + wi, float3(0.0f, 0.0f, 1.0f), halfValid);
    if (halfValid == 0u) return float3(0.0f);
    const NclsMetalBudgetedFrame halfFrame =
        nclsMetalBudgetedOrthonormalFrame(halfVector);
    const float3 difference = nclsMetalBudgetedToFrame(halfFrame, wi);
    const NclsMetalBudgetedFrame featureFrame = prepared.frames[1];
    const float3 woFeature = nclsMetalBudgetedToFrame(featureFrame, wo);
    const float3 wiFeature = nclsMetalBudgetedToFrame(featureFrame, wi);
    float features[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    [unroll]
    for (uint channel = 0u; channel < 3u; ++channel)
    {
        features[channel] = wo[channel];
        features[3u + channel] = wi[channel];
        features[6u + channel] = halfVector[channel];
        features[9u + channel] = difference[channel];
        features[12u + channel] = woFeature[channel];
        features[15u + channel] = wiFeature[channel];
    }
    features[18] = dot(wo, wi);
    features[19] = 1.0f;
    [unroll]
    for (uint index = 0u; index < 24u; ++index)
        features[20u + index] = prepared.semantic[index];
    float hidden0[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    float hidden1[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    float hidden2[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    float output[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    nclsMetalBudgetedDense(
        NCLS_METAL_BUDGETED_W_EVALUATOR_LAYERS_0_WEIGHT,
        NCLS_METAL_BUDGETED_W_EVALUATOR_LAYERS_0_BIAS,
        44u, 64u, 1u, features, hidden0);
    nclsMetalBudgetedDense(
        NCLS_METAL_BUDGETED_W_EVALUATOR_LAYERS_1_WEIGHT,
        NCLS_METAL_BUDGETED_W_EVALUATOR_LAYERS_1_BIAS,
        64u, 64u, 1u, hidden0, hidden1);
    nclsMetalBudgetedDense(
        NCLS_METAL_BUDGETED_W_EVALUATOR_LAYERS_2_WEIGHT,
        NCLS_METAL_BUDGETED_W_EVALUATOR_LAYERS_2_BIAS,
        64u, 64u, 1u, hidden1, hidden2);
    nclsMetalBudgetedDense(
        NCLS_METAL_BUDGETED_W_EVALUATOR_LAYERS_3_WEIGHT,
        NCLS_METAL_BUDGETED_W_EVALUATOR_LAYERS_3_BIAS,
        64u, 6u, 0u, hidden2, output);
    const float3 positive = float3(
        nclsMetalBudgetedSoftplus(output[0]),
        nclsMetalBudgetedSoftplus(output[1]),
        nclsMetalBudgetedSoftplus(output[2]));
    const uint mode = prepared.identityAndFlags[3] >> 16u;
    if (mode == 1u) return positive;
    const float3 gate = 1.0f / (1.0f + exp(-float3(
        output[3], output[4], output[5])));
    const float3 analytic = nclsMetalBudgetedAnalyticLobe(
        prepared, 0u, wo, wi, halfVector)
        + nclsMetalBudgetedAnalyticLobe(
            prepared, 1u, wo, wi, halfVector);
    const float3 result = positive + gate * analytic;
    return all(isfinite(result)) ? result : float3(0.0f);
}

#endif
