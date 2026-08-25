#ifndef NCLS_SCATTERING_GGX_SLANG
#define NCLS_SCATTERING_GGX_SLANG

#include "common.slang"

float nclsGgxLambda(float3 w, float2 alpha)
{
    const float cosThetaSquared = w.z * w.z;
    if (cosThetaSquared <= 1e-20f) return 1e20f;
    const float projectedRoughnessSquared = alpha.x * alpha.x * w.x * w.x
        + alpha.y * alpha.y * w.y * w.y;
    return 0.5f * (sqrt(1.0f + projectedRoughnessSquared / cosThetaSquared) - 1.0f);
}

float nclsGgxG1(float3 w, float2 alpha)
{
    return 1.0f / (1.0f + nclsGgxLambda(w, alpha));
}

float nclsGgxG2(float3 wi, float3 wo, float2 alpha)
{
    return 1.0f / (1.0f + nclsGgxLambda(wi, alpha) + nclsGgxLambda(wo, alpha));
}

float nclsGgxD(float3 h, float2 alpha)
{
    if (h.z <= 0.0f) return 0.0f;
    const float hx = h.x / alpha.x;
    const float hy = h.y / alpha.y;
    const float denominator = hx * hx + hy * hy + h.z * h.z;
    return 1.0f / (NCLS_PI * alpha.x * alpha.y * denominator * denominator);
}

float nclsGgxVisibleNormalPdf(float3 w, float3 h, float2 alpha)
{
    return abs(w.z) > NCLS_MIN_COS && w.z * dot(w, h) > 0.0f
        ? nclsGgxG1(w, alpha) * abs(dot(w, h)) * nclsGgxD(h, alpha) / abs(w.z)
        : 0.0f;
}

float3 nclsSampleGgxVisibleNormal(float3 w, float2 alpha, float2 u)
{
    float3 wh = nclsSafeNormalize(float3(alpha.x * w.x, alpha.y * w.y, w.z));
    if (wh.z < 0.0f) wh = -wh;

    const float3 tangentX = dot(wh.xy, wh.xy) > 0.0f
        ? nclsSafeNormalize(float3(-wh.y, wh.x, 0.0f))
        : float3(1.0f, 0.0f, 0.0f);
    const float3 tangentY = cross(wh, tangentX);
    const float radius = sqrt(u.x);
    const float phi = 2.0f * NCLS_PI * u.y;
    const float t1 = radius * cos(phi);
    float t2 = radius * sin(phi);
    const float blend = 0.5f * (1.0f + wh.z);
    t2 = (1.0f - blend) * sqrt(max(0.0f, 1.0f - t1 * t1)) + blend * t2;

    const float3 nh = t1 * tangentX + t2 * tangentY
        + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * wh;
    return nclsSafeNormalize(float3(alpha.x * nh.x, alpha.y * nh.y, max(0.0f, nh.z)));
}

struct NclsNoncenteredGgxParams
{
    float2 alpha;
    float rho;
    float2 meanSlope;
};

bool nclsNoncenteredGgxParamsValid(NclsNoncenteredGgxParams params)
{
    return nclsIsFiniteFloat2(params.alpha)
        && isfinite(params.rho)
        && nclsIsFiniteFloat2(params.meanSlope)
        && params.alpha.x >= 1e-4f
        && params.alpha.y >= 1e-4f
        && abs(params.rho) < 1.0f;
}

float nclsNoncenteredGgxPdf(
    NclsNoncenteredGgxParams params,
    float3 wo,
    float3 wi)
{
    if (!nclsNoncenteredGgxParamsValid(params)
        || !nclsIsFiniteFloat3(wo) || !nclsIsFiniteFloat3(wi)
        || wo.z <= NCLS_MIN_COS || wi.z <= NCLS_MIN_COS)
        return 0.0f;
    const float3 sum = wo + wi;
    if (dot(sum, sum) <= 1e-20f) return 0.0f;
    float3 halfVector = nclsSafeNormalize(sum);
    halfVector *= halfVector.z >= 0.0f ? 1.0f : -1.0f;
    if (halfVector.z <= 1e-4f) return 0.0f;
    const float woDotHalf = dot(wo, halfVector);
    if (abs(woDotHalf) <= 1e-8f) return 0.0f;

    float slopeX = -halfVector.x / halfVector.z - params.meanSlope.x;
    float slopeY = -halfVector.y / halfVector.z - params.meanSlope.y;
    const float sqrtOneMinusRho = sqrt(max(1.0f - params.rho * params.rho, 1e-12f));
    const float normalization = 1.0f
        / (params.alpha.x * params.alpha.y * sqrtOneMinusRho);
    const float slopeXStandard = slopeX / params.alpha.x;
    const float slopeYStandard = (
        params.alpha.x * slopeY - params.rho * params.alpha.y * slopeX)
        * normalization;
    const float radiusSquared = slopeXStandard * slopeXStandard
        + slopeYStandard * slopeYStandard;
    const float denominator = 1.0f + radiusSquared;
    const float p22Standard = 1.0f / (NCLS_PI * denominator * denominator);
    const float p22 = p22Standard * normalization;
    const float halfPdf = p22
        / (halfVector.z * halfVector.z * halfVector.z);
    const float value = halfPdf / (4.0f * abs(woDotHalf));
    return isfinite(value) && value >= 0.0f ? value : 0.0f;
}

NclsDirectionalSample nclsSampleNoncenteredGgx(
    NclsNoncenteredGgxParams params,
    float3 wo,
    float2 u)
{
    if (!nclsNoncenteredGgxParamsValid(params)
        || !nclsIsFiniteFloat3(wo) || wo.z <= NCLS_MIN_COS
        || !nclsIsUnitRandom(u.x) || !nclsIsUnitRandom(u.y))
        return nclsInvalidDirectionalSample();
    const float oneMinusU = max(1.0f - u.x, 1e-12f);
    const float radius = sqrt(u.x) / sqrt(oneMinusU);
    const float phi = 2.0f * NCLS_PI * u.y;
    const float slopeXStandard = radius * cos(phi);
    const float slopeYStandard = radius * sin(phi);
    const float sqrtOneMinusRho = sqrt(max(1.0f - params.rho * params.rho, 1e-12f));
    const float slopeX = params.alpha.x * slopeXStandard + params.meanSlope.x;
    const float slopeY = params.alpha.y
        * (params.rho * slopeXStandard + sqrtOneMinusRho * slopeYStandard)
        + params.meanSlope.y;
    const float3 halfVector = nclsSafeNormalize(float3(-slopeX, -slopeY, 1.0f));
    const float3 direction = 2.0f * dot(wo, halfVector) * halfVector - wo;
    if (!nclsIsFiniteFloat3(direction)) return nclsInvalidDirectionalSample();
    if (direction.z <= NCLS_MIN_COS || dot(wo, halfVector) <= 0.0f)
        return nclsNullDirectionalSample(direction);
    return nclsContinuousDirectionalSample(
        direction, nclsNoncenteredGgxPdf(params, wo, direction));
}

#endif
