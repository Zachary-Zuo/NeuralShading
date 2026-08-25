#ifndef NCLS_SCATTERING_COSINE_SLANG
#define NCLS_SCATTERING_COSINE_SLANG

#include "frame.slang"

float3 nclsSampleCosineHemisphere(float2 u)
{
    const float radius = sqrt(u.x);
    const float phi = 2.0f * NCLS_PI * u.y;
    const float2 xy = radius * float2(cos(phi), sin(phi));
    return float3(xy, sqrt(max(0.0f, 1.0f - u.x)));
}

float nclsCosineHemispherePdf(float3 direction)
{
    return nclsIsFiniteFloat3(direction) && direction.z > NCLS_MIN_COS
        ? direction.z * NCLS_INV_PI
        : 0.0f;
}

NclsDirectionalSample nclsSampleCosineDirection(float2 u)
{
    if (!nclsIsUnitRandom(u.x) || !nclsIsUnitRandom(u.y))
        return nclsInvalidDirectionalSample();
    const float3 direction = nclsSampleCosineHemisphere(u);
    return nclsContinuousDirectionalSample(direction, nclsCosineHemispherePdf(direction));
}

float nclsTiltedCosinePdf(float3 direction, float2 slope)
{
    if (!nclsIsFiniteFloat2(slope) || !nclsIsFiniteFloat3(direction)
        || direction.z <= NCLS_MIN_COS)
        return 0.0f;
    const NclsDirectionalFrame frame = nclsDirectionalFrameFromSlope(slope);
    const float localCosine = dot(direction, frame.normal);
    return localCosine > 0.0f ? localCosine * NCLS_INV_PI : 0.0f;
}

NclsDirectionalSample nclsSampleTiltedCosine(float2 slope, float2 u)
{
    if (!nclsIsFiniteFloat2(slope) || !nclsIsUnitRandom(u.x) || !nclsIsUnitRandom(u.y))
        return nclsInvalidDirectionalSample();
    const NclsDirectionalFrame frame = nclsDirectionalFrameFromSlope(slope);
    const float3 direction = nclsSafeNormalize(
        nclsDirectionalFrameToWorld(frame, nclsSampleCosineHemisphere(u)));
    if (!nclsIsFiniteFloat3(direction)) return nclsInvalidDirectionalSample();
    if (direction.z <= NCLS_MIN_COS) return nclsNullDirectionalSample(direction);
    return nclsContinuousDirectionalSample(direction, nclsTiltedCosinePdf(direction, slope));
}

#endif
