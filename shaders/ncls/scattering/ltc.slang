#ifndef NCLS_SCATTERING_LTC_SLANG
#define NCLS_SCATTERING_LTC_SLANG

#include "cosine.slang"

struct NclsLtcTransform : IDifferentiable
{
    float2 inverseScale;
    float3 shear;
    float angle;
};

bool nclsLtcTransformValid(NclsLtcTransform transform)
{
    const float determinant = transform.inverseScale.x * transform.inverseScale.y;
    return nclsIsFiniteFloat2(transform.inverseScale)
        && nclsIsFiniteFloat3(transform.shear)
        && isfinite(transform.angle)
        && transform.inverseScale.x > 1e-6f
        && transform.inverseScale.y > 1e-6f
        && isfinite(determinant)
        && determinant > 1e-12f;
}

[Differentiable]
float3 nclsLtcQueryVectorUnchecked(NclsLtcTransform transform, no_diff float3 direction)
{
    const float cosine = cos(transform.angle);
    const float sine = sin(transform.angle);
    const float rotatedX = cosine * direction.x + sine * direction.y;
    const float rotatedY = -sine * direction.x + cosine * direction.y;
    return float3(
        transform.inverseScale.x * rotatedX
            + transform.shear.x * rotatedY
            + transform.shear.y * direction.z,
        transform.inverseScale.y * rotatedY + transform.shear.z * direction.z,
        direction.z);
}

[Differentiable]
float nclsLtcBasisUnchecked(NclsLtcTransform transform, no_diff float3 direction)
{
    const float3 query = nclsLtcQueryVectorUnchecked(transform, direction);
    const float normSquared = max(dot(query, query), 1e-10f);
    const float determinant = transform.inverseScale.x * transform.inverseScale.y;
    return determinant * max(query.z, 0.0f)
        / (NCLS_PI * normSquared * normSquared);
}

float nclsLtcPdf(NclsLtcTransform transform, float3 direction)
{
    if (!nclsLtcTransformValid(transform) || !nclsIsFiniteFloat3(direction)
        || direction.z <= NCLS_MIN_COS)
        return 0.0f;
    const float value = nclsLtcBasisUnchecked(transform, direction);
    return isfinite(value) && value >= 0.0f ? value : 0.0f;
}

[Differentiable]
float3 nclsLtcResponseCosUnchecked(
    NclsLtcTransform transform,
    float3 amplitude,
    no_diff float3 direction)
{
    return nclsLtcBasisUnchecked(transform, direction) * amplitude;
}

float3 nclsLtcInverseMap(NclsLtcTransform transform, float3 direction)
{
    const float z = direction.z;
    const float y = (direction.y - transform.shear.z * z) / transform.inverseScale.y;
    const float x = (direction.x - transform.shear.x * y - transform.shear.y * z)
        / transform.inverseScale.x;
    return nclsSafeNormalize(nclsRotateTangentFromLocal(float3(x, y, z), transform.angle));
}

NclsDirectionalSample nclsSampleLtc(NclsLtcTransform transform, float2 u)
{
    if (!nclsLtcTransformValid(transform)
        || !nclsIsUnitRandom(u.x) || !nclsIsUnitRandom(u.y))
        return nclsInvalidDirectionalSample();
    const float3 direction = nclsLtcInverseMap(transform, nclsSampleCosineHemisphere(u));
    if (!nclsIsFiniteFloat3(direction)) return nclsInvalidDirectionalSample();
    if (direction.z <= NCLS_MIN_COS) return nclsNullDirectionalSample(direction);
    return nclsContinuousDirectionalSample(direction, nclsLtcPdf(transform, direction));
}

#endif
