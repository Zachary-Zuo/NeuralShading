#ifndef NCLS_SCATTERING_COMMON_SLANG
#define NCLS_SCATTERING_COMMON_SLANG

#include "../contracts/scattering_contract.slang"

// 公共散射数学只使用 outward、local +z 约定；PDF 都相对于立体角。
static const float NCLS_PI = 3.14159265358979323846f;
static const float NCLS_INV_PI = 0.31830988618379067154f;
static const float NCLS_MIN_COS = 1e-6f;
static const float NCLS_UNIT_RANDOM_MAX = 0.99999994f;

struct NclsDirectionalSample
{
    float3 direction;
    float pdf;
    uint valid;
    uint nullEvent;
};

float3 nclsSafeNormalize(float3 value)
{
    const float lengthSquared = dot(value, value);
    return lengthSquared > 1e-20f && isfinite(lengthSquared)
        ? value * rsqrt(lengthSquared)
        : float3(0.0f, 0.0f, 1.0f);
}

bool nclsIsFiniteFloat2(float2 value)
{
    return all(isfinite(value));
}

bool nclsIsFiniteFloat3(float3 value)
{
    return all(isfinite(value));
}

bool nclsIsUnitRandom(float value)
{
    return isfinite(value) && value >= 0.0f && value < 1.0f;
}

float nclsClampUnitRandom(float value)
{
    return min(max(value, 0.0f), NCLS_UNIT_RANDOM_MAX);
}

NclsDirectionalSample nclsInvalidDirectionalSample()
{
    NclsDirectionalSample result = {};
    result.direction = float3(0.0f, 0.0f, 1.0f);
    return result;
}

NclsDirectionalSample nclsNullDirectionalSample(float3 direction)
{
    if (!nclsIsFiniteFloat3(direction)) return nclsInvalidDirectionalSample();
    NclsDirectionalSample result = {};
    result.direction = direction;
    result.valid = 1u;
    result.nullEvent = 1u;
    return result;
}

NclsDirectionalSample nclsContinuousDirectionalSample(float3 direction, float pdf)
{
    if (!nclsIsFiniteFloat3(direction) || !isfinite(pdf) || pdf <= 0.0f)
        return nclsInvalidDirectionalSample();
    NclsDirectionalSample result = {};
    result.direction = direction;
    result.pdf = pdf;
    result.valid = 1u;
    return result;
}

#endif
