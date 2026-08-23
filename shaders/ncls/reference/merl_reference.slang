#ifndef NCLS_MERL_REFERENCE_SLANG
#define NCLS_MERL_REFERENCE_SLANG

static const uint NCLS_MERL_THETA_HALF = 90u;
static const uint NCLS_MERL_THETA_DIFFERENCE = 90u;
static const uint NCLS_MERL_PHI_DIFFERENCE = 180u;

float3 nclsMerlRotateZ(float3 value, float angle)
{
    const float cosine = cos(angle);
    const float sine = sin(angle);
    return float3(cosine * value.x - sine * value.y, sine * value.x + cosine * value.y, value.z);
}

float3 nclsMerlRotateY(float3 value, float angle)
{
    const float cosine = cos(angle);
    const float sine = sin(angle);
    return float3(cosine * value.x + sine * value.z, value.y, -sine * value.x + cosine * value.z);
}

uint nclsMerlIndex(float3 viewDirection, float3 lightDirection)
{
    const float3 halfDirection = normalize(viewDirection + lightDirection);
    const float thetaHalf = acos(clamp(halfDirection.z, -1.0f, 1.0f));
    const float phiHalf = atan2(halfDirection.y, halfDirection.x);
    float3 difference = nclsMerlRotateZ(lightDirection, -phiHalf);
    difference = nclsMerlRotateY(difference, -thetaHalf);
    const float thetaDifference = acos(clamp(difference.z, -1.0f, 1.0f));
    float phiDifference = atan2(difference.y, difference.x);
    if (phiDifference < 0.0f) phiDifference += 3.14159265358979323846f;
    const uint thetaHalfIndex = min(
        uint(sqrt(max(thetaHalf, 0.0f) / (0.5f * 3.14159265358979323846f)) * float(NCLS_MERL_THETA_HALF)),
        NCLS_MERL_THETA_HALF - 1u);
    const uint thetaDifferenceIndex = min(
        uint(thetaDifference / (0.5f * 3.14159265358979323846f) * float(NCLS_MERL_THETA_DIFFERENCE)),
        NCLS_MERL_THETA_DIFFERENCE - 1u);
    const uint phiDifferenceIndex = min(
        uint(phiDifference / 3.14159265358979323846f * float(NCLS_MERL_PHI_DIFFERENCE)),
        NCLS_MERL_PHI_DIFFERENCE - 1u);
    return (thetaHalfIndex * NCLS_MERL_THETA_DIFFERENCE + thetaDifferenceIndex)
        * NCLS_MERL_PHI_DIFFERENCE + phiDifferenceIndex;
}

float3 nclsEvalMerlReference(
    StructuredBuffer<float3> brdfTable,
    float3 viewDirection,
    float3 lightDirection)
{
    if (viewDirection.z <= 0.0f || lightDirection.z <= 0.0f) return float3(0.0f);
    if (dot(viewDirection + lightDirection, viewDirection + lightDirection) <= 1e-20f) return float3(0.0f);
    return brdfTable[nclsMerlIndex(viewDirection, lightDirection)] * lightDirection.z;
}

#endif
