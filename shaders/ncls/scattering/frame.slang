#ifndef NCLS_SCATTERING_FRAME_SLANG
#define NCLS_SCATTERING_FRAME_SLANG

#include "common.slang"

struct NclsDirectionalFrame
{
    float3 tangent;
    float3 bitangent;
    float3 normal;
};

float3 nclsFrameToLocal(const NclsShadingFrame frame, const float3 direction)
{
    return float3(
        dot(direction, frame.tangent),
        dot(direction, frame.bitangent),
        dot(direction, frame.normal));
}

float3 nclsFrameToWorld(const NclsShadingFrame frame, const float3 direction)
{
    return direction.x * frame.tangent
        + direction.y * frame.bitangent
        + direction.z * frame.normal;
}

float3 nclsRotateTangentToLocal(float3 direction, float angle)
{
    const float cosine = cos(angle);
    const float sine = sin(angle);
    return float3(
        cosine * direction.x + sine * direction.y,
        -sine * direction.x + cosine * direction.y,
        direction.z);
}

float3 nclsRotateTangentFromLocal(float3 direction, float angle)
{
    const float cosine = cos(angle);
    const float sine = sin(angle);
    return float3(
        cosine * direction.x - sine * direction.y,
        sine * direction.x + cosine * direction.y,
        direction.z);
}

NclsDirectionalFrame nclsDirectionalFrameFromNormal(float3 normal)
{
    NclsDirectionalFrame frame;
    frame.normal = nclsSafeNormalize(normal);
    const float3 axis = abs(frame.normal.x) < 0.999f
        ? float3(1.0f, 0.0f, 0.0f)
        : float3(0.0f, 1.0f, 0.0f);
    frame.tangent = nclsSafeNormalize(axis - frame.normal * dot(frame.normal, axis));
    frame.bitangent = cross(frame.normal, frame.tangent);
    return frame;
}

NclsDirectionalFrame nclsDirectionalFrameFromSlope(float2 slope)
{
    return nclsDirectionalFrameFromNormal(
        nclsSafeNormalize(float3(-slope.x, -slope.y, 1.0f)));
}

float3 nclsDirectionalFrameToLocal(NclsDirectionalFrame frame, float3 direction)
{
    return float3(
        dot(direction, frame.tangent),
        dot(direction, frame.bitangent),
        dot(direction, frame.normal));
}

float3 nclsDirectionalFrameToWorld(NclsDirectionalFrame frame, float3 direction)
{
    return direction.x * frame.tangent
        + direction.y * frame.bitangent
        + direction.z * frame.normal;
}

#endif
