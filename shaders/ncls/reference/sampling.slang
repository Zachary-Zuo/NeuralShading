#ifndef NCLS_REFERENCE_SAMPLING_SLANG
#define NCLS_REFERENCE_SAMPLING_SLANG

// LayerStack reference 只拥有随机数与体相函数；表面散射数学由公共层提供。
#include "../scattering/cosine.slang"
#include "../scattering/fresnel.slang"
#include "../scattering/ggx.slang"

uint nclsPcgHash(uint value)
{
    uint state = value * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct NclsRng
{
    uint state;

    __init(uint2 key)
    {
        state = nclsPcgHash(key.x ^ nclsPcgHash(key.y + 0x9e3779b9u));
    }

    [mutating] uint nextUInt()
    {
        state = state * 747796405u + 2891336453u;
        return nclsPcgHash(state);
    }

    [mutating] float nextFloat()
    {
        return min(float(nextUInt() >> 8u) * (1.0f / 16777216.0f), 0.99999994f);
    }

    [mutating] float2 nextFloat2()
    {
        return float2(nextFloat(), nextFloat());
    }
};

// 旧 RNG 入口保留为薄包装；随机数消费仍恰好是一次 nextFloat2()。
float3 nclsSampleGgxVisibleNormal(float3 w, float2 alpha, inout NclsRng rng)
{
    const float2 u = rng.nextFloat2();
    return nclsSampleGgxVisibleNormal(w, alpha, u);
}

float3 nclsSampleHenyeyGreenstein(float3 propagationDirection, float g, inout NclsRng rng)
{
    g = clamp(g, -0.99f, 0.99f);
    const float2 u = rng.nextFloat2();
    float cosTheta;
    if (abs(g) < 1e-3f)
    {
        cosTheta = 1.0f - 2.0f * u.x;
    }
    else
    {
        const float ratio = (1.0f - g * g) / (1.0f + g - 2.0f * g * u.x);
        cosTheta = (1.0f + g * g - ratio * ratio) / (2.0f * g);
    }
    cosTheta = clamp(cosTheta, -1.0f, 1.0f);
    const float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi = 2.0f * NCLS_PI * u.y;

    const float3 axis = nclsSafeNormalize(propagationDirection);
    const float3 tangent = abs(axis.z) < 0.999f
        ? nclsSafeNormalize(cross(float3(0.0f, 0.0f, 1.0f), axis))
        : float3(1.0f, 0.0f, 0.0f);
    const float3 bitangent = cross(axis, tangent);
    return nclsSafeNormalize(
        sinTheta * cos(phi) * tangent + sinTheta * sin(phi) * bitangent + cosTheta * axis);
}

float nclsHenyeyGreensteinPdf(float3 incomingPropagation, float3 outgoingPropagation, float g)
{
    g = clamp(g, -0.99f, 0.99f);
    const float cosine = clamp(dot(incomingPropagation, outgoingPropagation), -1.0f, 1.0f);
    const float denominator = max(1.0f + g * g - 2.0f * g * cosine, 1e-8f);
    return (1.0f - g * g) / (4.0f * NCLS_PI * denominator * sqrt(denominator));
}

#endif
