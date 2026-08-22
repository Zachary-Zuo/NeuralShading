#ifndef NCLS_SAMPLING_SLANG
#define NCLS_SAMPLING_SLANG

// Self-contained sampling and microfacet utilities for the layered teacher.
// GGX visible-normal sampling follows Heitz's 2018 formulation.

static const float NCLS_PI = 3.14159265358979323846f;
static const float NCLS_INV_PI = 0.31830988618379067154f;
static const float NCLS_MIN_COS = 1e-6f;

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

float3 nclsSafeNormalize(float3 value)
{
    const float lengthSquared = dot(value, value);
    return lengthSquared > 1e-20f ? value * rsqrt(lengthSquared) : float3(0.0f, 0.0f, 1.0f);
}

float3 nclsSampleCosineHemisphere(float2 u)
{
    const float radius = sqrt(u.x);
    const float phi = 2.0f * NCLS_PI * u.y;
    const float2 xy = radius * float2(cos(phi), sin(phi));
    return float3(xy, sqrt(max(0.0f, 1.0f - u.x)));
}

float nclsGgxLambda(float3 w, float2 alpha)
{
    const float cosThetaSquared = w.z * w.z;
    if (cosThetaSquared <= 1e-20f) return 1e20f;
    const float projectedRoughnessSquared = alpha.x * alpha.x * w.x * w.x + alpha.y * alpha.y * w.y * w.y;
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
    return abs(w.z) > NCLS_MIN_COS
        ? nclsGgxG1(w, alpha) * abs(dot(w, h)) * nclsGgxD(h, alpha) / abs(w.z)
        : 0.0f;
}

float3 nclsSampleGgxVisibleNormal(float3 w, float2 alpha, inout NclsRng rng)
{
    float3 wh = nclsSafeNormalize(float3(alpha.x * w.x, alpha.y * w.y, w.z));
    if (wh.z < 0.0f) wh = -wh;

    const float3 tangentX = dot(wh.xy, wh.xy) > 0.0f
        ? nclsSafeNormalize(float3(-wh.y, wh.x, 0.0f))
        : float3(1.0f, 0.0f, 0.0f);
    const float3 tangentY = cross(wh, tangentX);
    const float2 u = rng.nextFloat2();
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

float nclsFresnelDielectric(float etaIOverEtaT, float cosThetaI)
{
    cosThetaI = clamp(abs(cosThetaI), 0.0f, 1.0f);
    const float sinThetaTSquared = etaIOverEtaT * etaIOverEtaT * max(0.0f, 1.0f - cosThetaI * cosThetaI);
    if (sinThetaTSquared >= 1.0f) return 1.0f;
    const float cosThetaT = sqrt(max(0.0f, 1.0f - sinThetaTSquared));
    const float rs = (etaIOverEtaT * cosThetaI - cosThetaT) / (etaIOverEtaT * cosThetaI + cosThetaT);
    const float rp = (etaIOverEtaT * cosThetaT - cosThetaI) / (etaIOverEtaT * cosThetaT + cosThetaI);
    return 0.5f * (rs * rs + rp * rp);
}

float nclsFresnelConductorChannel(float eta, float k, float cosThetaI)
{
    cosThetaI = clamp(abs(cosThetaI), 0.0f, 1.0f);
    const float cosSquared = cosThetaI * cosThetaI;
    const float sinSquared = max(1.0f - cosSquared, 0.0f);
    const float sinFourth = sinSquared * sinSquared;
    const float inner = eta * eta - k * k - sinSquared;
    const float aSquaredPlusBSquared = sqrt(max(inner * inner + 4.0f * eta * eta * k * k, 0.0f));
    const float a = sqrt(max(0.5f * (aSquaredPlusBSquared + inner), 0.0f));
    const float rs = (aSquaredPlusBSquared + cosSquared - 2.0f * a * cosThetaI)
        / (aSquaredPlusBSquared + cosSquared + 2.0f * a * cosThetaI);
    const float rp = (cosSquared * aSquaredPlusBSquared + sinFourth - 2.0f * a * cosThetaI * sinSquared)
        / (cosSquared * aSquaredPlusBSquared + sinFourth + 2.0f * a * cosThetaI * sinSquared);
    return 0.5f * (rs + rs * rp);
}

float3 nclsFresnelConductor(float3 eta, float3 k, float cosThetaI)
{
    return float3(
        nclsFresnelConductorChannel(eta.x, k.x, cosThetaI),
        nclsFresnelConductorChannel(eta.y, k.y, cosThetaI),
        nclsFresnelConductorChannel(eta.z, k.z, cosThetaI));
}

#endif
