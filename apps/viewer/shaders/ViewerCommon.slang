#ifndef NCLS_VIEWER_COMMON_SLANG
#define NCLS_VIEWER_COMMON_SLANG

static const float NCLS_VIEWER_PI = 3.14159265358979323846f;
// 高动态环境的 path-tracing 对照统一使用多样本 MIS。reference 与
// ScatteringPackage path tracer 必须共享这个数，避免 estimator 漂移。
static const uint NCLS_VIEWER_ENVIRONMENT_NEE_SAMPLE_COUNT = 4u;

uint nclsViewerHash(uint value)
{
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    return value ^ (value >> 16u);
}

float nclsViewerHashFloat(uint value)
{
    return float(nclsViewerHash(value) >> 8u) * (1.0f / 16777216.0f);
}

float nclsViewerRadicalInverse(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;
}

float2 nclsViewerHammersley(uint index, uint count)
{
    return float2((float(index) + 0.5f) / float(count), nclsViewerRadicalInverse(index));
}

float nclsViewerSdRoundBox(float3 point, float3 bounds, float radius)
{
    const float3 q = abs(point) - bounds + radius;
    return min(max(q.x, max(q.y, q.z)), 0.0f) + length(max(q, 0.0f)) - radius;
}

float nclsViewerSdCylinder(float3 point, float radius, float halfHeight)
{
    const float2 q = abs(float2(length(point.xz), point.y)) - float2(radius, halfHeight);
    return min(max(q.x, q.y), 0.0f) + length(max(q, 0.0f));
}

float nclsViewerSceneSdf(float3 point, uint objectMode)
{
    if (objectMode == 0u) return length(point) - 1.0f;
    if (objectMode == 1u)
    {
        const float sphere = length(point) - 1.0f;
        const float torus = length(float2(length(point.xz) - 0.72f, point.y - 0.14f)) - 0.055f;
        const float carvedBall = max(sphere, -torus);
        const float pedestal = nclsViewerSdCylinder(point - float3(0.0f, -1.04f, 0.0f), 0.62f, 0.10f);
        return min(carvedBall, pedestal);
    }
    float shape = nclsViewerSdRoundBox(point, float3(0.72f, 0.88f, 0.62f), 0.24f);
    const float crown = length(point - float3(0.0f, 0.68f, 0.0f)) - 0.48f;
    shape = min(shape, crown);
    const float detail = 0.018f * sin(11.0f * point.x) * sin(9.0f * point.y) * sin(13.0f * point.z);
    return shape + detail;
}

bool nclsViewerTrace(float3 origin, float3 direction, uint objectMode, out float distance)
{
    distance = 0.05f;
    [loop]
    for (uint step = 0u; step < 128u; ++step)
    {
        const float value = nclsViewerSceneSdf(origin + distance * direction, objectMode);
        if (value < 5e-4f) return true;
        distance += max(value * 0.72f, 2e-4f);
        if (distance > 20.0f) break;
    }
    return false;
}

float3 nclsViewerSdfNormal(float3 point, uint objectMode)
{
    const float epsilon = 8e-4f;
    const float2 e = float2(epsilon, 0.0f);
    return normalize(float3(
        nclsViewerSceneSdf(point + e.xyy, objectMode) - nclsViewerSceneSdf(point - e.xyy, objectMode),
        nclsViewerSceneSdf(point + e.yxy, objectMode) - nclsViewerSceneSdf(point - e.yxy, objectMode),
        nclsViewerSceneSdf(point + e.yyx, objectMode) - nclsViewerSceneSdf(point - e.yyx, objectMode)));
}

float3 nclsViewerTangent(float3 normal)
{
    const float3 axis = abs(normal.y) < 0.98f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    return normalize(cross(axis, normal));
}

float3 nclsViewerToLocal(float3 direction, float3 normal, float3 tangent)
{
    const float3 bitangent = cross(normal, tangent);
    return float3(dot(direction, tangent), dot(direction, bitangent), dot(direction, normal));
}

float3 nclsViewerToWorld(float3 direction, float3 normal, float3 tangent)
{
    const float3 bitangent = cross(normal, tangent);
    return direction.x * tangent + direction.y * bitangent + direction.z * normal;
}

float3 nclsViewerUniformHemisphere(float2 u)
{
    const float z = u.x;
    const float radius = sqrt(max(0.0f, 1.0f - z * z));
    const float phi = 2.0f * NCLS_VIEWER_PI * u.y;
    return float3(radius * cos(phi), radius * sin(phi), z);
}

float3 nclsViewerUniformSphere(float2 u)
{
    const float z = 2.0f * u.x - 1.0f;
    const float radius = sqrt(max(0.0f, 1.0f - z * z));
    const float phi = 2.0f * NCLS_VIEWER_PI * u.y;
    return float3(radius * cos(phi), radius * sin(phi), z);
}

float2 nclsViewerLatLongUv(float3 direction, float rotation)
{
    const float longitude = atan2(direction.z, direction.x) + rotation;
    const float latitude = acos(clamp(direction.y, -1.0f, 1.0f));
    return float2(frac(longitude / (2.0f * NCLS_VIEWER_PI) + 0.5f), latitude / NCLS_VIEWER_PI);
}

float nclsViewerLuminance(float3 value)
{
    return dot(value, float3(0.2126f, 0.7152f, 0.0722f));
}

float3 nclsViewerAces(float3 value)
{
    const float3 mapped = (value * (2.51f * value + 0.03f)) / (value * (2.43f * value + 0.59f) + 0.14f);
    return saturate(mapped);
}

#endif
