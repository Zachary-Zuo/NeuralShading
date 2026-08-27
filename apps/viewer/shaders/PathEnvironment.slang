#ifndef NCLS_VIEWER_PATH_ENVIRONMENT_SLANG
#define NCLS_VIEWER_PATH_ENVIRONMENT_SLANG

#include "PathEnvironmentMath.slang"

// 本文件由 reference/package path tracer 在声明同名 environment 资源与
// constant buffer 后 include。两条 PT 因而共用完全相同的 CDF、PDF 与 MIS math。

float3 nclsViewerEnvironmentRadiance(float3 direction)
{
    return gEnvironment.SampleLevel(
        gLinearSampler, nclsViewerLatLongUv(direction, gEnvironmentRotation), 0.0f).rgb
        * gEnvironmentIntensity;
}

uint nclsViewerSearchEnvironmentMarginalCdf(float value)
{
    uint lower = 0u;
    uint upper = gEnvironmentSamplingDimensions.y;
    [loop]
    while (lower + 1u < upper)
    {
        const uint middle = (lower + upper) >> 1u;
        if (gEnvironmentMarginalCdf[middle] <= value) lower = middle;
        else upper = middle;
    }
    return min(lower, gEnvironmentSamplingDimensions.y - 1u);
}

uint nclsViewerSearchEnvironmentConditionalCdf(uint row, float value)
{
    const uint width = gEnvironmentSamplingDimensions.x;
    const uint offset = row * (width + 1u);
    uint lower = 0u;
    uint upper = width;
    [loop]
    while (lower + 1u < upper)
    {
        const uint middle = (lower + upper) >> 1u;
        if (gEnvironmentConditionalCdf[offset + middle] <= value) lower = middle;
        else upper = middle;
    }
    return min(lower, width - 1u);
}

float nclsViewerEnvironmentPdf(float3 direction)
{
    if (any(gEnvironmentSamplingDimensions == uint2(0u)))
        return 1.0f / (4.0f * NCLS_VIEWER_PI);
    const float2 uv = nclsViewerLatLongUv(direction, gEnvironmentRotation);
    const uint width = gEnvironmentSamplingDimensions.x;
    const uint height = gEnvironmentSamplingDimensions.y;
    const uint column = min(uint(uv.x * float(width)), width - 1u);
    const uint row = min(uint(uv.y * float(height)), height - 1u);
    const float rowProbability = max(
        gEnvironmentMarginalCdf[row + 1u] - gEnvironmentMarginalCdf[row], 0.0f);
    const uint offset = row * (width + 1u);
    const float columnProbability = max(
        gEnvironmentConditionalCdf[offset + column + 1u]
        - gEnvironmentConditionalCdf[offset + column], 0.0f);
    const float sineTheta = max(sin(NCLS_VIEWER_PI * uv.y), 1e-7f);
    return rowProbability * columnProbability * float(width * height)
        / (2.0f * NCLS_VIEWER_PI * NCLS_VIEWER_PI * sineTheta);
}

void nclsViewerSampleEnvironment(inout NclsRng rng, out float3 direction, out float pdf)
{
    if (any(gEnvironmentSamplingDimensions == uint2(0u)))
    {
        const float2 u = rng.nextFloat2();
        const float y = 2.0f * u.x - 1.0f;
        const float radius = sqrt(max(1.0f - y * y, 0.0f));
        const float longitude = 2.0f * NCLS_VIEWER_PI * u.y;
        direction = float3(radius * cos(longitude), y, radius * sin(longitude));
        pdf = 1.0f / (4.0f * NCLS_VIEWER_PI);
        return;
    }
    const uint row = nclsViewerSearchEnvironmentMarginalCdf(rng.nextFloat());
    const uint column = nclsViewerSearchEnvironmentConditionalCdf(row, rng.nextFloat());
    const float2 uv = (float2(column, row) + rng.nextFloat2())
        / float2(gEnvironmentSamplingDimensions);
    const float theta = NCLS_VIEWER_PI * uv.y;
    const float longitude = 2.0f * NCLS_VIEWER_PI * (uv.x - 0.5f) - gEnvironmentRotation;
    const float sineTheta = sin(theta);
    direction = float3(sineTheta * cos(longitude), cos(theta), sineTheta * sin(longitude));
    pdf = nclsViewerEnvironmentPdf(direction);
}

#endif
