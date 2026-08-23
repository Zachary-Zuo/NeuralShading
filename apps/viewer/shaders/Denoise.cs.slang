#include "ViewerCommon.slang"

Texture2D<float4> gInput;
Texture2D<float4> gRawReference;
Texture2D<float4> gPositionDepth;
Texture2D<float4> gNormal;
Texture2D<uint> gMaterialId;
RWTexture2D<float4> gOutput;

cbuffer DenoiseCB
{
    uint2 gFrameDim;
    uint gReferenceSpp;
    uint gStepWidth;
};

// A small a-trous cross-bilateral filter. This is deliberately a display-only
// estimate: the unfiltered running mean remains the reference used by metrics
// and captures. Position, normal, and source-material ID prevent the preview
// from bleeding across visible geometry boundaries.
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFrameDim)) return;

    const float4 centerRaw = gRawReference[pixel];
    const float4 centerPosition = gPositionDepth[pixel];
    const float3 centerNormal = normalize(gNormal[pixel].xyz);
    const uint centerMaterial = gMaterialId[pixel];
    const float centerLuminance = nclsViewerLuminance(centerRaw.rgb);
    const float centerVariance = max(centerRaw.a - centerLuminance * centerLuminance, 0.0f)
        / max(float(gReferenceSpp), 1.0f);
    const float lowSampleScale = 2.0f * max(centerLuminance, 0.1f)
        / sqrt(max(float(gReferenceSpp), 1.0f));
    const float colorScale = max(0.05f, 3.0f * sqrt(centerVariance) + lowSampleScale);
    const bool centerBackground = centerPosition.w < 0.0f;

    static const float kernel[5] = {1.0f, 4.0f, 6.0f, 4.0f, 1.0f};
    float3 weightedColor = float3(0.0f);
    float weightSum = 0.0f;
    [unroll]
    for (int y = -2; y <= 2; ++y)
    {
        [unroll]
        for (int x = -2; x <= 2; ++x)
        {
            const int2 sourcePixel = clamp(
                int2(pixel) + int2(x, y) * int(gStepWidth),
                int2(0), int2(gFrameDim) - 1);
            const float4 sampleRaw = gRawReference[sourcePixel];
            const float4 samplePosition = gPositionDepth[sourcePixel];
            const bool sampleBackground = samplePosition.w < 0.0f;
            if (sampleBackground != centerBackground) continue;
            if (!centerBackground && gMaterialId[sourcePixel] != centerMaterial) continue;

            const float3 sampleColor = gInput[sourcePixel].rgb;
            const float sampleLuminance = nclsViewerLuminance(sampleRaw.rgb);
            float weight = kernel[x + 2] * kernel[y + 2];
            weight *= exp(-abs(sampleLuminance - centerLuminance) / colorScale);
            if (!centerBackground)
            {
                const float3 sampleNormal = normalize(gNormal[sourcePixel].xyz);
                const float normalAgreement = max(dot(centerNormal, sampleNormal), 0.0f);
                weight *= pow(normalAgreement, 32.0f);
                const float depthScale = max(0.0025f * centerPosition.w, 1e-4f);
                weight *= exp(-abs(samplePosition.w - centerPosition.w) / depthScale);
            }
            weightedColor += weight * sampleColor;
            weightSum += weight;
        }
    }
    gOutput[pixel] = float4(
        weightSum > 0.0f ? weightedColor / weightSum : gInput[pixel].rgb,
        centerRaw.a);
}
