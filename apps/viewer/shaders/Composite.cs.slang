#include "ViewerCommon.slang"

Texture2D<float4> gReference;
Texture2D<float4> gApproximation;
SamplerState gLinearSampler;
RWTexture2D<float4> gComparisonLinear;
RWTexture2D<float4> gDisplay;

cbuffer CompositeCB
{
    uint2 gOutputDim;
    float gSplit;
    uint gComparisonMode;
    float gExposure;
    float gDifferenceScale;
    uint gHasApproximation;
    uint _padding;
};

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gOutputDim)) return;
    const float2 uv = (float2(pixel) + 0.5f) / float2(gOutputDim);
    float3 linear;
    if (gHasApproximation == 0u)
    {
        linear = gReference.SampleLevel(gLinearSampler, uv, 0.0f).rgb;
    }
    else if (gComparisonMode == 0u)
    {
        if (uv.x < gSplit)
        {
            const float2 sourceUv = float2(uv.x / max(gSplit, 1e-4f), uv.y);
            linear = gReference.SampleLevel(gLinearSampler, sourceUv, 0.0f).rgb;
        }
        else
        {
            const float2 sourceUv = float2((uv.x - gSplit) / max(1.0f - gSplit, 1e-4f), uv.y);
            linear = gApproximation.SampleLevel(gLinearSampler, sourceUv, 0.0f).rgb;
        }
    }
    else
    {
        const float3 reference = gReference.SampleLevel(gLinearSampler, uv, 0.0f).rgb;
        const float3 approximation = gApproximation.SampleLevel(gLinearSampler, uv, 0.0f).rgb;
        const float3 absoluteDifference = abs(reference - approximation);
        if (gComparisonMode == 1u) linear = absoluteDifference;
        else if (gComparisonMode == 2u) linear = absoluteDifference / max(max(abs(reference), abs(approximation)), 0.01f);
        else linear = absoluteDifference * gDifferenceScale;
    }
    gComparisonLinear[pixel] = float4(linear, 1.0f);
    float3 display = nclsViewerAces(max(linear, 0.0f) * exp2(gExposure));
    display = pow(display, 1.0f / 2.2f);
    if (gHasApproximation != 0u && gComparisonMode == 0u && abs(uv.x - gSplit) < 1.5f / float(gOutputDim.x))
        display = float3(0.85f);
    gDisplay[pixel] = float4(display, 1.0f);
}
