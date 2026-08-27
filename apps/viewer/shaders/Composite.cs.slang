#include "ViewerCommon.slang"

Texture2D<float4> gSlot0;
Texture2D<float4> gSlot1;
SamplerState gLinearSampler;
RWTexture2D<float4> gComparisonLinear;
RWTexture2D<float4> gDisplay;

cbuffer CompositeCB
{
    uint2 gOutputDim;
    uint gComparisonMode;
    float gExposure;
    float gDifferenceScale;
    uint2 gSlotReady;
};

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gOutputDim)) return;
    const float2 uv = (float2(pixel) + 0.5f) / float2(gOutputDim);
    float3 linear;
    float3 comparisonLinear;
    if (gComparisonMode == 0u)
    {
        const uint panelWidth = gOutputDim.x / 2u;
        const uint dividerWidth = gOutputDim.x - 2u * panelWidth;
        if (pixel.x < panelWidth)
        {
            const float2 sourceUv = float2((float(pixel.x) + 0.5f) / float(panelWidth), uv.y);
            comparisonLinear = gSlotReady.x != 0u
                ? gSlot0.SampleLevel(gLinearSampler, sourceUv, 0.0f).rgb
                : float3(0.12f, 0.02f, 0.02f);
            linear = comparisonLinear;
        }
        else if (pixel.x >= panelWidth + dividerWidth)
        {
            const float2 sourceUv = float2((float(pixel.x - panelWidth - dividerWidth) + 0.5f) / float(panelWidth), uv.y);
            comparisonLinear = gSlotReady.y != 0u
                ? gSlot1.SampleLevel(gLinearSampler, sourceUv, 0.0f).rgb
                : float3(0.12f, 0.02f, 0.02f);
            linear = comparisonLinear;
        }
        else { comparisonLinear = float3(0.85f); linear = comparisonLinear; }
    }
    else
    {
        const float3 reference = gSlotReady.x != 0u
            ? gSlot0.SampleLevel(gLinearSampler, uv, 0.0f).rgb : float3(0.0f);
        const float3 approximation = gSlotReady.y != 0u
            ? gSlot1.SampleLevel(gLinearSampler, uv, 0.0f).rgb : float3(0.0f);
        const float3 absoluteDifference = abs(reference - approximation);
        if (gComparisonMode == 1u) comparisonLinear = absoluteDifference;
        else if (gComparisonMode == 2u) comparisonLinear = absoluteDifference / max(max(abs(reference), abs(approximation)), 0.01f);
        else comparisonLinear = absoluteDifference * gDifferenceScale;
        linear = comparisonLinear;
    }
    gComparisonLinear[pixel] = float4(comparisonLinear, 1.0f);
    float3 display = nclsViewerAces(max(linear, 0.0f) * exp2(gExposure));
    display = pow(display, 1.0f / 2.2f);
    gDisplay[pixel] = float4(display, 1.0f);
}
