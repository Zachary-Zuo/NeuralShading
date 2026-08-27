#include "ViewerCommon.slang"

Texture2D<float4> gSlot0;
Texture2D<float4> gSlot1;
SamplerState gLinearSampler;
RWTexture2D<float4> gComparisonLinear;
RWTexture2D<float4> gDisplay;
RWTexture2D<float4> gDifferenceLinear;
RWTexture2D<float4> gDifferenceDisplay;

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
    const uint panelWidth = gOutputDim.x / 2u;
    const uint dividerWidth = gOutputDim.x - 2u * panelWidth;
    const bool leftPanel = pixel.x < panelWidth;
    const bool rightPanel = pixel.x >= panelWidth + dividerWidth;
    const uint panelX = leftPanel ? pixel.x
        : rightPanel ? pixel.x - panelWidth - dividerWidth : 0u;
    const float2 panelUv = float2(
        (float(panelX) + 0.5f) / float(panelWidth),
        uv.y);
    const float3 reference = gSlotReady.x != 0u
        ? gSlot0.SampleLevel(gLinearSampler, panelUv, 0.0f).rgb : float3(0.0f);
    const float3 approximation = gSlotReady.y != 0u
        ? gSlot1.SampleLevel(gLinearSampler, panelUv, 0.0f).rgb : float3(0.0f);
    const float3 absoluteDifference = abs(reference - approximation);
    float3 linear;
    float3 comparisonLinear;
    if (gComparisonMode == 0u)
    {
        if (leftPanel)
        {
            comparisonLinear = gSlotReady.x != 0u
                ? reference
                : float3(0.12f, 0.02f, 0.02f);
            linear = comparisonLinear;
        }
        else if (rightPanel)
        {
            comparisonLinear = gSlotReady.y != 0u
                ? approximation
                : float3(0.12f, 0.02f, 0.02f);
            linear = comparisonLinear;
        }
        else { comparisonLinear = float3(0.85f); linear = comparisonLinear; }
    }
    else
    {
        if (!leftPanel && !rightPanel) comparisonLinear = float3(0.85f);
        else if (gComparisonMode == 1u) comparisonLinear = absoluteDifference;
        else if (gComparisonMode == 2u) comparisonLinear = absoluteDifference / max(max(abs(reference), abs(approximation)), 0.01f);
        else comparisonLinear = absoluteDifference * gDifferenceScale;
        linear = comparisonLinear;
    }
    gComparisonLinear[pixel] = float4(comparisonLinear, 1.0f);
    float3 display = nclsViewerAces(max(linear, 0.0f) * exp2(gExposure));
    display = pow(display, 1.0f / 2.2f);
    gDisplay[pixel] = float4(display, 1.0f);
    if (leftPanel)
    {
        gDifferenceLinear[pixel] = float4(absoluteDifference, 1.0f);
        float3 differenceDisplay = nclsViewerAces(absoluteDifference * exp2(gExposure));
        differenceDisplay = pow(differenceDisplay, 1.0f / 2.2f);
        gDifferenceDisplay[pixel] = float4(differenceDisplay, 1.0f);
    }
}
