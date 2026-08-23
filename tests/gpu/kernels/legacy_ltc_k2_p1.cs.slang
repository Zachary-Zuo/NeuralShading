#include "../../../shaders/ncls/backends/legacy_ltc_k2/p1_compiler.slang"

StructuredBuffer<NclsLayerStackIR> gMaterials;
StructuredBuffer<float> gWeights;
StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gLights;
RWStructuredBuffer<float4> gOutput;
uniform uint gWidth;
uniform uint gLightCount;

[numthreads(64, 1, 1)]
void evaluateLegacyLtcK2P1(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gLightCount) return;
    const float3 view = normalize(gViews[0].xyz);
    const NclsLegacyLtcK2PackedState state = nclsCompileLegacyLtcK2P1(
        gMaterials[0],
        view,
        gWeights,
        gWidth);
    const float3 value = nclsLegacyLtcK2ResponseCos(
        state,
        view,
        normalize(gLights[index].xyz));
    gOutput[index] = float4(value, 1.0f);
}
