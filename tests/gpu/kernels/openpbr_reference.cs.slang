#include "../../../shaders/ncls/reference/openpbr_reference.slang"

StructuredBuffer<float> gResolvedInputs;
StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gLights;
RWStructuredBuffer<float4> gOutput;
uniform uint gQueryCount;

[numthreads(64, 1, 1)]
void evaluateOpenPbrReference(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gQueryCount) return;
    const float3 response = nclsEvalOpenPbrReference(
        gResolvedInputs,
        gViews[index].xyz,
        gLights[index].xyz);
    gOutput[index] = float4(response, 1.0f);
}
