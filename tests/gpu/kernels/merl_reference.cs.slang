#include "../../../shaders/ncls/reference/merl_reference.slang"

StructuredBuffer<float3> gBrdfTable;
StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gLights;
RWStructuredBuffer<float4> gOutput;
uniform uint gQueryCount;

[numthreads(64, 1, 1)]
void evaluateMerlReference(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gQueryCount) return;
    const float3 response = nclsEvalMerlReference(
        gBrdfTable,
        normalize(gViews[index].xyz),
        normalize(gLights[index].xyz));
    gOutput[index] = float4(response, 1.0f);
}
