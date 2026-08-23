#include "../reference/merl_reference.slang"

StructuredBuffer<float3> gBrdfTable;
StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gLights;
RWStructuredBuffer<float4> gOutput;
uniform uint gQueryCount;

[numthreads(64, 1, 1)]
void evaluateReference(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gQueryCount) return;
    gOutput[index] = float4(nclsEvalMerlReference(
        gBrdfTable, normalize(gViews[index].xyz), normalize(gLights[index].xyz)), 1.0f);
}
