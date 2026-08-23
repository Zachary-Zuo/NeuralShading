#include "../reference/openpbr_reference.slang"

StructuredBuffer<float> gResolvedInputs;
StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gLights;
RWStructuredBuffer<float4> gOutput;
uniform uint gQueryCount;

[numthreads(64, 1, 1)]
void evaluateReference(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gQueryCount) return;
    const float3 view = normalize(gViews[index].xyz);
    const float3 light = normalize(gLights[index].xyz);
    const OpenPBR_PreparedBsdf prepared = nclsPrepareOpenPbrReference(gResolvedInputs, view);
    const float3 response = openpbr_get_sum_of_diffuse_specular(openpbr_eval(prepared, light));
    gOutput[index] = float4(response, openpbr_pdf(prepared, light));
}
