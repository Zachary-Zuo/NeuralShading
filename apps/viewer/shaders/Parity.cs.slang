#include "../../../shaders/ncls/backends/film_m1/film_m1.slang"

StructuredBuffer<float> gWeights;
StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gLights;
RWStructuredBuffer<float4> gOutput;
uniform uint gLightCount;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gLightCount) return;
    const float3 view = normalize(gViews[0].xyz);
    const float3 light = normalize(gLights[index].xyz);
    const NclsFilmM1State state = nclsFilmM1Prepare(view, gWeights);
    gOutput[index] = float4(nclsFilmM1ResponseCos(state, view, light, gWeights), 1.0f);
}
