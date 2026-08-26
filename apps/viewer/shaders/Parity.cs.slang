#include "MethodBackend.slang"

StructuredBuffer<uint> gSharedWeights;
StructuredBuffer<NclsMethodCompiledMaterial> gCompiledMaterials;
StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gLights;
RWStructuredBuffer<float4> gOutput;
uniform uint gLightCount;
uniform uint gCompiledMaterialIndex;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gLightCount) return;
    const float3 view = normalize(gViews[0].xyz);
    const float3 light = normalize(gLights[index].xyz);
    const NclsScatteringContext context = nclsViewerMethodContext(
        float3(0.0f), float3(0.0f, 0.0f, 1.0f),
        float3(1.0f, 0.0f, 0.0f), view, 0u);
    const NclsMethodBackend backend = nclsCreateMethodBackend(gSharedWeights);
    const NclsMethodState state = backend.prepare(
        context, gCompiledMaterials[gCompiledMaterialIndex]);
    NclsViewerSampleGenerator sampleGenerator = {index + 1u};
    gOutput[index] = float4(nclsScatteringEvalToResponseCos(
        state.evaluate(light, sampleGenerator), context.surface.shadingFrame, light), 1.0f);
}
