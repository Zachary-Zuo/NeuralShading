#include "PackageBackend.slang"

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
        float3(1.0f, 0.0f, 0.0f), view,
        float2(0.0f), float2(0.0f), float2(0.0f),
        nclsViewerHashFloat(index + 1u), 0u);
    const NclsPackageBackend backend = nclsCreatePackageBackend();
    const NclsPackageState state = backend.prepare(
        context, nclsLoadPackageMaterial(gCompiledMaterialIndex));
    NclsViewerSampleGenerator sampleGenerator = {index + 1u};
    gOutput[index] = float4(state.evaluate(light, sampleGenerator).f, 1.0f);
}
