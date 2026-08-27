#include "../../../../apps/viewer/shaders/PackageBackend.slang"

StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gLights;
StructuredBuffer<float4> gUv;
StructuredBuffer<float4> gGradients;
RWStructuredBuffer<float4> gOutput;
uniform uint gQueryCount;
uniform uint gCompiledMaterialIndex;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gQueryCount) return;
    const float3 view = normalize(gViews[index].xyz);
    const float3 light = normalize(gLights[index].xyz);
    const float4 gradient = gGradients[index];
    const NclsScatteringContext context = nclsViewerMethodContext(
        float3(0.0f), float3(0.0f, 0.0f, 1.0f),
        float3(1.0f, 0.0f, 0.0f), view,
        gUv[index].xy, gradient.xy, gradient.zw,
        nclsViewerHashFloat(index + 1u), 0u);
    const NclsPackageBackend backend = nclsCreatePackageBackend();
    const NclsPackageState state = backend.prepare(
        context, nclsLoadPackageMaterial(gCompiledMaterialIndex));
    NclsViewerSampleGenerator sampleGenerator = {index + 1u};
    gOutput[index] = float4(nclsScatteringEvalToResponseCos(
        state.evaluate(light, sampleGenerator), context.surface.shadingFrame, light), 1.0f);
}
