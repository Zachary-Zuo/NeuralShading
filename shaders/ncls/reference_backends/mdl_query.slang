// Formal batch query 直接调用 canonical MDL scattering state；动态 target code
// 由 host 以 NclsMdlGenerated 模块提供。
#include "mdl.slang"
#include "../contracts/response_measure.slang"

StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gLights;
StructuredBuffer<float4> gPositions;
StructuredBuffer<float4> gUv;
RWStructuredBuffer<float4> gOutput;
uniform uint gQueryCount;

struct NclsMdlQuerySampleGenerator : ISampleGenerator
{
    uint state;
    [mutating] uint next()
    {
        state = state * 747796405u + 2891336453u;
        return state;
    }
};

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint index = dispatchThreadId.x;
    if (index >= gQueryCount) return;

    NclsShadingFrame frame = {};
    frame.normal = float3(0.0f, 0.0f, 1.0f);
    frame.tangent = float3(1.0f, 0.0f, 0.0f);
    frame.bitangent = float3(0.0f, 1.0f, 0.0f);
    NclsSurfaceInteraction surface = {};
    surface.position = gPositions[index].xyz;
    surface.geometricNormal = frame.normal;
    surface.shadingFrame = frame;
    surface.uv = gUv[index].xy;
    surface.frontFacing = 1u;
    NclsScatteringContext context = {};
    context.surface = surface;
    context.woWorld = normalize(gViews[index].xyz);
    context.transportMode = (uint)NclsTransportMode::Radiance;
    context.componentMask = (uint)NclsScatteringEvent::Reflection
        | (uint)NclsScatteringEvent::Transmission;
    const NclsMdlReferenceBackend backend = {};
    const NclsMdlReferenceState state = backend.prepare(context, {0u});
    NclsMdlQuerySampleGenerator sampleGenerator = {index + 1u};
    const NclsScatteringEval evaluation = state.evaluate(
        normalize(gLights[index].xyz), sampleGenerator);
    const float3 response = nclsScatteringEvalToResponseCos(
        evaluation, frame, normalize(gLights[index].xyz));
    gOutput[index] = float4(response, evaluation.pdf.forward);
}
