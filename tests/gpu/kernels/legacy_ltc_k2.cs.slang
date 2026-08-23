#include "../../../shaders/ncls/backends/legacy_ltc_k2/legacy_ltc_k2.slang"
#include "../../../shaders/ncls/contracts/response_measure.slang"

StructuredBuffer<NclsLegacyLtcK2PackedState> gStates;
StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gLights;
RWStructuredBuffer<float4> gOutput;
uniform uint gLightCount;

struct ContractTestSampleGenerator : ISampleGenerator
{
    uint state;

    [mutating]
    uint next()
    {
        state = state * 1664525u + 1013904223u;
        return state;
    }
};

[numthreads(64, 1, 1)]
void evaluateLegacyLtcK2(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gLightCount) return;
    const float3 value = nclsLegacyLtcK2ResponseCos(
        gStates[0],
        normalize(gViews[0].xyz),
        normalize(gLights[index].xyz));
    gOutput[index] = float4(value, 1.0f);
}

[numthreads(64, 1, 1)]
void evaluateLegacyLtcK2ThroughContract(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gLightCount) return;
    NclsScatteringContext context = {};
    context.surface.geometricNormal = float3(0.0f, 0.0f, 1.0f);
    context.surface.shadingFrame.normal = float3(0.0f, 0.0f, 1.0f);
    context.surface.shadingFrame.tangent = float3(1.0f, 0.0f, 0.0f);
    context.surface.shadingFrame.bitangent = float3(0.0f, 1.0f, 0.0f);
    context.surface.frontFacing = 1u;
    context.woWorld = normalize(gViews[0].xyz);
    context.transportMode = (uint)NclsTransportMode::Radiance;
    context.componentMask = (uint)NclsScatteringEvent::Reflection;
    LegacyLtcK2Backend backend;
    LegacyLtcK2State state = backend.prepare(context, gStates[0]);
    ContractTestSampleGenerator sampleGenerator = {index + 1u};
    const float3 wiWorld = normalize(gLights[index].xyz);
    const NclsScatteringEval evaluation = state.evaluate(wiWorld, sampleGenerator);
    gOutput[index] = float4(
        nclsScatteringEvalToResponseCos(evaluation, context.surface.shadingFrame, wiWorld),
        1.0f);
}
