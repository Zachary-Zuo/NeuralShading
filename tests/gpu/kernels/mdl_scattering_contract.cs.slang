#include "../../../shaders/ncls/reference_backends/mdl.slang"

StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gPositions;
StructuredBuffer<float4> gUv;
StructuredBuffer<float4> gXi;
RWStructuredBuffer<float4> gDirectionPdf;
RWStructuredBuffer<float4> gWeightValid;
uniform uint gQueryCount;

struct NclsMdlProbeSampleGenerator : ISampleGenerator
{
    float4 value;
    uint index;
    [mutating] uint next()
    {
        const float selected = value[index++];
        return uint(selected * 16777216.0f) << 8u;
    }
};

NclsScatteringContext nclsMdlProbeContext(uint index)
{
    NclsScatteringContext context = {};
    context.surface.position = gPositions[index].xyz;
    context.surface.geometricNormal = float3(0.0f, 0.0f, 1.0f);
    context.surface.shadingFrame.normal = float3(0.0f, 0.0f, 1.0f);
    context.surface.shadingFrame.tangent = float3(1.0f, 0.0f, 0.0f);
    context.surface.shadingFrame.bitangent = float3(0.0f, 1.0f, 0.0f);
    context.surface.uv = gUv[index].xy;
    context.surface.frontFacing = 1u;
    context.woWorld = normalize(gViews[index].xyz);
    context.transportMode = (uint)NclsTransportMode::Radiance;
    context.componentMask = (uint)NclsScatteringEvent::Reflection
        | (uint)NclsScatteringEvent::Transmission;
    return context;
}

[numthreads(64, 1, 1)]
void sampleMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint index = dispatchThreadId.x;
    if (index >= gQueryCount) return;
    const NclsMdlReferenceBackend backend = {};
    const NclsMdlReferenceState state = backend.prepare(nclsMdlProbeContext(index), {0u});
    NclsMdlProbeSampleGenerator generator = {gXi[index], 0u};
    NclsScatteringSample sampled = {};
    const bool valid = state.sample(sampled, generator);
    gDirectionPdf[index] = float4(sampled.wiWorld, sampled.pdf.forward);
    gWeightValid[index] = float4(sampled.weight, valid && sampled.valid != 0u ? 1.0f : 0.0f);
}
