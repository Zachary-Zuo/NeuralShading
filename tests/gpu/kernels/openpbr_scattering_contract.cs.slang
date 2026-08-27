#define NCLS_REFERENCE_BACKEND_NO_PACKAGE_BINDING 1
#include "../../../shaders/ncls/reference_backends/openpbr.slang"

StructuredBuffer<float> gResolvedInputs;
StructuredBuffer<float4> gViews;
RWStructuredBuffer<float4> gDirectionPdf;
RWStructuredBuffer<float4> gSampleWeight;
RWStructuredBuffer<float4> gEvaluateWeight;
RWStructuredBuffer<float4> gPdfAgreement;
RWStructuredBuffer<uint4> gEventAgreement;
RWStructuredBuffer<float4> gNativeDirectionPdf;
RWStructuredBuffer<float4> gNativeSampleWeight;
uniform uint gQueryCount;

struct NclsOpenPbrProbeSampleGenerator : ISampleGenerator
{
    uint state;
    [mutating] uint next()
    {
        state = state * 1664525u + 1013904223u;
        return state;
    }
};

[numthreads(64, 1, 1)]
void sampleOpenPbrContract(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint index = dispatchThreadId.x;
    if (index >= gQueryCount) return;
    NclsScatteringContext context = {};
    context.surface.geometricNormal = float3(0.0f, 0.0f, 1.0f);
    context.surface.shadingFrame.normal = float3(0.0f, 0.0f, 1.0f);
    context.surface.shadingFrame.tangent = float3(1.0f, 0.0f, 0.0f);
    context.surface.shadingFrame.bitangent = float3(0.0f, 1.0f, 0.0f);
    context.surface.frontFacing = 1u;
    context.woWorld = normalize(gViews[index].xyz);
    context.transportMode = (uint)NclsTransportMode::Radiance;
    context.componentMask = (uint)NclsScatteringEvent::Reflection
        | (uint)NclsScatteringEvent::Transmission;

    const NclsOpenPbrReferenceBackend backend = {};
    const NclsOpenPbrReferenceState state = backend.prepare(context, gResolvedInputs);
    NclsOpenPbrProbeSampleGenerator generator = {index ^ 0x9e3779b9u};
    NclsScatteringSample sample = {};
    const bool valid = state.sample(sample, generator);
    gDirectionPdf[index] = float4(sample.wiWorld, sample.pdf.forward);
    gSampleWeight[index] = float4(sample.weight, valid && sample.valid != 0u ? 1.0f : 0.0f);

    const NclsScatteringEval evaluation = state.evaluate(sample.wiWorld, generator);
    const NclsScatteringPdf independentPdf = state.pdf(sample.wiWorld);
    const float cosine = abs(dot(context.surface.shadingFrame.normal, sample.wiWorld));
    const float denominator = max(evaluation.pdf.forward, 1e-30f);
    gEvaluateWeight[index] = float4(
        evaluation.f * cosine / denominator,
        evaluation.valid != 0u ? 1.0f : 0.0f);
    gPdfAgreement[index] = float4(
        sample.pdf.reverse,
        independentPdf.forward,
        independentPdf.reverse,
        evaluation.pdf.forward);
    gEventAgreement[index] = uint4(
        sample.eventFlags, evaluation.eventFlags, sample.valid, evaluation.valid);

    NclsOpenPbrProbeSampleGenerator nativeGenerator = {index ^ 0x9e3779b9u};
    const float3 nativeRandom = sampleNext3D(nativeGenerator);
    float3 nativeDirection;
    OpenPBR_DiffuseSpecular nativeWeight;
    float nativePdf;
    OpenPBR_BsdfLobeType nativeType;
    openpbr_sample(
        state.prepared, nativeRandom, nativeDirection, nativeWeight, nativePdf, nativeType);
    gNativeDirectionPdf[index] = float4(nativeDirection, nativePdf);
    gNativeSampleWeight[index] = float4(
        openpbr_get_sum_of_diffuse_specular(nativeWeight), nativePdf > 0.0f ? 1.0f : 0.0f);
}
