#define NCLS_REFERENCE_BACKEND_NO_PACKAGE_BINDING 1
#include "../../../shaders/ncls/reference_backends/layer_stack.slang"
#include "../../../shaders/ncls/reference_backends/merl.slang"
#include "../../../shaders/ncls/reference_backends/materialx.slang"
#undef NCLS_REFERENCE_BACKEND_NO_PACKAGE_BINDING

StructuredBuffer<float4> gViews;
StructuredBuffer<NclsLayerStackIR> gLayerMaterials;
StructuredBuffer<float3> gMerlMaterial;
StructuredBuffer<float> gMaterialXInputs;
Texture2D<float4> gMaterialXBaseColor;
Texture2D<float4> gMaterialXRoughness;
Texture2D<float4> gMaterialXMetalness;
Texture2D<float4> gMaterialXNormalMap;
SamplerState gMaterialXSampler;

RWStructuredBuffer<float4> gDirectionPdf;
RWStructuredBuffer<float4> gSampleWeightValid;
RWStructuredBuffer<float4> gEvaluateWeightValid;
RWStructuredBuffer<float4> gPdfAgreement;
RWStructuredBuffer<uint4> gEventAgreement;
uniform uint gQueryCount;

struct NclsReferenceContractSampleGenerator : ISampleGenerator
{
    uint state;

    [mutating] uint next()
    {
        state = state * 747796405u + 2891336453u;
        return state;
    }
};

NclsScatteringContext nclsReferenceContractContext(float3 viewWorld)
{
    NclsShadingFrame frame = {};
    frame.normal = float3(0.0f, 0.0f, 1.0f);
    frame.tangent = float3(1.0f, 0.0f, 0.0f);
    frame.bitangent = float3(0.0f, 1.0f, 0.0f);
    NclsSurfaceInteraction surface = {};
    surface.position = float3(0.0f);
    surface.geometricNormal = frame.normal;
    surface.shadingFrame = frame;
    surface.uv = float2(0.37f, 0.61f);
    surface.uvDx = float2(1.0f / 1024.0f, 0.0f);
    surface.uvDy = float2(0.0f, 1.0f / 1024.0f);
    surface.frontFacing = 1u;
    NclsScatteringContext context = {};
    context.surface = surface;
    context.woWorld = normalize(viewWorld);
    context.transportMode = (uint)NclsTransportMode::Radiance;
    context.componentMask = (uint)NclsScatteringEvent::Reflection
        | (uint)NclsScatteringEvent::Transmission;
    context.filterRandom = 0.5f;
    return context;
}

void nclsWriteReferenceContractResult<T : INclsScatteringState>(
    uint index,
    T state,
    NclsScatteringSample sample,
    bool sampled,
    inout NclsReferenceContractSampleGenerator sampleGenerator)
{
    NclsScatteringEval evaluation = {};
    NclsScatteringPdf independentPdf = {};
    float3 evaluatedWeight = float3(0.0f);
    if (sampled && sample.valid != 0u)
    {
        evaluation = state.evaluate(sample.wiWorld, sampleGenerator);
        independentPdf = state.pdf(sample.wiWorld);
        if (evaluation.valid != 0u && independentPdf.forward > 0.0f)
            evaluatedWeight = evaluation.f
                * abs(sample.wiWorld.z)
                / independentPdf.forward;
    }
    gDirectionPdf[index] = float4(sample.wiWorld, sample.pdf.forward);
    gSampleWeightValid[index] = float4(sample.weight, float(sample.valid));
    gEvaluateWeightValid[index] = float4(evaluatedWeight, float(evaluation.valid));
    gPdfAgreement[index] = float4(
        independentPdf.forward, independentPdf.reverse,
        evaluation.pdf.forward, evaluation.pdf.reverse);
    gEventAgreement[index] = uint4(
        sample.eventFlags, evaluation.eventFlags, sample.valid, evaluation.valid);
}

[numthreads(64, 1, 1)]
void sampleLayerStackContract(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gQueryCount) return;
    const NclsScatteringContext context = nclsReferenceContractContext(gViews[index].xyz);
    const NclsLayerStackReferenceBackend backend = {24u};
    const NclsLayerStackReferenceState state = backend.prepare(context, gLayerMaterials[0]);
    NclsReferenceContractSampleGenerator sampleGenerator = {index ^ 0x51ed270bu};
    NclsScatteringSample sample = {};
    const bool sampled = state.sample(sample, sampleGenerator);
    nclsWriteReferenceContractResult(index, state, sample, sampled, sampleGenerator);
}

[numthreads(64, 1, 1)]
void sampleMerlContract(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gQueryCount) return;
    const NclsScatteringContext context = nclsReferenceContractContext(gViews[index].xyz);
    const NclsMerlReferenceBackend backend = {};
    const NclsMerlReferenceState state = backend.prepare(context, gMerlMaterial);
    NclsReferenceContractSampleGenerator sampleGenerator = {index ^ 0x51ed270bu};
    NclsScatteringSample sample = {};
    const bool sampled = state.sample(sample, sampleGenerator);
    nclsWriteReferenceContractResult(index, state, sample, sampled, sampleGenerator);
}

[numthreads(64, 1, 1)]
void sampleMaterialXContract(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gQueryCount) return;
    const NclsScatteringContext context = nclsReferenceContractContext(gViews[index].xyz);
    const NclsMaterialXCompiledMaterial material = {
        gMaterialXInputs,
        gMaterialXBaseColor,
        gMaterialXRoughness,
        gMaterialXMetalness,
        gMaterialXNormalMap};
    const NclsMaterialXReferenceBackend backend = {gMaterialXSampler};
    const NclsMaterialXReferenceState state = backend.prepare(context, material);
    NclsReferenceContractSampleGenerator sampleGenerator = {index ^ 0x51ed270bu};
    NclsScatteringSample sample = {};
    const bool sampled = state.sample(sample, sampleGenerator);
    nclsWriteReferenceContractResult(index, state, sample, sampled, sampleGenerator);
}
