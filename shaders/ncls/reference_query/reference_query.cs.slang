#ifndef NCLS_REFERENCE_PROGRAM_HEADER
#error NCLS_REFERENCE_PROGRAM_HEADER must name one canonical reference backend
#endif

#include NCLS_REFERENCE_PROGRAM_HEADER

#ifndef NCLS_REFERENCE_MAX_EVALUATION_SAMPLES
#define NCLS_REFERENCE_MAX_EVALUATION_SAMPLES 256
#endif

struct NclsReferenceQuerySampleGenerator : ISampleGenerator
{
    uint state;

    [mutating] uint next()
    {
        state = state * 747796405u + 2891336453u;
        return state;
    }
};

StructuredBuffer<float4> gNclsQueryPosition;
StructuredBuffer<float4> gNclsQueryGeometricNormal;
StructuredBuffer<float4> gNclsQueryShadingNormal;
StructuredBuffer<float4> gNclsQueryTangent;
StructuredBuffer<float4> gNclsQueryUv;
StructuredBuffer<float4> gNclsQueryUvDerivatives;
StructuredBuffer<float4> gNclsQueryWo;
StructuredBuffer<float4> gNclsQueryWi;
StructuredBuffer<uint4> gNclsQueryMeta;

RWStructuredBuffer<float4> gNclsQueryFValid;
RWStructuredBuffer<float4> gNclsQueryPdf;
RWStructuredBuffer<float4> gNclsQueryEventValid;
RWStructuredBuffer<float4> gNclsQuerySampleDirectionPdf;
RWStructuredBuffer<float4> gNclsQuerySampleWeightEta;
uniform uint gNclsQueryCount;
uniform uint gNclsEvaluationSamples;

NclsScatteringContext nclsReferenceQueryContext(uint index)
{
    const float3 normal = normalize(gNclsQueryShadingNormal[index].xyz);
    const float3 tangent = normalize(gNclsQueryTangent[index].xyz);
    NclsShadingFrame frame = {};
    frame.normal = normal;
    frame.tangent = tangent;
    frame.bitangent = normalize(cross(normal, tangent));
    const float4 uvDerivatives = gNclsQueryUvDerivatives[index];
    const uint4 meta = gNclsQueryMeta[index];
    NclsSurfaceInteraction surface = {};
    surface.position = gNclsQueryPosition[index].xyz;
    surface.geometricNormal = normalize(gNclsQueryGeometricNormal[index].xyz);
    surface.shadingFrame = frame;
    surface.uv = gNclsQueryUv[index].xy;
    surface.uvDx = uvDerivatives.xy;
    surface.uvDy = uvDerivatives.zw;
    surface.materialInstanceId = meta.x;
    surface.primitiveId = index;
    surface.frontFacing = 1u;
    NclsScatteringContext context = {};
    context.surface = surface;
    context.woWorld = normalize(gNclsQueryWo[index].xyz);
    context.transportMode = (uint)NclsTransportMode::Radiance;
    context.componentMask = (uint)NclsScatteringEvent::Reflection
        | (uint)NclsScatteringEvent::Transmission;
    context.filterRandom = asfloat(meta.z);
    return context;
}

[numthreads(64, 1, 1)]
void evaluateReference(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gNclsQueryCount) return;
    const uint4 meta = gNclsQueryMeta[index];
    const NclsPackageBackend backend = nclsCreatePackageBackend();
    NclsPackageState state = backend.prepare(
        nclsReferenceQueryContext(index), nclsLoadPackageMaterial(meta.x));
    float3 sum = float3(0.0f);
    NclsScatteringPdf firstPdf = {};
    uint firstEvent = 0u;
    uint valid = 1u;
    [loop]
    for (uint sampleIndex = 0u; sampleIndex < NCLS_REFERENCE_MAX_EVALUATION_SAMPLES; ++sampleIndex)
    {
        if (sampleIndex >= gNclsEvaluationSamples) break;
        NclsReferenceQuerySampleGenerator sampleGenerator = {
            meta.y ^ (sampleIndex * 0x9e3779b9u)};
        const NclsScatteringEval value = state.evaluate(
            normalize(gNclsQueryWi[index].xyz), sampleGenerator);
        if (value.valid == 0u || !all(isfinite(value.f)) || any(value.f < 0.0f))
        {
            valid = 0u;
            break;
        }
        if (sampleIndex == 0u)
        {
            firstPdf = value.pdf;
            firstEvent = value.eventFlags;
        }
        else if (
            value.eventFlags != firstEvent
            || value.pdf.forward != firstPdf.forward
            || value.pdf.reverse != firstPdf.reverse)
        {
            valid = 0u;
            break;
        }
        sum += value.f;
    }
    const float3 mean = valid != 0u ? sum / float(gNclsEvaluationSamples) : float3(0.0f);
    gNclsQueryFValid[index] = float4(mean, float(valid));
    gNclsQueryPdf[index] = float4(firstPdf.forward, firstPdf.reverse, 0.0f, 0.0f);
    gNclsQueryEventValid[index] = float4(float(firstEvent), float(valid), 0.0f, 0.0f);
}

[numthreads(64, 1, 1)]
void sampleReference(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gNclsQueryCount) return;
    const uint4 meta = gNclsQueryMeta[index];
    const NclsPackageBackend backend = nclsCreatePackageBackend();
    NclsPackageState state = backend.prepare(
        nclsReferenceQueryContext(index), nclsLoadPackageMaterial(meta.x));
    NclsReferenceQuerySampleGenerator sampleGenerator = {meta.y};
    NclsScatteringSample value = {};
    const bool sampled = state.sample(value, sampleGenerator);
    const uint valid = sampled && value.valid != 0u ? 1u : 0u;
    gNclsQuerySampleDirectionPdf[index] = float4(value.wiWorld, value.pdf.forward);
    gNclsQuerySampleWeightEta[index] = float4(value.weight, value.eta);
    gNclsQueryPdf[index] = float4(value.pdf.forward, value.pdf.reverse, 0.0f, 0.0f);
    gNclsQueryEventValid[index] = float4(float(value.eventFlags), float(valid), 0.0f, 0.0f);
}

[numthreads(64, 1, 1)]
void pdfReference(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gNclsQueryCount) return;
    const uint4 meta = gNclsQueryMeta[index];
    const NclsPackageBackend backend = nclsCreatePackageBackend();
    NclsPackageState state = backend.prepare(
        nclsReferenceQueryContext(index), nclsLoadPackageMaterial(meta.x));
    const NclsScatteringPdf value = state.pdf(normalize(gNclsQueryWi[index].xyz));
    const uint valid = isfinite(value.forward) && isfinite(value.reverse)
        && value.forward >= 0.0f && value.reverse >= 0.0f ? 1u : 0u;
    gNclsQueryPdf[index] = float4(value.forward, value.reverse, 0.0f, 0.0f);
    gNclsQueryEventValid[index] = float4(0.0f, float(valid), 0.0f, 0.0f);
}
