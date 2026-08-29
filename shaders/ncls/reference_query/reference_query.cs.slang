#ifndef NCLS_REFERENCE_PROGRAM_HEADER
#error NCLS_REFERENCE_PROGRAM_HEADER must name one canonical reference backend
#endif

#include NCLS_REFERENCE_PROGRAM_HEADER

#ifndef NCLS_REFERENCE_MAX_EVALUATION_SAMPLES
#define NCLS_REFERENCE_MAX_EVALUATION_SAMPLES 256
#endif

#ifndef NCLS_REFERENCE_MAX_FOOTPRINT_SAMPLES
#define NCLS_REFERENCE_MAX_FOOTPRINT_SAMPLES 64
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
uniform uint gNclsFootprintSamples;
uniform uint gNclsSourceExecutionMode;

float nclsRadicalInverseBase2(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;
}

float2 nclsFootprintOffset(uint sampleIndex, uint sampleCount)
{
    if (sampleCount == 1u) return float2(0.0f);
    return float2(
        (float(sampleIndex) + 0.5f) / float(sampleCount) - 0.5f,
        nclsRadicalInverseBase2(sampleIndex) - 0.5f);
}

NclsScatteringContext nclsReferenceQueryContext(uint index, float2 footprintOffset)
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
    surface.uv = gNclsQueryUv[index].xy
        + uvDerivatives.xy * footprintOffset.x
        + uvDerivatives.zw * footprintOffset.y;
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
    float3 sum = float3(0.0f);
    NclsScatteringPdf sumPdf = {};
    uint firstEvent = 0u;
    uint valid = 1u;
    [loop]
    for (uint footprintIndex = 0u; footprintIndex < NCLS_REFERENCE_MAX_FOOTPRINT_SAMPLES; ++footprintIndex)
    {
        if (footprintIndex >= gNclsFootprintSamples) break;
        const float2 offset = nclsFootprintOffset(footprintIndex, gNclsFootprintSamples);
        NclsPackageState state = backend.prepare(
            nclsReferenceQueryContext(index, offset), nclsLoadPackageMaterial(meta.x));
        float3 stochasticSum = float3(0.0f);
        NclsScatteringPdf footprintPdf = {};
        uint footprintEvent = 0u;
        [loop]
        for (uint sampleIndex = 0u; sampleIndex < NCLS_REFERENCE_MAX_EVALUATION_SAMPLES; ++sampleIndex)
        {
            if (sampleIndex >= gNclsEvaluationSamples) break;
            if (gNclsSourceExecutionMode == 0u && sampleIndex > 0u)
            {
                state = backend.prepare(
                    nclsReferenceQueryContext(index, offset),
                    nclsLoadPackageMaterial(meta.x));
            }
            NclsReferenceQuerySampleGenerator sampleGenerator = {
                meta.y ^ (footprintIndex * 0x85ebca6bu) ^ (sampleIndex * 0x9e3779b9u)};
            const NclsScatteringEval value = state.evaluate(
                normalize(gNclsQueryWi[index].xyz), sampleGenerator);
            if (
                value.valid == 0u
                || !all(isfinite(value.f))
                || any(value.f < 0.0f)
                || !isfinite(value.pdf.forward)
                || !isfinite(value.pdf.reverse)
                || value.pdf.forward < 0.0f
                || value.pdf.reverse < 0.0f)
            {
                valid = 0u;
                break;
            }
            if (sampleIndex == 0u)
            {
                footprintPdf = value.pdf;
                footprintEvent = value.eventFlags;
            }
            else if (
                value.eventFlags != footprintEvent
                || value.pdf.forward != footprintPdf.forward
                || value.pdf.reverse != footprintPdf.reverse)
            {
                valid = 0u;
                break;
            }
            stochasticSum += value.f;
        }
        if (valid == 0u) break;
        if (footprintIndex == 0u)
        {
            firstEvent = footprintEvent;
        }
        else if (footprintEvent != firstEvent)
        {
            valid = 0u;
            break;
        }
        sum += stochasticSum / float(gNclsEvaluationSamples);
        sumPdf.forward += footprintPdf.forward;
        sumPdf.reverse += footprintPdf.reverse;
    }
    const float invalidValue = asfloat(0x7fc00000u);
    const float3 mean = valid != 0u
        ? sum / float(gNclsFootprintSamples)
        : float3(invalidValue);
    const float inverseFootprintCount = 1.0f / float(gNclsFootprintSamples);
    gNclsQueryFValid[index] = float4(mean, float(valid));
    gNclsQueryPdf[index] = valid != 0u
        ? float4(
            sumPdf.forward * inverseFootprintCount,
            sumPdf.reverse * inverseFootprintCount,
            0.0f,
            0.0f)
        : float4(invalidValue);
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
        nclsReferenceQueryContext(index, float2(0.0f)), nclsLoadPackageMaterial(meta.x));
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
        nclsReferenceQueryContext(index, float2(0.0f)), nclsLoadPackageMaterial(meta.x));
    const NclsScatteringPdf value = state.pdf(normalize(gNclsQueryWi[index].xyz));
    const uint valid = isfinite(value.forward) && isfinite(value.reverse)
        && value.forward >= 0.0f && value.reverse >= 0.0f ? 1u : 0u;
    gNclsQueryPdf[index] = float4(value.forward, value.reverse, 0.0f, 0.0f);
    gNclsQueryEventValid[index] = float4(0.0f, float(valid), 0.0f, 0.0f);
}
