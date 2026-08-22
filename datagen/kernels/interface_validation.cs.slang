#include "../../teacher/interfaces.slang"

StructuredBuffer<LayerStack> gStacks;
StructuredBuffer<float4> gIncidentDirections;
StructuredBuffer<float4> gOutgoingDirections;
RWStructuredBuffer<float4> gEval;
RWStructuredBuffer<float4> gSampleMean;
RWStructuredBuffer<float4> gSampleMeanSquare;
RWStructuredBuffer<float4> gSampleDiagnostics;

uniform uint gQueryCount;
uniform uint gSampleCount;
uniform uint gSeed;
uniform uint gSampleModes;
uniform float gEtaI;
uniform float gEtaT;

[numthreads(64, 1, 1)]
void validateInterfaces(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint queryIndex = dispatchThreadID.x;
    if (queryIndex >= gQueryCount) return;

    const LayerInterface layer = gStacks[queryIndex].layers[0];
    const float3 wi = normalize(gIncidentDirections[queryIndex].xyz);
    const float3 wo = normalize(gOutgoingDirections[queryIndex].xyz);
    const float3 f = nclsEvalInterface(layer, wi, wo, gEtaI, gEtaT);
    const float pdf = nclsInterfacePdf(layer, wi, wo, gEtaI, gEtaT, gSampleModes);
    gEval[queryIndex] = float4(f, pdf);

    float3 sum = float3(0.0f);
    float3 sumSquare = float3(0.0f);
    float maxPdfMismatch = 0.0f;
    uint validCount = 0u;
    for (uint sampleIndex = 0u; sampleIndex < gSampleCount; ++sampleIndex)
    {
        NclsRng rng = NclsRng(uint2(queryIndex ^ gSeed, sampleIndex + 1664525u * gSeed));
        const LayerBsdfSample sample = nclsSampleInterface(layer, wi, gEtaI, gEtaT, gSampleModes, rng);
        if (sample.valid == 0u) continue;
        const float reevaluatedPdf = nclsInterfacePdf(layer, wi, sample.direction, gEtaI, gEtaT, gSampleModes);
        maxPdfMismatch = max(maxPdfMismatch, abs(sample.pdf - reevaluatedPdf));
        const float3 weight = sample.f * abs(sample.direction.z) / sample.pdf;
        sum += weight;
        sumSquare += weight * weight;
        validCount++;
    }

    const float inverseSampleCount = 1.0f / max(1u, gSampleCount);
    gSampleMean[queryIndex] = float4(sum * inverseSampleCount, 1.0f);
    gSampleMeanSquare[queryIndex] = float4(sumSquare * inverseSampleCount, 1.0f);
    gSampleDiagnostics[queryIndex] = float4(maxPdfMismatch, float(validCount), float(gSampleCount), 0.0f);
}
