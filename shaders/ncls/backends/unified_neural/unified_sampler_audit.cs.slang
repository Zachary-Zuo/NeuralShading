#include "unified_neural_core.slang"
#include "../nvidia_neural_appearance/nvidia_neural_appearance_core.slang"
#include "../nvidia_neural_appearance/nvidia_matched_ltc_sampler.slang"

struct NclsUnifiedAuditPrepared
{
    float values[NCLS_UNIFIED_PREPARE_OUTPUT];
};

StructuredBuffer<NclsUnifiedAuditPrepared> gPrepared;
StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gInput;
RWStructuredBuffer<float4> gOutput0;
RWStructuredBuffer<float4> gOutput1;
uniform uint gCount;
uniform uint gCaseIndex;
uniform uint gSampler;
uniform uint gMethod;

float nclsAuditPdf(
    NclsUnifiedAuditPrepared prepared,
    float3 wo,
    float3 wi)
{
    if (gMethod == 0u)
    {
        return gSampler == 0u
            ? nclsUnifiedNvidiaPreparedPdf(prepared.values, wo, wi)
            : nclsUnifiedLtcPreparedPdf(prepared.values, wi);
    }
    if (gSampler == 0u)
    {
        float raw[NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT];
        [unroll]
        for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT; ++index)
            raw[index] = prepared.values[14u + index];
        return nclsNvidiaNeuralSamplerPdf(raw, wo, wi);
    }
    float raw[NCLS_NVIDIA_MATCHED_LTC_RAW_COUNT];
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_MATCHED_LTC_RAW_COUNT; ++index)
        raw[index] = prepared.values[14u + index];
    return nclsNvidiaMatchedLtcPdf(raw, wi);
}

NclsDirectionalSample nclsAuditSample(
    NclsUnifiedAuditPrepared prepared,
    float3 wo,
    float3 u,
    out uint component)
{
    if (gMethod == 0u)
    {
        return gSampler == 0u
            ? nclsSampleUnifiedNvidiaPrepared(prepared.values, wo, u, component)
            : nclsSampleUnifiedLtcPrepared(prepared.values, u, component);
    }
    if (gSampler == 0u)
    {
        float raw[NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT];
        [unroll]
        for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT; ++index)
            raw[index] = prepared.values[14u + index];
        return nclsSampleNvidiaNeuralSampler(raw, wo, u, component);
    }
    float raw[NCLS_NVIDIA_MATCHED_LTC_RAW_COUNT];
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_MATCHED_LTC_RAW_COUNT; ++index)
        raw[index] = prepared.values[14u + index];
    return nclsSampleNvidiaMatchedLtc(raw, u, component);
}

[numthreads(64, 1, 1)]
void queryUnifiedLearnedSampler(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    const float3 wo = gViews[gCaseIndex].xyz;
    const float3 wi = gInput[index].xyz;
    const float pdf = nclsAuditPdf(gPrepared[gCaseIndex], wo, wi);
    gOutput0[index] = float4(pdf, 0.0f, 0.0f, 0.0f);
    gOutput1[index] = float4(0.0f);
}

[numthreads(64, 1, 1)]
void sampleUnifiedLearnedSampler(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    const float3 wo = gViews[gCaseIndex].xyz;
    const float3 u = gInput[index].xyz;
    uint component = 0u;
    const NclsDirectionalSample sample = nclsAuditSample(
        gPrepared[gCaseIndex], wo, u, component);
    const float reevaluated = sample.valid != 0u && sample.nullEvent == 0u
        ? nclsAuditPdf(gPrepared[gCaseIndex], wo, sample.direction)
        : 0.0f;
    gOutput0[index] = float4(sample.direction, sample.pdf);
    gOutput1[index] = float4(reevaluated, sample.valid, sample.nullEvent, component);
}
