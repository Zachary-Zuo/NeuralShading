#include "../../../shaders/ncls/backends/unified_neural/unified_neural_core.slang"

StructuredBuffer<float4> gInput;
RWStructuredBuffer<float4> gOutput0;
RWStructuredBuffer<float4> gOutput1;
uniform uint gCount;
uniform uint gMode;

NclsLayerInterfaceIR nclsUnifiedTestTop()
{
    NclsLayerInterfaceIR top = {};
    top.kind = (uint)NclsInterfaceKind::Diffuse;
    top.colorR = 0.5f;
    top.colorG = 0.4f;
    top.colorB = 0.3f;
    return top;
}

void nclsUnifiedTestPrepared(out float prepared[NCLS_UNIFIED_PREPARE_OUTPUT])
{
    [unroll] for (uint index = 0u; index < NCLS_UNIFIED_PREPARE_OUTPUT; ++index)
        prepared[index] = 0.0f;
}

[numthreads(64, 1, 1)]
void evaluateUnifiedNeural(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    const float3 wo = normalize(float3(0.25f, -0.1f, 0.963f));
    const float3 wi = normalize(gInput[index].xyz);
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT];
    nclsUnifiedTestPrepared(prepared);
    const float3 scale = float3(0.1f);
    const float3 direct = nclsUnifiedDecodeDirect(float3(0.0f), scale);
    const float3 core = nclsUnifiedTopInterface(nclsUnifiedTestTop(), wo, wi);
    const float3 residual = nclsUnifiedDecodeResidual(float3(0.0f), prepared, scale, core);
    gOutput0[index] = float4(direct, all(isfinite(direct)) ? 1.0f : 0.0f);
    gOutput1[index] = float4(residual, all(isfinite(residual)) ? 1.0f : 0.0f);
}

[numthreads(64, 1, 1)]
void queryUnifiedSampler(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    const float3 wo = normalize(float3(0.25f, -0.1f, 0.963f));
    const float3 wi = normalize(gInput[index].xyz);
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT];
    nclsUnifiedTestPrepared(prepared);
    const float nvidia = nclsUnifiedNvidiaPreparedPdf(prepared, wo, wi);
    const float ltc = nclsUnifiedLtcPreparedPdf(prepared, wi);
    gOutput0[index] = float4(nvidia, ltc, 0.0f, 1.0f);
    gOutput1[index] = float4(0.0f);
}

[numthreads(64, 1, 1)]
void sampleUnifiedSampler(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    const float3 wo = normalize(float3(0.25f, -0.1f, 0.963f));
    const float3 u = frac(gInput[index].xyz);
    float prepared[NCLS_UNIFIED_PREPARE_OUTPUT];
    nclsUnifiedTestPrepared(prepared);
    uint nvidiaComponent;
    uint ltcComponent;
    const NclsDirectionalSample nvidia = nclsSampleUnifiedNvidiaPrepared(
        prepared, wo, u, nvidiaComponent);
    const NclsDirectionalSample ltc = nclsSampleUnifiedLtcPrepared(
        prepared, u, ltcComponent);
    const float nvidiaReevaluated = nvidia.valid != 0u && nvidia.nullEvent == 0u
        ? nclsUnifiedNvidiaPreparedPdf(prepared, wo, nvidia.direction) : 0.0f;
    const float ltcReevaluated = ltc.valid != 0u && ltc.nullEvent == 0u
        ? nclsUnifiedLtcPreparedPdf(prepared, ltc.direction) : 0.0f;
    gOutput0[index] = float4(nvidia.pdf, nvidiaReevaluated, nvidia.valid, nvidia.nullEvent);
    gOutput1[index] = float4(ltc.pdf, ltcReevaluated, ltc.valid, ltc.nullEvent);
}
