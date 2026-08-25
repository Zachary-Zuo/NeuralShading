#ifndef NCLS_LTC_K2_ANALYTIC_CONTROL_CORE_SLANG
#define NCLS_LTC_K2_ANALYTIC_CONTROL_CORE_SLANG

// ltc-k2-analytic-control：exact top-interface response + 两个公共 LTC basis。
// 这是 Falcor-free optimized-code control，不在本任务注册 viewer fallback。
#include "../../contracts/layer_stack_ir.slang"
#include "../../reference/interfaces.slang"
#include "../../scattering/ltc.slang"

static const uint NCLS_LTC_K2_ANALYTIC_CONTROL_LOBE_COUNT = 2u;

struct NclsLtcK2AnalyticControlLobe
{
    float3 amplitude;
    NclsLtcTransform transform;
};

struct NclsLtcK2AnalyticControlState
{
    NclsLayerInterfaceIR top;
    NclsLtcK2AnalyticControlLobe residual[NCLS_LTC_K2_ANALYTIC_CONTROL_LOBE_COUNT];
};

float3 nclsLtcK2AnalyticControlResponseCos(
    NclsLtcK2AnalyticControlState state,
    float3 woLocal,
    float3 wiLocal)
{
    if (!nclsIsFiniteFloat3(woLocal) || !nclsIsFiniteFloat3(wiLocal)
        || woLocal.z <= NCLS_MIN_COS || wiLocal.z <= NCLS_MIN_COS)
        return float3(0.0f);
    const float etaBelow = state.top.kind == (uint)NclsInterfaceKind::RoughDielectric
        ? state.top.relativeIor
        : 1.0f;
    float3 response = nclsEvalInterface(state.top, woLocal, wiLocal, 1.0f, etaBelow)
        * wiLocal.z;
    [unroll]
    for (uint index = 0u; index < NCLS_LTC_K2_ANALYTIC_CONTROL_LOBE_COUNT; ++index)
    {
        const NclsLtcK2AnalyticControlLobe lobe = state.residual[index];
        if (!nclsLtcTransformValid(lobe.transform)
            || !nclsIsFiniteFloat3(lobe.amplitude))
            return float3(0.0f);
        response += nclsLtcResponseCosUnchecked(
            lobe.transform, max(lobe.amplitude, float3(0.0f)), wiLocal);
    }
    return all(isfinite(response)) ? response : float3(0.0f);
}

float3 nclsLtcK2AnalyticControlEvaluateF(
    NclsLtcK2AnalyticControlState state,
    float3 woLocal,
    float3 wiLocal)
{
    return wiLocal.z > NCLS_MIN_COS
        ? nclsLtcK2AnalyticControlResponseCos(state, woLocal, wiLocal) / wiLocal.z
        : float3(0.0f);
}

#endif
