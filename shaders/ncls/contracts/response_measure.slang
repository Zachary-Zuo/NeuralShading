#ifndef NCLS_RESPONSE_MEASURE_SLANG
#define NCLS_RESPONSE_MEASURE_SLANG

#include "scattering_contract.slang"

// 公共 scattering backend 的 evaluate() 返回纯 BSDF f，不含余弦。
// Falcor IBSDF::eval 的 renderer adapter 需要 f * abs(dot(light, Ns))；
// 绝对值让同一个 transport response 同时覆盖 reflection 与 transmission。
float nclsAbsoluteLightCosine(const NclsShadingFrame frame, const float3 wiWorld)
{
    return abs(dot(frame.normal, wiWorld));
}

float3 nclsScatteringEvalToResponseCos(
    const NclsScatteringEval evaluation,
    const NclsShadingFrame frame,
    const float3 wiWorld)
{
    return evaluation.valid != 0u
        ? evaluation.f * nclsAbsoluteLightCosine(frame, wiWorld)
        : float3(0.0f);
}

// Falcor IBSDF::eval(wi, wo) 的 wo 对应 NCLS 的入射光方向 wiWorld。
// 该转换只存在于 renderer integration，不是 training evaluator target。
float3 nclsScatteringEvalToFalcorIbsdfEval(
    const NclsScatteringEval evaluation,
    const NclsShadingFrame frame,
    const float3 nclsWiWorld)
{
    return nclsScatteringEvalToResponseCos(evaluation, frame, nclsWiWorld);
}

#endif
