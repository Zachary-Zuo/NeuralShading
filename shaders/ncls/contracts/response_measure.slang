#ifndef NCLS_RESPONSE_MEASURE_SLANG
#define NCLS_RESPONSE_MEASURE_SLANG

#include "scattering_contract.slang"

// 公共 scattering backend 的 evaluate() 返回纯 BSDF f，不含余弦。
// 数据集的 response_cos 以及 Falcor IBSDF::eval 都返回 f * dot(light, Ns)。
float nclsPositiveLightCosine(const NclsShadingFrame frame, const float3 wiWorld)
{
    return max(dot(frame.normal, wiWorld), 0.0f);
}

float3 nclsScatteringEvalToResponseCos(
    const NclsScatteringEval evaluation,
    const NclsShadingFrame frame,
    const float3 wiWorld)
{
    return evaluation.valid != 0u
        ? evaluation.f * nclsPositiveLightCosine(frame, wiWorld)
        : float3(0.0f);
}

// Falcor IBSDF::eval(wi, wo) 的 wo 对应 NCLS 的入射光方向 wiWorld，
// 所以它与 reference dataset 的 response_cos 使用同一个 adapter。
float3 nclsScatteringEvalToFalcorIbsdfEval(
    const NclsScatteringEval evaluation,
    const NclsShadingFrame frame,
    const float3 nclsWiWorld)
{
    return nclsScatteringEvalToResponseCos(evaluation, frame, nclsWiWorld);
}

#endif
