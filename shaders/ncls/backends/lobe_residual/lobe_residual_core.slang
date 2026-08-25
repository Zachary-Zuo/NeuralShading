#ifndef NCLS_LOBE_RESIDUAL_CORE_SLANG
#define NCLS_LOBE_RESIDUAL_CORE_SLANG

// lobe-residual 候选的 Falcor-free 核心（docs/research/p1_v2_plan.md §1.1 / S2.1）。
// 同一份源被 SlangPy（训练）与 Falcor（部署）编译；权重偏移由 Python 运行时写入
// NclsLobeResidualParams，Slang 内不写偏移常量。K 与 correction 是仅有的两个配置轴。
#include "../../contracts/layer_stack_ir.slang"
#include "../../reference/interfaces.slang"
#include "../../reference/sampling.slang"

#include "lobe_residual_mlp.slang"

// prepare 的可微输出：训练直接用它 evaluate，部署先 pack 为 NclsLobeResidualState。
struct NclsLobeResidualDecoded : IDifferentiable
{
    NclsLobeResidualLobe lobes[NCLS_LOBE_RESIDUAL_K];
    float hidden[NCLS_LOBE_RESIDUAL_HIDDEN];
};

// 部署 state：K=2 时 none 48 B、log32 64 B。顶层 IR 不进 state，evaluate 时按 materialSlot 读取。
struct NclsLobeResidualState
{
    uint materialSlot;
    uint lobeBits[NCLS_LOBE_RESIDUAL_LOBE_WORDS];
#if NCLS_LOBE_RESIDUAL_CORRECTION
    uint hiddenBits[NCLS_LOBE_RESIDUAL_HIDDEN / 2u];
#endif
    uint flags;      // 低 8 位 K，第 8 位 correction
    uint reserved;
};

// (16 + 7) → 64 → 64 → 8 + 9K。φ(wo) = (x, y, z, x², y², xy, sinθ)。
[Differentiable]
NclsLobeResidualDecoded nclsLobeResidualPrepare(
    float latent[NCLS_LOBE_RESIDUAL_LATENT],
    no_diff float3 woLocal,
    NclsLobeResidualParams params)
{
    float input[NCLS_LOBE_RESIDUAL_WIDTH];
    [unroll]
    for (uint index = 0u; index < NCLS_LOBE_RESIDUAL_LATENT; ++index)
        input[index] = latent[index];
    input[NCLS_LOBE_RESIDUAL_LATENT + 0u] = woLocal.x;
    input[NCLS_LOBE_RESIDUAL_LATENT + 1u] = woLocal.y;
    input[NCLS_LOBE_RESIDUAL_LATENT + 2u] = woLocal.z;
    input[NCLS_LOBE_RESIDUAL_LATENT + 3u] = woLocal.x * woLocal.x;
    input[NCLS_LOBE_RESIDUAL_LATENT + 4u] = woLocal.y * woLocal.y;
    input[NCLS_LOBE_RESIDUAL_LATENT + 5u] = woLocal.x * woLocal.y;
    input[NCLS_LOBE_RESIDUAL_LATENT + 6u] = sqrt(max(1.0f - woLocal.z * woLocal.z, 0.0f));
    float hidden0[NCLS_LOBE_RESIDUAL_WIDTH];
    float hidden1[NCLS_LOBE_RESIDUAL_WIDTH];
    float raw[NCLS_LOBE_RESIDUAL_WIDTH];
    nclsLobeResidualDense(params, params.prepareWeight0, params.prepareBias0,
        NCLS_LOBE_RESIDUAL_LATENT + NCLS_LOBE_RESIDUAL_WO_FEATURES, NCLS_LOBE_RESIDUAL_WIDTH, true, input, hidden0);
    nclsLobeResidualDense(params, params.prepareWeight1, params.prepareBias1,
        NCLS_LOBE_RESIDUAL_WIDTH, NCLS_LOBE_RESIDUAL_WIDTH, true, hidden0, hidden1);
    nclsLobeResidualDense(params, params.prepareWeight2, params.prepareBias2,
        NCLS_LOBE_RESIDUAL_WIDTH, NCLS_LOBE_RESIDUAL_RAW, false, hidden1, raw);
    NclsLobeResidualDecoded decoded;
    [unroll]
    for (uint index = 0u; index < NCLS_LOBE_RESIDUAL_HIDDEN; ++index)
        decoded.hidden[index] = raw[index];
    [unroll]
    for (uint lobe = 0u; lobe < NCLS_LOBE_RESIDUAL_K; ++lobe)
        decoded.lobes[lobe] = nclsLobeResidualDecodeLobe(
            raw, NCLS_LOBE_RESIDUAL_HIDDEN + NCLS_LOBE_RESIDUAL_LOBE_PARAMS * lobe);
    return decoded;
}

#if NCLS_LOBE_RESIDUAL_CORRECTION
// Δ(h, ψ(wi))：(8 + 6) → 32 → 3，ψ = (wi, half-slope xy, wi.z)；输出 clamp(·, −2, 2)。
[Differentiable]
float3 nclsLobeResidualCorrection(
    NclsLobeResidualDecoded decoded,
    no_diff float3 woLocal,
    no_diff float3 wiLocal,
    NclsLobeResidualParams params)
{
    float input[NCLS_LOBE_RESIDUAL_WIDTH];
    [unroll]
    for (uint index = 0u; index < NCLS_LOBE_RESIDUAL_HIDDEN; ++index)
        input[index] = decoded.hidden[index];
    const float3 halfVector = no_diff nclsSafeNormalize(woLocal + wiLocal);
    const float inverseZ = 1.0f / max(halfVector.z, 1e-3f);
    input[NCLS_LOBE_RESIDUAL_HIDDEN + 0u] = wiLocal.x;
    input[NCLS_LOBE_RESIDUAL_HIDDEN + 1u] = wiLocal.y;
    input[NCLS_LOBE_RESIDUAL_HIDDEN + 2u] = wiLocal.z;
    input[NCLS_LOBE_RESIDUAL_HIDDEN + 3u] = clamp(halfVector.x * inverseZ, -8.0f, 8.0f);
    input[NCLS_LOBE_RESIDUAL_HIDDEN + 4u] = clamp(halfVector.y * inverseZ, -8.0f, 8.0f);
    input[NCLS_LOBE_RESIDUAL_HIDDEN + 5u] = wiLocal.z;
    float hidden[NCLS_LOBE_RESIDUAL_WIDTH];
    float output[NCLS_LOBE_RESIDUAL_WIDTH];
    nclsLobeResidualDense(params, params.correctionWeight0, params.correctionBias0,
        NCLS_LOBE_RESIDUAL_HIDDEN + NCLS_LOBE_RESIDUAL_PSI, NCLS_LOBE_RESIDUAL_CORRECTION_WIDTH, true, input, hidden);
    nclsLobeResidualDense(params, params.correctionWeight1, params.correctionBias1,
        NCLS_LOBE_RESIDUAL_CORRECTION_WIDTH, 3u, false, hidden, output);
    return clamp(float3(output[0], output[1], output[2]), -2.0f, 2.0f);
}
#endif

// f = f_top(精确顶层反射) + Σ_k lobe_k·cos / wi.z，log32 时再乘 exp(Δ)。全程无非负截断。
[Differentiable]
float3 nclsLobeResidualEvaluateF(
    NclsLobeResidualDecoded decoded,
    no_diff NclsLayerInterfaceIR top,
    no_diff float3 woLocal,
    no_diff float3 wiLocal,
    NclsLobeResidualParams params)
{
    if (woLocal.z <= NCLS_MIN_COS || wiLocal.z <= NCLS_MIN_COS) return float3(0.0f);
    const float etaBelow = top.kind == (uint)NclsInterfaceKind::RoughDielectric ? top.relativeIor : 1.0f;
    // interfaces.slang 的形参顺序是 (layer, wi=视线, wo=光源)，与 NCLS 的 wo/wi 相反。
    float3 f = no_diff nclsEvalInterface(top, woLocal, wiLocal, 1.0f, etaBelow);
    float3 residualCos = float3(0.0f);
    [unroll]
    for (uint lobe = 0u; lobe < NCLS_LOBE_RESIDUAL_K; ++lobe)
        residualCos += nclsLobeResidualLobeResponseCos(decoded.lobes[lobe], wiLocal);
    f += residualCos / wiLocal.z;
#if NCLS_LOBE_RESIDUAL_CORRECTION
    f *= exp(nclsLobeResidualCorrection(decoded, woLocal, wiLocal, params));
#endif
    return f;
}

// TODO(S2.3)：混合 proposal p = w_0·p_top + Σ_k w_k·p_k。p_top = nclsInterfacePdf(top, woLocal, wiLocal,
// 1, etaBelow, NCLS_SAMPLE_REFLECTION)；p_k 即 nclsLobeResidualLobeResponseCos 的 basis 项；w 由 prepare 按能量归一。
float nclsLobeResidualPdf(
    NclsLobeResidualDecoded decoded,
    NclsLayerInterfaceIR top,
    float3 woLocal,
    float3 wiLocal)
{
    return 0.0f;
}

// TODO(S2.3)：u.x 选分量；顶层走 S2.2 的预抽随机数 VNDF 重载，lobe 走 LTC 逆变换 wi = normalize(M_k·cosine 样本)。
bool nclsLobeResidualSample(
    NclsLobeResidualDecoded decoded,
    NclsLayerInterfaceIR top,
    float3 woLocal,
    float3 u,
    out float3 wiLocal,
    out float pdf)
{
    wiLocal = float3(0.0f, 0.0f, 1.0f);
    pdf = 0.0f;
    return false;
}

#endif
