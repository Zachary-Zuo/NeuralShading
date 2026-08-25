#ifndef NCLS_LOBE_RESIDUAL_MLP_SLANG
#define NCLS_LOBE_RESIDUAL_MLP_SLANG

// lobe-residual 的原语：配置轴、权重缓冲与偏移、dense 层、lobe 解码与 LTC lobe 响应。
// 由 lobe_residual_core.slang 包含；本文件同样不依赖任何 Falcor 模块。
#include "../../reference/sampling.slang"

#ifndef NCLS_LOBE_RESIDUAL_K
#define NCLS_LOBE_RESIDUAL_K 2u
#endif
#ifndef NCLS_LOBE_RESIDUAL_CORRECTION // 0 = none，1 = log32
#define NCLS_LOBE_RESIDUAL_CORRECTION 0
#endif
// 训练时由 SlangPy 定义为可微张量类型与对应读取写法（取 P1.0 spike 编译成功的候选）。
#ifndef NCLS_LOBE_RESIDUAL_WEIGHTS_T
#define NCLS_LOBE_RESIDUAL_WEIGHTS_T StructuredBuffer<float>
#endif
#ifndef NCLS_LOBE_RESIDUAL_WEIGHT_READ
#define NCLS_LOBE_RESIDUAL_WEIGHT_READ(weights, index) (no_diff weights[index])
#endif

static const uint NCLS_LOBE_RESIDUAL_LATENT = 16u;
static const uint NCLS_LOBE_RESIDUAL_WO_FEATURES = 7u;
static const uint NCLS_LOBE_RESIDUAL_WIDTH = 64u;
static const uint NCLS_LOBE_RESIDUAL_HIDDEN = 8u;
static const uint NCLS_LOBE_RESIDUAL_LOBE_PARAMS = 9u;
static const uint NCLS_LOBE_RESIDUAL_RAW =
    NCLS_LOBE_RESIDUAL_HIDDEN + NCLS_LOBE_RESIDUAL_LOBE_PARAMS * NCLS_LOBE_RESIDUAL_K;
static const uint NCLS_LOBE_RESIDUAL_PSI = 6u;
static const uint NCLS_LOBE_RESIDUAL_CORRECTION_WIDTH = 32u;
static const uint NCLS_LOBE_RESIDUAL_LOBE_WORDS = (NCLS_LOBE_RESIDUAL_LOBE_PARAMS * NCLS_LOBE_RESIDUAL_K + 1u) / 2u;

typedef NCLS_LOBE_RESIDUAL_WEIGHTS_T NclsLobeResidualWeights;

// 权重缓冲 + 各层在缓冲内的 float 偏移；行主序 [out][in]，bias 紧随其后由 Python 决定。
struct NclsLobeResidualParams
{
    NclsLobeResidualWeights weights;
    uint prepareWeight0;     // [64][16 + 7]
    uint prepareBias0;
    uint prepareWeight1;     // [64][64]
    uint prepareBias1;
    uint prepareWeight2;     // [8 + 9K][64]
    uint prepareBias2;
    uint correctionWeight0;  // [32][8 + 6]，仅 log32
    uint correctionBias0;
    uint correctionWeight1;  // [3][32]
    uint correctionBias1;
};

struct NclsLobeResidualLobe : IDifferentiable
{
    float3 amplitude;
    float2 inverseScale;
    float3 shear;
    float angle;
};

[Differentiable]
float nclsLobeResidualWeight(NclsLobeResidualParams params, no_diff uint index)
{
    return NCLS_LOBE_RESIDUAL_WEIGHT_READ(params.weights, index);
}

[Differentiable]
float nclsLobeResidualSoftplus(float x)
{
    return log(1.0f + exp(-abs(x))) + max(x, 0.0f);
}

[Differentiable]
float nclsLobeResidualSilu(float x)
{
    return x / (1.0f + exp(-x));
}

// 一层 dense：output[row] = act(bias[row] + Σ weight[row][column]·input[column])；上界为常量。
[Differentiable]
void nclsLobeResidualDense(
    NclsLobeResidualParams params,
    no_diff uint weightOffset,
    no_diff uint biasOffset,
    no_diff uint inputCount,
    no_diff uint outputCount,
    no_diff bool activate,
    float input[NCLS_LOBE_RESIDUAL_WIDTH],
    out float output[NCLS_LOBE_RESIDUAL_WIDTH])
{
    [MaxIters(NCLS_LOBE_RESIDUAL_WIDTH)]
    for (uint row = 0u; row < outputCount; ++row)
    {
        float value = nclsLobeResidualWeight(params, biasOffset + row);
        [MaxIters(NCLS_LOBE_RESIDUAL_WIDTH)]
        for (uint column = 0u; column < inputCount; ++column)
            value += nclsLobeResidualWeight(params, weightOffset + row * inputCount + column) * input[column];
        output[row] = activate ? nclsLobeResidualSilu(value) : value;
    }
}

// 解码沿用 p1_compiler.slang:203-211：softplus / exp(clamp ±3) / 3·tanh / π·tanh，幅值非负由 softplus 保证。
[Differentiable]
NclsLobeResidualLobe nclsLobeResidualDecodeLobe(float raw[NCLS_LOBE_RESIDUAL_WIDTH], no_diff uint offset)
{
    NclsLobeResidualLobe lobe;
    lobe.amplitude = float3(
        nclsLobeResidualSoftplus(raw[offset + 0u]),
        nclsLobeResidualSoftplus(raw[offset + 1u]),
        nclsLobeResidualSoftplus(raw[offset + 2u]));
    lobe.inverseScale = exp(clamp(float2(raw[offset + 3u], raw[offset + 4u]), -3.0f, 3.0f));
    lobe.shear = 3.0f * tanh(float3(raw[offset + 5u], raw[offset + 6u], raw[offset + 7u]));
    lobe.angle = NCLS_PI * tanh(raw[offset + 8u]);
    return lobe;
}

// 与 legacy_ltc_k2.slang:40 nclsLegacyLtcK2LobeResponseCos 同形，返回 f·cos。
[Differentiable]
float3 nclsLobeResidualLobeResponseCos(NclsLobeResidualLobe lobe, no_diff float3 wiLocal)
{
    const float cosine = cos(lobe.angle);
    const float sine = sin(lobe.angle);
    const float rotatedX = cosine * wiLocal.x + sine * wiLocal.y;
    const float rotatedY = -sine * wiLocal.x + cosine * wiLocal.y;
    const float qx = lobe.inverseScale.x * rotatedX + lobe.shear.x * rotatedY + lobe.shear.y * wiLocal.z;
    const float qy = lobe.inverseScale.y * rotatedY + lobe.shear.z * wiLocal.z;
    const float qz = wiLocal.z;
    const float norm2 = max(qx * qx + qy * qy + qz * qz, 1e-10f);
    const float determinant = lobe.inverseScale.x * lobe.inverseScale.y;
    const float basis = determinant * max(qz, 0.0f) / (NCLS_PI * norm2 * norm2);
    return basis * lobe.amplitude;
}

#endif
