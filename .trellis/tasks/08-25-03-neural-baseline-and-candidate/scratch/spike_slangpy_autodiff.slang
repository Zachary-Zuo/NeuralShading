// 03 spike：最小 [Differentiable] 核，只由同目录 spike_slangpy_autodiff.py 经 SlangPy 编译。
// 形态：16 维输入 → 64 宽 SiLU → 9 个 raw → 解码为一个 LTC lobe → 按
// legacy_ltc_k2.slang:40 nclsLegacyLtcK2LobeResponseCos 的形式求 f·cos。
// @WEIGHT_TENSOR@ / @WEIGHT_READ@ 由 Python 按候选写法替换，用来探测 slangpy 携带的 slang 语法。
import slangpy;

static const uint SPIKE_INPUT = 16u;
static const uint SPIKE_WIDTH = 64u;
static const uint SPIKE_RAW = 9u;
static const uint SPIKE_W0 = 0u;
static const uint SPIKE_B0 = SPIKE_W0 + SPIKE_WIDTH * SPIKE_INPUT;
static const uint SPIKE_W1 = SPIKE_B0 + SPIKE_WIDTH;
static const uint SPIKE_B1 = SPIKE_W1 + SPIKE_RAW * SPIKE_WIDTH;
static const uint SPIKE_PARAMETER_COUNT = SPIKE_B1 + SPIKE_RAW;
static const float SPIKE_PI = 3.14159265358979323846f;

typedef @WEIGHT_TENSOR@ SpikeWeights;

struct SpikeLobe : IDifferentiable
{
    float3 amplitude;
    float2 inverseScale;
    float3 shear;
    float angle;
};

[Differentiable]
float spikeSoftplus(float x)
{
    return log(1.0f + exp(-abs(x))) + max(x, 0.0f);
}

[Differentiable]
float spikeSilu(float x)
{
    return x / (1.0f + exp(-x));
}

[Differentiable]
float spikeWeight(SpikeWeights weights, no_diff uint index)
{
    return @WEIGHT_READ@;
}

// 与 p1_compiler.slang:203-211 / torch_eval.decode_ltc_residual 同一解码。
[Differentiable]
SpikeLobe spikeDecode(float raw[SPIKE_RAW])
{
    SpikeLobe lobe;
    lobe.amplitude = float3(spikeSoftplus(raw[0]), spikeSoftplus(raw[1]), spikeSoftplus(raw[2]));
    lobe.inverseScale = exp(clamp(float2(raw[3], raw[4]), -3.0f, 3.0f));
    lobe.shear = 3.0f * tanh(float3(raw[5], raw[6], raw[7]));
    lobe.angle = SPIKE_PI * tanh(raw[8]);
    return lobe;
}

[Differentiable]
float3 spikeLobeResponseCos(SpikeLobe lobe, no_diff float3 wi)
{
    const float cosine = cos(lobe.angle);
    const float sine = sin(lobe.angle);
    const float rotatedX = cosine * wi.x + sine * wi.y;
    const float rotatedY = -sine * wi.x + cosine * wi.y;
    const float qx = lobe.inverseScale.x * rotatedX + lobe.shear.x * rotatedY + lobe.shear.y * wi.z;
    const float qy = lobe.inverseScale.y * rotatedY + lobe.shear.z * wi.z;
    const float qz = wi.z;
    const float norm2 = max(qx * qx + qy * qy + qz * qz, 1e-10f);
    const float determinant = lobe.inverseScale.x * lobe.inverseScale.y;
    const float basis = determinant * max(qz, 0.0f) / (SPIKE_PI * norm2 * norm2);
    return basis * lobe.amplitude;
}

// lobe 参数梯度探针：Python 端与 torch_eval.eval_ltc_residual 的 autograd 对照。
[Differentiable]
float3 spikeLobeProbe(float3 amplitude, float2 inverseScale, float3 shear, float angle, no_diff float3 wi)
{
    SpikeLobe lobe = { amplitude, inverseScale, shear, angle };
    return spikeLobeResponseCos(lobe, wi);
}

// S2.1 的 NclsLobeResidualParams 把权重张量放进 struct；探测 struct 包装是否保留可微性。
struct SpikeParams
{
    SpikeWeights weights;
    uint bias;
};

[Differentiable]
float spikeStructProbe(SpikeParams params, float x)
{
    return spikeWeight(params.weights, params.bias) * x;
}

// 权重梯度探针：Python 端用 float64 有限差分对照。
[Differentiable]
float[SPIKE_WIDTH] spikeLinear0(float weights[SPIKE_WIDTH][SPIKE_INPUT], float bias[SPIKE_WIDTH], float input[SPIKE_INPUT])
{
    float result[SPIKE_WIDTH];
    for (uint row = 0u; row < SPIKE_WIDTH; ++row)
    {
        float value = bias[row];
        for (uint column = 0u; column < SPIKE_INPUT; ++column)
            value += weights[row][column] * input[column];
        result[row] = value;
    }
    return result;
}

[Differentiable]
float[SPIKE_WIDTH] spikeActivate(float input[SPIKE_WIDTH])
{
    float result[SPIKE_WIDTH];
    for (uint index = 0u; index < SPIKE_WIDTH; ++index)
        result[index] = spikeSilu(input[index]);
    return result;
}

[Differentiable]
float[SPIKE_RAW] spikeLinear1(float weights[SPIKE_RAW][SPIKE_WIDTH], float bias[SPIKE_RAW], float input[SPIKE_WIDTH])
{
    float result[SPIKE_RAW];
    for (uint row = 0u; row < SPIKE_RAW; ++row)
    {
        float value = bias[row];
        for (uint column = 0u; column < SPIKE_WIDTH; ++column)
            value += weights[row][column] * input[column];
        result[row] = value;
    }
    return result;
}

[Differentiable]
float3 spikeDecodeEvaluate(float raw[SPIKE_RAW], no_diff float3 wi)
{
    return spikeLobeResponseCos(spikeDecode(raw), wi);
}

[Differentiable]
float3 spikeEvaluate(SpikeWeights weights, float4 x0, float4 x1, float4 x2, float4 x3, no_diff float3 wi)
{
    float input[SPIKE_INPUT] = {
        x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w,
        x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w,
    };
    float hidden[SPIKE_WIDTH];
    [MaxIters(SPIKE_WIDTH)]
    for (uint row = 0u; row < SPIKE_WIDTH; ++row)
    {
        float value = spikeWeight(weights, SPIKE_B0 + row);
        [MaxIters(SPIKE_INPUT)]
        for (uint column = 0u; column < SPIKE_INPUT; ++column)
            value += spikeWeight(weights, SPIKE_W0 + row * SPIKE_INPUT + column) * input[column];
        hidden[row] = spikeSilu(value);
    }
    float raw[SPIKE_RAW];
    [MaxIters(SPIKE_RAW)]
    for (uint row = 0u; row < SPIKE_RAW; ++row)
    {
        float value = spikeWeight(weights, SPIKE_B1 + row);
        [MaxIters(SPIKE_WIDTH)]
        for (uint column = 0u; column < SPIKE_WIDTH; ++column)
            value += spikeWeight(weights, SPIKE_W1 + row * SPIKE_WIDTH + column) * hidden[column];
        raw[row] = value;
    }
    return spikeLobeResponseCos(spikeDecode(raw), wi);
}
