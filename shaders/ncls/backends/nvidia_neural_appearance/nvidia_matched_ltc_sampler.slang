#ifndef NCLS_NVIDIA_MATCHED_LTC_SAMPLER_SLANG
#define NCLS_NVIDIA_MATCHED_LTC_SAMPLER_SLANG

// 08-25-03 matched sampler adaptation：冻结 NVIDIA evaluator 后单独训练 LTC-K2。
#include "nvidia_neural_appearance_layout.slang"
#include "../../scattering/ltc.slang"
#include "../../scattering/mixture.slang"

static const uint NCLS_NVIDIA_MATCHED_LTC_RAW_COUNT = 13u;
static const float NCLS_NVIDIA_MATCHED_LTC_SAFETY_WEIGHT = 0.03125f;

struct NclsNvidiaMatchedLtcProposal
{
    NclsLtcTransform lobe0;
    NclsLtcTransform lobe1;
    float learnedWeight;
    uint valid;
};

[Differentiable]
float nclsNvidiaMatchedLtcDot11(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT])
{
    float value = bias;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_SAMPLER_INPUT; ++index)
        value += weights[index] * input[index];
    return value;
}

[Differentiable]
float nclsNvidiaMatchedLtcDot32(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    float value = bias;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH; ++index)
        value += weights[index] * input[index];
    return value;
}

[Differentiable]
float nclsNvidiaMatchedLtcRelu(float value)
{
    return max(value, 0.0f);
}

[Differentiable]
float[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT] nclsNvidiaMatchedLtcInput(
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT],
    no_diff float3 wo)
{
    float result[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT];
    result[0] = wo.x;
    result[1] = wo.y;
    result[2] = wo.z;
    [unroll]
    for (uint index = 0u; index < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++index)
        result[3u + index] = latent[index];
    return result;
}

[Differentiable]
float nclsNvidiaMatchedLtcTanhApprox(float value)
{
    return value / sqrt(1.0f + value * value);
}

NclsNvidiaMatchedLtcProposal nclsNvidiaMatchedLtcDecode(
    float raw[NCLS_NVIDIA_MATCHED_LTC_RAW_COUNT])
{
    NclsNvidiaMatchedLtcProposal result = {};
    result.lobe0.inverseScale = exp(clamp(float2(raw[0], raw[1]), -4.0f, 4.0f));
    result.lobe0.shear = 2.0f * float3(
        nclsNvidiaMatchedLtcTanhApprox(raw[2]),
        nclsNvidiaMatchedLtcTanhApprox(raw[3]),
        nclsNvidiaMatchedLtcTanhApprox(raw[4]));
    result.lobe0.angle = NCLS_PI * nclsNvidiaMatchedLtcTanhApprox(raw[5]);
    result.lobe1.inverseScale = exp(clamp(float2(raw[6], raw[7]), -4.0f, 4.0f));
    result.lobe1.shear = 2.0f * float3(
        nclsNvidiaMatchedLtcTanhApprox(raw[8]),
        nclsNvidiaMatchedLtcTanhApprox(raw[9]),
        nclsNvidiaMatchedLtcTanhApprox(raw[10]));
    result.lobe1.angle = NCLS_PI * nclsNvidiaMatchedLtcTanhApprox(raw[11]);
    result.learnedWeight = 0.5f + 0.5f * nclsNvidiaMatchedLtcTanhApprox(raw[12]);
    result.valid = nclsLtcTransformValid(result.lobe0)
        && nclsLtcTransformValid(result.lobe1) ? 1u : 0u;
    return result;
}

float nclsNvidiaMatchedLtcPdfDecoded(
    NclsNvidiaMatchedLtcProposal params,
    float3 wi)
{
    if (params.valid == 0u || wi.z <= NCLS_MIN_COS) return 0.0f;
    const float learnedMass = 1.0f - NCLS_NVIDIA_MATCHED_LTC_SAFETY_WEIGHT;
    return NCLS_NVIDIA_MATCHED_LTC_SAFETY_WEIGHT * nclsCosineHemispherePdf(wi)
        + learnedMass * (
            params.learnedWeight * nclsLtcPdf(params.lobe0, wi)
            + (1.0f - params.learnedWeight) * nclsLtcPdf(params.lobe1, wi));
}

float nclsNvidiaMatchedLtcPdf(
    float raw[NCLS_NVIDIA_MATCHED_LTC_RAW_COUNT],
    float3 wi)
{
    return nclsNvidiaMatchedLtcPdfDecoded(nclsNvidiaMatchedLtcDecode(raw), wi);
}

[Differentiable]
float nclsNvidiaMatchedLtcPdfTraining(
    float raw[NCLS_NVIDIA_MATCHED_LTC_RAW_COUNT],
    no_diff float3 wi)
{
    NclsLtcTransform lobe0;
    lobe0.inverseScale = exp(clamp(float2(raw[0], raw[1]), -4.0f, 4.0f));
    lobe0.shear = 2.0f * float3(
        nclsNvidiaMatchedLtcTanhApprox(raw[2]),
        nclsNvidiaMatchedLtcTanhApprox(raw[3]),
        nclsNvidiaMatchedLtcTanhApprox(raw[4]));
    lobe0.angle = NCLS_PI * nclsNvidiaMatchedLtcTanhApprox(raw[5]);
    NclsLtcTransform lobe1;
    lobe1.inverseScale = exp(clamp(float2(raw[6], raw[7]), -4.0f, 4.0f));
    lobe1.shear = 2.0f * float3(
        nclsNvidiaMatchedLtcTanhApprox(raw[8]),
        nclsNvidiaMatchedLtcTanhApprox(raw[9]),
        nclsNvidiaMatchedLtcTanhApprox(raw[10]));
    lobe1.angle = NCLS_PI * nclsNvidiaMatchedLtcTanhApprox(raw[11]);
    const float learnedWeight = 0.5f
        + 0.5f * nclsNvidiaMatchedLtcTanhApprox(raw[12]);
    const float learnedMass = 1.0f - NCLS_NVIDIA_MATCHED_LTC_SAFETY_WEIGHT;
    return NCLS_NVIDIA_MATCHED_LTC_SAFETY_WEIGHT
        * no_diff(nclsCosineHemispherePdf(wi))
        + learnedMass * (
            learnedWeight * nclsLtcBasisUnchecked(lobe0, wi)
            + (1.0f - learnedWeight) * nclsLtcBasisUnchecked(lobe1, wi));
}

NclsDirectionalSample nclsSampleNvidiaMatchedLtc(
    float raw[NCLS_NVIDIA_MATCHED_LTC_RAW_COUNT],
    float3 u,
    out uint component)
{
    const NclsNvidiaMatchedLtcProposal params = nclsNvidiaMatchedLtcDecode(raw);
    component = 0u;
    if (params.valid == 0u) return nclsInvalidDirectionalSample();
    const float learnedMass = 1.0f - NCLS_NVIDIA_MATCHED_LTC_SAFETY_WEIGHT;
    const float3 weights = float3(
        NCLS_NVIDIA_MATCHED_LTC_SAFETY_WEIGHT,
        learnedMass * params.learnedWeight,
        learnedMass * (1.0f - params.learnedWeight));
    const NclsMixture3Selection selection = nclsSelectMixture3(weights, u.x);
    if (selection.valid == 0u) return nclsInvalidDirectionalSample();
    component = selection.component;
    NclsDirectionalSample sample = component == 0u
        ? nclsSampleCosineDirection(u.yz)
        : (component == 1u
            ? nclsSampleLtc(params.lobe0, u.yz)
            : nclsSampleLtc(params.lobe1, u.yz));
    if (sample.valid != 0u && sample.nullEvent == 0u)
        sample.pdf = nclsNvidiaMatchedLtcPdfDecoded(params, sample.direction);
    return sample;
}

// SlangPy 0.43.1 的 callable active-mask 缓存要求训练/推理使用不同身份。
[Differentiable]
float[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT] nclsNvidiaMatchedLtcInputTraining(
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT],
    no_diff float3 wo)
{
    return nclsNvidiaMatchedLtcInput(latent, wo);
}

float[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT] nclsNvidiaMatchedLtcInputInference(
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT],
    float3 wo)
{
    return nclsNvidiaMatchedLtcInput(latent, wo);
}

[Differentiable]
float nclsNvidiaMatchedLtcDot11Training(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT])
{
    return nclsNvidiaMatchedLtcDot11(weights, bias, input);
}

float nclsNvidiaMatchedLtcDot11Inference(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_INPUT])
{
    return nclsNvidiaMatchedLtcDot11(weights, bias, input);
}

[Differentiable]
float nclsNvidiaMatchedLtcDot32ATraining(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    return nclsNvidiaMatchedLtcDot32(weights, bias, input);
}

[Differentiable]
float nclsNvidiaMatchedLtcDot32BTraining(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    return nclsNvidiaMatchedLtcDot32(weights, bias, input);
}

[Differentiable]
float nclsNvidiaMatchedLtcDot32OutTraining(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    return nclsNvidiaMatchedLtcDot32(weights, bias, input);
}

float nclsNvidiaMatchedLtcDot32AInference(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    return nclsNvidiaMatchedLtcDot32(weights, bias, input);
}

float nclsNvidiaMatchedLtcDot32BInference(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    return nclsNvidiaMatchedLtcDot32(weights, bias, input);
}

float nclsNvidiaMatchedLtcDot32OutInference(
    float weights[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH],
    float bias,
    float input[NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH])
{
    return nclsNvidiaMatchedLtcDot32(weights, bias, input);
}

[Differentiable]
float nclsNvidiaMatchedLtcReluTraining(float value)
{
    return nclsNvidiaMatchedLtcRelu(value);
}

float nclsNvidiaMatchedLtcReluInference(float value)
{
    return nclsNvidiaMatchedLtcRelu(value);
}

#endif
