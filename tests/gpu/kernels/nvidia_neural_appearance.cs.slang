#include "../../../shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_pack.slang"
#include "../../../shaders/ncls/backends/nvidia_neural_appearance/nvidia_matched_ltc_sampler.slang"

StructuredBuffer<float4> gInput;
RWStructuredBuffer<float4> gOutput0;
RWStructuredBuffer<float4> gOutput1;
uniform uint gCount;

[numthreads(64, 1, 1)]
void evaluateNvidiaNeuralAppearance(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    const float3 wo = normalize(float3(0.25f, -0.1f, 0.963f));
    const float3 wi = normalize(gInput[index].xyz);
    const float3 response = nclsNvidiaNeuralDecodeResponseInference(float3(0.0f));
    const float3 bareF = nclsNvidiaNeuralResponseToBareFInference(response, wi);
    float raw[NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT] = {};
    const float pdf = nclsNvidiaNeuralSamplerPdf(raw, wo, wi);
    gOutput0[index] = float4(response, pdf);
    gOutput1[index] = float4(bareF, all(isfinite(bareF)) ? 1.0f : 0.0f);
}

[numthreads(64, 1, 1)]
void inspectNvidiaNeuralInput(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    const float3 wo = normalize(float3(0.0f, 0.0f, 1.0f));
    const float3 wi = normalize(gInput[index].xyz);
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT];
    [unroll] for (uint i = 0u; i < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++i)
        latent[i] = 0.1f * float(i);
    float frameRaw[NCLS_NVIDIA_NEURAL_FRAME_RAW_COUNT] = {
        0.2f, -0.1f, 0.3f, -0.4f, 0.5f, 0.1f,
        -0.2f, 0.3f, -0.1f, 0.2f, -0.4f, 0.6f};
    const float[NCLS_NVIDIA_NEURAL_EVALUATE_INPUT] input =
        nclsNvidiaNeuralEvaluationInputInference(latent, frameRaw, wo, wi);
    gOutput0[index] = float4(input[0], input[1], input[2], input[3]);
    gOutput1[index] = float4(input[4], input[5], input[6], input[7]);
}

[numthreads(64, 1, 1)]
void inspectNvidiaNeuralPacking(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    NclsNvidiaNeuralCompiledMaterial material;
    [unroll]
    for (uint word = 0u; word < 4u; ++word)
        material.latentBits[word] = nclsNvidiaNeuralPackHalf2(
            0.1f * float(2u * word), 0.1f * float(2u * word + 1u));
    material.layoutVersion = 1u;
    material.flags = 0u;
    material.reserved[0] = 0u;
    material.reserved[1] = 0u;
    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT];
    nclsNvidiaNeuralCompiledLatent(material, latent);
    NclsNvidiaNeuralPrepared prepared;
    [unroll]
    for (uint valueIndex = 0u; valueIndex < NCLS_NVIDIA_NEURAL_LATENT_COUNT; ++valueIndex)
        prepared.latent[valueIndex] = latent[valueIndex];
    [unroll]
    for (uint frameIndex = 0u; frameIndex < 2u; ++frameIndex)
    {
        prepared.frames[frameIndex].tangent = float3(1.0f, 0.0f, 0.0f);
        prepared.frames[frameIndex].bitangent = float3(0.0f, 1.0f, 0.0f);
        prepared.frames[frameIndex].normal = float3(0.0f, 0.0f, 1.0f);
    }
    [unroll]
    for (uint valueIndex = 0u; valueIndex < 6u; ++valueIndex)
        prepared.projectedView[valueIndex] = -0.05f * float(valueIndex + 1u);
    [unroll]
    for (uint valueIndex = 0u; valueIndex < NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT; ++valueIndex)
        prepared.samplerRaw[valueIndex] = 0.025f * float(valueIndex + 1u);
    const NclsNvidiaNeuralPrepared unpacked = nclsNvidiaNeuralUnpackPrepared(
        nclsNvidiaNeuralPackPrepared(prepared));
    gOutput0[index] = float4(
        unpacked.latent[0], unpacked.latent[1],
        unpacked.latent[6], unpacked.latent[7]);
    gOutput1[index] = float4(
        unpacked.projectedView[0], unpacked.projectedView[5],
        unpacked.samplerRaw[0], unpacked.samplerRaw[8]);
}

[numthreads(64, 1, 1)]
void inspectNvidiaMatchedLtc(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    float raw[NCLS_NVIDIA_MATCHED_LTC_RAW_COUNT] = {};
    const float3 wi = normalize(gInput[index].xyz);
    const float pdf = nclsNvidiaMatchedLtcPdf(raw, wi);
    uint component = 0u;
    const NclsDirectionalSample sample = nclsSampleNvidiaMatchedLtc(
        raw, float3(0.73f, 0.31f, 0.67f), component);
    gOutput0[index] = float4(pdf, sample.pdf, float(component), float(sample.valid));
    gOutput1[index] = float4(
        sample.direction,
        nclsNvidiaMatchedLtcPdf(raw, sample.direction));
}
