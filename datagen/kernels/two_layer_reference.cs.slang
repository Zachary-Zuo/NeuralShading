#include "../../teacher/two_layer_reference.slang"

StructuredBuffer<LayerStack> gStacks;
StructuredBuffer<float4> gViewDirections;
StructuredBuffer<float4> gLightDirections;
RWStructuredBuffer<float4> gMean;
RWStructuredBuffer<float4> gMeanSquare;

uniform uint gQueryCount;
uniform uint gSampleCount;
uniform uint gSeed;
uniform uint gMaxDepth;

[numthreads(64, 1, 1)]
void evaluateDirectCoat(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint queryIndex = dispatchThreadID.x;
    if (queryIndex >= gQueryCount) return;

    const LayerStack stack = gStacks[queryIndex];
    const float3 viewDirection = normalize(gViewDirections[queryIndex].xyz);
    const float3 lightDirection = normalize(gLightDirections[queryIndex].xyz);
    const uint2 rngKey = uint2(queryIndex ^ gSeed, gSeed * 1664525u);
    const float3 value = evalDirectCoat(stack, viewDirection, lightDirection, rngKey);
    gMean[queryIndex] = float4(value, 1.0f);
    gMeanSquare[queryIndex] = float4(value * value, 1.0f);
}

[numthreads(64, 1, 1)]
void sampleDirectCoatTransmission(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint queryIndex = dispatchThreadID.x;
    if (queryIndex >= gQueryCount) return;

    const LayerInterface coat = gStacks[queryIndex].layers[0];
    const float3 viewDirection = normalize(gViewDirections[queryIndex].xyz);
    AnisotropicGGX coatDistribution = AnisotropicGGX(float2(coat.roughnessX, coat.roughnessY));
    PBRTDielectricBSDF top = {coatDistribution, coat.etaR};
    const BSDFContext context = viewDirection.z > 0.0f
        ? BSDFContext(1.0f, top.getEta())
        : BSDFContext(top.getEta(), 1.0f);

    float3 sum = float3(0.0f);
    float3 sumSquare = float3(0.0f);
    for (uint sampleIndex = 0u; sampleIndex < gSampleCount; ++sampleIndex)
    {
        UniformSampleGenerator sg = UniformSampleGenerator(
            uint2(queryIndex ^ gSeed, sampleIndex + gSeed * 1664525u), 0);
        float3 sampledDirection;
        float pdf;
        float3 weight;
        uint lobeType;
        if (top.sample(viewDirection, sampledDirection, pdf, weight, lobeType, sg, context)
            && viewDirection.z * sampledDirection.z < 0.0f)
        {
            sum += weight;
            sumSquare += weight * weight;
        }
    }
    const float inverseSampleCount = 1.0f / max(1u, gSampleCount);
    gMean[queryIndex] = float4(sum * inverseSampleCount, 1.0f);
    gMeanSquare[queryIndex] = float4(sumSquare * inverseSampleCount, 1.0f);
}

[numthreads(64, 1, 1)]
void sampleDirectCoatReflection(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint queryIndex = dispatchThreadID.x;
    if (queryIndex >= gQueryCount) return;

    const LayerInterface coat = gStacks[queryIndex].layers[0];
    const float3 viewDirection = normalize(gViewDirections[queryIndex].xyz);
    AnisotropicGGX coatDistribution = AnisotropicGGX(float2(coat.roughnessX, coat.roughnessY));
    PBRTDielectricBSDF top = {coatDistribution, coat.etaR};
    const BSDFContext context = viewDirection.z > 0.0f
        ? BSDFContext(1.0f, top.getEta())
        : BSDFContext(top.getEta(), 1.0f);

    float3 sum = float3(0.0f);
    float3 sumSquare = float3(0.0f);
    for (uint sampleIndex = 0u; sampleIndex < gSampleCount; ++sampleIndex)
    {
        UniformSampleGenerator sg = UniformSampleGenerator(
            uint2(queryIndex ^ gSeed, sampleIndex + gSeed * 1664525u), 0);
        float3 sampledDirection;
        float pdf;
        float3 weight;
        uint lobeType;
        if (top.sample(viewDirection, sampledDirection, pdf, weight, lobeType, sg, context)
            && viewDirection.z * sampledDirection.z > 0.0f)
        {
            sum += weight;
            sumSquare += weight * weight;
        }
    }
    const float inverseSampleCount = 1.0f / max(1u, gSampleCount);
    gMean[queryIndex] = float4(sum * inverseSampleCount, 1.0f);
    gMeanSquare[queryIndex] = float4(sumSquare * inverseSampleCount, 1.0f);
}

[numthreads(64, 1, 1)]
void evaluateTwoLayerReference(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint queryIndex = dispatchThreadID.x;
    if (queryIndex >= gQueryCount) return;

    const LayerStack stack = gStacks[queryIndex];
    const float3 viewDirection = normalize(gViewDirections[queryIndex].xyz);
    const float3 lightDirection = normalize(gLightDirections[queryIndex].xyz);

    float3 sum = float3(0.0f);
    float3 sumSquare = float3(0.0f);
    for (uint sampleIndex = 0; sampleIndex < gSampleCount; ++sampleIndex)
    {
        const uint2 rngKey = uint2(queryIndex ^ gSeed, sampleIndex + gSeed * 1664525u);
        const float3 value = evalTwoLayerReference(
            stack,
            viewDirection,
            lightDirection,
            rngKey,
            gMaxDepth);
        sum += value;
        sumSquare += value * value;
    }

    const float invSampleCount = 1.0f / max(1u, gSampleCount);
    gMean[queryIndex] = float4(sum * invSampleCount, 1.0f);
    gMeanSquare[queryIndex] = float4(sumSquare * invSampleCount, 1.0f);
}
