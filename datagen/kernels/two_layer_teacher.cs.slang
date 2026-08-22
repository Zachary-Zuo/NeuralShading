#include "../../teacher/layered_walk.slang"

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
void evaluateTwoLayerTeacher(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint queryIndex = dispatchThreadID.x;
    if (queryIndex >= gQueryCount) return;

    const LayerStack stack = gStacks[queryIndex];
    const float3 viewDirection = normalize(gViewDirections[queryIndex].xyz);
    const float3 lightDirection = normalize(gLightDirections[queryIndex].xyz);
    float3 sum = float3(0.0f);
    float3 sumSquare = float3(0.0f);
    for (uint sampleIndex = 0u; sampleIndex < gSampleCount; ++sampleIndex)
    {
        const uint2 rngKey = uint2(queryIndex ^ gSeed, sampleIndex + gSeed * 1664525u);
        const float3 value = nclsEvalTwoLayerPbrtWalk(stack, viewDirection, lightDirection, rngKey, gMaxDepth);
        sum += value;
        sumSquare += value * value;
    }

    const float inverseSampleCount = 1.0f / max(1u, gSampleCount);
    gMean[queryIndex] = float4(sum * inverseSampleCount, 1.0f);
    gMeanSquare[queryIndex] = float4(sumSquare * inverseSampleCount, 1.0f);
}
