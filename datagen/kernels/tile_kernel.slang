#include "../../teacher/layered_walk.slang"

StructuredBuffer<LayerStack> gStacks;
RWStructuredBuffer<uint> gValidation;

StructuredBuffer<float4> gViewDirections;
StructuredBuffer<float4> gLightDirections;
StructuredBuffer<uint> gTileSeeds;
RWStructuredBuffer<float4> gMeanA;
RWStructuredBuffer<float4> gMeanSquareA;
RWStructuredBuffer<float4> gMeanB;
RWStructuredBuffer<float4> gMeanSquareB;

uniform uint gQueryCount;
uniform uint gLightCount;
uniform uint gLightIndexOffset;
uniform uint gSampleCountPerHalf;
uniform uint gSampleOffset;
uniform uint gSeed;
uniform uint gMaxDepth;

// First compute milestone: prove Python-packed LayerStack payloads are read
// with the expected layout before the random-walk evaluator is connected.
[numthreads(64, 1, 1)]
void validateStackLayout(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    const LayerStack stack = gStacks[index];
    const bool valid = stack.magic == NCLS_SCHEMA_MAGIC
        && stack.version == NCLS_SCHEMA_VERSION
        && stack.layerCount >= 1
        && stack.layerCount <= NCLS_MAX_LAYERS
        && stack.mediumCount + 1 == stack.layerCount;
    gValidation[index] = valid ? 1 : 0;
}

[numthreads(64, 1, 1)]
void generateTileBatch(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint queryIndex = dispatchThreadID.x;
    if (queryIndex >= gQueryCount) return;

    const uint tileIndex = queryIndex / gLightCount;
    const uint lightBufferIndex = queryIndex - tileIndex * gLightCount;
    const uint lightIndex = lightBufferIndex + gLightIndexOffset;
    const LayerStack stack = gStacks[tileIndex];
    const float3 viewDirection = normalize(gViewDirections[tileIndex].xyz);
    const float3 lightDirection = normalize(gLightDirections[lightBufferIndex].xyz);
    const uint tileSeed = gTileSeeds[tileIndex] ^ gSeed;
    float3 sumA = float3(0.0f);
    float3 sumSquareA = float3(0.0f);
    float3 sumB = float3(0.0f);
    float3 sumSquareB = float3(0.0f);

    for (uint sampleIndex = 0u; sampleIndex < gSampleCountPerHalf; ++sampleIndex)
    {
        const uint globalSampleIndex = sampleIndex + gSampleOffset;
        const uint2 keyA = uint2(lightIndex ^ tileSeed, globalSampleIndex + tileSeed * 1664525u);
        const uint seedB = tileSeed ^ 0xa511e9b3u;
        const uint2 keyB = uint2(lightIndex ^ seedB, globalSampleIndex + seedB * 22695477u);
        const float3 valueA = nclsEvalTeacher(
            stack, viewDirection, lightDirection, keyA, gMaxDepth);
        const float3 valueB = nclsEvalTeacher(
            stack, viewDirection, lightDirection, keyB, gMaxDepth);
        sumA += valueA;
        sumSquareA += valueA * valueA;
        sumB += valueB;
        sumSquareB += valueB * valueB;
    }

    const float inverseSampleCount = 1.0f / max(1u, gSampleCountPerHalf);
    gMeanA[queryIndex] = float4(sumA * inverseSampleCount, 1.0f);
    gMeanSquareA[queryIndex] = float4(sumSquareA * inverseSampleCount, 1.0f);
    gMeanB[queryIndex] = float4(sumB * inverseSampleCount, 1.0f);
    gMeanSquareB[queryIndex] = float4(sumSquareB * inverseSampleCount, 1.0f);
}

[numthreads(64, 1, 1)]
void generateDirectTopBatch(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint queryIndex = dispatchThreadID.x;
    if (queryIndex >= gQueryCount) return;
    const uint tileIndex = queryIndex / gLightCount;
    const uint lightBufferIndex = queryIndex - tileIndex * gLightCount;
    const LayerStack stack = gStacks[tileIndex];
    const float3 viewDirection = normalize(gViewDirections[tileIndex].xyz);
    const float3 lightDirection = normalize(gLightDirections[lightBufferIndex].xyz);
    const LayerInterface top = stack.layers[0];
    const float etaBelow = top.type == (uint)LayerType::RoughDielectric ? top.etaR : 1.0f;
    const float3 value = nclsEvalInterfaceTransport(
        top, viewDirection, lightDirection, 1.0f, etaBelow, true) * lightDirection.z;
    gMeanA[queryIndex] = float4(value, 1.0f);
}
