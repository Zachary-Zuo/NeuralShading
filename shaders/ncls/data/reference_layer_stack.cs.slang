#include "../reference/random_walk_reference.slang"

StructuredBuffer<NclsLayerStackIR> gMaterialStates;
StructuredBuffer<float4> gViewDirections;
StructuredBuffer<float4> gLightDirections;
StructuredBuffer<uint> gQueryGroupSeeds;
RWStructuredBuffer<uint> gValidation;
RWStructuredBuffer<float4> gMeanA;
RWStructuredBuffer<float4> gMeanSquareA;
RWStructuredBuffer<float4> gMeanB;
RWStructuredBuffer<float4> gMeanSquareB;

uniform uint gQueryCount;
uniform uint gLightCount;
uniform uint gLightIndexOffset;
uniform uint gSampleCountPerReplica;
uniform uint gSampleOffset;
uniform uint gSeed;
uniform uint gMaxDepth;

[numthreads(64, 1, 1)]
void validateReferenceMaterialLayout(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    const NclsLayerStackIR material = gMaterialStates[index];
    const bool valid = material.magic == NCLS_LAYER_STACK_IR_MAGIC
        && material.abiVersion == NCLS_LAYER_STACK_IR_VERSION
        && material.interfaceCount >= 1u
        && material.interfaceCount <= NCLS_MAX_INTERFACES
        && material.mediumCount + 1u == material.interfaceCount;
    gValidation[index] = valid ? 1u : 0u;
}

[numthreads(64, 1, 1)]
void evaluateReferenceQueryGroups(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint queryIndex = dispatchThreadID.x;
    if (queryIndex >= gQueryCount) return;

    const uint queryGroupIndex = queryIndex / gLightCount;
    const uint lightBufferIndex = queryIndex - queryGroupIndex * gLightCount;
    const uint lightIndex = lightBufferIndex + gLightIndexOffset;
    const NclsLayerStackIR material = gMaterialStates[queryGroupIndex];
    const float3 viewDirection = normalize(gViewDirections[queryGroupIndex].xyz);
    const float3 lightDirection = normalize(gLightDirections[lightBufferIndex].xyz);
    const uint queryGroupSeed = gQueryGroupSeeds[queryGroupIndex] ^ gSeed;
    float3 sumA = float3(0.0f);
    float3 sumSquareA = float3(0.0f);
    float3 sumB = float3(0.0f);
    float3 sumSquareB = float3(0.0f);

    for (uint sampleIndex = 0u; sampleIndex < gSampleCountPerReplica; ++sampleIndex)
    {
        const uint globalSampleIndex = sampleIndex + gSampleOffset;
        const uint2 keyA = uint2(lightIndex ^ queryGroupSeed, globalSampleIndex + queryGroupSeed * 1664525u);
        const uint seedB = queryGroupSeed ^ 0xa511e9b3u;
        const uint2 keyB = uint2(lightIndex ^ seedB, globalSampleIndex + seedB * 22695477u);
        const float3 valueA = nclsEvalRandomWalkReference(
            material, viewDirection, lightDirection, keyA, gMaxDepth);
        const float3 valueB = nclsEvalRandomWalkReference(
            material, viewDirection, lightDirection, keyB, gMaxDepth);
        sumA += valueA;
        sumSquareA += valueA * valueA;
        sumB += valueB;
        sumSquareB += valueB * valueB;
    }

    const float inverseSampleCount = 1.0f / max(1u, gSampleCountPerReplica);
    gMeanA[queryIndex] = float4(sumA * inverseSampleCount, 1.0f);
    gMeanSquareA[queryIndex] = float4(sumSquareA * inverseSampleCount, 1.0f);
    gMeanB[queryIndex] = float4(sumB * inverseSampleCount, 1.0f);
    gMeanSquareB[queryIndex] = float4(sumSquareB * inverseSampleCount, 1.0f);
}
