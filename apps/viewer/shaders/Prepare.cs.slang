#include "MethodBackend.slang"

StructuredBuffer<uint> gSharedWeights;
StructuredBuffer<NclsMethodCompiledMaterial> gCompiledMaterials;
Texture2D<float4> gPositionDepth;
Texture2D<float4> gNormal;
Texture2D<float4> gTangent;
Texture2D<float4> gViewDirection;
Texture2D<uint> gMaterialId;
RWStructuredBuffer<NclsMethodPackedState> gStates;

cbuffer PrepareCB
{
    uint2 gFrameDim;
    uint gUseScene;
    uint gTargetMaterialId;
    uint gCompiledMaterialIndex;
};

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFrameDim)) return;
    if (gUseScene != 0u && gMaterialId[pixel] != gTargetMaterialId + 1u) return;
    const uint index = pixel.y * gFrameDim.x + pixel.x;
    NclsMethodPackedState state = {};
    if (gPositionDepth[pixel].w >= 0.0f)
    {
        const float3 position = gPositionDepth[pixel].xyz;
        const float3 normal = normalize(gNormal[pixel].xyz);
        const float3 tangent = normalize(gTangent[pixel].xyz);
        const float3 woWorld = normalize(gViewDirection[pixel].xyz);
        const NclsScatteringContext context = nclsViewerMethodContext(
            position, normal, tangent, woWorld, gTargetMaterialId);
        const NclsMethodBackend backend = nclsCreateMethodBackend(gSharedWeights);
        state = nclsPackMethodState(backend.prepare(
            context, gCompiledMaterials[gCompiledMaterialIndex]));
    }
    gStates[index] = state;
}
