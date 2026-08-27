#include "PackageBackend.slang"

StructuredBuffer<uint> gSharedWeights;
StructuredBuffer<NclsMethodCompiledMaterial> gCompiledMaterials;
Texture2D<float4> gPositionDepth;
Texture2D<float4> gNormal;
Texture2D<float4> gTangent;
Texture2D<float4> gViewDirection;
Texture2D<uint> gMaterialId;
Texture2D<float4> gTexCoord;
Texture2D<float4> gTexCoordGrad;
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
        const float2 uv = gTexCoord[pixel].xy;
        const float4 uvGrad = gTexCoordGrad[pixel];
        const NclsScatteringContext context = nclsViewerMethodContext(
            position, normal, tangent, woWorld,
            uv, uvGrad.xy, uvGrad.zw, nclsViewerHashFloat(index),
            gTargetMaterialId);
        const NclsPackageBackend backend = nclsCreatePackageBackend();
        state = nclsPackMethodState(backend.prepare(
            context, gCompiledMaterials[gCompiledMaterialIndex]));
    }
    gStates[index] = state;
}
