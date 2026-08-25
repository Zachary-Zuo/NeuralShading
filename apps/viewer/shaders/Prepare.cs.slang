#include "ViewerCommon.slang"
#include "../../../shaders/ncls/backends/film_m1/film_m1.slang"

StructuredBuffer<float> gWeights;
Texture2D<float4> gPositionDepth;
Texture2D<float4> gNormal;
Texture2D<float4> gTangent;
Texture2D<float4> gViewDirection;
Texture2D<uint> gMaterialId;
RWStructuredBuffer<NclsFilmM1State> gStates;

cbuffer PrepareCB
{
    uint2 gFrameDim;
    uint gUseScene;
    uint gTargetMaterialId;
};

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFrameDim)) return;
    if (gUseScene != 0u && gMaterialId[pixel] != gTargetMaterialId + 1u) return;
    const uint index = pixel.y * gFrameDim.x + pixel.x;
    NclsFilmM1State state = {};
    if (gPositionDepth[pixel].w >= 0.0f)
    {
        const float3 normal = normalize(gNormal[pixel].xyz);
        const float3 tangent = normalize(gTangent[pixel].xyz);
        const float3 woWorld = normalize(gViewDirection[pixel].xyz);
        const float3 woLocal = nclsViewerToLocal(woWorld, normal, tangent);
        state = nclsFilmM1Prepare(woLocal, gWeights);
    }
    gStates[index] = state;
}
