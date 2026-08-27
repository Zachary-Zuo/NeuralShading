#include "../../../apps/viewer/shaders/PathSurfaceMath.slang"

RWStructuredBuffer<float4> gOutput;
RWStructuredBuffer<float4> gSurfaceOutput;

[numthreads(7, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    const float commonScale = index == 1u ? 10000.0f : 1.0f;
    const float3 cameraV = float3(0.0f, 0.3443276f, 0.0f) * commonScale;
    const float3 cameraW = float3(0.0f, 0.0f, 1.0f) * commonScale;
    const uint frameHeight = index == 2u ? 480u : 240u;
    const float hitDistance = index == 3u
        ? 8.747364f
        : (index == 5u ? 2.186841f : 4.373682f);
    const float3 rayDirection = index == 4u
        ? normalize(float3(1.0f, -0.1f, 0.2f))
        : normalize(float3(0.2f, -0.1f, 1.0f));
    const float spread = nclsViewerPrimaryRayConeSpreadAngle(
        cameraV, cameraW, frameHeight);
    const float triangleLod = index == 6u
        ? asfloat(0x7f800000u)
        : -2.7890625f;
    const NclsViewerPathUvFootprint footprint = nclsViewerPathUvFootprint(
        triangleLod,
        spread * hitDistance,
        rayDirection,
        float3(0.0f, 0.0f, 1.0f));
    gOutput[index] = float4(
        spread,
        footprint.textureIndependentLod,
        footprint.uvDx.x,
        footprint.uvDy.y);

    if (index == 0u)
    {
        const NclsViewerPathSurface base = nclsViewerPreparePathSurfaceFields(
            float3(1.0f, 2.0f, 3.0f),
            float3(0.0f, 0.0f, -1.0f),
            float3(0.0f, 0.0f, -1.0f),
            float3(0.0f, 0.0f, 1.0f),
            float2(0.2f, 0.8f),
            -2.7890625f,
            7u,
            float3(0.0f, 0.0f, 1.0f),
            float3(0.0f, 0.0f, -1.0f),
            spread * 4.373682f,
            0u);
        const NclsViewerPathSurface flipped = nclsViewerPreparePathSurfaceFields(
            float3(1.0f, 2.0f, 3.0f),
            float3(0.0f, 0.0f, -1.0f),
            float3(0.0f, 0.0f, -1.0f),
            float3(0.0f, 0.0f, 1.0f),
            float2(0.2f, 0.8f),
            -2.7890625f,
            7u,
            float3(0.0f, 0.0f, 1.0f),
            float3(0.0f, 0.0f, -1.0f),
            spread * 4.373682f,
            1u);
        gSurfaceOutput[0] = float4(
            base.uv, float(base.frontFacing), dot(base.faceNormal, float3(0.0f, 0.0f, 1.0f)));
        gSurfaceOutput[1] = float4(
            flipped.uv, dot(flipped.normal, flipped.tangent), flipped.tangent.x);
    }
}
