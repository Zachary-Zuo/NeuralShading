#include "ViewerCommon.slang"

RWTexture2D<float4> gPositionDepth;
RWTexture2D<float4> gNormal;
RWTexture2D<float4> gTangent;
RWTexture2D<float4> gViewDirection;
RWTexture2D<float4> gMaterialXTexCoord;
RWTexture2D<float4> gMaterialXTexCoordGrad;
RWTexture2D<uint> gInstanceId;
RWTexture2D<uint> gMaterialId;

cbuffer VisibilityCB
{
    uint2 gFrameDim;
    uint gObjectMode;
    uint gClearOnly;
    float gVerticalFovRadians;
    float3 gCameraPosition;
    float _padding0;
    float3 gCameraTarget;
    float _padding1;
};

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFrameDim)) return;
    const float2 uv = (float2(pixel) + 0.5f) / float2(gFrameDim);
    const float2 screen = float2(2.0f * uv.x - 1.0f, 1.0f - 2.0f * uv.y);
    const float3 forward = normalize(gCameraTarget - gCameraPosition);
    const float3 right = normalize(cross(forward, float3(0.0f, 1.0f, 0.0f)));
    const float3 up = cross(right, forward);
    const float tangentFov = tan(0.5f * gVerticalFovRadians);
    const float aspect = float(gFrameDim.x) / float(gFrameDim.y);
    const float3 rayDirection = normalize(
        forward + screen.x * aspect * tangentFov * right + screen.y * tangentFov * up);
    if (gClearOnly != 0u)
    {
        gPositionDepth[pixel] = float4(0.0f, 0.0f, 0.0f, -1.0f);
        gNormal[pixel] = float4(0.0f, 1.0f, 0.0f, 0.0f);
        gTangent[pixel] = float4(1.0f, 0.0f, 0.0f, 0.0f);
        gViewDirection[pixel] = float4(-rayDirection, 0.0f);
        gMaterialXTexCoord[pixel] = float4(0.0f);
        gMaterialXTexCoordGrad[pixel] = float4(0.0f);
        gInstanceId[pixel] = 0u;
        gMaterialId[pixel] = 0u;
        return;
    }
    float distance;
    if (!nclsViewerTrace(gCameraPosition, rayDirection, gObjectMode, distance))
    {
        gPositionDepth[pixel] = float4(0.0f, 0.0f, 0.0f, -1.0f);
        gNormal[pixel] = float4(0.0f, 1.0f, 0.0f, 0.0f);
        gTangent[pixel] = float4(1.0f, 0.0f, 0.0f, 0.0f);
        gViewDirection[pixel] = float4(-rayDirection, 0.0f);
        gMaterialXTexCoord[pixel] = float4(0.0f);
        gMaterialXTexCoordGrad[pixel] = float4(0.0f);
        gInstanceId[pixel] = 0u;
        gMaterialId[pixel] = 0u;
        return;
    }
    const float3 position = gCameraPosition + distance * rayDirection;
    const float3 normal = nclsViewerSdfNormal(position, gObjectMode);
    const float3 tangent = nclsViewerTangent(normal);
    gPositionDepth[pixel] = float4(position, distance);
    gNormal[pixel] = float4(normal, 0.0f);
    gTangent[pixel] = float4(tangent, 0.0f);
    gViewDirection[pixel] = float4(-rayDirection, 0.0f);
    gMaterialXTexCoord[pixel] = float4(0.0f);
    gMaterialXTexCoordGrad[pixel] = float4(0.0f);
    gInstanceId[pixel] = 1u;
    gMaterialId[pixel] = 1u;
}
