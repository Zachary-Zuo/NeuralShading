import Scene.Raster;
#include "PathSurfaceMath.slang"
import Rendering.Materials.TexLODHelpers;

#ifndef USE_ALPHA_TEST
#define USE_ALPHA_TEST 0
#endif

cbuffer SceneVisibilityCB
{
    uint gFlipTexCoordV;
};

VSOut vsMain(VSIn input)
{
    return defaultVS(input);
}

struct PSOut
{
    float4 positionDepth : SV_TARGET0;
    float4 normal : SV_TARGET1;
    float4 tangent : SV_TARGET2;
    float4 viewDirection : SV_TARGET3;
    float4 texCoord : SV_TARGET4;
    float4 texCoordGrad : SV_TARGET5;
    uint instanceId : SV_TARGET6;
    uint materialId : SV_TARGET7;
};

[earlydepthstencil]
PSOut psMain(VSOut input, uint triangleIndex : SV_PrimitiveID)
{
    const float3 faceNormal = gScene.getFaceNormalW(input.instanceID, triangleIndex);
    const VertexData vertex = prepareVertexData(input, faceNormal);
    let lod = ImplicitLodTextureSampler();

#if USE_ALPHA_TEST
    if (gScene.materials.alphaTest(vertex, input.materialID, lod)) discard;
#endif

    const float3 viewDirection = normalize(gScene.camera.getPosition() - vertex.posW);
    const NclsViewerPathSurface surface = nclsViewerPreparePathSurfaceFields(
        vertex.posW, faceNormal, vertex.normalW, vertex.tangentW.xyz, vertex.texC,
        0.0f, input.materialID, viewDirection, -viewDirection, 0.0f, gFlipTexCoordV);
    const float2 texCoord = surface.uv;
    PSOut output;
    output.positionDepth = float4(vertex.posW, length(gScene.camera.getPosition() - vertex.posW));
    output.normal = float4(surface.normal, surface.faceNormal.x);
    output.tangent = float4(surface.tangent, surface.faceNormal.y);
    output.viewDirection = float4(viewDirection, surface.faceNormal.z);
    output.texCoord = float4(texCoord, float(surface.frontFacing), 0.0f);
    output.texCoordGrad = float4(ddx(texCoord), ddy(texCoord));
    output.instanceId = input.instanceID.index + 1u;
    output.materialId = input.materialID + 1u;
    return output;
}
