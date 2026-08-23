import Scene.Raster;
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
    const float3 normal = normalize(vertex.normalW);
    float3 tangent = normalize(vertex.tangentW.xyz);
    tangent -= normal * dot(normal, tangent);
    if (dot(tangent, tangent) < 1e-10f)
    {
        const float signZ = normal.z < 0.0f ? -1.0f : 1.0f;
        const float a = -1.0f / (signZ + normal.z);
        const float b = normal.x * normal.y * a;
        tangent = float3(1.0f + signZ * normal.x * normal.x * a, signZ * b, -signZ * normal.x);
    }
    else tangent = normalize(tangent);

    float2 texCoord = vertex.texC;
    if (gFlipTexCoordV != 0u) texCoord.y = 1.0f - texCoord.y;

    PSOut output;
    output.positionDepth = float4(vertex.posW, length(gScene.camera.getPosition() - vertex.posW));
    output.normal = float4(normal, 0.0f);
    output.tangent = float4(tangent, 0.0f);
    output.viewDirection = float4(viewDirection, 0.0f);
    output.texCoord = float4(texCoord, 0.0f, 0.0f);
    output.texCoordGrad = float4(ddx(texCoord), ddy(texCoord));
    output.instanceId = input.instanceID.index + 1u;
    output.materialId = input.materialID + 1u;
    return output;
}
