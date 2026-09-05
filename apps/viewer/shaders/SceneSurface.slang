#ifndef NCLS_VIEWER_SCENE_SURFACE_SLANG
#define NCLS_VIEWER_SCENE_SURFACE_SLANG

#include "PathSurfaceMath.slang"
#include "../../../shaders/ncls/contracts/scattering_backend.slang"

bool nclsViewerDecodeRasterSurface(
    float4 positionDepth, float4 normal, float4 tangent, float4 viewDirection,
    float4 texCoord, float4 texCoordGrad, uint encodedMaterialId,
    out NclsViewerPathSurface surface)
{
    surface = {};
    if (positionDepth.w < 0.0f || encodedMaterialId == 0u) return false;
    surface.position = positionDepth.xyz;
    surface.faceNormal = float3(normal.w, tangent.w, viewDirection.w);
    surface.normal = normal.xyz;
    surface.tangent = tangent.xyz;
    surface.uv = texCoord.xy;
    surface.uvDx = texCoordGrad.xy;
    surface.uvDy = texCoordGrad.zw;
    surface.frontFacing = uint(texCoord.z);
    surface.materialId = encodedMaterialId - 1u;
    return true;
}

NclsScatteringContext nclsViewerSurfaceContext(NclsViewerPathSurface surface, float3 wo, float filterRandom)
{
    NclsScatteringContext context = {};
    context.surface.position = surface.position;
    context.surface.geometricNormal = surface.faceNormal;
    context.surface.shadingFrame.normal = surface.normal;
    context.surface.shadingFrame.tangent = surface.tangent;
    context.surface.shadingFrame.bitangent = cross(surface.normal, surface.tangent);
    context.surface.uv = surface.uv;
    context.surface.uvDx = surface.uvDx;
    context.surface.uvDy = surface.uvDy;
    context.surface.materialInstanceId = surface.materialId;
    context.surface.frontFacing = surface.frontFacing;
    context.woWorld = normalize(wo);
    context.transportMode = (uint)NclsTransportMode::Radiance;
    context.componentMask = (uint)NclsScatteringEvent::Reflection | (uint)NclsScatteringEvent::Transmission;
    context.filterRandom = filterRandom;
    return context;
}

#endif
