#ifndef NCLS_VIEWER_PATH_SURFACE_MATH_SLANG
#define NCLS_VIEWER_PATH_SURFACE_MATH_SLANG

#include "ViewerCommon.slang"

struct NclsViewerPathUvFootprint
{
    float textureIndependentLod;
    float2 uvDx;
    float2 uvDy;
};

struct NclsViewerPathSurface
{
    float3 position;
    float3 faceNormal;
    float3 normal;
    float3 tangent;
    float2 uv;
    float2 uvDx;
    float2 uvDy;
    uint materialId;
    uint frontFacing;
};

/** Return the world-space cone spread of one primary pixel.

    Falcor's cameraU/cameraV/cameraW vectors share the camera focal-distance
    scale. Primary-ray directions are normalized, but a ray-cone angle must be
    formed from their ratio or it changes when that common scale changes.
*/
float nclsViewerPrimaryRayConeSpreadAngle(
    float3 cameraV,
    float3 cameraW,
    uint frameHeight)
{
    const float imagePlaneSlope = length(cameraV) / max(length(cameraW), 1e-8f);
    return 2.0f * imagePlaneSlope / float(max(frameHeight, 1u));
}

/** Convert a propagated world-space ray cone to normalized UV gradients.

    triangleUvWorldLog2Scale is Falcor's texture-dimension-independent
    coneTexLODValue. Texture owners add their own dimensions when consuming
    the normalized gradients, so the same footprint is valid for MaterialX
    textures and neural latent grids.
*/
NclsViewerPathUvFootprint nclsViewerPathUvFootprint(
    float triangleUvWorldLog2Scale,
    float rayConeWidth,
    float3 rayDirection,
    float3 geometricNormal)
{
    NclsViewerPathUvFootprint result;
    const float normalProjection = max(
        abs(dot(rayDirection, normalize(geometricNormal))), 1e-4f);
    const float rawLod = triangleUvWorldLog2Scale
        + log2(max(abs(rayConeWidth) / normalProjection, 1e-8f));
    // A malformed/degenerate geometry differential falls back to one full UV
    // period. This is finite and conservative instead of selecting an
    // arbitrary sharp texel or propagating NaN into a package.
    result.textureIndependentLod = isfinite(rawLod) ? rawLod : 0.0f;
    const float uvFootprint = clamp(
        exp2(result.textureIndependentLod), 1e-8f, 1.0f);
    result.uvDx = float2(uvFootprint, 0.0f);
    result.uvDy = float2(0.0f, uvFootprint);
    return result;
}

NclsViewerPathSurface nclsViewerPreparePathSurfaceFields(
    float3 position,
    float3 faceNormal,
    float3 normal,
    float3 tangent,
    float2 uv,
    float triangleUvWorldLog2Scale,
    uint materialId,
    float3 viewWorld,
    float3 rayDirection,
    float rayConeWidth,
    uint flipTexCoordV)
{
    NclsViewerPathSurface surface;
    surface.position = position;
    surface.faceNormal = normalize(faceNormal);
    surface.normal = normalize(normal);
    surface.tangent = tangent;
    surface.uv = uv;
    surface.materialId = materialId;
    surface.frontFacing = dot(viewWorld, surface.faceNormal) >= 0.0f ? 1u : 0u;
    if (flipTexCoordV != 0u) surface.uv.y = 1.0f - surface.uv.y;
    if (surface.frontFacing == 0u) surface.faceNormal = -surface.faceNormal;
    if (dot(viewWorld, surface.normal) < 0.0f) surface.normal = -surface.normal;
    surface.tangent -= surface.normal * dot(surface.normal, surface.tangent);
    surface.tangent = dot(surface.tangent, surface.tangent) > 1e-10f
        ? normalize(surface.tangent)
        : nclsViewerTangent(surface.normal);

    const NclsViewerPathUvFootprint footprint = nclsViewerPathUvFootprint(
        triangleUvWorldLog2Scale, rayConeWidth, rayDirection, surface.faceNormal);
    surface.uvDx = footprint.uvDx;
    surface.uvDy = footprint.uvDy;
    return surface;
}

#endif
