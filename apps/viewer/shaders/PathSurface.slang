#ifndef NCLS_VIEWER_PATH_SURFACE_SLANG
#define NCLS_VIEWER_PATH_SURFACE_SLANG

#include "ViewerCommon.slang"
#include "PathSurfaceMath.slang"

bool nclsViewerLoadPathVertexData(
    const HitInfo hit,
    const Ray ray,
    out VertexData vertex,
    out uint materialId)
{
    vertex = {};
    materialId = 0u;
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        vertex = gScene.getVertexDataRayCones(
            triangleHit.instanceID,
            triangleHit.primitiveIndex,
            triangleHit.getBarycentricWeights());
        materialId = gScene.getMaterialID(triangleHit.instanceID);
        return true;
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedHit = hit.getDisplacedTriangleHit();
        vertex = gScene.getVertexData(displacedHit, -ray.dir);
        vertex.coneTexLODValue = asfloat(0x7f800000u);
        materialId = gScene.getMaterialID(displacedHit.instanceID);
        return true;
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        vertex = gScene.getVertexDataFromCurve(curveHit);
        vertex.coneTexLODValue = asfloat(0x7f800000u);
        materialId = gScene.getMaterialID(curveHit.instanceID);
        return true;
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfHit = hit.getSDFGridHit();
        vertex = gScene.getVertexDataFromSDFGrid(sdfHit, ray.origin, ray.dir);
        vertex.coneTexLODValue = asfloat(0x7f800000u);
        materialId = gScene.getMaterialID(sdfHit.instanceID);
        return true;
    }
#endif
    return false;
}

NclsViewerPathSurface nclsViewerPreparePathSurface(
    const VertexData vertex,
    uint materialId,
    float3 viewWorld,
    float3 rayDirection,
    float rayConeWidth,
    uint flipTexCoordV)
{
    return nclsViewerPreparePathSurfaceFields(
        vertex.posW,
        vertex.faceNormalW,
        vertex.normalW,
        vertex.tangentW.xyz,
        vertex.texC,
        vertex.coneTexLODValue,
        materialId,
        viewWorld,
        rayDirection,
        rayConeWidth,
        flipTexCoordV);
}

#endif
