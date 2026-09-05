#include "../../../apps/viewer/shaders/SceneSurface.slang"
#include "../../../shaders/ncls/scattering/frame.slang"

RWStructuredBuffer<float4> gOutput;
[numthreads(4, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint id = dispatchThreadID.x;
    const float sign = id < 2u ? 1.0f : -1.0f;
    const float3 wo = normalize(float3(0.2f, -0.1f, 1.0f));
    const NclsViewerPathSurface ray = nclsViewerPreparePathSurfaceFields(
        float3(1, 2, 3), float3(0, 0, sign), sign * normalize(float3(0.3f, 0.4f, 1.0f)),
        float3(0.8f, 0.1f, 0.2f), float2(0.2f, 0.8f), -2.0f, id,
        wo, -wo, 0.01f, id & 1u);
    NclsViewerPathSurface raster;
    const bool valid = nclsViewerDecodeRasterSurface(float4(ray.position, 1.0f),
        float4(ray.normal, ray.faceNormal.x), float4(ray.tangent, ray.faceNormal.y),
        float4(wo, ray.faceNormal.z), float4(ray.uv, float(ray.frontFacing), 0.0f),
        float4(ray.uvDx, ray.uvDy), id + 1u, raster);
    const NclsScatteringContext pt = nclsViewerSurfaceContext(ray, wo, 0.25f);
    const NclsScatteringContext deferred = nclsViewerSurfaceContext(raster, wo, 0.25f);
    const float3 wi = normalize(float3(-0.25f, 0.2f, 0.9f));
    const float3 ptLocal = nclsFrameToLocal(pt.surface.shadingFrame, wi);
    const float3 deferredLocal = nclsFrameToLocal(deferred.surface.shadingFrame, wi);
    gOutput[id * 4u] = float4(ptLocal - deferredLocal, float(valid));
    gOutput[id * 4u + 1u] = float4(deferred.surface.uv, float(deferred.surface.materialInstanceId), float(deferred.surface.frontFacing));
    gOutput[id * 4u + 2u] = float4(deferred.surface.geometricNormal, dot(raster.normal, raster.tangent));
    NclsViewerPathSurface emptySurface;
    const bool background = nclsViewerDecodeRasterSurface(float4(0,0,0,-1), float4(0), float4(0), float4(0), float4(0), float4(0), 0u, emptySurface);
    const bool sentinel = nclsViewerDecodeRasterSurface(float4(0,0,0,1), float4(0), float4(0), float4(0), float4(0), float4(0), 0u, emptySurface);
    gOutput[id * 4u + 3u] = float4(deferred.surface.uvDx - pt.surface.uvDx, float(background), float(sentinel));
}
