#ifndef NCLS_VIEWER_PACKAGE_PROGRAM_SLANG
#define NCLS_VIEWER_PACKAGE_PROGRAM_SLANG

#include "ViewerCommon.slang"
#include "../../../shaders/ncls/contracts/response_measure.slang"

#ifndef NCLS_PACKAGE_PROGRAM_HEADER
#error NCLS_PACKAGE_PROGRAM_HEADER must be declared by the ScatteringPackage specialization
#endif
#include NCLS_PACKAGE_PROGRAM_HEADER

struct NclsViewerSampleGenerator : ISampleGenerator
{
    uint state;

    [mutating]
    uint next()
    {
        state = state * 1664525u + 1013904223u;
        return state;
    }
};

NclsScatteringContext nclsViewerMethodContext(
    float3 position,
    float3 normal,
    float3 tangent,
    float3 woWorld,
    float2 uv,
    float2 uvDx,
    float2 uvDy,
    float filterRandom,
    uint materialId)
{
    NclsScatteringContext result = {};
    result.surface.position = position;
    result.surface.geometricNormal = normal;
    result.surface.shadingFrame.normal = normal;
    result.surface.shadingFrame.tangent = tangent;
    result.surface.shadingFrame.bitangent = cross(normal, tangent);
    result.surface.uv = uv;
    result.surface.uvDx = uvDx;
    result.surface.uvDy = uvDy;
    result.surface.materialInstanceId = materialId;
    result.surface.frontFacing = 1u;
    result.woWorld = woWorld;
    result.transportMode = (uint)NclsTransportMode::Radiance;
    result.componentMask = (uint)NclsScatteringEvent::Reflection;
    result.filterRandom = filterRandom;
    return result;
}

#endif
