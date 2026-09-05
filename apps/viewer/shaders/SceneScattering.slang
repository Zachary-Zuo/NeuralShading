#ifndef NCLS_VIEWER_SCENE_SCATTERING_SLANG
#define NCLS_VIEWER_SCENE_SCATTERING_SLANG

#include "SceneReferenceProgram.slang"
#include "SceneSurface.slang"
#include "../../../shaders/ncls/contracts/response_measure.slang"
#ifdef NCLS_PACKAGE_PROGRAM_HEADER
#include "PackageBackend.slang"
#endif

SamplerState gMaterialXSampler;
cbuffer SceneMaterialCB
{
    uint gMaterialCount;
    uint gActiveMaterialId;
    uint gCompiledMaterialIndex;
    uint gMaxLayerWalkDepth;
};

// The renderer sees one prepared scene state. Native resource/math dispatch stays here.
struct NclsViewerSceneState : INclsScatteringState
{
    NclsSceneReferenceState source;
#ifdef NCLS_PACKAGE_PROGRAM_HEADER
    NclsPackageState package;
    NclsScatteringContext packageContext;
    bool usePackage;
#endif
    NclsScatteringEval evaluate<S : ISampleGenerator>(float3 wi, inout S sg)
    {
#ifdef NCLS_PACKAGE_PROGRAM_HEADER
        if (usePackage) return package.evaluate(wi, sg);
#endif
        return source.evaluate(wi, sg);
    }
    bool sample<S : ISampleGenerator>(out NclsScatteringSample result, inout S sg)
    {
#ifdef NCLS_PACKAGE_PROGRAM_HEADER
        if (usePackage) return package.sample(result, sg);
#endif
        return source.sample(result, sg);
    }
    NclsScatteringPdf pdf(float3 wi)
    {
#ifdef NCLS_PACKAGE_PROGRAM_HEADER
        if (usePackage) return package.pdf(wi);
#endif
        return source.pdf(wi);
    }
    NclsSurfaceInteraction surfaceInteraction()
    {
#ifdef NCLS_PACKAGE_PROGRAM_HEADER
        if (usePackage) return packageContext.surface;
#endif
        return source.surfaceInteraction();
    }
    float3 emission()
    {
#ifdef NCLS_PACKAGE_PROGRAM_HEADER
        if (usePackage) return float3(0.0f);
#endif
        return source.emission();
    }
    float3 illuminate(float3 response, float3 radiance)
    {
#ifdef NCLS_PACKAGE_PROGRAM_HEADER
        if (usePackage) return response * radiance;
#endif
        return source.illuminate(response, radiance);
    }
};

NclsViewerSceneState nclsViewerPrepareScattering(NclsScatteringContext context)
{
    NclsViewerSceneState result = {};
#ifdef NCLS_PACKAGE_PROGRAM_HEADER
    result.usePackage = context.surface.materialInstanceId == gActiveMaterialId;
    if (result.usePackage)
    {
        result.packageContext = context;
        result.package = nclsCreatePackageBackend().prepare(context, nclsLoadPackageMaterial(gCompiledMaterialIndex));
        return result;
    }
#endif
    const NclsSceneReferenceBackend backend = {gMaxLayerWalkDepth, gMaterialXSampler};
    result.source = backend.prepare(context, nclsLoadSceneReferenceMaterial(context.surface.materialInstanceId, gMaterialCount));
    return result;
}
#endif
