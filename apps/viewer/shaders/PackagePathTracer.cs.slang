#include "Scene/SceneDefines.slangh"
import Scene.RaytracingInline;
import Utils.Geometry.GeometryHelpers;

#include "PackageBackend.slang"
#include "PathSurface.slang"
#include "PathSampleGenerator.slang"
#include "../../../shaders/ncls/reference/sampling.slang"

Texture2D<float4> gPreviousPackage;
Texture2D<float4> gEnvironment;
StructuredBuffer<float> gEnvironmentMarginalCdf;
StructuredBuffer<float> gEnvironmentConditionalCdf;
SamplerState gLinearSampler;
RWTexture2D<float4> gNextPackage;
RWStructuredBuffer<uint> gPackageNoiseStats;

cbuffer PackagePathTracerCB
{
    uint2 gFrameDim;
    uint2 gDispatchOffset;
    uint gFrameIndex;
    uint gPackageSpp;
    uint gSamplesThisFrame;
    uint gMaxSceneBounces;
    uint gResetAccumulation;
    uint gFlipTexCoordV;
    uint gCompiledMaterialIndex;
    uint2 gEnvironmentSamplingDimensions;
    uint gUseEnvironment;
    float gEnvironmentRotation;
    float gEnvironmentIntensity;
    uint gUseSun;
    float3 gSunDirection;
    float gSunIntensity;
    float3 gSunColor;
    uint gUsePoint;
    float3 gPointPosition;
    float gPointIntensity;
    float3 gPointColor;
    uint gUseRectangle;
    float3 gRectangleCenter;
    float _padding0;
    float3 gRectangleAxisU;
    float _padding1;
    float3 gRectangleAxisV;
    float gRectangleIntensity;
    float3 gRectangleColor;
    float _padding2;
};

#include "PathEnvironment.slang"

struct NclsPackageSurface
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

NclsPackageSurface nclsPreparePackageSurface(
    const VertexData vertex,
    uint materialId,
    float3 viewWorld,
    float3 rayDirection,
    float rayConeWidth)
{
    NclsPackageSurface surface;
    const NclsViewerPathSurface pathSurface = nclsViewerPreparePathSurface(
        vertex, materialId, viewWorld, rayDirection, rayConeWidth, gFlipTexCoordV);
    surface.position = pathSurface.position;
    surface.faceNormal = pathSurface.faceNormal;
    surface.normal = pathSurface.normal;
    surface.tangent = pathSurface.tangent;
    surface.uv = pathSurface.uv;
    surface.uvDx = pathSurface.uvDx;
    surface.uvDy = pathSurface.uvDy;
    surface.materialId = pathSurface.materialId;
    surface.frontFacing = pathSurface.frontFacing;
    return surface;
}

NclsScatteringContext nclsPackageContext(
    const NclsPackageSurface surface,
    float3 woWorld,
    float filterRandom)
{
    NclsScatteringContext context = nclsViewerMethodContext(
        surface.position,
        surface.normal,
        surface.tangent,
        woWorld,
        surface.uv,
        surface.uvDx,
        surface.uvDy,
        filterRandom,
        surface.materialId);
    context.surface.geometricNormal = surface.faceNormal;
    context.surface.frontFacing = surface.frontFacing;
    context.componentMask = uint(NclsScatteringEvent::Reflection)
        | uint(NclsScatteringEvent::Transmission);
    return context;
}

bool nclsPackageVisible(SceneRayQuery<1> query, float3 origin, float3 direction, float maximumDistance)
{
    return query.traceVisibilityRay(Ray(origin, direction, 0.0f, maximumDistance));
}

float3 nclsPackageDirectOrigin(const NclsPackageSurface surface, float3 direction)
{
    return computeRayOrigin(
        surface.position,
        dot(direction, surface.faceNormal) >= 0.0f ? surface.faceNormal : -surface.faceNormal);
}

float3 nclsPackageEvaluateResponse(
    NclsPackageState state,
    const NclsScatteringContext context,
    float3 wiWorld,
    uint seed)
{
    NclsViewerSampleGenerator sampleGenerator = {seed};
    return nclsScatteringEvalToResponseCos(
        state.evaluate(wiWorld, sampleGenerator), context.surface.shadingFrame, wiWorld);
}

float3 nclsPackageDirectLighting(
    SceneRayQuery<1> query,
    const NclsPackageSurface surface,
    const NclsScatteringContext context,
    NclsPackageState state,
    uint globalSample,
    uint2 pixel,
    uint depth,
    bool includeEnvironmentBsdfSamples,
    inout NclsRng rng)
{
    float3 radiance = float3(0.0f);
    uint queryIndex = 0u;
    if (gUseSun != 0u)
    {
        const float3 lightWorld = normalize(gSunDirection);
        if (nclsPackageVisible(query, nclsPackageDirectOrigin(surface, lightWorld), lightWorld, 3.402823466e+38f))
            radiance += nclsPackageEvaluateResponse(
                state, context, lightWorld,
                nclsViewerHash(globalSample ^ depth ^ queryIndex++))
                * gSunColor * gSunIntensity;
    }
    if (gUsePoint != 0u)
    {
        const float3 toLight = gPointPosition - surface.position;
        const float distanceSquared = max(dot(toLight, toLight), 1e-8f);
        const float distance = sqrt(distanceSquared);
        const float3 lightWorld = toLight / distance;
        if (nclsPackageVisible(
            query, nclsPackageDirectOrigin(surface, lightWorld), lightWorld,
            max(distance - 1e-4f, 0.0f)))
            radiance += nclsPackageEvaluateResponse(
                state, context, lightWorld,
                nclsViewerHash(globalSample ^ depth ^ queryIndex++))
                * gPointColor * (gPointIntensity / distanceSquared);
    }
    if (gUseRectangle != 0u)
    {
        const float2 u = rng.nextFloat2();
        const float3 lightPoint = gRectangleCenter
            + (2.0f * u.x - 1.0f) * gRectangleAxisU
            + (2.0f * u.y - 1.0f) * gRectangleAxisV;
        const float3 toLight = lightPoint - surface.position;
        const float distanceSquared = max(dot(toLight, toLight), 1e-8f);
        const float distance = sqrt(distanceSquared);
        const float3 lightWorld = toLight / distance;
        const float3 lightNormal = normalize(cross(gRectangleAxisU, gRectangleAxisV));
        const float lightCosine = max(dot(lightNormal, -lightWorld), 0.0f);
        const float area = 4.0f * length(cross(gRectangleAxisU, gRectangleAxisV));
        if (lightCosine > 0.0f && area > 0.0f
            && nclsPackageVisible(
                query, nclsPackageDirectOrigin(surface, lightWorld), lightWorld,
                max(distance - 1e-4f, 0.0f)))
        {
            const float inversePdf = area * lightCosine / distanceSquared;
            radiance += nclsPackageEvaluateResponse(
                state, context, lightWorld,
                nclsViewerHash(globalSample ^ depth ^ queryIndex++))
                * gRectangleColor * gRectangleIntensity * inversePdf;
        }
    }
    if (gUseEnvironment != 0u)
    {
        [loop]
        for (uint environmentSample = 0u;
             environmentSample < NCLS_VIEWER_ENVIRONMENT_NEE_SAMPLE_COUNT;
             ++environmentSample)
        {
            float3 lightWorld;
            float lightPdf;
            nclsViewerSampleEnvironment(rng, lightWorld, lightPdf);
            if (nclsPackageVisible(
                query, nclsPackageDirectOrigin(surface, lightWorld), lightWorld,
                3.402823466e+38f))
            {
                const NclsScatteringPdf bsdfPdf = state.pdf(lightWorld);
                const float lightTechniquePdf = nclsViewerEnvironmentLightTechniquePdf(
                    lightPdf);
                const float misWeight = nclsViewerEnvironmentLightMisWeight(
                    lightPdf, bsdfPdf.forward);
                radiance += nclsPackageEvaluateResponse(
                    state, context, lightWorld,
                    nclsViewerHash(globalSample ^ depth ^ queryIndex++))
                    * nclsViewerEnvironmentRadiance(lightWorld)
                    * (misWeight / max(lightTechniquePdf, 1e-8f));
            }
        }

        if (includeEnvironmentBsdfSamples)
        {
            [loop]
            for (uint bsdfSample = 0u;
                 bsdfSample < NCLS_VIEWER_ENVIRONMENT_BSDF_SAMPLE_COUNT;
                 ++bsdfSample)
            {
                NclsViewerPathSampleGenerator sampleGenerator =
                    nclsViewerCreatePathSampleGenerator(
                        pixel,
                        globalSample * NCLS_VIEWER_ENVIRONMENT_BSDF_SAMPLE_COUNT
                            + bsdfSample,
                        depth,
                        0x243f6a88u);
                NclsScatteringSample scatter = {};
                if (!state.sample(scatter, sampleGenerator) || scatter.valid == 0u) continue;
                const float3 directionWorld = normalize(scatter.wiWorld);
                if (!nclsPackageVisible(
                    query, nclsPackageDirectOrigin(surface, directionWorld), directionWorld,
                    3.402823466e+38f)) continue;
                const bool delta = (scatter.eventFlags & (uint)NclsScatteringEvent::Delta) != 0u;
                const float lightPdf = nclsViewerEnvironmentPdf(directionWorld);
                const float misWeight = nclsViewerEnvironmentBsdfMisWeight(
                    scatter.pdf.forward, lightPdf, delta);
                radiance += scatter.weight * nclsViewerEnvironmentRadiance(directionWorld)
                    * (misWeight / float(NCLS_VIEWER_ENVIRONMENT_BSDF_SAMPLE_COUNT));
            }
        }
    }
    return radiance;
}

float2 nclsPackageR2Sample(uint index, uint scramble)
{
    const float2 shift = float2(
        nclsViewerHashFloat(scramble),
        nclsViewerHashFloat(scramble ^ 0x68bc21ebu));
    return frac(shift + (float(index) + 0.5f)
        * float2(0.7548776662466927f, 0.5698402909980532f));
}

Ray nclsPackagePrimaryRay(uint2 pixel, uint globalSample)
{
    const uint linearPixel = pixel.y * gFrameDim.x + pixel.x;
    const float2 subpixel = nclsPackageR2Sample(globalSample, nclsViewerHash(linearPixel));
    const float2 screen = (float2(pixel) + subpixel) / float2(gFrameDim);
    const float2 ndc = float2(2.0f, -2.0f) * screen + float2(-1.0f, 1.0f);
    const float3 direction = normalize(
        ndc.x * gScene.camera.data.cameraU
        + ndc.y * gScene.camera.data.cameraV
        + gScene.camera.data.cameraW);
    const float inverseCosine = 1.0f / dot(normalize(gScene.camera.data.cameraW), direction);
    return Ray(
        gScene.camera.getPosition(), direction,
        gScene.camera.data.nearZ * inverseCosine,
        gScene.camera.data.farZ * inverseCosine);
}

float3 nclsTracePackageSuffix(
    Ray ray,
    float3 throughput,
    float rayConeWidth,
    float rayConeSpreadAngle,
    uint2 pixel,
    uint pathSampleIndex,
    uint branchSeed,
    float initialBsdfPdf,
    bool initialDelta)
{
    NclsRng rng = NclsRng(uint2(branchSeed, pathSampleIndex));
    SceneRayQuery<1> query;
    float3 radiance = float3(0.0f);
    const NclsPackageBackend backend = nclsCreatePackageBackend();
    const NclsPackageCompiledMaterial material = nclsLoadPackageMaterial(gCompiledMaterialIndex);

    [loop]
    for (uint depth = 1u; depth <= max(gMaxSceneBounces, 1u); ++depth)
    {
        float hitDistance;
        const HitInfo hit = query.traceRay(ray, hitDistance);
        if (!hit.isValid())
        {
            if (depth == 1u && gUseEnvironment != 0u)
            {
                const float lightPdf = nclsViewerEnvironmentPdf(ray.dir);
                const float misWeight = nclsViewerEnvironmentBsdfMisWeight(
                    initialBsdfPdf, lightPdf, initialDelta);
                radiance += throughput * nclsViewerEnvironmentRadiance(ray.dir)
                    * misWeight;
            }
            break;
        }
        if (depth > gMaxSceneBounces) break;

        VertexData vertex;
        uint materialId;
        if (!nclsViewerLoadPathVertexData(hit, ray, vertex, materialId)) break;
        rayConeWidth += rayConeSpreadAngle * hitDistance;
        const float3 viewWorld = -ray.dir;
        const NclsPackageSurface surface = nclsPreparePackageSurface(
            vertex, materialId, viewWorld, ray.dir, rayConeWidth);
        const NclsScatteringContext context = nclsPackageContext(
            surface, viewWorld,
            nclsViewerHashFloat(branchSeed ^ nclsViewerHash(depth + 0x51ed270bu)));
        const NclsPackageState state = backend.prepare(context, material);
        radiance += throughput * nclsPackageDirectLighting(
            query, surface, context, state, pathSampleIndex, pixel, depth, true, rng);
        if (depth >= gMaxSceneBounces) break;

        NclsViewerPathSampleGenerator sampleGenerator =
            nclsViewerCreatePathSampleGenerator(
                pixel, pathSampleIndex, depth, 0x13198a2eu);
        NclsScatteringSample scatter = {};
        if (!state.sample(scatter, sampleGenerator) || scatter.valid == 0u) break;
        throughput *= scatter.weight;
        if (!all(isfinite(throughput)) || !any(throughput > 0.0f)) break;
        if (depth >= 3u)
        {
            const float survival = clamp(
                max(throughput.x, max(throughput.y, throughput.z)), 0.05f, 0.95f);
            if (rng.nextFloat() >= survival) break;
            throughput /= survival;
        }

        const float3 directionWorld = normalize(scatter.wiWorld);
        const float3 nextOrigin = nclsPackageDirectOrigin(surface, directionWorld);
        ray = Ray(nextOrigin, directionWorld);
    }
    return max(radiance, 0.0f);
}

float3 nclsTracePackagePath(uint2 pixel, uint globalSample)
{
    const uint linearPixel = pixel.y * gFrameDim.x + pixel.x;
    const uint seed = nclsViewerHash(linearPixel ^ nclsViewerHash(globalSample + 0x9e3779b9u));
    NclsRng rng = NclsRng(uint2(seed, globalSample));
    SceneRayQuery<1> query;
    const Ray primaryRay = nclsPackagePrimaryRay(pixel, globalSample);
    float hitDistance;
    const HitInfo hit = query.traceRay(primaryRay, hitDistance);
    if (!hit.isValid())
        return gUseEnvironment != 0u
            ? nclsViewerEnvironmentRadiance(primaryRay.dir)
            : float3(0.025f, 0.03f, 0.04f);

    VertexData vertex;
    uint materialId;
    if (!nclsViewerLoadPathVertexData(hit, primaryRay, vertex, materialId))
        return float3(0.0f);
    const float rayConeSpreadAngle = nclsViewerPrimaryRayConeSpreadAngle(
        gScene.camera.data.cameraV, gScene.camera.data.cameraW, gFrameDim.y);
    const float rayConeWidth = rayConeSpreadAngle * hitDistance;
    const float3 viewWorld = -primaryRay.dir;
    const NclsPackageSurface surface = nclsPreparePackageSurface(
        vertex, materialId, viewWorld, primaryRay.dir, rayConeWidth);
    const NclsScatteringContext context = nclsPackageContext(
        surface, viewWorld,
        nclsViewerHashFloat(seed ^ nclsViewerHash(0x51ed270bu)));
    const NclsPackageBackend backend = nclsCreatePackageBackend();
    const NclsPackageCompiledMaterial material = nclsLoadPackageMaterial(gCompiledMaterialIndex);
    const NclsPackageState state = backend.prepare(context, material);
    float3 radiance = nclsPackageDirectLighting(
        query, surface, context, state, globalSample, pixel, 0u, false, rng);

    [loop]
    for (uint bsdfSample = 0u;
         bsdfSample < NCLS_VIEWER_PRIMARY_PATH_SAMPLE_COUNT;
         ++bsdfSample)
    {
        const uint pathSampleIndex = globalSample
            * NCLS_VIEWER_PRIMARY_PATH_SAMPLE_COUNT + bsdfSample;
        NclsViewerPathSampleGenerator sampleGenerator =
            nclsViewerCreatePathSampleGenerator(
                pixel, pathSampleIndex, 0u, 0x243f6a88u);
        NclsScatteringSample scatter = {};
        if (!state.sample(scatter, sampleGenerator) || scatter.valid == 0u) continue;
        const float3 throughput = scatter.weight
            / float(NCLS_VIEWER_PRIMARY_PATH_SAMPLE_COUNT);
        if (!all(isfinite(throughput)) || !any(throughput > 0.0f)) continue;
        const float3 directionWorld = normalize(scatter.wiWorld);
        const Ray ray = Ray(
            nclsPackageDirectOrigin(surface, directionWorld), directionWorld);
        const bool delta = (scatter.eventFlags & (uint)NclsScatteringEvent::Delta) != 0u;
        const uint branchSeed = nclsViewerHash(
            seed ^ nclsViewerHash(pathSampleIndex + 0xa4093822u));
        radiance += nclsTracePackageSuffix(
            ray,
            throughput,
            rayConeWidth,
            rayConeSpreadAngle,
            pixel,
            pathSampleIndex,
            branchSeed,
            scatter.pdf.forward,
            delta);
    }
    return max(radiance, 0.0f);
}

groupshared float gPackageRelativeStandardError[64];

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    const uint2 pixel = dispatchThreadID.xy + gDispatchOffset;
    const bool validPixel = all(pixel < gFrameDim);
    float3 batchSum = float3(0.0f);
    float batchLuminanceSquare = 0.0f;
    if (validPixel)
    {
        [loop]
        for (uint sampleIndex = 0u; sampleIndex < gSamplesThisFrame; ++sampleIndex)
        {
            const uint globalSample = gPackageSpp + sampleIndex;
            const float3 sampleRadiance = nclsTracePackagePath(pixel, globalSample);
            batchSum += sampleRadiance;
            const float luminance = nclsViewerLuminance(sampleRadiance);
            batchLuminanceSquare += luminance * luminance;
        }
    }

    float4 accumulated = float4(0.0f);
    if (validPixel)
    {
        const float inverseBatch = 1.0f / max(1u, gSamplesThisFrame);
        if (gResetAccumulation != 0u || gPackageSpp == 0u)
            accumulated = float4(batchSum * inverseBatch, batchLuminanceSquare * inverseBatch);
        else
        {
            const float4 previous = gPreviousPackage[pixel];
            const float oldCount = float(gPackageSpp);
            const float newCount = oldCount + float(gSamplesThisFrame);
            accumulated = float4(
                (previous.rgb * oldCount + batchSum) / newCount,
                (previous.a * oldCount + batchLuminanceSquare) / newCount);
        }
        gNextPackage[pixel] = accumulated;
    }

    const float sampleCount = float(
        (gResetAccumulation != 0u || gPackageSpp == 0u)
        ? gSamplesThisFrame : gPackageSpp + gSamplesThisFrame);
    const float meanLuminance = nclsViewerLuminance(accumulated.rgb);
    const float variance = max(accumulated.a - meanLuminance * meanLuminance, 0.0f);
    const float standardError = sqrt(variance / max(sampleCount, 1.0f));
    gPackageRelativeStandardError[groupIndex] = validPixel
        ? min(standardError / max(abs(meanLuminance), 0.01f), 16.0f)
        : 0.0f;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32u; stride > 0u; stride >>= 1u)
    {
        if (groupIndex < stride)
            gPackageRelativeStandardError[groupIndex]
                += gPackageRelativeStandardError[groupIndex + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (groupIndex == 0u)
    {
        const uint groupPixels = min(8u, gFrameDim.x - pixel.x)
            * min(8u, gFrameDim.y - pixel.y);
        const float groupMean = gPackageRelativeStandardError[0] / max(groupPixels, 1u);
        InterlockedAdd(gPackageNoiseStats[0], uint(groupMean * 4096.0f));
        InterlockedAdd(gPackageNoiseStats[1], 1u);
    }
}
