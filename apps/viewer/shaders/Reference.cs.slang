#include "ViewerCommon.slang"
#include "../../../shaders/ncls/reference/random_walk_reference.slang"
#include "../../../shaders/ncls/reference/merl_reference.slang"
#include "../../../shaders/ncls/reference/openpbr_reference.slang"
#include "../../../shaders/ncls/reference/materialx_standard_surface_reference.slang"

StructuredBuffer<NclsLayerStackIR> gMaterials;
StructuredBuffer<float3> gMerlBrdf;
StructuredBuffer<float> gOpenPbrInputs;
StructuredBuffer<float> gMaterialXInputs;
Texture2D<float4> gMaterialXBaseColor;
Texture2D<float4> gMaterialXRoughness;
Texture2D<float4> gMaterialXMetalness;
Texture2D<float4> gMaterialXNormalMap;
Texture2D<float4> gPositionDepth;
Texture2D<float4> gNormal;
Texture2D<float4> gTangent;
Texture2D<float4> gViewDirection;
Texture2D<float4> gMaterialXTexCoord;
Texture2D<float4> gMaterialXTexCoordGrad;
Texture2D<uint> gMaterialId;
Texture2D<float4> gPreviousReference;
Texture2D<float4> gEnvironment;
SamplerState gLinearSampler;
SamplerState gMaterialXSampler;
RWTexture2D<float4> gNextReference;

cbuffer ReferenceCB
{
    uint2 gFrameDim;
    uint gFrameIndex;
    uint gReferenceFamily;
    uint gUseRasterGeometry;
    uint gTargetMaterialId;
    uint gInitializeOutput;
    uint gOpenPbrColorSpace;
    uint gReferenceSpp;
    uint gSamplesThisFrame;
    uint gMaxDepth;
    uint gResetAccumulation;
    uint gAccumulate;
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

float3 nclsReferenceLocal(
    NclsLayerStackIR material,
    NclsMaterialXResolvedSurface materialXSurface,
    float3 woLocal,
    float3 wiLocal,
    uint2 key)
{
    if (gReferenceFamily == 1u) return nclsEvalMerlReference(gMerlBrdf, woLocal, wiLocal);
    if (gReferenceFamily == 2u) return nclsEvalOpenPbrReference(gOpenPbrInputs, woLocal, wiLocal);
    if (gReferenceFamily == 3u)
        return nclsEvalMaterialXStandardSurface(gMaterialXInputs, materialXSurface, woLocal, wiLocal);
    return max(nclsEvalRandomWalkReference(material, woLocal, wiLocal, key, gMaxDepth), 0.0f);
}

float3 nclsOpenPbrAcesCgToLinearSrgb(float3 value)
{
    return float3(
        1.70505099f * value.x - 0.62179212f * value.y - 0.08325887f * value.z,
        -0.13025642f * value.x + 1.14080474f * value.y - 0.01054832f * value.z,
        -0.02400336f * value.x - 0.12896898f * value.y + 1.15297233f * value.z);
}

float3 nclsOpenPbrLinearSrgbToAcesCg(float3 value)
{
    return float3(
        0.61309740f * value.x + 0.33952315f * value.y + 0.04737945f * value.z,
        0.07019372f * value.x + 0.91635388f * value.y + 0.01345240f * value.z,
        0.02061560f * value.x + 0.10956978f * value.y + 0.86981464f * value.z);
}

float3 nclsReferenceIlluminate(float3 response, float3 linearSrgbRadiance)
{
    if (gReferenceFamily != 2u) return response * linearSrgbRadiance;
    if (gOpenPbrColorSpace == 0u) return response * linearSrgbRadiance;
    return nclsOpenPbrAcesCgToLinearSrgb(response * nclsOpenPbrLinearSrgbToAcesCg(linearSrgbRadiance));
}

float3 nclsReferenceDirection(
    NclsLayerStackIR material,
    NclsMaterialXResolvedSurface materialXSurface,
    float3 woLocal,
    float3 wiWorld,
    float3 normal,
    float3 tangent,
    uint2 key)
{
    const float3 wiLocal = nclsViewerToLocal(normalize(wiWorld), normal, tangent);
    if (wiLocal.z <= 0.0f && gReferenceFamily != 2u) return float3(0.0f);
    return nclsReferenceLocal(material, materialXSurface, woLocal, wiLocal, key);
}

float3 nclsEnvironment(float3 direction)
{
    return gEnvironment.SampleLevel(gLinearSampler, nclsViewerLatLongUv(direction, gEnvironmentRotation), 0.0f).rgb
        * gEnvironmentIntensity;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFrameDim)) return;
    if (gUseRasterGeometry != 0u)
    {
        const uint encodedMaterialId = gMaterialId[pixel];
        if (encodedMaterialId == 0u)
        {
            if (gInitializeOutput == 0u) return;
        }
        else if (encodedMaterialId != gTargetMaterialId + 1u) return;
    }
    const float4 positionDepth = gPositionDepth[pixel];
    const float3 viewWorld = normalize(gViewDirection[pixel].xyz);
    if (positionDepth.w < 0.0f)
    {
        const float3 background = gUseEnvironment != 0u ? nclsEnvironment(-viewWorld) : float3(0.025f, 0.03f, 0.04f);
        gNextReference[pixel] = float4(background, nclsViewerLuminance(background) * nclsViewerLuminance(background));
        return;
    }

    const NclsLayerStackIR material = gMaterials[0];
    const float3 position = positionDepth.xyz;
    const float3 geometricNormal = normalize(gNormal[pixel].xyz);
    const float3 geometricTangent = normalize(gTangent[pixel].xyz);
    NclsMaterialXResolvedSurface materialXSurface;
    materialXSurface.baseColor = float3(0.0f);
    materialXSurface.metalness = 0.0f;
    materialXSurface.specularRoughness = 0.0f;
    float3 normal = geometricNormal;
    float3 tangent = geometricTangent;
    if (gReferenceFamily == 3u)
    {
        float2 materialUv;
        float2 uvDx;
        float2 uvDy;
        if (gUseRasterGeometry != 0u)
        {
            materialUv = gMaterialXTexCoord[pixel].xy;
            const float4 gradients = gMaterialXTexCoordGrad[pixel];
            uvDx = gradients.xy;
            uvDy = gradients.zw;
        }
        else
        {
            materialUv = nclsMaterialXSphereUv(position);
            const uint2 maximumPixel = gFrameDim - 1u;
            // Analytic-SDF fallback for interactive use without an explicit mesh.
            const uint2 pixelX = uint2(min(pixel.x ^ 1u, maximumPixel.x), pixel.y);
            const uint2 pixelY = uint2(pixel.x, min(pixel.y ^ 1u, maximumPixel.y));
            const float4 positionX = gPositionDepth[pixelX];
            const float4 positionY = gPositionDepth[pixelY];
            const float2 uvX = positionX.w >= 0.0f ? nclsMaterialXSphereUv(positionX.xyz) : materialUv;
            const float2 uvY = positionY.w >= 0.0f ? nclsMaterialXSphereUv(positionY.xyz) : materialUv;
            uvDx = (pixel.x & 1u) == 0u ? uvX - materialUv : materialUv - uvX;
            uvDy = (pixel.y & 1u) == 0u ? uvY - materialUv : materialUv - uvY;
            uvDx.x -= round(uvDx.x);
            uvDy.x -= round(uvDy.x);
        }
        materialXSurface = nclsResolveMaterialXSurface(
            gMaterialXInputs,
            gMaterialXBaseColor,
            gMaterialXRoughness,
            gMaterialXMetalness,
            gMaterialXSampler,
            materialUv,
            uvDx,
            uvDy);
        normal = nclsMaterialXShadingNormal(
            gMaterialXInputs,
            gMaterialXNormalMap,
            gMaterialXSampler,
            materialUv,
            uvDx,
            uvDy,
            geometricNormal,
            geometricTangent);
        normal = nclsMxForwardFacingNormal(normal, viewWorld);
        tangent = geometricTangent - dot(geometricTangent, normal) * normal;
        tangent = dot(tangent, tangent) > 1e-10f ? normalize(tangent) : nclsViewerTangent(normal);
    }
    const float3 woLocal = nclsViewerToLocal(viewWorld, normal, tangent);
    float3 batchSum = float3(0.0f);
    float batchLuminanceSquare = 0.0f;
    const uint linearPixel = pixel.y * gFrameDim.x + pixel.x;
    [loop]
    for (uint sampleIndex = 0u; sampleIndex < gSamplesThisFrame; ++sampleIndex)
    {
        const uint globalSample = gReferenceSpp + sampleIndex;
        const uint seed = nclsViewerHash(linearPixel ^ nclsViewerHash(globalSample + 1664525u * gFrameIndex));
        NclsRng lightRng = NclsRng(uint2(seed, seed ^ 0xa511e9b3u));
        float3 radiance = float3(0.0f);
        uint query = 0u;
        if (gUseSun != 0u)
        {
            const float3 response = nclsReferenceDirection(
                material, materialXSurface, woLocal, normalize(gSunDirection), normal, tangent,
                uint2(seed ^ query++, globalSample));
            radiance += nclsReferenceIlluminate(response, gSunColor * gSunIntensity);
        }
        if (gUsePoint != 0u)
        {
            const float3 toLight = gPointPosition - position;
            const float distanceSquared = max(dot(toLight, toLight), 1e-4f);
            const float3 response = nclsReferenceDirection(
                material, materialXSurface, woLocal, toLight, normal, tangent,
                uint2(seed ^ query++, globalSample));
            radiance += nclsReferenceIlluminate(response, gPointColor * (gPointIntensity / distanceSquared));
        }
        if (gUseRectangle != 0u)
        {
            const float2 u = lightRng.nextFloat2();
            const float3 lightPoint = gRectangleCenter
                + (2.0f * u.x - 1.0f) * gRectangleAxisU
                + (2.0f * u.y - 1.0f) * gRectangleAxisV;
            const float3 toLight = lightPoint - position;
            const float distanceSquared = max(dot(toLight, toLight), 1e-4f);
            const float3 wiWorld = normalize(toLight);
            const float3 lightNormal = normalize(cross(gRectangleAxisU, gRectangleAxisV));
            const float lightCosine = max(dot(lightNormal, -wiWorld), 0.0f);
            const float area = 4.0f * length(cross(gRectangleAxisU, gRectangleAxisV));
            const float3 response = nclsReferenceDirection(
                material, materialXSurface, woLocal, wiWorld, normal, tangent,
                uint2(seed ^ query++, globalSample));
            radiance += nclsReferenceIlluminate(
                response,
                gRectangleColor * gRectangleIntensity * (area * lightCosine / distanceSquared));
        }
        if (gUseEnvironment != 0u)
        {
            const float3 wiLocal = gReferenceFamily == 2u
                ? nclsViewerUniformSphere(lightRng.nextFloat2())
                : nclsViewerUniformHemisphere(lightRng.nextFloat2());
            const float3 wiWorld = nclsViewerToWorld(wiLocal, normal, tangent);
            const float3 response = nclsReferenceLocal(
                material, materialXSurface, woLocal, wiLocal, uint2(seed ^ query++, globalSample));
            radiance += nclsReferenceIlluminate(response, nclsEnvironment(wiWorld))
                * (gReferenceFamily == 2u ? 4.0f : 2.0f) * NCLS_VIEWER_PI;
        }
        if (gReferenceFamily == 2u)
        {
            const float3 emission = nclsOpenPbrEmission(gOpenPbrInputs, woLocal);
            radiance += gOpenPbrColorSpace == 0u ? emission : nclsOpenPbrAcesCgToLinearSrgb(emission);
        }
        if (gReferenceFamily == 3u) radiance += nclsMaterialXEmission(gMaterialXInputs);
        batchSum += radiance;
        const float luminance = nclsViewerLuminance(radiance);
        batchLuminanceSquare += luminance * luminance;
    }
    const float inverseBatch = 1.0f / max(1u, gSamplesThisFrame);
    const float3 batchMean = batchSum * inverseBatch;
    const float batchMoment = batchLuminanceSquare * inverseBatch;
    if (gAccumulate == 0u || gResetAccumulation != 0u || gReferenceSpp == 0u)
    {
        gNextReference[pixel] = float4(batchMean, batchMoment);
        return;
    }
    const float4 previous = gPreviousReference[pixel];
    const float oldCount = float(gReferenceSpp);
    const float newCount = oldCount + float(gSamplesThisFrame);
    gNextReference[pixel] = float4(
        (previous.rgb * oldCount + batchSum) / newCount,
        (previous.a * oldCount + batchLuminanceSquare) / newCount);
}
