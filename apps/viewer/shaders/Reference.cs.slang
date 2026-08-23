#include "ViewerCommon.slang"
#include "../../../shaders/ncls/reference/random_walk_reference.slang"

StructuredBuffer<NclsLayerStackIR> gMaterials;
Texture2D<float4> gPositionDepth;
Texture2D<float4> gNormal;
Texture2D<float4> gTangent;
Texture2D<float4> gViewDirection;
Texture2D<float4> gPreviousReference;
Texture2D<float4> gEnvironment;
SamplerState gLinearSampler;
RWTexture2D<float4> gNextReference;

cbuffer ReferenceCB
{
    uint2 gFrameDim;
    uint gFrameIndex;
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

float3 nclsReferenceDirection(
    NclsLayerStackIR material,
    float3 woLocal,
    float3 wiWorld,
    float3 normal,
    float3 tangent,
    uint2 key)
{
    const float3 wiLocal = nclsViewerToLocal(normalize(wiWorld), normal, tangent);
    if (wiLocal.z <= 0.0f) return float3(0.0f);
    return max(nclsEvalRandomWalkReference(material, woLocal, wiLocal, key, gMaxDepth), 0.0f);
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
    const float3 normal = normalize(gNormal[pixel].xyz);
    const float3 tangent = normalize(gTangent[pixel].xyz);
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
                material, woLocal, normalize(gSunDirection), normal, tangent,
                uint2(seed ^ query++, globalSample));
            radiance += response * gSunColor * gSunIntensity;
        }
        if (gUsePoint != 0u)
        {
            const float3 toLight = gPointPosition - position;
            const float distanceSquared = max(dot(toLight, toLight), 1e-4f);
            const float3 response = nclsReferenceDirection(
                material, woLocal, toLight, normal, tangent,
                uint2(seed ^ query++, globalSample));
            radiance += response * gPointColor * (gPointIntensity / distanceSquared);
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
            const float3 lightNormal = -normalize(cross(gRectangleAxisU, gRectangleAxisV));
            const float lightCosine = max(dot(lightNormal, -wiWorld), 0.0f);
            const float area = 4.0f * length(cross(gRectangleAxisU, gRectangleAxisV));
            const float3 response = nclsReferenceDirection(
                material, woLocal, wiWorld, normal, tangent,
                uint2(seed ^ query++, globalSample));
            radiance += response * gRectangleColor * gRectangleIntensity
                * (area * lightCosine / distanceSquared);
        }
        if (gUseEnvironment != 0u)
        {
            const float3 wiLocal = nclsViewerUniformHemisphere(lightRng.nextFloat2());
            const float3 wiWorld = nclsViewerToWorld(wiLocal, normal, tangent);
            const float3 response = max(nclsEvalRandomWalkReference(
                material, woLocal, wiLocal, uint2(seed ^ query++, globalSample), gMaxDepth), 0.0f);
            radiance += response * nclsEnvironment(wiWorld) * (2.0f * NCLS_VIEWER_PI);
        }
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
