#include "PackageBackend.slang"

StructuredBuffer<NclsMethodPackedState> gStates;
StructuredBuffer<uint> gSharedWeights;
StructuredBuffer<NclsMethodCompiledMaterial> gCompiledMaterials;
Texture2D<float4> gPositionDepth;
Texture2D<float4> gNormal;
Texture2D<float4> gTangent;
Texture2D<float4> gViewDirection;
Texture2D<float4> gEnvironment;
SamplerState gLinearSampler;
RWTexture2D<float4> gApproximation;

cbuffer ApproximationCB
{
    uint2 gFrameDim;
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
    uint gEnvironmentQueryBudget;
    uint gRectangleQueryBudget;
    uint gCompiledMaterialIndex;
};

float3 nclsApproximationDirection(
    NclsMethodPackedState packedState,
    NclsScatteringContext context,
    float3 wiWorld,
    uint seed)
{
    const NclsMethodState state = nclsUnpackMethodState(
        context,
        gCompiledMaterials[gCompiledMaterialIndex],
        packedState,
        gSharedWeights);
    NclsViewerSampleGenerator sampleGenerator = {seed};
    const float3 direction = normalize(wiWorld);
    return nclsScatteringEvalToResponseCos(
        state.evaluate(direction, sampleGenerator),
        context.surface.shadingFrame,
        direction);
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
        gApproximation[pixel] = float4(background, 1.0f);
        return;
    }
    const uint index = pixel.y * gFrameDim.x + pixel.x;
    const NclsMethodPackedState state = gStates[index];
    const float3 position = positionDepth.xyz;
    const float3 normal = normalize(gNormal[pixel].xyz);
    const float3 tangent = normalize(gTangent[pixel].xyz);
    const NclsScatteringContext context = nclsViewerMethodContext(
        position, normal, tangent, viewWorld, 0u);
    float3 radiance = float3(0.0f);
    if (gUseSun != 0u)
        radiance += nclsApproximationDirection(state, context, normalize(gSunDirection), index + 1u)
            * gSunColor * gSunIntensity;
    if (gUsePoint != 0u)
    {
        const float3 toLight = gPointPosition - position;
        const float distanceSquared = max(dot(toLight, toLight), 1e-4f);
        radiance += nclsApproximationDirection(state, context, toLight, index + 2u)
            * gPointColor * (gPointIntensity / distanceSquared);
    }
    if (gUseRectangle != 0u)
    {
        const float3 lightNormal = normalize(cross(gRectangleAxisU, gRectangleAxisV));
        const float area = 4.0f * length(cross(gRectangleAxisU, gRectangleAxisV));
        [loop]
        for (uint sampleIndex = 0u; sampleIndex < gRectangleQueryBudget; ++sampleIndex)
        {
            const float2 u = nclsViewerHammersley(sampleIndex, gRectangleQueryBudget);
            const float3 lightPoint = gRectangleCenter
                + (2.0f * u.x - 1.0f) * gRectangleAxisU
                + (2.0f * u.y - 1.0f) * gRectangleAxisV;
            const float3 toLight = lightPoint - position;
            const float distanceSquared = max(dot(toLight, toLight), 1e-4f);
            const float3 wiWorld = normalize(toLight);
            const float lightCosine = max(dot(lightNormal, -wiWorld), 0.0f);
            radiance += nclsApproximationDirection(
                state, context, wiWorld, index + 17u * sampleIndex + 3u)
                * gRectangleColor * gRectangleIntensity
                * (area * lightCosine / (distanceSquared * float(gRectangleQueryBudget)));
        }
    }
    if (gUseEnvironment != 0u)
    {
        // A shared deterministic quadrature avoids pixel-scrambled salt-and-pepper noise
        // on the realtime side. This is bounded lighting work, not temporal accumulation.
        float3 environmentIntegral = float3(0.0f);
        [loop]
        for (uint sampleIndex = 0u; sampleIndex < gEnvironmentQueryBudget; ++sampleIndex)
        {
            const float2 u = nclsViewerHammersley(sampleIndex, gEnvironmentQueryBudget);
            const float3 wiWorld = nclsViewerToWorld(nclsViewerUniformHemisphere(u), normal, tangent);
            environmentIntegral += nclsApproximationDirection(
                state, context, wiWorld, index + 31u * sampleIndex + 5u)
                * nclsEnvironment(wiWorld);
        }
        radiance += environmentIntegral * (2.0f * NCLS_VIEWER_PI / float(gEnvironmentQueryBudget));
    }
    gApproximation[pixel] = float4(max(radiance, 0.0f), 1.0f);
}

