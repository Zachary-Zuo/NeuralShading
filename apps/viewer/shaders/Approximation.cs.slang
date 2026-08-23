#include "ViewerCommon.slang"
#include "../../../shaders/ncls/backends/legacy_ltc_k2/legacy_ltc_k2.slang"
#include "../../../shaders/ncls/contracts/response_measure.slang"

StructuredBuffer<NclsLegacyLtcK2PackedState> gStates;
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
};

struct NclsViewerSampleGenerator : ISampleGenerator
{
    uint state;

    [mutating]
    uint next()
    {
        state = nclsViewerHash(state);
        return state;
    }
};

float3 nclsApproximationDirection(
    NclsLegacyLtcK2PackedState state,
    float3 woLocal,
    float3 wiWorld,
    float3 normal,
    float3 tangent)
{
    NclsScatteringContext context = {};
    context.surface.geometricNormal = normal;
    context.surface.shadingFrame.normal = normal;
    context.surface.shadingFrame.tangent = tangent;
    context.surface.shadingFrame.bitangent = cross(normal, tangent);
    context.surface.frontFacing = 1u;
    context.woWorld = nclsViewerToWorld(woLocal, normal, tangent);
    context.transportMode = (uint)NclsTransportMode::Radiance;
    context.componentMask = (uint)NclsScatteringEvent::Reflection;
    LegacyLtcK2Backend backend;
    LegacyLtcK2State prepared = backend.prepare(context, state);
    NclsViewerSampleGenerator sampleGenerator = {nclsViewerHash(asuint(woLocal.x) ^ asuint(woLocal.y))};
    const float3 direction = normalize(wiWorld);
    const NclsScatteringEval evaluation = prepared.evaluate(direction, sampleGenerator);
    return nclsScatteringEvalToResponseCos(evaluation, context.surface.shadingFrame, direction);
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
    const NclsLegacyLtcK2PackedState state = gStates[index];
    const float3 position = positionDepth.xyz;
    const float3 normal = normalize(gNormal[pixel].xyz);
    const float3 tangent = normalize(gTangent[pixel].xyz);
    const float3 woLocal = nclsViewerToLocal(viewWorld, normal, tangent);
    float3 radiance = float3(0.0f);
    if (gUseSun != 0u)
        radiance += nclsApproximationDirection(state, woLocal, normalize(gSunDirection), normal, tangent)
            * gSunColor * gSunIntensity;
    if (gUsePoint != 0u)
    {
        const float3 toLight = gPointPosition - position;
        const float distanceSquared = max(dot(toLight, toLight), 1e-4f);
        radiance += nclsApproximationDirection(state, woLocal, toLight, normal, tangent)
            * gPointColor * (gPointIntensity / distanceSquared);
    }
    if (gUseRectangle != 0u)
    {
        static const uint kRectangleSamples = 8u;
        const float3 lightNormal = -normalize(cross(gRectangleAxisU, gRectangleAxisV));
        const float area = 4.0f * length(cross(gRectangleAxisU, gRectangleAxisV));
        [unroll]
        for (uint sampleIndex = 0u; sampleIndex < kRectangleSamples; ++sampleIndex)
        {
            const float2 u = nclsViewerHammersley(sampleIndex, kRectangleSamples);
            const float3 lightPoint = gRectangleCenter
                + (2.0f * u.x - 1.0f) * gRectangleAxisU
                + (2.0f * u.y - 1.0f) * gRectangleAxisV;
            const float3 toLight = lightPoint - position;
            const float distanceSquared = max(dot(toLight, toLight), 1e-4f);
            const float3 wiWorld = normalize(toLight);
            const float lightCosine = max(dot(lightNormal, -wiWorld), 0.0f);
            radiance += nclsApproximationDirection(state, woLocal, wiWorld, normal, tangent)
                * gRectangleColor * gRectangleIntensity
                * (area * lightCosine / (distanceSquared * float(kRectangleSamples)));
        }
    }
    if (gUseEnvironment != 0u)
    {
        // A shared deterministic quadrature avoids pixel-scrambled salt-and-pepper noise
        // on the realtime side. This is bounded lighting work, not temporal accumulation.
        static const uint kEnvironmentSamples = 64u;
        float3 environmentIntegral = float3(0.0f);
        [unroll]
        for (uint sampleIndex = 0u; sampleIndex < kEnvironmentSamples; ++sampleIndex)
        {
            const float2 u = nclsViewerHammersley(sampleIndex, kEnvironmentSamples);
            const float3 wiWorld = nclsViewerToWorld(nclsViewerUniformHemisphere(u), normal, tangent);
            environmentIntegral += nclsApproximationDirection(state, woLocal, wiWorld, normal, tangent)
                * nclsEnvironment(wiWorld);
        }
        radiance += environmentIntegral * (2.0f * NCLS_VIEWER_PI / float(kEnvironmentSamples));
    }
    gApproximation[pixel] = float4(max(radiance, 0.0f), 1.0f);
}
