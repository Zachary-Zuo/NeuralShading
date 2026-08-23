#include "../reference/materialx_standard_surface_reference.slang"

StructuredBuffer<float> gInputs;
Texture2D<float4> gBaseColor;
Texture2D<float4> gRoughness;
Texture2D<float4> gMetalness;
Texture2D<float4> gNormalMap;
SamplerState gMaterialSampler;
StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gLights;
StructuredBuffer<float4> gUv;
StructuredBuffer<float4> gUvGrad;
RWStructuredBuffer<float4> gOutput;
uniform uint gQueryCount;

float3 toShadingFrame(float3 value, float3 normal, float3 tangent)
{
    const float3 bitangent = normalize(cross(normal, tangent));
    return float3(dot(value, tangent), dot(value, bitangent), dot(value, normal));
}

[numthreads(64, 1, 1)]
void evaluateReference(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gQueryCount) return;
    const float2 uv = gUv[index].xy;
    const float4 gradients = gUvGrad[index];
    const NclsMaterialXResolvedSurface surface = nclsResolveMaterialXSurface(
        gInputs, gBaseColor, gRoughness, gMetalness, gMaterialSampler,
        uv, gradients.xy, gradients.zw);
    const float3 geometricNormal = float3(0.0f, 0.0f, 1.0f);
    const float3 geometricTangent = float3(1.0f, 0.0f, 0.0f);
    const float3 viewGeometric = normalize(gViews[index].xyz);
    float3 normal = nclsMaterialXShadingNormal(
        gInputs, gNormalMap, gMaterialSampler, uv, gradients.xy, gradients.zw,
        geometricNormal, geometricTangent);
    normal = nclsMxForwardFacingNormal(normal, viewGeometric);
    float3 tangent = geometricTangent - dot(geometricTangent, normal) * normal;
    tangent = dot(tangent, tangent) > 1e-10f ? normalize(tangent) : float3(1.0f, 0.0f, 0.0f);
    const float3 view = toShadingFrame(viewGeometric, normal, tangent);
    const float3 light = toShadingFrame(normalize(gLights[index].xyz), normal, tangent);
    const float3 response = view.z > 0.0f && light.z > 0.0f
        ? nclsEvalMaterialXStandardSurface(gInputs, surface, view, light)
        : float3(0.0f);
    gOutput[index] = float4(response, 1.0f);
}
