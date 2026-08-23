#include "ViewerCommon.slang"
#include "../../../shaders/ncls/backends/legacy_ltc_k2/p1_compiler.slang"

StructuredBuffer<NclsLayerStackIR> gMaterials;
StructuredBuffer<float> gWeights;
Texture2D<float4> gPositionDepth;
Texture2D<float4> gNormal;
Texture2D<float4> gTangent;
Texture2D<float4> gViewDirection;
Texture2D<uint> gMaterialId;
RWStructuredBuffer<NclsLegacyLtcK2PackedState> gStates;

cbuffer PrepareCB
{
    uint2 gFrameDim;
    uint gMethodMode;
    uint gWidth;
    uint gUseScene;
    uint gTargetMaterialId;
};

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFrameDim)) return;
    if (gUseScene != 0u && gMaterialId[pixel] != gTargetMaterialId + 1u) return;
    const uint index = pixel.y * gFrameDim.x + pixel.x;
    NclsLegacyLtcK2PackedState state = {};
    if (gPositionDepth[pixel].w >= 0.0f)
    {
        const NclsLayerStackIR material = gMaterials[0];
        const float3 normal = normalize(gNormal[pixel].xyz);
        const float3 tangent = normalize(gTangent[pixel].xyz);
        if (gMethodMode == 1u)
        {
            NclsScatteringContext context = {};
            context.surface.position = gPositionDepth[pixel].xyz;
            context.surface.geometricNormal = normal;
            context.surface.shadingFrame.normal = normal;
            context.surface.shadingFrame.tangent = tangent;
            context.surface.shadingFrame.bitangent = cross(normal, tangent);
            context.surface.frontFacing = 1u;
            context.woWorld = normalize(gViewDirection[pixel].xyz);
            context.transportMode = (uint)NclsTransportMode::Radiance;
            context.componentMask = (uint)NclsScatteringEvent::Reflection;
            LegacyLtcK2P1Backend backend;
            backend.weights = gWeights;
            backend.width = gWidth;
            const LegacyLtcK2State prepared = backend.prepare(context, material);
            state = prepared.packed;
        }
        else
        {
            state.magic = NCLS_LEGACY_LTC_K2_MAGIC;
            state.version = NCLS_LEGACY_LTC_K2_VERSION;
            state.residualLobeCount = NCLS_LEGACY_LTC_K2_LOBE_COUNT;
            state.flags = 0u;
            state.directTop = material.interfaces[0];
        }
    }
    gStates[index] = state;
}
