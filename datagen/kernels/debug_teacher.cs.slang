#include "../../teacher/layered_walk.slang"

StructuredBuffer<LayerStack> gStacks;
StructuredBuffer<float4> gViewDirections;
StructuredBuffer<float4> gLightDirections;
RWStructuredBuffer<float4> gResult;
RWStructuredBuffer<uint> gDebugCode;
RWStructuredBuffer<float4> gAuxiliary0;
RWStructuredBuffer<float4> gAuxiliary1;

uniform uint gKeyX;
uniform uint gKeyY;
uniform uint gMaxDepth;

[numthreads(1, 1, 1)]
void debugTeacherSample(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint debugCode;
    const float3 value = nclsEvalLayerStackWalkDiagnostic(
        gStacks[0],
        normalize(gViewDirections[0].xyz),
        normalize(gLightDirections[0].xyz),
        uint2(gKeyX, gKeyY),
        gMaxDepth,
        debugCode);
    gResult[0] = float4(value, 1.0f);
    gDebugCode[0] = debugCode;
    NclsRng ballisticRng = NclsRng(uint2(
        gKeyX ^ 0x6a09e667u,
        gKeyY ^ 0xbb67ae85u));
    const NclsBallisticPath viewBallistic = nclsSampleBallisticToBase(
        gStacks[0], normalize(gViewDirections[0].xyz), true, ballisticRng);
    const NclsBallisticPath lightBallistic = nclsSampleBallisticToBase(
        gStacks[0], normalize(gLightDirections[0].xyz), false, ballisticRng);
    gAuxiliary0[0] = float4(viewBallistic.directionAtBase, float(viewBallistic.valid));
    gAuxiliary1[0] = float4(lightBallistic.directionAtBase, float(lightBallistic.valid));
}
