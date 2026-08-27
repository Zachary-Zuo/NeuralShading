#include "../../../apps/viewer/shaders/PathEnvironmentMath.slang"

RWStructuredBuffer<float4> gOutput;

[numthreads(4, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    const float lightPdf = index == 0u ? 0.25f : 0.5f;
    const float bsdfPdf = index == 0u ? 0.25f : (index == 1u ? 0.125f : 0.0f);
    const bool delta = index == 3u;
    gOutput[index] = float4(
        nclsViewerEnvironmentLightTechniquePdf(lightPdf),
        nclsViewerEnvironmentBsdfTechniquePdf(bsdfPdf),
        nclsViewerEnvironmentLightMisWeight(lightPdf, bsdfPdf),
        nclsViewerEnvironmentBsdfMisWeight(bsdfPdf, lightPdf, delta));
}
