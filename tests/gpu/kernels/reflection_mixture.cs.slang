#include "../../../shaders/ncls/scattering/reflection_mixture.slang"

RWStructuredBuffer<float4> gDirectionPdf;
RWStructuredBuffer<float4> gCheck;
uniform uint gCount;
uniform uint gMode;

float nclsProbeRadicalInverse(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;
}

NclsReflectionMixture nclsProbeProposal()
{
    NclsReflectionMixture proposal = {};
    proposal.cosineWeight = 0.2f;
    proposal.ggxWeights = float4(0.5f, 0.25f, 0.15f, 0.0f);
    proposal.alphaX = float4(0.025f, 0.15f, 0.55f, 1.0f);
    proposal.alphaY = float4(0.08f, 0.35f, 0.30f, 1.0f);
    proposal.rotations = float4(0.3f, -0.7f, 1.1f, 0.0f);
    proposal.ggxCount = 3u;
    return proposal;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint index = dispatchThreadId.x;
    if (index >= gCount) return;
    const NclsReflectionMixture proposal = nclsProbeProposal();
    const float3 wo = normalize(float3(0.45f, -0.20f, 0.87f));
    if (gMode == 0u)
    {
        const float2 u = float2(
            (float(index) + 0.5f) / float(gCount),
            nclsProbeRadicalInverse(index ^ 0x51ed270bu));
        uint component;
        const NclsDirectionalSample sampled = nclsSampleReflectionMixture(
            proposal, wo, u, component);
        gDirectionPdf[index] = float4(sampled.direction, sampled.pdf);
        const float checkedPdf = sampled.valid != 0u && sampled.nullEvent == 0u
            ? nclsReflectionMixturePdf(proposal, wo, sampled.direction)
            : 0.0f;
        gCheck[index] = float4(
            checkedPdf, float(sampled.valid), float(sampled.nullEvent), float(component));
    }
    else
    {
        const uint width = 1024u;
        const uint y = index / width;
        const uint x = index - y * width;
        const float z = (float(y) + 0.5f) / 512.0f;
        const float phi = 2.0f * NCLS_PI * (float(x) + 0.5f) / float(width);
        const float radius = sqrt(max(1.0f - z * z, 0.0f));
        const float3 wi = float3(radius * cos(phi), radius * sin(phi), z);
        gDirectionPdf[index] = float4(wi, nclsReflectionMixturePdf(proposal, wo, wi));
        gCheck[index] = float4(0.0f);
    }
}
