#include "../../../shaders/ncls/scattering/nvidia_proposal.slang"

RWStructuredBuffer<float4> gOutput;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= 16u) return;
    float raw[NCLS_NVIDIA_PROPOSAL_RAW_COUNT] = {
        -0.3f, 0.4f, 0.2f, 0.1f, -0.2f, 0.3f, -0.1f, 0.2f, -0.2f
    };
    const NclsNvidiaProposalParams params = nclsDecodeNvidiaProposal(raw);
    const float3 wo = normalize(float3(0.2f, -0.1f, 1.0f));
    const float2 u = float2(
        (float(index) + 0.5f) / 16.0f,
        frac((float(index) + 0.5f) * 0.61803398875f));
    uint component;
    const NclsDirectionalSample sample = nclsSampleNvidiaProposal(
        params, wo, u, component);
    const float directPdf = sample.valid != 0u && sample.nullEvent == 0u
        ? nclsNvidiaProposalPdf(params, wo, sample.direction)
        : 0.0f;
    gOutput[index] = float4(sample.direction, sample.pdf);
    gOutput[16u + index] = float4(
        directPdf, float(sample.valid), float(sample.nullEvent), float(component));
}
