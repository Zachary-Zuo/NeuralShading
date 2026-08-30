#include "../../../shaders/ncls/scattering/metal_fused_proposal.slang"

RWStructuredBuffer<float4> gSample;
RWStructuredBuffer<float4> gCheck;
RWStructuredBuffer<float4> gQuery;
uniform uint gCount;

NclsMetalFusedProposal nclsProbeMetalProposal(uint probeIndex)
{
    NclsMetalFusedProposal proposal = {};
    proposal.valid = 1u;
    [unroll]
    for (uint component = 0u; component < NCLS_METAL_PROPOSAL_COMPONENT_COUNT; ++component)
    {
        proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_NORMALIZED_WEIGHT]
            = 0.02f + 0.002f * float(component)
            + 0.0005f * float((probeIndex + 3u * component) % 17u);
        proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_ALPHA_X]
            = 0.05f + 0.003f * float(component)
            + 0.0004f * float((2u * probeIndex + component) % 29u);
        proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_ALPHA_Y]
            = 0.08f + 0.004f * float(component)
            + 0.0003f * float((probeIndex + 2u * component) % 31u);
        proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_ROTATION_RADIANS]
            = -0.5f + 0.1f * float(component)
            + 0.001f * float((probeIndex + 5u * component) % 23u);
        proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_ACTIVE] = 1.0f;
        proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_FRAME_INDEX]
            = float(nclsMetalProposalFrameIndex(component));
        proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_DISTRIBUTION_ID]
            = float(nclsMetalProposalDistribution(component));
        proposal.state[component][NCLS_METAL_PROPOSAL_FIELD_ENERGY_CLUE]
            = 0.5f + 0.01f * float((probeIndex + component) % 19u);
    }
    const float angles[NCLS_METAL_PROPOSAL_FRAME_COUNT] = {
        0.0f, 0.35f, -0.45f, 0.7f
    };
    [unroll]
    for (uint frame = 0u; frame < NCLS_METAL_PROPOSAL_FRAME_COUNT; ++frame)
    {
        const float angle = angles[frame]
            + 0.0007f * float(probeIndex % 41u);
        const float cosine = cos(angle);
        const float sine = sin(angle);
        proposal.frames[frame][0] = float3(cosine, 0.0f, -sine);
        proposal.frames[frame][1] = float3(0.0f, 1.0f, 0.0f);
        proposal.frames[frame][2] = float3(sine, 0.0f, cosine);
    }
    return proposal;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    const NclsMetalFusedProposal proposal = nclsProbeMetalProposal(index);
    const float3 wo = normalize(float3(
        0.15f + 0.002f * float(index % 53u),
        -0.23f + 0.0015f * float(index % 71u),
        1.0f));
    const float2 u = float2(
        (float(index) + 0.5f) / float(gCount),
        frac((float(index) + 0.5f) * 0.61803398875f));
    const NclsMetalFusedProposalSample sampled = nclsSampleMetalFusedProposal(
        proposal, wo, u);
    const float directPdf = sampled.valid != 0u
        ? nclsMetalFusedProposalPdf(proposal, wo, sampled.direction)
        : 0.0f;
    gSample[index] = float4(sampled.direction, sampled.forwardPdf);
    gCheck[index] = float4(
        sampled.reversePdf,
        directPdf,
        float(sampled.valid),
        float(sampled.component));
    const float queryZ = 0.01f + 0.98f
        * ((float(index) + 0.5f) / float(gCount));
    const float queryPhi = 2.0f * NCLS_PI
        * frac((float(index) + 0.25f) * 0.41421356237f);
    const float queryRadius = sqrt(max(1.0f - queryZ * queryZ, 0.0f));
    const float3 queryDirection = float3(
        queryRadius * cos(queryPhi), queryRadius * sin(queryPhi), queryZ);
    gQuery[index] = float4(
        queryDirection,
        nclsMetalFusedProposalPdf(proposal, wo, queryDirection));
}
