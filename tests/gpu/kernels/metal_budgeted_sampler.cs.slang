#include "../../../shaders/ncls/contracts/scattering_contract.slang"
StructuredBuffer<uint> gNclsRuntimeWeights;
StructuredBuffer<uint> gNclsCompiledMaterials;
#include "../../../shaders/ncls/backends/metal_budgeted/metal_budgeted_common.slang"
#include "../../../shaders/ncls/backends/metal_budgeted/metal_budgeted_sampler.slang"

StructuredBuffer<float4> gStates;
StructuredBuffer<float4> gQueries;
RWStructuredBuffer<float4> gOutput;
cbuffer TestCB { uint gCount; uint gIntegrate; };

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint i = tid.x;
    if (i >= gCount) return;
    const uint base = uint(gQueries[2u*i+1u].w) * 10u;
    NclsMetalBudgetedPrepared state = {};
    state.valid = uint(gStates[base+9u].w);
    [unroll]
    for (uint c = 0u; c < 3u; ++c)
        [unroll]
        for (uint k = 0u; k < 4u; ++k) state.proposal[4u*c+k] = gStates[base+c][k];
    [unroll]
    for (uint c = 0u; c < 2u; ++c)
    {
        state.frames[c].tangent = gStates[base+3u+3u*c].xyz;
        state.frames[c].bitangent = gStates[base+4u+3u*c].xyz;
        state.frames[c].normal = gStates[base+5u+3u*c].xyz;
    }
    const float3 wo = gStates[base+9u].xyz;
    if (gIntegrate != 0u)
    {
        const NclsScatteringPdf pdf = nclsMetalBudgetedProposalPdf(state, wo, gQueries[2u*i].xyz);
        gOutput[3u*i] = float4(pdf.forward, pdf.reverse, 0, 0);
        return;
    }
    float3 wi;
    NclsScatteringPdf pdf;
    const bool valid = nclsMetalBudgetedSampleProposal(state, wo, gQueries[2u*i].xy, wi, pdf);
    const NclsScatteringPdf independent = nclsMetalBudgetedProposalPdf(state, wo, wi);
    gOutput[3u*i] = float4(wi, float(valid));
    gOutput[3u*i+1u] = float4(pdf.forward, pdf.reverse, independent.forward, independent.reverse);
    const NclsScatteringPdf query = nclsMetalBudgetedProposalPdf(state, wo, gQueries[2u*i+1u].xyz);
    gOutput[3u*i+2u] = float4(query.forward, query.reverse, 0, 0);
}
