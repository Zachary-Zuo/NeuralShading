#ifndef NCLS_METAL_FUSED_SLANG
#define NCLS_METAL_FUSED_SLANG

#include "../../contracts/scattering_backend.slang"
#include "../../scattering/frame.slang"

StructuredBuffer<uint> gNclsRuntimeWeights;
StructuredBuffer<uint> gNclsMetalDomains;
StructuredBuffer<uint> gNclsMetalRecords;
StructuredBuffer<uint> gNclsMetalHighGrid;
StructuredBuffer<uint> gNclsMetalLowGrid;
StructuredBuffer<uint> gNclsMetalGridScales;
StructuredBuffer<uint> gNclsMetalAssetAdapter;
SamplerState gNclsMetalGridSampler;
RWStructuredBuffer<uint> gNclsMetalRawParameters;
RWStructuredBuffer<uint> gNclsCompiledMaterials;

#include "../../scattering/metal_fused_proposal.slang"
#include "metal_fused_evaluator.slang"
#ifndef NCLS_METAL_RUNTIME_ONLY
#include "metal_fused_compiler.slang"
#endif

struct NclsMetalContractState : INclsScatteringState
{
    NclsScatteringContext context;
    NclsMetalCompiledMaterial material;
    NclsMetalPrepared prepared;

    NclsScatteringEval evaluate<S : ISampleGenerator>(float3 wiWorld, inout S sampleGenerator)
    {
        NclsScatteringEval result = {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        if (prepared.valid == 0u || wo.z <= NCLS_MIN_COS || wi.z <= NCLS_MIN_COS) return result;
        result.f = nclsMetalEvaluateF(prepared, wo, wi);
        result.pdf.forward = nclsMetalFusedProposalPdf(prepared.proposal, wo, wi);
        result.pdf.reverse = nclsMetalFusedProposalPdf(prepared.proposal, wi, wo);
        result.eventFlags = (uint)NclsScatteringEvent::Reflection | (uint)NclsScatteringEvent::Glossy;
        result.valid = all(isfinite(result.f)) && result.pdf.forward >= 0.0f
            && result.pdf.reverse >= 0.0f ? 1u : 0u;
        return result;
    }

    bool sample<S : ISampleGenerator>(out NclsScatteringSample result, inout S sampleGenerator)
    {
        result = {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const NclsMetalFusedProposalSample proposal = nclsSampleMetalFusedProposal(
            prepared.proposal, wo, sampleNext2D(sampleGenerator));
        if (proposal.valid == 0u || proposal.forwardPdf <= 0.0f) return false;
        result.wiWorld = nclsFrameToWorld(context.surface.shadingFrame, proposal.direction);
        const float3 f = nclsMetalEvaluateF(prepared, wo, proposal.direction);
        result.weight = f * max(proposal.direction.z, 0.0f) / proposal.forwardPdf;
        result.pdf.forward = proposal.forwardPdf;
        result.pdf.reverse = proposal.reversePdf;
        result.eta = 1.0f;
        result.eventFlags = (uint)NclsScatteringEvent::Reflection | (uint)NclsScatteringEvent::Glossy;
        result.valid = all(isfinite(result.weight)) ? 1u : 0u;
        return result.valid != 0u;
    }

    NclsScatteringPdf pdf(float3 wiWorld)
    {
        NclsScatteringPdf result = {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        if (prepared.valid == 0u || wo.z <= NCLS_MIN_COS || wi.z <= NCLS_MIN_COS) return result;
        result.forward = nclsMetalFusedProposalPdf(prepared.proposal, wo, wi);
        result.reverse = nclsMetalFusedProposalPdf(prepared.proposal, wi, wo);
        return result;
    }
};

struct NclsMetalBackend : INclsScatteringBackend
{
    typedef NclsMetalCompiledMaterial CompiledMaterial;
    typedef NclsMetalContractState State;

    State prepare(NclsScatteringContext context, CompiledMaterial material)
    {
        return {context, material, nclsMetalPrepare(context, material)};
    }
};

typedef NclsMetalBackend NclsPackageBackend;
typedef NclsMetalCompiledMaterial NclsPackageCompiledMaterial;
typedef NclsMetalContractState NclsPackageState;

NclsPackageBackend nclsCreatePackageBackend() { return {}; }
NclsPackageCompiledMaterial nclsLoadPackageMaterial(uint index)
{
    NclsPackageCompiledMaterial result;
    result.baseWord = index * NCLS_METAL_COMPILED_WORD_COUNT;
    return result;
}

#endif
