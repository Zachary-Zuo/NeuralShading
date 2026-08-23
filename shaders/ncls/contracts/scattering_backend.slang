#ifndef NCLS_SCATTERING_BACKEND_SLANG
#define NCLS_SCATTERING_BACKEND_SLANG

#include "scattering_contract.slang"
import Utils.Sampling.SampleGeneratorInterface;

// 这些接口只通过编译期 specialization 使用，不在热路径创建 existential value。
interface INclsScatteringState
{
    NclsScatteringEval evaluate<S : ISampleGenerator>(
        float3 wiWorld,
        inout S sampleGenerator);

    bool sample<S : ISampleGenerator>(
        out NclsScatteringSample result,
        inout S sampleGenerator);

    NclsScatteringPdf pdf(float3 wiWorld);
}

interface INclsScatteringBackend
{
    associatedtype CompiledMaterial;
    associatedtype State : INclsScatteringState;

    State prepare(
        NclsScatteringContext context,
        CompiledMaterial material);
}

#endif
