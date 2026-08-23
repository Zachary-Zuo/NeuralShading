#include "../../../shaders/ncls/contracts/scattering_backend.slang"
#include "../../../shaders/ncls/contracts/response_measure.slang"

struct TestCompiledMaterial
{
    float3 color;
};

struct TestScatteringState : INclsScatteringState
{
    float3 color;

    NclsScatteringEval evaluate<S : ISampleGenerator>(
        float3 wiWorld,
        inout S sampleGenerator)
    {
        NclsScatteringEval result = {};
        result.f = color;
        result.eventFlags = (uint)NclsScatteringEvent::Reflection | (uint)NclsScatteringEvent::Diffuse;
        result.valid = 1u;
        return result;
    }

    bool sample<S : ISampleGenerator>(
        out NclsScatteringSample result,
        inout S sampleGenerator)
    {
        result = {};
        return false;
    }

    NclsScatteringPdf pdf(float3 wiWorld)
    {
        return {0.0f, 0.0f};
    }
};

struct TestScatteringBackend : INclsScatteringBackend
{
    typedef TestCompiledMaterial CompiledMaterial;
    typedef TestScatteringState State;

    State prepare(NclsScatteringContext context, CompiledMaterial material)
    {
        return {material.color};
    }
};

[numthreads(1, 1, 1)]
void compileScatteringContract(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TestScatteringBackend backend;
    TestCompiledMaterial material = {float3(0.5f)};
    NclsScatteringContext context = {};
    context.surface.shadingFrame.normal = float3(0.0f, 0.0f, 1.0f);
    TestScatteringState state = backend.prepare(context, material);
    NclsScatteringEval evaluation = {};
    evaluation.f = state.color;
    evaluation.valid = 1u;
    const float3 responseCos = nclsScatteringEvalToFalcorIbsdfEval(
        evaluation,
        context.surface.shadingFrame,
        float3(0.0f, 0.0f, 1.0f));
}
