#ifndef NCLS_BENCH_PACKAGE_HEADER
#error NCLS_BENCH_PACKAGE_HEADER must name a frozen ScatteringPackage entry module
#endif

#include NCLS_BENCH_PACKAGE_HEADER

#ifndef NCLS_BENCH_ADAPTER_NVIDIA_V1
#define NCLS_BENCH_ADAPTER_NVIDIA_V1 0
#endif
#ifndef NCLS_BENCH_ADAPTER_METAL_FULL_V1
#define NCLS_BENCH_ADAPTER_METAL_FULL_V1 0
#endif
#ifndef NCLS_BENCH_ADAPTER_COMMON_PACKED
#define NCLS_BENCH_ADAPTER_COMMON_PACKED 0
#endif
#ifndef NCLS_BENCH_ADAPTER_MDL_REFERENCE
#define NCLS_BENCH_ADAPTER_MDL_REFERENCE 0
#endif

struct NclsBenchmarkSampleGenerator : ISampleGenerator
{
    uint state;

    [mutating]
    uint next()
    {
        state = state * 1664525u + 1013904223u;
        return state;
    }
};

#if NCLS_BENCH_ADAPTER_NVIDIA_V1
typedef NclsMethodPackedState NclsBenchmarkPreparedState;

NclsBenchmarkPreparedState nclsBenchmarkPack(NclsPackageState state)
{
    return nclsPackMethodState(state);
}

NclsPackageState nclsBenchmarkUnpack(
    NclsScatteringContext context,
    NclsPackageCompiledMaterial material,
    NclsBenchmarkPreparedState prepared)
{
    return nclsUnpackMethodState(context, material, prepared, gNclsRuntimeWeights);
}
#elif NCLS_BENCH_ADAPTER_METAL_FULL_V1
// The historical Metal package predates the common packed-state helper. Its
// pure-data prepared payload is nevertheless public in the frozen module, so
// keep the adapter at the package boundary instead of re-running prepare().
typedef NclsMetalPrepared NclsBenchmarkPreparedState;

NclsBenchmarkPreparedState nclsBenchmarkPack(NclsPackageState state)
{
    return state.prepared;
}

NclsPackageState nclsBenchmarkUnpack(
    NclsScatteringContext context,
    NclsPackageCompiledMaterial material,
    NclsBenchmarkPreparedState prepared)
{
    return {context, material, prepared};
}
#elif NCLS_BENCH_ADAPTER_COMMON_PACKED
typedef NclsMethodPackedState NclsBenchmarkPreparedState;

NclsBenchmarkPreparedState nclsBenchmarkPack(NclsPackageState state)
{
    return nclsPackMethodState(state);
}

NclsPackageState nclsBenchmarkUnpack(
    NclsScatteringContext context,
    NclsPackageCompiledMaterial material,
    NclsBenchmarkPreparedState prepared)
{
    return nclsUnpackMethodState(context, material, prepared, gNclsRuntimeWeights);
}
#elif NCLS_BENCH_ADAPTER_MDL_REFERENCE
// MDL's public prepare() only captures the common context and the 16 B
// material record. The expensive target-code state is intentionally formed by
// evaluate/sample/pdf, matching the authoritative MDL backend semantics.
typedef NclsMdlCompiledMaterial NclsBenchmarkPreparedState;

NclsBenchmarkPreparedState nclsBenchmarkPack(NclsPackageState state)
{
    return state.material;
}

NclsPackageState nclsBenchmarkUnpack(
    NclsScatteringContext context,
    NclsPackageCompiledMaterial material,
    NclsBenchmarkPreparedState prepared)
{
    return {context, prepared};
}
#else
#error The benchmark needs a lossless prepared-state adapter for this package
#endif

StructuredBuffer<float4> gBenchmarkViews;
StructuredBuffer<float4> gBenchmarkLights;
StructuredBuffer<float4> gBenchmarkUvAndDx;
StructuredBuffer<float4> gBenchmarkDyAndRandom;
RWStructuredBuffer<NclsBenchmarkPreparedState> gBenchmarkPrepared;
RWStructuredBuffer<float4> gBenchmarkOutput;

uniform uint gBenchmarkCount;
uniform uint gBenchmarkDirectionCount;
uniform uint gBenchmarkCompiledMaterialIndex;
uniform uint gBenchmarkSeed;

NclsScatteringContext nclsBenchmarkContext(uint index)
{
    const float4 uvAndDx = gBenchmarkUvAndDx[index];
    const float4 dyAndRandom = gBenchmarkDyAndRandom[index];
    NclsScatteringContext result = {};
    result.surface.position = float3(0.0f);
    result.surface.geometricNormal = float3(0.0f, 0.0f, 1.0f);
    result.surface.shadingFrame.normal = float3(0.0f, 0.0f, 1.0f);
    result.surface.shadingFrame.tangent = float3(1.0f, 0.0f, 0.0f);
    result.surface.shadingFrame.bitangent = float3(0.0f, 1.0f, 0.0f);
    result.surface.uv = uvAndDx.xy;
    result.surface.uvDx = uvAndDx.zw;
    result.surface.uvDy = dyAndRandom.xy;
    result.surface.materialInstanceId = gBenchmarkCompiledMaterialIndex;
    result.surface.primitiveId = index;
    result.surface.frontFacing = 1u;
    result.woWorld = normalize(gBenchmarkViews[index].xyz);
    result.transportMode = (uint)NclsTransportMode::Radiance;
    result.componentMask = (uint)NclsScatteringEvent::Reflection;
    result.filterRandom = dyAndRandom.z;
    return result;
}

NclsPackageState nclsBenchmarkPrepare(uint index)
{
    const NclsPackageBackend backend = nclsCreatePackageBackend();
    return backend.prepare(
        nclsBenchmarkContext(index),
        nclsLoadPackageMaterial(gBenchmarkCompiledMaterialIndex));
}

NclsPackageState nclsBenchmarkPreparedState(uint index)
{
    return nclsBenchmarkUnpack(
        nclsBenchmarkContext(index),
        nclsLoadPackageMaterial(gBenchmarkCompiledMaterialIndex),
        gBenchmarkPrepared[index]);
}

[numthreads(64, 1, 1)]
void prepareOnly(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gBenchmarkCount) return;
    gBenchmarkPrepared[index] = nclsBenchmarkPack(nclsBenchmarkPrepare(index));
}

[numthreads(64, 1, 1)]
void evaluateOnly(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gBenchmarkCount) return;
    NclsPackageState state = nclsBenchmarkPreparedState(index);
    NclsBenchmarkSampleGenerator generator = {gBenchmarkSeed ^ index};
    const NclsScatteringEval value = state.evaluate(
        normalize(gBenchmarkLights[index * gBenchmarkDirectionCount].xyz), generator);
    gBenchmarkOutput[index] = float4(value.f, value.pdf.forward);
}

[numthreads(64, 1, 1)]
void prepareEvaluate(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gBenchmarkCount) return;
    NclsPackageState state = nclsBenchmarkPrepare(index);
    NclsBenchmarkSampleGenerator generator = {gBenchmarkSeed ^ index};
    float4 total = float4(0.0f);
    [loop]
    for (uint direction = 0u; direction < gBenchmarkDirectionCount; ++direction)
    {
        const NclsScatteringEval value = state.evaluate(
            normalize(gBenchmarkLights[index * gBenchmarkDirectionCount + direction].xyz),
            generator);
        total += float4(value.f, value.pdf.forward);
    }
    gBenchmarkOutput[index] = total;
}

[numthreads(64, 1, 1)]
void sampleOnly(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gBenchmarkCount) return;
    NclsPackageState state = nclsBenchmarkPreparedState(index);
    NclsBenchmarkSampleGenerator generator = {gBenchmarkSeed ^ index};
    NclsScatteringSample value = {};
    const bool sampled = state.sample(value, generator);
    gBenchmarkOutput[index] = float4(
        sampled ? value.weight : float3(0.0f), value.pdf.forward);
}

[numthreads(64, 1, 1)]
void pdfOnly(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gBenchmarkCount) return;
    NclsPackageState state = nclsBenchmarkPreparedState(index);
    const NclsScatteringPdf value = state.pdf(
        normalize(gBenchmarkLights[index * gBenchmarkDirectionCount].xyz));
    gBenchmarkOutput[index] = float4(value.forward, value.reverse, 0.0f, 0.0f);
}
