#ifndef NCLS_VIEWER_PATH_SAMPLE_GENERATOR_SLANG
#define NCLS_VIEWER_PATH_SAMPLE_GENERATOR_SLANG

import Utils.Sampling.UniformSampleGenerator;

#include "ViewerCommon.slang"

// Viewer path sampling uses Falcor's production UniformSampleGenerator
// (SplitMix64-seeded xoshiro128**). Every material still consumes the canonical
// ISampleGenerator interface; the wrapper only assigns a stable, independent
// pixel/path/depth/stream identity to each native material sample call.
struct NclsViewerPathSampleGenerator : ISampleGenerator
{
    UniformSampleGenerator generator;

    [mutating]
    uint next() { return generator.next(); }
};

NclsViewerPathSampleGenerator nclsViewerCreatePathSampleGenerator(
    uint2 pixel,
    uint sampleIndex,
    uint depth,
    uint stream)
{
    const uint sequence = nclsViewerHash(
        sampleIndex ^ nclsViewerHash(depth + 0x9e3779b9u) ^ stream);
    NclsViewerPathSampleGenerator result = {
        UniformSampleGenerator(pixel, sequence)};
    return result;
}

#endif
