#include "../../../apps/viewer/shaders/PathSampleGenerator.slang"

RWStructuredBuffer<float4> gOutput;

[numthreads(4, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    NclsViewerPathSampleGenerator sampleGenerator =
        nclsViewerCreatePathSampleGenerator(
            uint2(17u, 23u), index, 1u, 0x243f6a88u);
    gOutput[index] = sampleNext4D(sampleGenerator);
}
