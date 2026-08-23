#include "../../../shaders/ncls/contracts/layer_stack_ir.slang"

StructuredBuffer<NclsLayerStackIR> gStacks;
RWStructuredBuffer<uint4> gOutput;

[numthreads(1, 1, 1)]
void validateLayerStackIrLayout(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const NclsLayerStackIR stack = gStacks[0];
    const NclsLayerInterfaceIR top = stack.interfaces[0];
    const NclsLayerInterfaceIR base = stack.interfaces[1];
    const NclsHomogeneousMediumIR medium = stack.media[0];

    gOutput[0] = uint4(stack.magic, stack.abiVersion, stack.interfaceCount, stack.mediumCount);
    gOutput[1] = uint4(top.kind, top.flags, asuint(top.alphaX), asuint(top.alphaY));
    gOutput[2] = uint4(asuint(top.relativeIor), asuint(top.tangentRotation), 0u, 0u);
    gOutput[3] = uint4(base.kind, asuint(base.alphaX), asuint(base.alphaY), asuint(base.etaR));
    gOutput[4] = uint4(asuint(base.etaG), asuint(base.etaB), asuint(base.kR), asuint(base.kG));
    gOutput[5] = uint4(asuint(base.kB), asuint(base.tangentRotation), 0u, 0u);
    gOutput[6] = uint4(asuint(medium.sigmaAR), asuint(medium.sigmaAG), asuint(medium.sigmaAB), asuint(medium.sigmaSR));
    gOutput[7] = uint4(asuint(medium.sigmaSG), asuint(medium.sigmaSB), asuint(medium.g), asuint(medium.thickness));
}
