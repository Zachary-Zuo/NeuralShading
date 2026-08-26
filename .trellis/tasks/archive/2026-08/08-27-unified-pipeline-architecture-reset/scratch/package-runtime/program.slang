#include "../../../../../shaders/ncls/contracts/scattering_contract.slang"

RWStructuredBuffer<float4> gOutput;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    NclsScatteringPdf pdf;
    pdf.forward = float(NCLS_SCATTERING_CONTRACT_VERSION);
    pdf.reverse = float(NclsTransportMode::Radiance);
    gOutput[0] = float4(pdf.forward, pdf.reverse, 0.0f, 1.0f);
}
