#include "../../closures/packet.slang"

StructuredBuffer<NclsClosurePacket> gPackets;
StructuredBuffer<float4> gViewDirections;
StructuredBuffer<float4> gLightDirections;
RWStructuredBuffer<float4> gOutput;

uniform uint gQueryCount;
uniform uint gLightCount;

[numthreads(64, 1, 1)]
void evaluateOraclePackets(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint queryIndex = dispatchThreadID.x;
    if (queryIndex >= gQueryCount) return;
    const uint packetIndex = queryIndex / gLightCount;
    const uint lightIndex = queryIndex - packetIndex * gLightCount;
    const float3 viewDirection = normalize(gViewDirections[packetIndex].xyz);
    const float3 lightDirection = normalize(gLightDirections[lightIndex].xyz);
    const float3 value = nclsEvalClosurePacket(
        gPackets[packetIndex], viewDirection, lightDirection);
    gOutput[queryIndex] = float4(value, 1.0f);
}
