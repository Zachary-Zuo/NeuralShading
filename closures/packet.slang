#ifndef NCLS_CLOSURE_PACKET_SLANG
#define NCLS_CLOSURE_PACKET_SLANG

#include "../teacher/interfaces.slang"

static const uint NCLS_PACKET_MAGIC = 0x504c434e;
static const uint NCLS_PACKET_VERSION = 1;
static const uint NCLS_PACKET_RESIDUAL_LOBES = 2;

// 48 bytes. The three float4 rows keep CPU/Slang StructuredBuffer layout explicit.
struct NclsLtcResidualLobe
{
    float amplitudeR;
    float amplitudeG;
    float amplitudeB;
    float reserved0;

    float inverseScaleX;
    float inverseScaleY;
    float shearX;
    float shearY;

    float shearZ;
    float angle;
    float reserved1;
    float reserved2;
};

// 176 bytes: 16-byte header + 64-byte atomic interface + two 48-byte LTC lobes.
struct NclsClosurePacket
{
    uint magic;
    uint version;
    uint residualLobeCount;
    uint flags;
    LayerInterface directTop;
    NclsLtcResidualLobe residual[NCLS_PACKET_RESIDUAL_LOBES];
};

float3 nclsEvalLtcResidual(const NclsLtcResidualLobe lobe, const float3 lightDirection)
{
    const float cosine = cos(lobe.angle);
    const float sine = sin(lobe.angle);
    const float rotatedX = cosine * lightDirection.x + sine * lightDirection.y;
    const float rotatedY = -sine * lightDirection.x + cosine * lightDirection.y;
    const float qx = lobe.inverseScaleX * rotatedX
        + lobe.shearX * rotatedY + lobe.shearY * lightDirection.z;
    const float qy = lobe.inverseScaleY * rotatedY + lobe.shearZ * lightDirection.z;
    const float qz = lightDirection.z;
    const float norm2 = max(qx * qx + qy * qy + qz * qz, 1e-10f);
    const float determinant = lobe.inverseScaleX * lobe.inverseScaleY;
    const float basis = determinant * max(qz, 0.0f) / (NCLS_PI * norm2 * norm2);
    return basis * float3(lobe.amplitudeR, lobe.amplitudeG, lobe.amplitudeB);
}

float3 nclsEvalClosurePacket(
    const NclsClosurePacket packet,
    const float3 viewDirection,
    const float3 lightDirection)
{
    if (packet.magic != NCLS_PACKET_MAGIC || packet.version != NCLS_PACKET_VERSION
        || packet.residualLobeCount != NCLS_PACKET_RESIDUAL_LOBES)
        return float3(0.0f);

    const LayerInterface top = packet.directTop;
    const float etaBelow = top.type == (uint)LayerType::RoughDielectric ? top.etaR : 1.0f;
    float3 value = nclsEvalInterfaceTransport(
        top, viewDirection, lightDirection, 1.0f, etaBelow, true) * max(lightDirection.z, 0.0f);
    [unroll]
    for (uint lobeIndex = 0u; lobeIndex < NCLS_PACKET_RESIDUAL_LOBES; ++lobeIndex)
        value += nclsEvalLtcResidual(packet.residual[lobeIndex], lightDirection);
    return value;
}

#endif
