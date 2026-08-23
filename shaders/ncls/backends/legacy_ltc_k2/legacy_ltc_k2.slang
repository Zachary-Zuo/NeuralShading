#ifndef NCLS_LEGACY_LTC_K2_SLANG
#define NCLS_LEGACY_LTC_K2_SLANG

#include "../../contracts/scattering_backend.slang"
#include "../../reference/interfaces.slang"

static const uint NCLS_LEGACY_LTC_K2_MAGIC = 0x324b544cu;
static const uint NCLS_LEGACY_LTC_K2_VERSION = 1u;
static const uint NCLS_LEGACY_LTC_K2_LOBE_COUNT = 2u;

struct NclsLegacyLtcK2Lobe
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

// 176 bytes；这是 legacy-ltc-k2 后端私有状态，不是公共 scattering packet。
struct NclsLegacyLtcK2PackedState
{
    uint magic;
    uint version;
    uint residualLobeCount;
    uint flags;
    NclsLayerInterfaceIR directTop;
    NclsLegacyLtcK2Lobe residual[NCLS_LEGACY_LTC_K2_LOBE_COUNT];
};

float3 nclsLegacyLtcK2LobeResponseCos(
    const NclsLegacyLtcK2Lobe lobe,
    const float3 wiLocal)
{
    const float cosine = cos(lobe.angle);
    const float sine = sin(lobe.angle);
    const float rotatedX = cosine * wiLocal.x + sine * wiLocal.y;
    const float rotatedY = -sine * wiLocal.x + cosine * wiLocal.y;
    const float qx = lobe.inverseScaleX * rotatedX
        + lobe.shearX * rotatedY + lobe.shearY * wiLocal.z;
    const float qy = lobe.inverseScaleY * rotatedY + lobe.shearZ * wiLocal.z;
    const float qz = wiLocal.z;
    const float norm2 = max(qx * qx + qy * qy + qz * qz, 1e-10f);
    const float determinant = lobe.inverseScaleX * lobe.inverseScaleY;
    const float basis = determinant * max(qz, 0.0f) / (NCLS_PI * norm2 * norm2);
    return basis * float3(lobe.amplitudeR, lobe.amplitudeG, lobe.amplitudeB);
}

float3 nclsLegacyLtcK2ResponseCos(
    const NclsLegacyLtcK2PackedState state,
    const float3 woLocal,
    const float3 wiLocal)
{
    if (state.magic != NCLS_LEGACY_LTC_K2_MAGIC
        || state.version != NCLS_LEGACY_LTC_K2_VERSION
        || state.residualLobeCount != NCLS_LEGACY_LTC_K2_LOBE_COUNT
        || state.flags != 0u)
        return float3(0.0f);
    const NclsLayerInterfaceIR top = state.directTop;
    const float etaBelow = top.kind == (uint)NclsInterfaceKind::RoughDielectric
        ? top.relativeIor
        : 1.0f;
    float3 value = nclsEvalInterface(top, woLocal, wiLocal, 1.0f, etaBelow)
        * max(wiLocal.z, 0.0f);
    [unroll]
    for (uint index = 0u; index < NCLS_LEGACY_LTC_K2_LOBE_COUNT; ++index)
        value += nclsLegacyLtcK2LobeResponseCos(state.residual[index], wiLocal);
    return value;
}

float3 nclsLegacyLtcK2Bsdf(
    const NclsLegacyLtcK2PackedState state,
    const float3 woLocal,
    const float3 wiLocal)
{
    return wiLocal.z > NCLS_MIN_COS
        ? nclsLegacyLtcK2ResponseCos(state, woLocal, wiLocal) / wiLocal.z
        : float3(0.0f);
}

float3 nclsLegacyLtcK2ToLocal(const NclsShadingFrame frame, const float3 direction)
{
    return float3(
        dot(direction, frame.tangent),
        dot(direction, frame.bitangent),
        dot(direction, frame.normal));
}

float3 nclsLegacyLtcK2ToWorld(const NclsShadingFrame frame, const float3 direction)
{
    return direction.x * frame.tangent + direction.y * frame.bitangent + direction.z * frame.normal;
}

struct LegacyLtcK2State : INclsScatteringState
{
    NclsScatteringContext context;
    NclsLegacyLtcK2PackedState packed;

    NclsScatteringEval evaluate<S : ISampleGenerator>(
        float3 wiWorld,
        inout S sampleGenerator)
    {
        NclsScatteringEval result = {};
        const float3 woLocal = nclsLegacyLtcK2ToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wiLocal = nclsLegacyLtcK2ToLocal(context.surface.shadingFrame, wiWorld);
        if (woLocal.z <= NCLS_MIN_COS || wiLocal.z <= NCLS_MIN_COS) return result;
        result.f = nclsLegacyLtcK2Bsdf(packed, woLocal, wiLocal);
        result.pdf = {wiLocal.z * NCLS_INV_PI, woLocal.z * NCLS_INV_PI};
        result.eventFlags = (uint)NclsScatteringEvent::Reflection | (uint)NclsScatteringEvent::Glossy;
        result.valid = all(isfinite(result.f)) ? 1u : 0u;
        return result;
    }

    bool sample<S : ISampleGenerator>(
        out NclsScatteringSample result,
        inout S sampleGenerator)
    {
        result = {};
        const float3 wiLocal = nclsSampleCosineHemisphere(sampleNext2D(sampleGenerator));
        result.wiWorld = nclsLegacyLtcK2ToWorld(context.surface.shadingFrame, wiLocal);
        const NclsScatteringEval evaluation = evaluate(result.wiWorld, sampleGenerator);
        if (evaluation.valid == 0u || evaluation.pdf.forward <= 0.0f) return false;
        result.weight = evaluation.f * wiLocal.z / evaluation.pdf.forward;
        result.pdf = evaluation.pdf;
        result.eta = 1.0f;
        result.eventFlags = evaluation.eventFlags;
        result.valid = 1u;
        return true;
    }

    NclsScatteringPdf pdf(float3 wiWorld)
    {
        const float3 woLocal = nclsLegacyLtcK2ToLocal(context.surface.shadingFrame, context.woWorld);
        const float3 wiLocal = nclsLegacyLtcK2ToLocal(context.surface.shadingFrame, wiWorld);
        return {
            max(wiLocal.z, 0.0f) * NCLS_INV_PI,
            max(woLocal.z, 0.0f) * NCLS_INV_PI,
        };
    }
};

struct LegacyLtcK2Backend : INclsScatteringBackend
{
    typedef NclsLegacyLtcK2PackedState CompiledMaterial;
    typedef LegacyLtcK2State State;

    State prepare(NclsScatteringContext context, CompiledMaterial material)
    {
        return {context, material};
    }
};

#endif
