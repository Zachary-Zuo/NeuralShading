#ifndef NCLS_REFERENCE_BACKENDS_OPENPBR_SLANG
#define NCLS_REFERENCE_BACKENDS_OPENPBR_SLANG

// OpenPBR 1.1.1 reference 的统一 scattering contract 包装。
#include "../reference/openpbr_reference.slang"
#include "../scattering/frame.slang"
#include "../contracts/scattering_backend.slang"

struct NclsOpenPbrReferenceState : INclsScatteringState
{
    NclsScatteringContext context;
    StructuredBuffer<float> material;
    OpenPBR_PreparedBsdf prepared;

    NclsScatteringEval evaluate<S : ISampleGenerator>(float3 wiWorld, inout S sg)
    {
        NclsScatteringEval result = {};
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        if (abs(wi.z) <= NCLS_MIN_COS) return result;
        result.f = openpbr_get_sum_of_diffuse_specular(openpbr_eval(prepared, wi)) / abs(wi.z);
        result.pdf.forward = openpbr_pdf(prepared, wi);
        result.pdf.reverse = openpbr_pdf(nclsPrepareOpenPbrReference(material, wi), wo);
        result.eventFlags = wi.z >= 0.0f ? (uint)NclsScatteringEvent::Reflection : (uint)NclsScatteringEvent::Transmission;
        result.valid = all(isfinite(result.f)) && all(result.f >= 0.0f)
            && result.pdf.forward >= 0.0f && result.pdf.reverse >= 0.0f ? 1u : 0u;
        return result;
    }

    bool sample<S : ISampleGenerator>(out NclsScatteringSample result, inout S sg)
    {
        result = {};
        const float3 randomValue = sampleNext3D(sg);
        float3 wi;
        OpenPBR_DiffuseSpecular weight;
        float forwardPdf;
        OpenPBR_BsdfLobeType sampledType;
        openpbr_sample(prepared, randomValue, wi, weight, forwardPdf, sampledType);
        const bool delta = bool(sampledType & OpenPBR_BsdfLobeTypeSpecular);
        if (!isfinite(forwardPdf) || (!delta && forwardPdf <= 0.0f)
            || (delta && forwardPdf < 0.0f) || abs(wi.z) <= NCLS_MIN_COS) return false;
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        result.wiWorld = nclsFrameToWorld(context.surface.shadingFrame, wi);
        // Preserve Adobe's native sample weight and accumulated PDF. Reconstructing
        // a very narrow, grazing microfacet half-vector from the rounded output wi
        // can catastrophically cancel tangential components in openpbr_pdf() even
        // though the sampler still has a stable half-vector and finite weight.
        result.weight = openpbr_get_sum_of_diffuse_specular(weight);
        result.pdf = {
            forwardPdf,
            delta ? 0.0f : openpbr_pdf(nclsPrepareOpenPbrReference(material, wi), wo)};
        result.eta = 1.0f;
        result.eventFlags = bool(sampledType & OpenPBR_BsdfLobeTypeTransmission)
            ? (uint)NclsScatteringEvent::Transmission
            : (uint)NclsScatteringEvent::Reflection;
        if (bool(sampledType & OpenPBR_BsdfLobeTypeDiffuse))
            result.eventFlags |= (uint)NclsScatteringEvent::Diffuse;
        if (bool(sampledType & OpenPBR_BsdfLobeTypeGlossy))
            result.eventFlags |= (uint)NclsScatteringEvent::Glossy;
        if (delta) result.eventFlags |= (uint)NclsScatteringEvent::Delta;
        result.valid = all(isfinite(result.weight)) && all(result.weight >= 0.0f)
            && any(result.weight > 0.0f) ? 1u : 0u;
        return result.valid != 0u;
    }

    NclsScatteringPdf pdf(float3 wiWorld)
    {
        const float3 wi = nclsFrameToLocal(context.surface.shadingFrame, wiWorld);
        if (abs(wi.z) <= NCLS_MIN_COS) return {};
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        return {openpbr_pdf(prepared, wi), openpbr_pdf(nclsPrepareOpenPbrReference(material, wi), wo)};
    }
};

struct NclsOpenPbrReferenceBackend : INclsScatteringBackend
{
    typedef StructuredBuffer<float> CompiledMaterial;
    typedef NclsOpenPbrReferenceState State;
    State prepare(NclsScatteringContext context, CompiledMaterial material)
    {
        const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
        return {context, material, nclsPrepareOpenPbrReference(material, wo)};
    }
};

#ifndef NCLS_REFERENCE_BACKEND_NO_PACKAGE_BINDING
typedef NclsOpenPbrReferenceBackend NclsPackageBackend;
typedef StructuredBuffer<float> NclsPackageCompiledMaterial;
typedef NclsOpenPbrReferenceState NclsPackageState;
StructuredBuffer<float> gNclsCompiledMaterialValues;
NclsPackageBackend nclsCreatePackageBackend() { return {}; }
NclsPackageCompiledMaterial nclsLoadPackageMaterial(uint index)
{
    // OpenPBR package 当前每个 material asset 有独立 77-float buffer。
    return gNclsCompiledMaterialValues;
}
#endif

#endif
