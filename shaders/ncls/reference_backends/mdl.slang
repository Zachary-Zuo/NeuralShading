#ifndef NCLS_REFERENCE_BACKENDS_MDL_SLANG
#define NCLS_REFERENCE_BACKENDS_MDL_SLANG

// NclsMdlGenerated 只包含锁定 SDK types、renderer callbacks 与该材质的 target code。
// 本文件是 formal query 与 viewer 共用的唯一 MDL scattering backend。
import NclsMdlGenerated;
#include "../contracts/scattering_backend.slang"

// 这些值属于锁定 mdl_target_code_types.hlsl 的 renderer ABI。预处理宏不会跨
// Slang module export，因此在 canonical backend 中显式命名一次。
static const uint NCLS_MDL_EVENT_ABSORB = 0u;
static const uint NCLS_MDL_EVENT_DIFFUSE = 1u;
static const uint NCLS_MDL_EVENT_GLOSSY = 2u;
static const uint NCLS_MDL_EVENT_SPECULAR = 4u;
static const uint NCLS_MDL_EVENT_TRANSMISSION = 16u;
static const int NCLS_MDL_ALLOW_REFLECT_AND_TRANSMIT = 3;

uint nclsMdlEventFlags(uint eventType)
{
    uint flags = 0u;
    flags |= (eventType & NCLS_MDL_EVENT_TRANSMISSION) != 0u
        ? (uint)NclsScatteringEvent::Transmission
        : (uint)NclsScatteringEvent::Reflection;
    if ((eventType & NCLS_MDL_EVENT_DIFFUSE) != 0u)
        flags |= (uint)NclsScatteringEvent::Diffuse;
    if ((eventType & NCLS_MDL_EVENT_GLOSSY) != 0u)
        flags |= (uint)NclsScatteringEvent::Glossy;
    if ((eventType & NCLS_MDL_EVENT_SPECULAR) != 0u)
        flags |= (uint)NclsScatteringEvent::Delta;
    return flags;
}

Shading_state_material nclsPrepareMdlTargetState(
    NclsSurfaceInteraction surface,
    float3 viewWorld)
{
    Shading_state_material state = (Shading_state_material)0;
    state.normal = normalize(surface.shadingFrame.normal);
    state.geom_normal = normalize(surface.geometricNormal);
    state.position = surface.position;
    state.animation_time = 0.0;
    state.text_coords[0] = float3(surface.uv, 0.0);
    state.tangent_u[0] = normalize(surface.shadingFrame.tangent);
    state.tangent_v[0] = normalize(surface.shadingFrame.bitangent);
    state.object_id = int(surface.primitiveId);
    state.meters_per_scene_unit = 1.0;
    state.arg_block_offset = 0;
    state.ro_data_segment_offset = 0;
    state.renderer_state.view_direction = normalize(viewWorld);
    state.world_to_object = float4x4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0);
    state.object_to_world = state.world_to_object;
    init(state);
    return state;
}

float nclsMdlTargetPdf(
    Shading_state_material state,
    float3 woWorld,
    float3 wiWorld)
{
    Bsdf_pdf_data data = (Bsdf_pdf_data)0;
    data.ior1 = float3(1.0);
    data.ior2 = ior(state);
    data.k1 = normalize(woWorld);
    data.k2 = normalize(wiWorld);
    data.flags = NCLS_MDL_ALLOW_REFLECT_AND_TRANSMIT;
    surface_scattering_pdf(data, state);
    return isfinite(data.pdf) && data.pdf >= 0.0 ? data.pdf : 0.0;
}

NclsScatteringPdf nclsMdlTargetPdfPair(
    Shading_state_material state,
    float3 woWorld,
    float3 wiWorld)
{
    return {
        nclsMdlTargetPdf(state, woWorld, wiWorld),
        nclsMdlTargetPdf(state, wiWorld, woWorld)};
}

struct NclsMdlCompiledMaterial
{
    uint reserved;
};

struct NclsMdlReferenceState : INclsScatteringState
{
    NclsScatteringContext context;

    NclsScatteringEval evaluate<S : ISampleGenerator>(float3 wiWorld, inout S sg)
    {
        NclsScatteringEval result = {};
        const Shading_state_material targetState = nclsPrepareMdlTargetState(
            context.surface, context.woWorld);
        // MDL target-code evaluate returns f * |dot(material Ns, wi)|.  The
        // public scattering contract reconstructs transport response with the
        // input NclsShadingFrame, so remove that input-frame cosine here.  This
        // keeps geometry.normal / normal-map semantics inside the equivalent f
        // instead of applying the material-normal cosine twice.
        const float lightCosine = abs(dot(context.surface.shadingFrame.normal, wiWorld));
        if (lightCosine <= 1e-8f) return result;
        Bsdf_evaluate_data data = (Bsdf_evaluate_data)0;
        data.ior1 = float3(1.0);
        data.ior2 = ior(targetState);
        data.k1 = normalize(context.woWorld);
        data.k2 = normalize(wiWorld);
        data.flags = NCLS_MDL_ALLOW_REFLECT_AND_TRANSMIT;
        surface_scattering_evaluate(data, targetState);
        result.f = (data.bsdf_diffuse + data.bsdf_glossy) / lightCosine;
        result.pdf = nclsMdlTargetPdfPair(targetState, context.woWorld, wiWorld);
        const bool transmission = dot(targetState.normal, data.k1)
            * dot(targetState.normal, data.k2) < 0.0f;
        result.eventFlags = transmission
            ? (uint)NclsScatteringEvent::Transmission | (uint)NclsScatteringEvent::Glossy
            : (uint)NclsScatteringEvent::Reflection | (uint)NclsScatteringEvent::Glossy;
        result.valid = all(isfinite(result.f)) && all(result.f >= 0.0f)
            && isfinite(result.pdf.forward) && isfinite(result.pdf.reverse) ? 1u : 0u;
        return result;
    }

    bool sample<S : ISampleGenerator>(out NclsScatteringSample result, inout S sg)
    {
        result = {};
        const Shading_state_material targetState = nclsPrepareMdlTargetState(
            context.surface, context.woWorld);
        Bsdf_sample_data data = (Bsdf_sample_data)0;
        data.ior1 = float3(1.0);
        data.ior2 = ior(targetState);
        data.k1 = normalize(context.woWorld);
        data.xi = sampleNext4D(sg);
        data.flags = NCLS_MDL_ALLOW_REFLECT_AND_TRANSMIT;
        surface_scattering_sample(data, targetState);

        const uint eventType = uint(data.event_type);
        if (eventType == NCLS_MDL_EVENT_ABSORB) return false;
        const float directionLengthSquared = dot(data.k2, data.k2);
        if (directionLengthSquared <= 1e-20f || !all(isfinite(data.k2))) return false;
        const uint eventFlags = nclsMdlEventFlags(eventType);
        const bool delta = (eventFlags & (uint)NclsScatteringEvent::Delta) != 0u;
        if (!isfinite(data.pdf) || (!delta && data.pdf <= 0.0) || (delta && data.pdf < 0.0))
            return false;
        result.wiWorld = data.k2 * rsqrt(directionLengthSquared);
        result.weight = data.bsdf_over_pdf;
        result.pdf.forward = data.pdf;
        result.pdf.reverse = delta ? 0.0f : nclsMdlTargetPdf(
            targetState, result.wiWorld, context.woWorld);
        result.eta = 1.0f;
        result.eventFlags = eventFlags;
        result.valid = all(isfinite(result.weight)) && all(result.weight >= 0.0f)
            && any(result.weight > 0.0f) ? 1u : 0u;
        return result.valid != 0u;
    }

    NclsScatteringPdf pdf(float3 wiWorld)
    {
        const Shading_state_material targetState = nclsPrepareMdlTargetState(
            context.surface, context.woWorld);
        return nclsMdlTargetPdfPair(targetState, context.woWorld, wiWorld);
    }
};

struct NclsMdlReferenceBackend : INclsScatteringBackend
{
    typedef NclsMdlCompiledMaterial CompiledMaterial;
    typedef NclsMdlReferenceState State;

    State prepare(NclsScatteringContext context, CompiledMaterial material)
    {
        return {context};
    }
};

#ifndef NCLS_REFERENCE_BACKEND_NO_PACKAGE_BINDING
typedef NclsMdlReferenceBackend NclsPackageBackend;
typedef NclsMdlCompiledMaterial NclsPackageCompiledMaterial;
typedef NclsMdlReferenceState NclsPackageState;
NclsPackageBackend nclsCreatePackageBackend() { return {}; }
NclsPackageCompiledMaterial nclsLoadPackageMaterial(uint index)
{
    return {index};
}
#endif

#endif
