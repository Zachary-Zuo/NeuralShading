// Appended to the material-specific named string module after the pinned MDL
// target-code types, project renderer callbacks, and SDK generated HLSL.

struct NclsMdlSample
{
    uint valid;
    float3 directionWorld;
    float3 weight;
    float pdf;
    uint eventType;
    uint transmission;
};

Shading_state_material nclsMdlPrepareState(
    float3 position,
    float3 normal,
    float3 tangent,
    float2 texCoord,
    float3 viewWorld)
{
    Shading_state_material state = (Shading_state_material)0;
    state.normal = normalize(normal);
    state.geom_normal = state.normal;
    state.position = position;
    state.animation_time = 0.0;
    state.text_coords[0] = float3(texCoord, 0.0);
    state.tangent_u[0] = normalize(tangent);
    state.tangent_v[0] = normalize(cross(state.normal, state.tangent_u[0]));
    state.object_id = 0;
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

float3 nclsMdlEvaluateSurface(
    float3 position,
    float3 normal,
    float3 tangent,
    float2 texCoord,
    float3 viewWorld,
    float3 lightWorld,
    out float pdf)
{
    const Shading_state_material state = nclsMdlPrepareState(
        position, normal, tangent, texCoord, viewWorld);

    Bsdf_evaluate_data data = (Bsdf_evaluate_data)0;
    data.ior1 = float3(1.0);
    data.ior2 = ior(state);
    data.k1 = normalize(viewWorld);
    data.k2 = normalize(lightWorld);
    data.flags = DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;
    surface_scattering_evaluate(data, state);

    Bsdf_pdf_data pdfData = (Bsdf_pdf_data)0;
    pdfData.ior1 = data.ior1;
    pdfData.ior2 = data.ior2;
    pdfData.k1 = data.k1;
    pdfData.k2 = data.k2;
    pdfData.flags = DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;
    surface_scattering_pdf(pdfData, state);
    pdf = pdfData.pdf;
    return data.bsdf_diffuse + data.bsdf_glossy;
}

float nclsMdlPdfSurface(
    float3 position,
    float3 normal,
    float3 tangent,
    float2 texCoord,
    float3 viewWorld,
    float3 lightWorld)
{
    const Shading_state_material state = nclsMdlPrepareState(
        position, normal, tangent, texCoord, viewWorld);
    Bsdf_pdf_data data = (Bsdf_pdf_data)0;
    data.ior1 = float3(1.0);
    data.ior2 = ior(state);
    data.k1 = normalize(viewWorld);
    data.k2 = normalize(lightWorld);
    data.flags = DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;
    surface_scattering_pdf(data, state);
    return data.pdf;
}

NclsMdlSample nclsMdlSampleSurface(
    float3 position,
    float3 normal,
    float3 tangent,
    float2 texCoord,
    float3 viewWorld,
    float4 xi)
{
    NclsMdlSample result = (NclsMdlSample)0;
    const Shading_state_material state = nclsMdlPrepareState(
        position, normal, tangent, texCoord, viewWorld);
    Bsdf_sample_data data = (Bsdf_sample_data)0;
    data.ior1 = float3(1.0);
    data.ior2 = ior(state);
    data.k1 = normalize(viewWorld);
    data.xi = xi;
    data.flags = DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;
    surface_scattering_sample(data, state);

    const float directionLengthSquared = dot(data.k2, data.k2);
    result.directionWorld = directionLengthSquared > 1e-20
        ? data.k2 * rsqrt(directionLengthSquared) : float3(0.0);
    result.weight = data.bsdf_over_pdf;
    result.pdf = data.pdf;
    result.eventType = uint(data.event_type);
    result.transmission = (result.eventType & uint(BSDF_EVENT_TRANSMISSION)) != 0u ? 1u : 0u;
    const bool delta = (result.eventType & uint(BSDF_EVENT_SPECULAR)) != 0u;
    const bool pdfValid = isfinite(result.pdf)
        && (result.pdf > 0.0 || (delta && result.pdf >= 0.0));
    result.valid = result.eventType != uint(BSDF_EVENT_ABSORB)
        && directionLengthSquared > 1e-20
        && all(isfinite(data.k2))
        && all(isfinite(result.weight))
        && any(result.weight > 0.0)
        && pdfValid
        ? 1u : 0u;
    return result;
}
