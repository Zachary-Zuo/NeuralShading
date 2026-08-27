// The host composes this adapter after the pinned MDL target-code types,
// project renderer callbacks, and material-specific generated HLSL.

StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gLights;
StructuredBuffer<float4> gPositions;
StructuredBuffer<float4> gUv;
RWStructuredBuffer<float4> gOutput;
uniform uint gQueryCount;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    if (index >= gQueryCount)
        return;

    Shading_state_material state = (Shading_state_material)0;
    state.normal = float3(0.0, 0.0, 1.0);
    state.geom_normal = state.normal;
    state.position = gPositions[index].xyz;
    state.animation_time = 0.0;
    state.text_coords[0] = float3(gUv[index].xy, 0.0);
    state.tangent_u[0] = float3(1.0, 0.0, 0.0);
    state.tangent_v[0] = float3(0.0, 1.0, 0.0);
    state.object_id = 0;
    state.meters_per_scene_unit = 1.0;
    state.arg_block_offset = 0;
    state.ro_data_segment_offset = 0;
    state.renderer_state.view_direction = normalize(gViews[index].xyz);
    state.world_to_object = float4x4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    state.object_to_world = state.world_to_object;

    // falcor2's MDL material setup invokes the link-unit init target.  This is
    // not interchangeable with the per-DF surface_scattering_init function:
    // init evaluates shared expressions and applies geometry.normal.
    init(state);

    Bsdf_evaluate_data data = (Bsdf_evaluate_data)0;
    data.ior1 = float3(1.0);
    data.ior2 = ior(state);
    data.k1 = normalize(gViews[index].xyz);
    data.k2 = normalize(gLights[index].xyz);
    data.flags = DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;
    surface_scattering_evaluate(data, state);

    Bsdf_pdf_data pdfData = (Bsdf_pdf_data)0;
    pdfData.ior1 = data.ior1;
    pdfData.ior2 = data.ior2;
    pdfData.k1 = data.k1;
    pdfData.k2 = data.k2;
    surface_scattering_pdf(pdfData, state);

    gOutput[index] = float4(data.bsdf_diffuse + data.bsdf_glossy, pdfData.pdf);
}
