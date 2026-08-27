// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module ncls_falcor2_mdl_oracle_query;

__exported import falcor2.render;
__exported import falcor2.utils;

public SurfaceInteraction ncls_mdl_surface(float3 position, float2 uv, float3 view_direction)
{
    SurfaceInteraction result = {};
    result.position_ws = position;
    result.wi_ws = view_direction;
    result.uv = uv;
    result.shading_frame_ws = Frame.identity();
    result.normal_ws = result.shading_frame_ws.normal;
    result.front_facing = dot(result.wi_ws, result.normal_ws) >= 0.0;
    result.ior = 1.0;
    return result;
}

public float4 ncls_mdl_evaluate<SampleGenerator : ISampleGenerator>(
    uint3 seeds,
    IMaterial material,
    float3 position,
    float2 uv,
    float3 view_direction,
    float3 light_direction
)
{
    SurfaceInteraction si = ncls_mdl_surface(position, uv, view_direction);
    var sample_generator = SampleGenerator(seeds.xy, seeds.z);
    let lod_sampler = ExplicitLodSampler(0);
    let instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);
    float3 value = instance.eval(si, light_direction, sample_generator);
    float pdf = instance.eval_pdf(si, light_direction);
    return float4(value, pdf);
}
