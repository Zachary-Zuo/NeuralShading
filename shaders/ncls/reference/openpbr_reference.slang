#ifndef NCLS_OPENPBR_REFERENCE_SLANG
#define NCLS_OPENPBR_REFERENCE_SLANG

Texture3D<float4> gOpenPbrIdealDielectricEnergy;
Texture2D<float4> gOpenPbrIdealDielectricAverage;
Texture2D<float4> gOpenPbrIdealDielectricRatio;
Texture3D<float4> gOpenPbrOpaqueDielectricEnergy;
Texture2D<float4> gOpenPbrOpaqueDielectricAverage;
Texture2D<float4> gOpenPbrIdealMetalEnergy;
Texture2D<float4> gOpenPbrIdealMetalAverage;
Texture2D<float4> gOpenPbrLtc;
SamplerState gOpenPbrLutSampler;

float4 nclsOpenPbrSample2D(int lutId, float2 uv);
float4 nclsOpenPbrSample3D(int lutId, float3 uvw);

#define OPENPBR_LANGUAGE_TARGET_SLANG 1
#define OPENPBR_USE_TEXTURE_LUTS 1
#define OPENPBR_SAMPLE_2D_TEXTURE(lut_id, uv) nclsOpenPbrSample2D(lut_id, uv)
#define OPENPBR_SAMPLE_3D_TEXTURE(lut_id, uvw) nclsOpenPbrSample3D(lut_id, uvw)
#include "../../../external/openpbr-bsdf/openpbr.h"

float4 nclsOpenPbrSample2D(int lutId, float2 uv)
{
    switch (lutId)
    {
    case 1: return gOpenPbrIdealDielectricAverage.SampleLevel(gOpenPbrLutSampler, uv, 0.0f);
    case 2: return gOpenPbrIdealDielectricRatio.SampleLevel(gOpenPbrLutSampler, uv, 0.0f);
    case 4: return gOpenPbrOpaqueDielectricAverage.SampleLevel(gOpenPbrLutSampler, uv, 0.0f);
    case 5: return gOpenPbrIdealMetalEnergy.SampleLevel(gOpenPbrLutSampler, uv, 0.0f);
    case 6: return gOpenPbrIdealMetalAverage.SampleLevel(gOpenPbrLutSampler, uv, 0.0f);
    case 7: return gOpenPbrLtc.SampleLevel(gOpenPbrLutSampler, uv, 0.0f);
    default: return float4(0.0f);
    }
}

float4 nclsOpenPbrSample3D(int lutId, float3 uvw)
{
    if (lutId == 0) return gOpenPbrIdealDielectricEnergy.SampleLevel(gOpenPbrLutSampler, uvw, 0.0f);
    if (lutId == 3) return gOpenPbrOpaqueDielectricEnergy.SampleLevel(gOpenPbrLutSampler, uvw, 0.0f);
    return float4(0.0f);
}

static const uint NCLS_OPENPBR_RESOLVED_FLOAT_COUNT = 77u;

float3 nclsOpenPbrReadFloat3(StructuredBuffer<float> values, inout uint cursor)
{
    const float3 result = float3(values[cursor], values[cursor + 1u], values[cursor + 2u]);
    cursor += 3u;
    return result;
}

float2 nclsOpenPbrReadFloat2(StructuredBuffer<float> values, inout uint cursor)
{
    const float2 result = float2(values[cursor], values[cursor + 1u]);
    cursor += 2u;
    return result;
}

OpenPBR_ResolvedInputs nclsOpenPbrResolvedInputs(StructuredBuffer<float> values)
{
    OpenPBR_ResolvedInputs inputs;
    uint cursor = 0u;
    inputs.base_weight = values[cursor++];
    inputs.base_color = nclsOpenPbrReadFloat3(values, cursor);
    inputs.base_diffuse_roughness = values[cursor++];
    inputs.base_metalness = values[cursor++];
    inputs.subsurface_weight = values[cursor++];
    inputs.subsurface_color = nclsOpenPbrReadFloat3(values, cursor);
    inputs.subsurface_radius = values[cursor++];
    inputs.subsurface_radius_scale = nclsOpenPbrReadFloat3(values, cursor);
    inputs.subsurface_scatter_anisotropy = values[cursor++];
    inputs.specular_weight = values[cursor++];
    inputs.specular_color = nclsOpenPbrReadFloat3(values, cursor);
    inputs.specular_roughness = values[cursor++];
    inputs.specular_roughness_anisotropy = values[cursor++];
    inputs.specular_ior = values[cursor++];
    inputs.specular_anisotropy_rotation_cos_sin = nclsOpenPbrReadFloat2(values, cursor);
    inputs.coat_weight = values[cursor++];
    inputs.coat_color = nclsOpenPbrReadFloat3(values, cursor);
    inputs.coat_roughness = values[cursor++];
    inputs.coat_roughness_anisotropy = values[cursor++];
    inputs.coat_ior = values[cursor++];
    inputs.coat_darkening = values[cursor++];
    inputs.coat_anisotropy_rotation_cos_sin = nclsOpenPbrReadFloat2(values, cursor);
    inputs.fuzz_weight = values[cursor++];
    inputs.fuzz_color = nclsOpenPbrReadFloat3(values, cursor);
    inputs.fuzz_roughness = values[cursor++];
    inputs.transmission_weight = values[cursor++];
    inputs.transmission_color = nclsOpenPbrReadFloat3(values, cursor);
    inputs.transmission_depth = values[cursor++];
    inputs.transmission_scatter = nclsOpenPbrReadFloat3(values, cursor);
    inputs.transmission_scatter_anisotropy = values[cursor++];
    inputs.transmission_dispersion_scale = values[cursor++];
    inputs.transmission_dispersion_abbe_number = values[cursor++];
    inputs.thin_film_weight = values[cursor++];
    inputs.thin_film_thickness = values[cursor++];
    inputs.thin_film_ior = values[cursor++];
    inputs.emission_luminance = values[cursor++];
    inputs.emission_color = nclsOpenPbrReadFloat3(values, cursor);
    inputs.geometry_opacity = values[cursor++];
    inputs.geometry_thin_walled = values[cursor++] != 0.0f;
    inputs.geometry_basis.t = nclsOpenPbrReadFloat3(values, cursor);
    inputs.geometry_basis.b = nclsOpenPbrReadFloat3(values, cursor);
    inputs.geometry_basis.n = nclsOpenPbrReadFloat3(values, cursor);
    inputs.geometry_coat_basis.t = nclsOpenPbrReadFloat3(values, cursor);
    inputs.geometry_coat_basis.b = nclsOpenPbrReadFloat3(values, cursor);
    inputs.geometry_coat_basis.n = nclsOpenPbrReadFloat3(values, cursor);
    return inputs;
}

OpenPBR_PreparedBsdf nclsPrepareOpenPbrReference(
    StructuredBuffer<float> values,
    float3 viewDirection)
{
    return openpbr_prepare(
        nclsOpenPbrResolvedInputs(values),
        float3(1.0f),
        OpenPBR_BaseRgbWavelengths_nm,
        OpenPBR_VacuumIor,
        viewDirection);
}

float3 nclsEvalOpenPbrReference(
    StructuredBuffer<float> values,
    float3 viewDirection,
    float3 lightDirection)
{
    const OpenPBR_PreparedBsdf prepared = nclsPrepareOpenPbrReference(values, viewDirection);
    return openpbr_get_sum_of_diffuse_specular(openpbr_eval(prepared, lightDirection));
}

float3 nclsOpenPbrEmission(StructuredBuffer<float> values, float3 viewDirection)
{
    return nclsPrepareOpenPbrReference(values, viewDirection).emission;
}

#endif
