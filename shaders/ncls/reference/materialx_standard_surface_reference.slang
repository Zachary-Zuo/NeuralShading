#ifndef NCLS_MATERIALX_STANDARD_SURFACE_REFERENCE_SLANG
#define NCLS_MATERIALX_STANDARD_SURFACE_REFERENCE_SLANG

// Locked MaterialX 1.39.4 standard_surface surface-response subset used by the
// curated Poly Haven documents. These equations follow the GLSL emitted by the
// pinned upstream generator. Source .mtlx documents and original textures remain
// the ground truth; this is a family-specific Falcor execution backend.

static const float NCLS_MX_PI = 3.14159265358979323846f;
static const float NCLS_MX_EPS = 1e-8f;

struct NclsMaterialXResolvedSurface
{
    float3 baseColor;
    float metalness;
    float specularRoughness;
};

float nclsMxSquare(float value) { return value * value; }
float3 nclsMxSquare(float3 value) { return value * value; }

float3 nclsMxSrgbTextureToLinearRec709(float3 value)
{
    const float3 linearSegment = value / 12.92f;
    const float3 powerSegment = pow(max((value + 0.055f) / 1.055f, 0.0f), 2.4f);
    return float3(
        value.x > 0.04045f ? powerSegment.x : linearSegment.x,
        value.y > 0.04045f ? powerSegment.y : linearSegment.y,
        value.z > 0.04045f ? powerSegment.z : linearSegment.z);
}

float3 nclsMxForwardFacingNormal(float3 normal, float3 view)
{
    return dot(normal, view) < 0.0f ? -normal : normal;
}

float nclsMxGgxNdf(float3 halfVector, float2 alpha)
{
    const float2 ellipse = halfVector.xy / alpha;
    const float denominator = dot(ellipse, ellipse) + nclsMxSquare(halfVector.z);
    return 1.0f / (NCLS_MX_PI * alpha.x * alpha.y * nclsMxSquare(denominator));
}

float nclsMxGgxSmithG2(float normalLight, float normalView, float alpha)
{
    const float alpha2 = nclsMxSquare(alpha);
    const float lambdaLight = sqrt(alpha2 + (1.0f - alpha2) * nclsMxSquare(normalLight));
    const float lambdaView = sqrt(alpha2 + (1.0f - alpha2) * nclsMxSquare(normalView));
    return 2.0f * normalLight * normalView
        / (lambdaLight * normalView + lambdaView * normalLight);
}

float3 nclsMxGgxDirectionalAlbedo(float normalView, float alpha, float3 f0, float3 f90)
{
    const float x = normalView;
    const float y = alpha;
    const float x2 = nclsMxSquare(x);
    const float y2 = nclsMxSquare(y);
    const float4 r = float4(0.1003f, 0.9345f, 1.0f, 1.0f)
        + float4(-0.6303f, -2.323f, -1.765f, 0.2281f) * x
        + float4(9.748f, 2.229f, 8.263f, 15.94f) * y
        + float4(-2.038f, -3.748f, 11.53f, -55.83f) * x * y
        + float4(29.34f, 1.424f, 28.96f, 13.08f) * x2
        + float4(-8.245f, -0.7684f, -7.507f, 41.26f) * y2
        + float4(-26.44f, 1.436f, -36.11f, 54.9f) * x2 * y
        + float4(19.99f, 0.2913f, 15.86f, 300.2f) * x * y2
        + float4(-5.448f, 0.6286f, 33.37f, -285.1f) * x2 * y2;
    const float2 ab = saturate(r.xy / r.zw);
    return f0 * ab.x + f90 * ab.y;
}

float3 nclsMxGgxEnergyCompensation(float normalView, float alpha, float3 singleScatterFresnel)
{
    const float singleScatterAlbedo = nclsMxGgxDirectionalAlbedo(
        normalView, alpha, float3(1.0f), float3(1.0f)).x;
    return 1.0f + singleScatterFresnel * (1.0f - singleScatterAlbedo) / singleScatterAlbedo;
}

float nclsMxIorToF0(float ior)
{
    return nclsMxSquare((ior - 1.0f) / (ior + 1.0f));
}

float nclsMxFresnelDielectric(float cosine, float ior)
{
    const float g2 = ior * ior + cosine * cosine - 1.0f;
    if (g2 < 0.0f) return 1.0f;
    const float g = sqrt(g2);
    return 0.5f * nclsMxSquare((g - cosine) / (g + cosine))
        * (1.0f + nclsMxSquare(((g + cosine) * cosine - 1.0f) / ((g - cosine) * cosine + 1.0f)));
}

float3 nclsMxFresnelConductor(float cosine, float3 ior, float3 extinction)
{
    const float cosine2 = nclsMxSquare(saturate(cosine));
    const float sine2 = 1.0f - cosine2;
    const float3 ior2 = ior * ior;
    const float3 extinction2 = extinction * extinction;
    const float3 t0 = ior2 - extinction2 - sine2;
    const float3 a2plusb2 = sqrt(t0 * t0 + 4.0f * ior2 * extinction2);
    const float3 t1 = a2plusb2 + cosine2;
    const float3 a = sqrt(max(0.5f * (a2plusb2 + t0), 0.0f));
    const float3 t2 = 2.0f * a * cosine;
    const float3 rs = (t1 - t2) / (t1 + t2);
    const float3 t3 = cosine2 * a2plusb2 + sine2 * sine2;
    const float3 t4 = t2 * sine2;
    const float3 rp = rs * (t3 - t4) / (t3 + t4);
    return 0.5f * (rp + rs);
}

void nclsMxArtisticIor(float3 reflectivity, float3 edgeColor, out float3 ior, out float3 extinction)
{
    const float3 r = clamp(reflectivity, 0.0f, 0.99f);
    const float3 root = sqrt(r);
    const float3 minimumIor = (1.0f - r) / (1.0f + r);
    const float3 maximumIor = (1.0f + root) / (1.0f - root);
    ior = lerp(maximumIor, minimumIor, edgeColor);
    const float3 plus = ior + 1.0f;
    const float3 minus = ior - 1.0f;
    const float3 extinction2 = max((plus * plus * r - minus * minus) / (1.0f - r), 0.0f);
    extinction = sqrt(extinction2);
}

float2 nclsMxRoughnessAnisotropy(float roughness, float anisotropy)
{
    const float squared = clamp(roughness * roughness, NCLS_MX_EPS, 1.0f);
    if (anisotropy <= 0.0f) return float2(squared);
    const float aspect = sqrt(1.0f - clamp(anisotropy, 0.0f, 0.98f));
    return float2(min(squared / aspect, 1.0f), squared * aspect);
}

float nclsMxOrenNayar(float normalView, float normalLight, float lightView, float roughness)
{
    const float s = lightView - normalLight * normalView;
    const float stinv = s > 0.0f ? s / max(normalLight, normalView) : 0.0f;
    const float sigma2 = roughness * roughness;
    const float a = 1.0f - 0.5f * (sigma2 / (sigma2 + 0.33f));
    const float b = 0.45f * sigma2 / (sigma2 + 0.09f);
    return a + b * stinv;
}

float2 nclsMaterialXSphereUv(float3 position)
{
    const float3 direction = normalize(position);
    return float2(
        frac(atan2(-direction.x, -direction.z) / (2.0f * NCLS_MX_PI)),
        asin(clamp(direction.y, -1.0f, 1.0f)) / NCLS_MX_PI + 0.5f);
}

NclsMaterialXResolvedSurface nclsResolveMaterialXSurface(
    StructuredBuffer<float> inputs,
    Texture2D<float4> baseColorTexture,
    Texture2D<float4> roughnessTexture,
    Texture2D<float4> metalnessTexture,
    SamplerState materialSampler,
    float2 uv,
    float2 uvDx,
    float2 uvDy)
{
    NclsMaterialXResolvedSurface result;
    result.baseColor = float3(inputs[1], inputs[2], inputs[3]);
    result.metalness = inputs[5];
    result.specularRoughness = inputs[12];
    if (inputs[6] != 0.0f)
        result.baseColor = nclsMxSrgbTextureToLinearRec709(
            baseColorTexture.SampleGrad(materialSampler, uv, uvDx, uvDy).rgb);
    if (inputs[7] != 0.0f)
        result.metalness = metalnessTexture.SampleGrad(materialSampler, uv, uvDx, uvDy).r;
    if (inputs[13] != 0.0f)
        result.specularRoughness = roughnessTexture.SampleGrad(materialSampler, uv, uvDx, uvDy).r;
    return result;
}

float3 nclsMaterialXShadingNormal(
    StructuredBuffer<float> inputs,
    Texture2D<float4> normalTexture,
    SamplerState materialSampler,
    float2 uv,
    float2 uvDx,
    float2 uvDy,
    float3 geometricNormal,
    float3 geometricTangent)
{
    if (inputs[18] == 0.0f) return geometricNormal;
    float3 value = normalTexture.SampleGrad(materialSampler, uv, uvDx, uvDy).rgb;
    value = dot(value, value) == 0.0f ? float3(0.0f, 0.0f, 1.0f) : value * 2.0f - 1.0f;
    const float normalScale = inputs[17];
    const float3 bitangent = cross(geometricNormal, geometricTangent);
    return normalize(
        geometricTangent * value.x * normalScale
        + bitangent * value.y * normalScale
        + geometricNormal * value.z);
}

float3 nclsEvalMaterialXStandardSurface(
    StructuredBuffer<float> inputs,
    NclsMaterialXResolvedSurface surface,
    float3 view,
    float3 light)
{
    const float normalView = clamp(view.z, NCLS_MX_EPS, 1.0f);
    const float normalLight = clamp(light.z, NCLS_MX_EPS, 1.0f);
    const float3 halfVector = normalize(light + view);
    const float viewHalf = clamp(dot(view, halfVector), NCLS_MX_EPS, 1.0f);
    const float anisotropy = inputs[15];
    const float rotation = inputs[16] * (2.0f * NCLS_MX_PI);
    const float cosine = cos(rotation);
    const float sine = sin(rotation);
    const float3 tangent = float3(cosine, sine, 0.0f);
    const float3 bitangent = float3(-sine, cosine, 0.0f);
    const float3 tangentHalf = float3(dot(halfVector, tangent), dot(halfVector, bitangent), halfVector.z);
    const float2 alpha = nclsMxRoughnessAnisotropy(surface.specularRoughness, anisotropy);
    const float averageAlpha = sqrt(alpha.x * alpha.y);
    const float distribution = nclsMxGgxNdf(tangentHalf, alpha);
    const float geometry = nclsMxGgxSmithG2(normalLight, normalView, averageAlpha);

    const float specularWeight = inputs[8];
    const float3 specularColor = max(float3(inputs[9], inputs[10], inputs[11]), 0.0f);
    const float specularIor = inputs[14];
    const float dielectricFresnel = nclsMxFresnelDielectric(viewHalf, specularIor);
    const float3 dielectricCompensation = nclsMxGgxEnergyCompensation(
        normalView, averageAlpha, float3(dielectricFresnel));
    const float dielectricF0 = nclsMxIorToF0(specularIor);
    const float3 dielectricAlbedo = nclsMxGgxDirectionalAlbedo(
        normalView, averageAlpha, float3(dielectricF0), float3(1.0f)) * dielectricCompensation;
    const float3 dielectricThroughput = 1.0f - dielectricAlbedo * specularWeight;
    const float3 dielectricResponse = distribution * dielectricFresnel * geometry
        * dielectricCompensation * specularColor * specularWeight / (4.0f * normalView);

    const float diffuseWeight = inputs[0];
    const float diffuseRoughness = inputs[4];
    const float lightView = clamp(dot(light, view), NCLS_MX_EPS, 1.0f);
    const float3 diffuseColor = max(surface.baseColor, 0.0f);
    const float3 diffuseResponse = nclsMxOrenNayar(
        normalView, normalLight, lightView, diffuseRoughness)
        * diffuseColor * diffuseWeight * normalLight / NCLS_MX_PI;
    const float3 dielectricLayer = dielectricResponse + diffuseResponse * dielectricThroughput;

    float3 conductorIor;
    float3 conductorExtinction;
    nclsMxArtisticIor(surface.baseColor * diffuseWeight, specularColor * specularWeight,
        conductorIor, conductorExtinction);
    const float3 conductorFresnel = nclsMxFresnelConductor(viewHalf, conductorIor, conductorExtinction);
    const float3 conductorCompensation = nclsMxGgxEnergyCompensation(
        normalView, averageAlpha, conductorFresnel);
    const float3 conductorResponse = distribution * conductorFresnel * geometry
        * conductorCompensation * surface.metalness / (4.0f * normalView);

    return (conductorResponse + dielectricLayer * saturate(1.0f - surface.metalness)) * inputs[23];
}

float3 nclsMaterialXEmission(StructuredBuffer<float> inputs)
{
    return inputs[19] * float3(inputs[20], inputs[21], inputs[22]) * inputs[23];
}

#endif
