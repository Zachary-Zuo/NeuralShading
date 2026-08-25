#ifndef NCLS_REFERENCE_INTERFACES_SLANG
#define NCLS_REFERENCE_INTERFACES_SLANG

#include "../contracts/layer_stack_ir.slang"
#include "sampling.slang"

static const uint NCLS_SAMPLE_REFLECTION = 1u;
static const uint NCLS_SAMPLE_TRANSMISSION = 2u;
static const uint NCLS_SAMPLE_ALL = 3u;

struct LayerBsdfSample
{
    float3 direction;
    float3 f;
    float3 weight;
    float pdf;
    uint isReflection;
    uint valid;
};

float2 nclsLayerAlpha(const NclsLayerInterfaceIR layer)
{
    return max(float2(layer.alphaX, layer.alphaY), float2(1e-3f));
}

float3 nclsLayerEta(const NclsLayerInterfaceIR layer)
{
    return float3(layer.etaR, layer.etaG, layer.etaB);
}

float3 nclsLayerK(const NclsLayerInterfaceIR layer)
{
    return float3(layer.kR, layer.kG, layer.kB);
}

float3 nclsLayerColor(const NclsLayerInterfaceIR layer)
{
    return float3(layer.colorR, layer.colorG, layer.colorB);
}

float nclsCharlieD(float normalDotHalf, float roughness)
{
    const float inverseAlpha = 1.0f / max(roughness, 1e-3f);
    const float sinThetaSquared = max(1.0f - normalDotHalf * normalDotHalf, 0.0078125f);
    return (2.0f + inverseAlpha) * pow(sinThetaSquared, 0.5f * inverseAlpha) / (2.0f * NCLS_PI);
}

float nclsSheenL(float cosine, float roughness)
{
    cosine = clamp(cosine, 0.0f, 1.0f);
    const float r = (1.0f - roughness) * (1.0f - roughness);
    const float oneMinusR = 1.0f - r;
    const float a = 25.3245f * r + 21.5473f * oneMinusR;
    const float b = 3.32435f * r + 3.82987f * oneMinusR;
    const float c = 0.16801f * r + 0.19823f * oneMinusR;
    const float d = -1.27393f * r - 1.97760f * oneMinusR;
    const float e = -4.85967f * r - 4.32054f * oneMinusR;
    return a / (1.0f + b * pow(cosine, c)) + d * cosine + e;
}

float nclsSheenLambda(float cosine, float roughness)
{
    cosine = clamp(cosine, 0.0f, 1.0f);
    return cosine < 0.5f
        ? exp(nclsSheenL(cosine, roughness))
        : exp(2.0f * nclsSheenL(0.5f, roughness) - nclsSheenL(1.0f - cosine, roughness));
}

float nclsSheenG(float wiCosine, float woCosine, float roughness)
{
    wiCosine = clamp(wiCosine, 0.0f, 1.0f);
    woCosine = clamp(woCosine, 0.0f, 1.0f);
    const float softened = pow(nclsSheenLambda(woCosine, roughness), 1.0f + 2.0f * pow(1.0f - woCosine, 8.0f));
    return 1.0f / (1.0f + softened + nclsSheenLambda(wiCosine, roughness));
}

float3 nclsEvalInterfaceLocal(
    const NclsLayerInterfaceIR layer,
    const float3 wi,
    const float3 wo,
    const float etaI,
    const float etaT)
{
    if (abs(wi.z) < NCLS_MIN_COS || abs(wo.z) < NCLS_MIN_COS) return float3(0.0f);

    if (layer.kind == (uint)NclsInterfaceKind::Diffuse)
    {
        return wi.z > 0.0f && wo.z > 0.0f ? nclsLayerColor(layer) * NCLS_INV_PI : float3(0.0f);
    }

    if (layer.kind == (uint)NclsInterfaceKind::Sheen)
    {
        if (wi.z <= 0.0f || wo.z <= 0.0f) return float3(0.0f);
        const float3 h = nclsSafeNormalize(wi + wo);
        const float roughness = max(layer.alphaX, 1e-3f);
        return nclsLayerColor(layer) * nclsCharlieD(h.z, roughness)
            * nclsSheenG(wi.z, wo.z, roughness) / (4.0f * wi.z * wo.z);
    }

    const float2 alpha = nclsLayerAlpha(layer);
    if (layer.kind == (uint)NclsInterfaceKind::RoughConductor)
    {
        if (wi.z <= 0.0f || wo.z <= 0.0f) return float3(0.0f);
        const float3 h = nclsSafeNormalize(wi + wo);
        return nclsFresnelConductor(nclsLayerEta(layer), nclsLayerK(layer), dot(wi, h))
            * nclsGgxD(h, alpha) * nclsGgxG2(wi, wo, alpha) / (4.0f * wi.z * wo.z);
    }

    if (layer.kind == (uint)NclsInterfaceKind::RoughDielectric)
    {
        const float relativeEta = etaT / etaI;
        const bool reflection = wi.z * wo.z > 0.0f;
        if (abs(relativeEta - 1.0f) < 1e-6f) return float3(0.0f);
        float3 h = reflection ? nclsSafeNormalize(wi + wo) : nclsSafeNormalize(wi / relativeEta + wo);
        h *= h.z >= 0.0f ? 1.0f : -1.0f;
        const float wiDotH = dot(wi, h);
        const float woDotH = dot(wo, h);
        if (wiDotH * wi.z < 0.0f || woDotH * wo.z < 0.0f) return float3(0.0f);
        const float fresnel = nclsFresnelDielectric(1.0f / relativeEta, wiDotH);
        const float common = nclsGgxD(h, alpha) * nclsGgxG2(wi, wo, alpha);
        if (reflection)
            return float3(common * fresnel / (4.0f * abs(wi.z * wo.z)));
        const float denominator = max(abs(woDotH + wiDotH / relativeEta), 1e-10f);
        return float3(common * (1.0f - fresnel)
            * abs(wiDotH * woDotH / (denominator * denominator * wi.z * wo.z)));
    }
    return float3(0.0f);
}

float nclsInterfacePdfLocal(
    const NclsLayerInterfaceIR layer,
    const float3 wi,
    const float3 wo,
    const float etaI,
    const float etaT,
    const uint sampleModes)
{
    if (abs(wi.z) < NCLS_MIN_COS || abs(wo.z) < NCLS_MIN_COS) return 0.0f;
    if (layer.kind == (uint)NclsInterfaceKind::Diffuse || layer.kind == (uint)NclsInterfaceKind::Sheen)
        return (sampleModes & NCLS_SAMPLE_REFLECTION) != 0u && wi.z > 0.0f && wo.z > 0.0f
            ? wo.z * NCLS_INV_PI : 0.0f;

    const float2 alpha = nclsLayerAlpha(layer);
    if (layer.kind == (uint)NclsInterfaceKind::RoughConductor)
    {
        if ((sampleModes & NCLS_SAMPLE_REFLECTION) == 0u || wi.z <= 0.0f || wo.z <= 0.0f) return 0.0f;
        const float3 h = nclsSafeNormalize(wi + wo);
        return nclsGgxVisibleNormalPdf(wi, h, alpha) / (4.0f * abs(dot(wi, h)));
    }

    if (layer.kind == (uint)NclsInterfaceKind::RoughDielectric)
    {
        const float relativeEta = etaT / etaI;
        const bool reflection = wi.z * wo.z > 0.0f;
        if (abs(relativeEta - 1.0f) < 1e-6f) return 0.0f;
        if (reflection && (sampleModes & NCLS_SAMPLE_REFLECTION) == 0u) return 0.0f;
        if (!reflection && (sampleModes & NCLS_SAMPLE_TRANSMISSION) == 0u) return 0.0f;
        float3 h = reflection ? nclsSafeNormalize(wi + wo) : nclsSafeNormalize(wi / relativeEta + wo);
        h *= h.z >= 0.0f ? 1.0f : -1.0f;
        const float wiDotH = dot(wi, h);
        const float woDotH = dot(wo, h);
        if (wiDotH * wi.z < 0.0f || woDotH * wo.z < 0.0f) return 0.0f;
        const float fresnel = nclsFresnelDielectric(1.0f / relativeEta, wiDotH);
        const float reflectionWeight = (sampleModes & NCLS_SAMPLE_REFLECTION) != 0u ? fresnel : 0.0f;
        const float transmissionWeight = (sampleModes & NCLS_SAMPLE_TRANSMISSION) != 0u ? 1.0f - fresnel : 0.0f;
        const float weightSum = reflectionWeight + transmissionWeight;
        if (weightSum <= 0.0f) return 0.0f;
        const float normalPdf = nclsGgxVisibleNormalPdf(wi, h, alpha);
        if (reflection)
            return normalPdf * (reflectionWeight / weightSum) / (4.0f * abs(wiDotH));
        const float denominator = max(abs(woDotH + wiDotH / relativeEta), 1e-10f);
        return normalPdf * (transmissionWeight / weightSum) * abs(woDotH) / (denominator * denominator);
    }
    return 0.0f;
}

float3 nclsEvalInterface(
    const NclsLayerInterfaceIR layer,
    const float3 wi,
    const float3 wo,
    const float etaI = 1.0f,
    const float etaT = 1.0f)
{
    return nclsEvalInterfaceLocal(
        layer,
        nclsRotateTangentToLocal(wi, layer.tangentRotation),
        nclsRotateTangentToLocal(wo, layer.tangentRotation),
        etaI,
        etaT);
}

float nclsInterfacePdf(
    const NclsLayerInterfaceIR layer,
    const float3 wi,
    const float3 wo,
    const float etaI = 1.0f,
    const float etaT = 1.0f,
    const uint sampleModes = NCLS_SAMPLE_ALL)
{
    return nclsInterfacePdfLocal(
        layer,
        nclsRotateTangentToLocal(wi, layer.tangentRotation),
        nclsRotateTangentToLocal(wo, layer.tangentRotation),
        etaI,
        etaT,
        sampleModes);
}

LayerBsdfSample nclsSampleInterface(
    const NclsLayerInterfaceIR layer,
    const float3 wiWorld,
    const float etaI,
    const float etaT,
    const uint sampleModes,
    inout NclsRng rng)
{
    LayerBsdfSample result = {};
    const float3 wi = nclsRotateTangentToLocal(wiWorld, layer.tangentRotation);
    if (abs(wi.z) <= NCLS_MIN_COS) return result;

    float3 wo = float3(0.0f);
    if (layer.kind == (uint)NclsInterfaceKind::Diffuse || layer.kind == (uint)NclsInterfaceKind::Sheen)
    {
        if ((sampleModes & NCLS_SAMPLE_REFLECTION) == 0u || wi.z <= 0.0f) return result;
        wo = nclsSampleCosineHemisphere(rng.nextFloat2());
    }
    else if (layer.kind == (uint)NclsInterfaceKind::RoughConductor)
    {
        if ((sampleModes & NCLS_SAMPLE_REFLECTION) == 0u || wi.z <= 0.0f) return result;
        const float2 u = rng.nextFloat2();
        const float3 h = nclsSampleGgxVisibleNormal(wi, nclsLayerAlpha(layer), u);
        wo = reflect(-wi, h);
        if (wo.z <= NCLS_MIN_COS) return result;
    }
    else if (layer.kind == (uint)NclsInterfaceKind::RoughDielectric)
    {
        const float relativeEta = etaT / etaI;
        if (abs(relativeEta - 1.0f) < 1e-6f)
        {
            if ((sampleModes & NCLS_SAMPLE_TRANSMISSION) == 0u) return result;
            wo = -wi;
            result.direction = nclsRotateTangentFromLocal(wo, layer.tangentRotation);
            result.f = float3(1.0f / abs(wo.z));
            result.weight = float3(1.0f);
            result.pdf = 1.0f;
            result.isReflection = 0u;
            result.valid = 1u;
            return result;
        }
        const float2 u = rng.nextFloat2();
        const float3 h = nclsSampleGgxVisibleNormal(wi, nclsLayerAlpha(layer), u);
        const float fresnel = nclsFresnelDielectric(1.0f / relativeEta, dot(wi, h));
        const float reflectionWeight = (sampleModes & NCLS_SAMPLE_REFLECTION) != 0u ? fresnel : 0.0f;
        const float transmissionWeight = (sampleModes & NCLS_SAMPLE_TRANSMISSION) != 0u ? 1.0f - fresnel : 0.0f;
        const float weightSum = reflectionWeight + transmissionWeight;
        if (weightSum <= 0.0f) return result;
        if (rng.nextFloat() * weightSum < reflectionWeight)
        {
            wo = reflect(-wi, h);
            if (wi.z * wo.z <= 0.0f) return result;
        }
        else
        {
            wo = refract(-wi, (wi.z >= 0.0f ? 1.0f : -1.0f) * h, 1.0f / relativeEta);
            if (wi.z * wo.z >= 0.0f || abs(wo.z) <= NCLS_MIN_COS) return result;
        }
    }
    else
    {
        return result;
    }

    result.direction = nclsRotateTangentFromLocal(wo, layer.tangentRotation);
    result.f = nclsEvalInterfaceLocal(layer, wi, wo, etaI, etaT);
    result.pdf = nclsInterfacePdfLocal(layer, wi, wo, etaI, etaT, sampleModes);
    result.isReflection = wi.z * wo.z > 0.0f ? 1u : 0u;
    if (layer.kind == (uint)NclsInterfaceKind::Diffuse || layer.kind == (uint)NclsInterfaceKind::Sheen)
    {
        result.weight = result.f * NCLS_PI;
    }
    else if (layer.kind == (uint)NclsInterfaceKind::RoughConductor)
    {
        const float3 h = nclsSafeNormalize(wi + wo);
        const float geometryRatio = nclsGgxG2(wi, wo, nclsLayerAlpha(layer))
            / max(nclsGgxG1(wi, nclsLayerAlpha(layer)), 1e-10f);
        result.weight = nclsFresnelConductor(
            nclsLayerEta(layer), nclsLayerK(layer), dot(wi, h)) * geometryRatio;
    }
    else if (layer.kind == (uint)NclsInterfaceKind::RoughDielectric)
    {
        const float relativeEta = etaT / etaI;
        float3 h = result.isReflection != 0u
            ? nclsSafeNormalize(wi + wo)
            : nclsSafeNormalize(wi / relativeEta + wo);
        h *= h.z >= 0.0f ? 1.0f : -1.0f;
        const float fresnel = nclsFresnelDielectric(1.0f / relativeEta, dot(wi, h));
        const float reflectionWeight = (sampleModes & NCLS_SAMPLE_REFLECTION) != 0u ? fresnel : 0.0f;
        const float transmissionWeight = (sampleModes & NCLS_SAMPLE_TRANSMISSION) != 0u
            ? 1.0f - fresnel
            : 0.0f;
        const float weightSum = reflectionWeight + transmissionWeight;
        const float branchProbability = result.isReflection != 0u
            ? reflectionWeight / max(weightSum, 1e-10f)
            : transmissionWeight / max(weightSum, 1e-10f);
        const float branchFactor = result.isReflection != 0u ? fresnel : 1.0f - fresnel;
        const float geometryRatio = nclsGgxG2(wi, wo, nclsLayerAlpha(layer))
            / max(nclsGgxG1(wi, nclsLayerAlpha(layer)), 1e-10f);
        result.weight = float3(branchFactor * geometryRatio / max(branchProbability, 1e-10f));
    }
    result.valid = result.pdf > 0.0f && isfinite(result.pdf)
        && all(isfinite(result.f)) && all(isfinite(result.weight)) && any(result.weight > 0.0f)
        ? 1u
        : 0u;
    return result;
}

float3 nclsEvalInterfaceTransport(
    const NclsLayerInterfaceIR layer,
    const float3 wi,
    const float3 wo,
    const float etaI,
    const float etaT,
    const bool radianceTransport)
{
    float3 f = nclsEvalInterface(layer, wi, wo, etaI, etaT);
    if (radianceTransport && layer.kind == (uint)NclsInterfaceKind::RoughDielectric && wi.z * wo.z < 0.0f)
    {
        const float relativeEta = etaT / etaI;
        f /= relativeEta * relativeEta;
    }
    return f;
}

LayerBsdfSample nclsSampleInterfaceTransport(
    const NclsLayerInterfaceIR layer,
    const float3 wi,
    const float etaI,
    const float etaT,
    const uint sampleModes,
    const bool radianceTransport,
    inout NclsRng rng)
{
    LayerBsdfSample result = nclsSampleInterface(layer, wi, etaI, etaT, sampleModes, rng);
    if (result.valid != 0u && result.isReflection == 0u && radianceTransport
        && layer.kind == (uint)NclsInterfaceKind::RoughDielectric)
    {
        const float relativeEta = etaT / etaI;
        result.f /= relativeEta * relativeEta;
        result.weight /= relativeEta * relativeEta;
    }
    return result;
}

#endif
