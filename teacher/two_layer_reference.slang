import Rendering.Materials.AnisotropicGGX;
import Rendering.Materials.LayeredBSDF;
import Rendering.Materials.PBRT.PBRTConductorMaterialInstance;
import Rendering.Materials.PBRT.PBRTDielectricMaterialInstance;
import Rendering.Materials.PBRT.PBRTDiffuseMaterialInstance;
import Utils.Sampling.UniformSampleGenerator;

#include "layer_types.slang"

float3 layerEta(const LayerInterface layer)
{
    return float3(layer.etaR, layer.etaG, layer.etaB);
}

float3 layerK(const LayerInterface layer)
{
    return float3(layer.kR, layer.kG, layer.kB);
}

float3 layerAlbedo(const LayerInterface layer)
{
    return float3(layer.albedoR, layer.albedoG, layer.albedoB);
}

float3 evalDirectCoat(
    const LayerStack stack,
    const float3 viewDirection,
    const float3 lightDirection,
    const uint2 rngKey)
{
    if (stack.layerCount < 1) return float3(0.0f);

    const LayerInterface coat = stack.layers[0];
    if (coat.type != (uint)LayerType::RoughDielectric) return float3(0.0f);

    UniformSampleGenerator sg = UniformSampleGenerator(rngKey, 0);
    AnisotropicGGX coatDistribution = AnisotropicGGX(float2(coat.roughnessX, coat.roughnessY));
    PBRTDielectricBSDF top = {coatDistribution, coat.etaR};
    const BSDFContext context = viewDirection.z > 0.0f
        ? BSDFContext(1.0f, top.getEta())
        : BSDFContext(top.getEta(), 1.0f);
    return top.eval(viewDirection, lightDirection, sg, context);
}

float3 evalTwoLayerReference(
    const LayerStack stack,
    const float3 viewDirection,
    const float3 lightDirection,
    const uint2 rngKey,
    const uint maxDepth)
{
    if (stack.layerCount != 2 || stack.mediumCount != 1) return float3(0.0f);

    const LayerInterface coat = stack.layers[0];
    const LayerInterface base = stack.layers[1];
    if (coat.type != (uint)LayerType::RoughDielectric) return float3(0.0f);

    UniformSampleGenerator sg = UniformSampleGenerator(rngKey, 0);
    AnisotropicGGX coatDistribution = AnisotropicGGX(float2(coat.roughnessX, coat.roughnessY));
    PBRTDielectricBSDF top = {coatDistribution, coat.etaR};
    const BSDFContext context = BSDFContext(1.0f, top.getEta());

    // The first reference slice uses a clear slab. Absorption/scattering is
    // connected when the two-interface surface walk is validated.
    if (base.type == (uint)LayerType::Diffuse)
    {
        PBRTDiffuseBSDF bottom = {layerAlbedo(base)};
        return LayeredBSDF(top, bottom, 0.0f, 0.0f, float3(0.0f), maxDepth)
            .eval(viewDirection, lightDirection, sg, context);
    }
    if (base.type == (uint)LayerType::RoughConductor)
    {
        AnisotropicGGX baseDistribution = AnisotropicGGX(float2(base.roughnessX, base.roughnessY));
        PBRTConductorBSDF bottom = {baseDistribution, layerEta(base), layerK(base)};
        return LayeredBSDF(top, bottom, 0.0f, 0.0f, float3(0.0f), maxDepth)
            .eval(viewDirection, lightDirection, sg, context);
    }
    return float3(0.0f);
}
