#ifndef NCLS_LEGACY_LTC_K2_P1_COMPILER_SLANG
#define NCLS_LEGACY_LTC_K2_P1_COMPILER_SLANG

#include "legacy_ltc_k2.slang"

static const uint NCLS_P1_MAX_WIDTH = 64u;
static const uint NCLS_P1_TYPE_WIDTH = 8u;
static const uint NCLS_P1_FEATURE_COUNT = 25u;
static const uint NCLS_P1_RAW_OUTPUT_COUNT = 18u;

float nclsP1Silu(float value)
{
    return value / (1.0f + exp(-value));
}

float nclsP1Sigmoid(float value)
{
    return 1.0f / (1.0f + exp(-value));
}

float nclsP1Softplus(float value)
{
    return value > 20.0f ? value : log(1.0f + exp(value));
}

float nclsP1RoughnessFeature(float value)
{
    return value > 0.0f ? 1.0f + log(max(value, 1e-3f)) / log(1000.0f) : 0.0f;
}

float nclsP1ContinuousFeature(
    const NclsLayerInterfaceIR interfaceRecord,
    const NclsHomogeneousMediumIR medium,
    const bool hasMedium,
    const uint featureIndex)
{
    const bool dielectric = interfaceRecord.kind == (uint)NclsInterfaceKind::RoughDielectric;
    const bool conductor = interfaceRecord.kind == (uint)NclsInterfaceKind::RoughConductor;
    const bool sheen = interfaceRecord.kind == (uint)NclsInterfaceKind::Sheen;
    const bool colored = interfaceRecord.kind == (uint)NclsInterfaceKind::Diffuse || sheen;
    const bool rotated = dielectric || conductor;
    if (featureIndex == 0u) return nclsP1RoughnessFeature((dielectric || conductor || sheen) ? interfaceRecord.alphaX : 0.0f);
    if (featureIndex == 1u) return nclsP1RoughnessFeature((dielectric || conductor || sheen) ? interfaceRecord.alphaY : 0.0f);
    if (featureIndex >= 2u && featureIndex < 5u)
        return dielectric ? interfaceRecord.relativeIor / 3.0f : 0.0f;
    if (featureIndex >= 5u && featureIndex < 8u)
    {
        const float3 eta = float3(interfaceRecord.etaR, interfaceRecord.etaG, interfaceRecord.etaB);
        return conductor ? eta[featureIndex - 5u] / 3.0f : 0.0f;
    }
    if (featureIndex >= 8u && featureIndex < 11u)
    {
        const float3 k = float3(interfaceRecord.kR, interfaceRecord.kG, interfaceRecord.kB);
        return conductor ? k[featureIndex - 8u] / 8.0f : 0.0f;
    }
    if (featureIndex >= 11u && featureIndex < 14u)
    {
        const float3 color = float3(interfaceRecord.colorR, interfaceRecord.colorG, interfaceRecord.colorB);
        return colored ? color[featureIndex - 11u] : 0.0f;
    }
    if (featureIndex == 14u) return rotated ? sin(interfaceRecord.tangentRotation) : 0.0f;
    if (featureIndex == 15u) return rotated ? cos(interfaceRecord.tangentRotation) : 0.0f;
    if (!hasMedium) return 0.0f;
    if (featureIndex >= 16u && featureIndex < 19u)
    {
        const float3 sigmaA = float3(medium.sigmaAR, medium.sigmaAG, medium.sigmaAB);
        return log(1.0f + sigmaA[featureIndex - 16u]) / log(7.0f);
    }
    if (featureIndex >= 19u && featureIndex < 22u)
    {
        const float3 sigmaS = float3(medium.sigmaSR, medium.sigmaSG, medium.sigmaSB);
        return log(1.0f + sigmaS[featureIndex - 19u]) / log(7.0f);
    }
    if (featureIndex == 22u) return medium.g;
    if (featureIndex == 23u) return log(1.0f + medium.thickness) / log(3.0f);
    if (featureIndex == 24u) return 1.0f;
    return 0.0f;
}

NclsLegacyLtcK2PackedState nclsCompileLegacyLtcK2P1(
    const NclsLayerStackIR material,
    const float3 woLocal,
    StructuredBuffer<float> weights,
    const uint width)
{
    NclsLegacyLtcK2PackedState packed = {};
    if (width < 8u || width > NCLS_P1_MAX_WIDTH || (width & 1u) != 0u)
        return packed;
    const uint viewWidth = width / 2u;
    const uint embeddingOffset = 0u;
    const uint encoder0WeightOffset = embeddingOffset + 4u * NCLS_P1_TYPE_WIDTH;
    const uint encoder0BiasOffset = encoder0WeightOffset + width * (NCLS_P1_TYPE_WIDTH + NCLS_P1_FEATURE_COUNT);
    const uint encoder1WeightOffset = encoder0BiasOffset + width;
    const uint encoder1BiasOffset = encoder1WeightOffset + width * width;
    const uint gruInputWeightOffset = encoder1BiasOffset + width;
    const uint gruHiddenWeightOffset = gruInputWeightOffset + 3u * width * width;
    const uint gruInputBiasOffset = gruHiddenWeightOffset + 3u * width * width;
    const uint gruHiddenBiasOffset = gruInputBiasOffset + 3u * width;
    const uint viewWeightOffset = gruHiddenBiasOffset + 3u * width;
    const uint viewBiasOffset = viewWeightOffset + viewWidth * 3u;
    const uint head0WeightOffset = viewBiasOffset + viewWidth;
    const uint head0BiasOffset = head0WeightOffset + width * (width + viewWidth);
    const uint head1WeightOffset = head0BiasOffset + width;
    const uint head1BiasOffset = head1WeightOffset + NCLS_P1_RAW_OUTPUT_COUNT * width;

    float hidden[NCLS_P1_MAX_WIDTH];
    float token0[NCLS_P1_MAX_WIDTH];
    float token1[NCLS_P1_MAX_WIDTH];
    float inputGates[3 * NCLS_P1_MAX_WIDTH];
    float hiddenGates[3 * NCLS_P1_MAX_WIDTH];
    [unroll]
    for (uint unit = 0u; unit < NCLS_P1_MAX_WIDTH; ++unit) hidden[unit] = 0.0f;

    for (uint interfaceIndex = 0u; interfaceIndex < material.interfaceCount; ++interfaceIndex)
    {
        const NclsLayerInterfaceIR interfaceRecord = material.interfaces[interfaceIndex];
        const bool hasMedium = interfaceIndex < material.mediumCount;
        const NclsHomogeneousMediumIR medium = hasMedium ? material.media[interfaceIndex] : (NclsHomogeneousMediumIR)0;
        for (uint row = 0u; row < width; ++row)
        {
            float value = weights[encoder0BiasOffset + row];
            for (uint column = 0u; column < NCLS_P1_TYPE_WIDTH; ++column)
            {
                const float inputValue = weights[embeddingOffset + interfaceRecord.kind * NCLS_P1_TYPE_WIDTH + column];
                value += weights[encoder0WeightOffset + row * (NCLS_P1_TYPE_WIDTH + NCLS_P1_FEATURE_COUNT) + column]
                    * inputValue;
            }
            for (uint feature = 0u; feature < NCLS_P1_FEATURE_COUNT; ++feature)
            {
                const float inputValue = nclsP1ContinuousFeature(interfaceRecord, medium, hasMedium, feature);
                value += weights[
                    encoder0WeightOffset
                    + row * (NCLS_P1_TYPE_WIDTH + NCLS_P1_FEATURE_COUNT)
                    + NCLS_P1_TYPE_WIDTH
                    + feature] * inputValue;
            }
            token0[row] = nclsP1Silu(value);
        }
        for (uint row = 0u; row < width; ++row)
        {
            float value = weights[encoder1BiasOffset + row];
            for (uint column = 0u; column < width; ++column)
                value += weights[encoder1WeightOffset + row * width + column] * token0[column];
            token1[row] = nclsP1Silu(value);
        }
        for (uint row = 0u; row < 3u * width; ++row)
        {
            float inputValue = weights[gruInputBiasOffset + row];
            float hiddenValue = weights[gruHiddenBiasOffset + row];
            for (uint column = 0u; column < width; ++column)
            {
                inputValue += weights[gruInputWeightOffset + row * width + column] * token1[column];
                hiddenValue += weights[gruHiddenWeightOffset + row * width + column] * hidden[column];
            }
            inputGates[row] = inputValue;
            hiddenGates[row] = hiddenValue;
        }
        for (uint unit = 0u; unit < width; ++unit)
        {
            const float reset = nclsP1Sigmoid(inputGates[unit] + hiddenGates[unit]);
            const float update = nclsP1Sigmoid(inputGates[width + unit] + hiddenGates[width + unit]);
            const float candidate = tanh(inputGates[2u * width + unit] + reset * hiddenGates[2u * width + unit]);
            hidden[unit] = candidate + update * (hidden[unit] - candidate);
        }
    }

    float viewHidden[NCLS_P1_MAX_WIDTH / 2];
    for (uint row = 0u; row < viewWidth; ++row)
    {
        float value = weights[viewBiasOffset + row];
        value += weights[viewWeightOffset + row * 3u + 0u] * woLocal.x;
        value += weights[viewWeightOffset + row * 3u + 1u] * woLocal.y;
        value += weights[viewWeightOffset + row * 3u + 2u] * woLocal.z;
        viewHidden[row] = nclsP1Silu(value);
    }
    for (uint row = 0u; row < width; ++row)
    {
        float value = weights[head0BiasOffset + row];
        for (uint column = 0u; column < width; ++column)
            value += weights[head0WeightOffset + row * (width + viewWidth) + column] * hidden[column];
        for (uint column = 0u; column < viewWidth; ++column)
            value += weights[head0WeightOffset + row * (width + viewWidth) + width + column] * viewHidden[column];
        token0[row] = nclsP1Silu(value);
    }
    float raw[NCLS_P1_RAW_OUTPUT_COUNT];
    for (uint row = 0u; row < NCLS_P1_RAW_OUTPUT_COUNT; ++row)
    {
        float value = weights[head1BiasOffset + row];
        for (uint column = 0u; column < width; ++column)
            value += weights[head1WeightOffset + row * width + column] * token0[column];
        raw[row] = value;
    }

    packed.magic = NCLS_LEGACY_LTC_K2_MAGIC;
    packed.version = NCLS_LEGACY_LTC_K2_VERSION;
    packed.residualLobeCount = NCLS_LEGACY_LTC_K2_LOBE_COUNT;
    packed.flags = 0u;
    packed.directTop = material.interfaces[0];
    [unroll]
    for (uint lobeIndex = 0u; lobeIndex < NCLS_LEGACY_LTC_K2_LOBE_COUNT; ++lobeIndex)
    {
        const uint offset = 9u * lobeIndex;
        packed.residual[lobeIndex].amplitudeR = nclsP1Softplus(raw[offset + 0u]);
        packed.residual[lobeIndex].amplitudeG = nclsP1Softplus(raw[offset + 1u]);
        packed.residual[lobeIndex].amplitudeB = nclsP1Softplus(raw[offset + 2u]);
        packed.residual[lobeIndex].inverseScaleX = exp(clamp(raw[offset + 3u], -3.0f, 3.0f));
        packed.residual[lobeIndex].inverseScaleY = exp(clamp(raw[offset + 4u], -3.0f, 3.0f));
        packed.residual[lobeIndex].shearX = 3.0f * tanh(raw[offset + 5u]);
        packed.residual[lobeIndex].shearY = 3.0f * tanh(raw[offset + 6u]);
        packed.residual[lobeIndex].shearZ = 3.0f * tanh(raw[offset + 7u]);
        packed.residual[lobeIndex].angle = NCLS_PI * tanh(raw[offset + 8u]);
    }
    return packed;
}

struct LegacyLtcK2P1Backend : INclsScatteringBackend
{
    typedef NclsLayerStackIR CompiledMaterial;
    typedef LegacyLtcK2State State;

    StructuredBuffer<float> weights;
    uint width;

    State prepare(NclsScatteringContext context, CompiledMaterial material)
    {
        const float3 woLocal = nclsFrameToLocal(
            context.surface.shadingFrame,
            context.woWorld);
        return {context, nclsCompileLegacyLtcK2P1(material, woLocal, weights, width)};
    }
};

#endif
