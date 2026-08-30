#ifndef NCLS_METAL_FUSED_PREPARE_SLANG
#define NCLS_METAL_FUSED_PREPARE_SLANG

#include "metal_fused_common.slang"

int nclsMetalGridInt8(StructuredBuffer<uint> grid, uint byteIndex)
{
    const uint bits = grid[byteIndex >> 2u];
    const uint value = (bits >> (8u * (byteIndex & 3u))) & 0xffu;
    return value < 128u ? int(value) : int(value) - 256;
}

float nclsMetalGridScale(uint halfIndex)
{
    const uint bits = gNclsMetalGridScales[halfIndex >> 1u];
    return f16tof32((halfIndex & 1u) == 0u ? bits & 0xffffu : bits >> 16u);
}

float nclsMetalAssetAdapter(uint index)
{
    const uint bits = gNclsMetalAssetAdapter[index >> 1u];
    return f16tof32((index & 1u) == 0u ? bits & 0xffffu : bits >> 16u);
}

int nclsMetalAddressIndex(int value, uint extent, uint addressMode)
{
    if (addressMode == 0u) return clamp(value, 0, int(extent) - 1);
    int wrapped = value % int(extent);
    return wrapped < 0 ? wrapped + int(extent) : wrapped;
}

float nclsMetalSampleGridChannel(
    StructuredBuffer<uint> grid,
    uint byteOffset,
    uint height,
    uint width,
    uint channel,
    uint scaleHalfOffset,
    uint addressMode,
    float2 uv)
{
    const float2 position = uv * float2(width, height) - 0.5f;
    const int2 low = int2(floor(position));
    const float2 weight = position - float2(low);
    const int x0 = nclsMetalAddressIndex(low.x, width, addressMode);
    const int y0 = nclsMetalAddressIndex(low.y, height, addressMode);
    const int x1 = nclsMetalAddressIndex(low.x + 1, width, addressMode);
    const int y1 = nclsMetalAddressIndex(low.y + 1, height, addressMode);
    const uint i00 = byteOffset + 8u * (uint(y0) * width + uint(x0)) + channel;
    const uint i10 = byteOffset + 8u * (uint(y0) * width + uint(x1)) + channel;
    const uint i01 = byteOffset + 8u * (uint(y1) * width + uint(x0)) + channel;
    const uint i11 = byteOffset + 8u * (uint(y1) * width + uint(x1)) + channel;
    const float scale = nclsMetalGridScale(scaleHalfOffset + channel);
    const float top = lerp(float(nclsMetalGridInt8(grid, i00)), float(nclsMetalGridInt8(grid, i10)), weight.x);
    const float bottom = lerp(float(nclsMetalGridInt8(grid, i01)), float(nclsMetalGridInt8(grid, i11)), weight.x);
    return lerp(top, bottom, weight.y) * scale;
}

void nclsMetalDecoderBlock(
    uint linear0Weight, uint linear0Bias,
    uint linear1Weight, uint linear1Bias,
    uint adapterScaleWeight, uint adapterScaleBias,
    uint adapterBiasWeight, uint adapterBiasBias,
    uint loraDownWeight, uint loraUpWeight,
    inout float value[NCLS_METAL_MAX_WIDTH])
{
    float first[NCLS_METAL_MAX_WIDTH] = {};
    float second[NCLS_METAL_MAX_WIDTH] = {};
    float loraDown[NCLS_METAL_MAX_WIDTH] = {};
    float lora[NCLS_METAL_MAX_WIDTH] = {};
    nclsMetalDense(linear0Weight, linear0Bias, 128u, 128u, 1u, value, first);
    nclsMetalDense(linear1Weight, linear1Bias, 128u, 128u, 0u, first, second);
    float adapter[NCLS_METAL_MAX_WIDTH] = {};
    [loop] for (uint index = 0u; index < 8u; ++index) adapter[index] = nclsMetalAssetAdapter(index);
    // Biasless LoRA layers are evaluated explicitly.
    [loop]
    for (uint row = 0u; row < 8u; ++row)
    {
        float sum = 0.0f;
        [loop] for (uint column = 0u; column < 128u; ++column)
            sum += nclsMetalWeight(loraDownWeight + row * 128u + column) * value[column];
        loraDown[row] = sum;
    }
    [loop]
    for (uint row = 0u; row < 128u; ++row)
    {
        float loraValue = 0.0f;
        float scaleValue = nclsMetalWeight(adapterScaleBias + row);
        float biasValue = nclsMetalWeight(adapterBiasBias + row);
        [loop]
        for (uint column = 0u; column < 8u; ++column)
        {
            loraValue += nclsMetalWeight(loraUpWeight + row * 8u + column) * loraDown[column];
            scaleValue += nclsMetalWeight(adapterScaleWeight + row * 8u + column) * adapter[column];
            biasValue += nclsMetalWeight(adapterBiasWeight + row * 8u + column) * adapter[column];
        }
        const float scale = 0.25f * tanh(scaleValue);
        const float bias = 0.25f * tanh(biasValue);
        value[row] = nclsMetalSilu(value[row] + second[row] * (1.0f + scale) + loraValue * scale + bias);
    }
}

void nclsMetalDecodeRecord(
    uint recordIndex,
    uint assetIndex,
    uint addressMode,
    float2 uv,
    out float structured[64])
{
    const uint base = 12u * recordIndex;
    const uint highHeight = gNclsMetalRecords[base + 2u];
    const uint highWidth = gNclsMetalRecords[base + 3u];
    const uint lowHeight = gNclsMetalRecords[base + 4u];
    const uint lowWidth = gNclsMetalRecords[base + 5u];
    const uint highOffset = gNclsMetalRecords[base + 6u];
    const uint lowOffset = gNclsMetalRecords[base + 7u];
    const uint scaleOffset = gNclsMetalRecords[base + 8u];
    const uint roleClass = gNclsMetalRecords[base + 9u];
    float input[NCLS_METAL_MAX_WIDTH] = {};
    [loop]
    for (uint channel = 0u; channel < 8u; ++channel)
    {
        input[channel] = nclsMetalSampleGridChannel(
            gNclsMetalHighGrid, highOffset, highHeight, highWidth, channel,
            scaleOffset, addressMode, uv);
        input[8u + channel] = nclsMetalSampleGridChannel(
            gNclsMetalLowGrid, lowOffset, lowHeight, lowWidth, channel,
            scaleOffset + 8u, addressMode, uv);
    }
    [loop]
    for (uint index = 0u; index < 32u; ++index)
    {
        input[16u + index] = nclsMetalWeight(
            NCLS_METAL_W_TEXTURE_CODEC_ROLE_EMBEDDING_WEIGHT + roleClass * 32u + index);
        input[48u + index] = nclsMetalWeight(
            NCLS_METAL_W_TEXTURE_CODEC_ASSET_EMBEDDING_WEIGHT + assetIndex * 32u + index);
    }
    float hidden[NCLS_METAL_MAX_WIDTH] = {};
    nclsMetalDense(
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_INPUT_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_INPUT_BIAS,
        80u, 128u, 0u, input, hidden);
    nclsMetalDecoderBlock(
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_0_LINEAR0_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_0_LINEAR0_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_0_LINEAR1_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_0_LINEAR1_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_0_ADAPTER_SCALE_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_0_ADAPTER_SCALE_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_0_ADAPTER_BIAS_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_0_ADAPTER_BIAS_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_0_LORA_DOWN_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_0_LORA_UP_WEIGHT, hidden);
    nclsMetalDecoderBlock(
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_1_LINEAR0_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_1_LINEAR0_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_1_LINEAR1_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_1_LINEAR1_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_1_ADAPTER_SCALE_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_1_ADAPTER_SCALE_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_1_ADAPTER_BIAS_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_1_ADAPTER_BIAS_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_1_LORA_DOWN_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_1_LORA_UP_WEIGHT, hidden);
    nclsMetalDecoderBlock(
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_2_LINEAR0_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_2_LINEAR0_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_2_LINEAR1_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_2_LINEAR1_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_2_ADAPTER_SCALE_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_2_ADAPTER_SCALE_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_2_ADAPTER_BIAS_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_2_ADAPTER_BIAS_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_2_LORA_DOWN_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_2_LORA_UP_WEIGHT, hidden);
    nclsMetalDecoderBlock(
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_3_LINEAR0_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_3_LINEAR0_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_3_LINEAR1_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_3_LINEAR1_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_3_ADAPTER_SCALE_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_3_ADAPTER_SCALE_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_3_ADAPTER_BIAS_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_3_ADAPTER_BIAS_BIAS,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_3_LORA_DOWN_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_DECODER_BLOCKS_3_LORA_UP_WEIGHT, hidden);
    float output[NCLS_METAL_MAX_WIDTH] = {};
    nclsMetalDense(
        NCLS_METAL_W_TEXTURE_CODEC_STRUCTURED_HEAD_WEIGHT,
        NCLS_METAL_W_TEXTURE_CODEC_STRUCTURED_HEAD_BIAS,
        128u, 64u, 0u, hidden, output);
    [loop] for (uint index = 0u; index < 64u; ++index) structured[index] = output[index];
}

void nclsMetalConditionedBlock(
    uint linear0Weight, uint linear0Bias,
    uint linear1Weight, uint linear1Bias,
    uint filmScaleWeight, uint filmScaleBias,
    uint filmBiasWeight, uint filmBiasBias,
    uint loraDownWeight, uint loraUpWeight,
    float condition[8],
    inout float value[NCLS_METAL_MAX_WIDTH])
{
    float first[NCLS_METAL_MAX_WIDTH] = {};
    float second[NCLS_METAL_MAX_WIDTH] = {};
    nclsMetalDense(linear0Weight, linear0Bias, 128u, 128u, 1u, value, first);
    nclsMetalDense(linear1Weight, linear1Bias, 128u, 128u, 0u, first, second);
    float down[8] = {};
    [loop]
    for (uint row = 0u; row < 8u; ++row)
        [loop]
        for (uint column = 0u; column < 128u; ++column)
            down[row] += nclsMetalWeight(loraDownWeight + row * 128u + column) * value[column];
    [loop]
    for (uint row = 0u; row < 128u; ++row)
    {
        float scaleValue = nclsMetalWeight(filmScaleBias + row);
        float biasValue = nclsMetalWeight(filmBiasBias + row);
        float lora = 0.0f;
        [loop]
        for (uint column = 0u; column < 8u; ++column)
        {
            scaleValue += nclsMetalWeight(filmScaleWeight + row * 8u + column) * condition[column];
            biasValue += nclsMetalWeight(filmBiasWeight + row * 8u + column) * condition[column];
            lora += nclsMetalWeight(loraUpWeight + row * 8u + column) * down[column];
        }
        const float scale = 0.25f * tanh(scaleValue);
        value[row] = nclsMetalSilu(value[row] + second[row] * (1.0f + scale)
            + lora * scale + 0.25f * tanh(biasValue));
    }
}

NclsMetalPrepared nclsMetalPrepare(
    NclsScatteringContext context,
    NclsMetalCompiledMaterial material)
{
    NclsMetalPrepared result = {};
    const uint base = material.baseWord;
    if (nclsMetalCompiledUint(base, NCLS_METAL_COMPILED_FLAGS) != NCLS_METAL_COMPILED_LAYOUT_VERSION)
        return result;
    const uint assetIndex = nclsMetalCompiledUint(base, NCLS_METAL_COMPILED_FLAGS + 1u);
    const uint domainCount = nclsMetalCompiledUint(base, NCLS_METAL_COMPILED_FLAGS + 2u);
    const uint maximumExtent = nclsMetalCompiledUint(base, NCLS_METAL_COMPILED_FLAGS + 4u);
    const uint maximumMip = nclsMetalCompiledUint(base, NCLS_METAL_COMPILED_FLAGS + 5u);
    float2 scale = float2(
        nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_ACCESS),
        nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_ACCESS + 1u));
    const float2 translate = float2(
        nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_ACCESS + 2u),
        nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_ACCESS + 3u));
    const float cosine = nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_ACCESS + 4u);
    const float sine = nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_ACCESS + 5u);
    float2 uv = context.surface.uv * scale;
    uv = float2(cosine * uv.x - sine * uv.y, sine * uv.x + cosine * uv.y) + translate;
    float2 dx = context.surface.uvDx * scale;
    dx = float2(cosine * dx.x - sine * dx.y, sine * dx.x + cosine * dx.y);
    float2 dy = context.surface.uvDy * scale;
    dy = float2(cosine * dy.x - sine * dy.y, sine * dy.x + cosine * dy.y);
    if (nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_ACCESS + 6u) > 0.5f) uv = frac(uv);
    const float rho = max(length(dx), length(dy)) * float(maximumExtent);
    const float lod = clamp(log2(max(rho, 1e-8f)), 0.0f, float(maximumMip));
    const uint lowerMip = uint(floor(lod));
    const float mipFraction = frac(lod);
    float lower[64] = {};
    float upper[64] = {};
    [MaxIters(9)]
    for (uint domain = 0u; domain < domainCount; ++domain)
    {
        const uint domainBase = 4u * domain;
        const uint firstRecord = gNclsMetalDomains[domainBase];
        const uint recordCount = gNclsMetalDomains[domainBase + 1u];
        const uint addressMode = gNclsMetalDomains[domainBase + 3u];
        const uint first = firstRecord + min(lowerMip, recordCount - 1u);
        const uint second = firstRecord + min(lowerMip + 1u, recordCount - 1u);
        float decodedLower[64];
        float decodedUpper[64];
        nclsMetalDecodeRecord(first, assetIndex, addressMode, uv, decodedLower);
        nclsMetalDecodeRecord(second, assetIndex, addressMode, uv, decodedUpper);
        [loop]
        for (uint index = 0u; index < 64u; ++index)
        {
            lower[index] += decodedLower[index];
            upper[index] += decodedUpper[index];
        }
    }
    [loop]
    for (uint index = 0u; index < 64u; ++index)
    {
        const float decoded = lerp(lower[index], upper[index], mipFraction) / float(max(domainCount, 1u));
        result.structured[index] = tanh(decoded + nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_SPATIAL + index));
        result.compiler[index] = nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_COMPILER + index);
    }
    float condition[NCLS_METAL_MAX_WIDTH] = {};
    [loop]
    for (uint index = 0u; index < 64u; ++index)
    {
        condition[index] = lerp(lower[index], upper[index], mipFraction) / float(max(domainCount, 1u));
        condition[64u + index] = nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_SPATIAL + index);
    }
    float frameInput[NCLS_METAL_MAX_WIDTH] = {};
    [loop] for (uint index = 0u; index < 128u; ++index) frameInput[index] = condition[index];
    [loop] for (uint index = 0u; index < 8u; ++index)
        frameInput[128u + index] = nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_FRAME + index);
    float frameHidden[NCLS_METAL_MAX_WIDTH] = {};
    float frameRaw[NCLS_METAL_MAX_WIDTH] = {};
    nclsMetalDense(NCLS_METAL_W_PREPARED_MODEL_FRAME_HEAD_0_WEIGHT,
        NCLS_METAL_W_PREPARED_MODEL_FRAME_HEAD_0_BIAS, 136u, 128u, 1u, frameInput, frameHidden);
    nclsMetalDense(NCLS_METAL_W_PREPARED_MODEL_FRAME_HEAD_2_WEIGHT,
        NCLS_METAL_W_PREPARED_MODEL_FRAME_HEAD_2_BIAS, 128u, 18u, 0u, frameHidden, frameRaw);
    result.frames[0] = nclsMetalBaseFrame();
    const float frameStrength = nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_TAIL_BOUNDS + 2u);
    uint frameValid = 1u;
    [loop]
    for (uint frameIndex = 0u; frameIndex < 3u; ++frameIndex)
    {
        const uint offset = 6u * frameIndex;
        uint normalValid;
        float3 normal = nclsMetalSafeNormalize(float3(
            frameRaw[offset + 3u], frameRaw[offset + 4u], frameRaw[offset + 5u]),
            float3(0.0f, 0.0f, 1.0f), normalValid);
        float3 tangentSeed = float3(frameRaw[offset], frameRaw[offset + 1u], frameRaw[offset + 2u]);
        tangentSeed -= dot(tangentSeed, normal) * normal;
        uint tangentValid;
        float3 tangent = nclsMetalSafeNormalize(tangentSeed, float3(1.0f, 0.0f, 0.0f), tangentValid);
        uint mixedNormalValid;
        normal = nclsMetalSafeNormalize(
            lerp(float3(0.0f, 0.0f, 1.0f), normal, frameStrength),
            float3(0.0f, 0.0f, 1.0f), mixedNormalValid);
        tangentSeed = lerp(float3(1.0f, 0.0f, 0.0f), tangent, frameStrength);
        tangentSeed -= dot(tangentSeed, normal) * normal;
        uint mixedTangentValid;
        tangent = nclsMetalSafeNormalize(tangentSeed, float3(1.0f, 0.0f, 0.0f), mixedTangentValid);
        result.frames[frameIndex + 1u] = {tangent, cross(normal, tangent), normal};
        frameValid &= normalValid & tangentValid & mixedNormalValid & mixedTangentValid;
    }
    float hidden[NCLS_METAL_MAX_WIDTH] = {};
    float output[NCLS_METAL_MAX_WIDTH] = {};
    nclsMetalDense(NCLS_METAL_W_PREPARED_MODEL_CORE_SPATIAL_HEAD_0_WEIGHT,
        NCLS_METAL_W_PREPARED_MODEL_CORE_SPATIAL_HEAD_0_BIAS, 128u, 128u, 1u, condition, hidden);
    nclsMetalDense(NCLS_METAL_W_PREPARED_MODEL_CORE_SPATIAL_HEAD_2_WEIGHT,
        NCLS_METAL_W_PREPARED_MODEL_CORE_SPATIAL_HEAD_2_BIAS, 128u, 54u, 0u, hidden, output);
    [loop]
    for (uint lobe = 0u; lobe < 6u; ++lobe)
    {
        const uint stateOffset = 9u * lobe;
        [loop] for (uint channel = 0u; channel < 3u; ++channel)
            result.core[stateOffset + channel] = clamp(
                nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_CORE + stateOffset + channel)
                * exp(0.25f * tanh(output[stateOffset + channel])), 0.0f, 4.0f);
        [loop] for (uint channel = 3u; channel < 5u; ++channel)
            result.core[stateOffset + channel] = clamp(
                nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_CORE + stateOffset + channel)
                * exp(0.35f * tanh(output[stateOffset + channel])), 0.01f, 1.0f);
        result.core[stateOffset + 5u] = clamp(nclsMetalCompiledFloat(base,
            NCLS_METAL_COMPILED_CORE + stateOffset + 5u) + 0.25f * tanh(output[stateOffset + 5u]), -1.0f, 1.0f);
        result.core[stateOffset + 6u] = nclsMetalCompiledFloat(base,
            NCLS_METAL_COMPILED_CORE + stateOffset + 6u) * nclsMetalSoftplus(output[stateOffset + 6u] + 1.0f);
        result.core[stateOffset + 7u] = clamp(nclsMetalCompiledFloat(base,
            NCLS_METAL_COMPILED_CORE + stateOffset + 7u) * (1.0f / (1.0f + exp(-(output[stateOffset + 7u] + 2.0f)))), 0.0f, 1.0f);
        result.core[stateOffset + 8u] = clamp(nclsMetalCompiledFloat(base,
            NCLS_METAL_COMPILED_CORE + stateOffset + 8u) + 0.25f * tanh(output[stateOffset + 8u]), 0.0f, 1.0f);
    }
    nclsMetalDense(NCLS_METAL_W_PREPARED_MODEL_RESIDUAL_SPATIAL_HEAD_0_WEIGHT,
        NCLS_METAL_W_PREPARED_MODEL_RESIDUAL_SPATIAL_HEAD_0_BIAS, 128u, 128u, 1u, condition, hidden);
    nclsMetalDense(NCLS_METAL_W_PREPARED_MODEL_RESIDUAL_SPATIAL_HEAD_2_WEIGHT,
        NCLS_METAL_W_PREPARED_MODEL_RESIDUAL_SPATIAL_HEAD_2_BIAS, 128u, 28u, 0u, hidden, output);
    [loop]
    for (uint lobe = 0u; lobe < 4u; ++lobe)
    {
        const uint stateOffset = 7u * lobe;
        [loop] for (uint channel = 0u; channel < 3u; ++channel)
            result.residual[stateOffset + channel] = nclsMetalCompiledFloat(base,
                NCLS_METAL_COMPILED_RESIDUAL + stateOffset + channel) * nclsMetalSoftplus(output[stateOffset + channel] + 1.0f);
        [loop] for (uint channel = 3u; channel < 5u; ++channel)
            result.residual[stateOffset + channel] = clamp(nclsMetalCompiledFloat(base,
                NCLS_METAL_COMPILED_RESIDUAL + stateOffset + channel) * exp(0.35f * tanh(output[stateOffset + channel])), 0.01f, 1.0f);
        result.residual[stateOffset + 5u] = clamp(nclsMetalCompiledFloat(base,
            NCLS_METAL_COMPILED_RESIDUAL + stateOffset + 5u) * (1.0f / (1.0f + exp(-(output[stateOffset + 5u] + 2.0f)))), 0.0f, 1.0f);
        result.residual[stateOffset + 6u] = clamp(nclsMetalCompiledFloat(base,
            NCLS_METAL_COMPILED_RESIDUAL + stateOffset + 6u) + 0.25f * tanh(output[stateOffset + 6u]), -1.0f, 1.0f);
    }
    const float3 wo = nclsFrameToLocal(context.surface.shadingFrame, context.woWorld);
    float viewInput[NCLS_METAL_MAX_WIDTH] = {};
    viewInput[0] = wo.x; viewInput[1] = wo.y; viewInput[2] = wo.z;
    [loop] for (uint index = 0u; index < 128u; ++index) viewInput[3u + index] = condition[index];
    nclsMetalDense(NCLS_METAL_W_PREPARED_MODEL_VIEW_ENCODER_0_WEIGHT,
        NCLS_METAL_W_PREPARED_MODEL_VIEW_ENCODER_0_BIAS, 131u, 128u, 1u, viewInput, hidden);
    nclsMetalDense(NCLS_METAL_W_PREPARED_MODEL_VIEW_ENCODER_2_WEIGHT,
        NCLS_METAL_W_PREPARED_MODEL_VIEW_ENCODER_2_BIAS, 128u, 64u, 1u, hidden, output);
    [loop] for (uint index = 0u; index < 64u; ++index) result.viewToken[index] = output[index];
    nclsMetalDense(NCLS_METAL_W_PREPARED_MODEL_PROPOSAL_SPATIAL_HEAD_WEIGHT,
        NCLS_METAL_W_PREPARED_MODEL_PROPOSAL_SPATIAL_HEAD_BIAS, 128u, 44u, 0u, condition, output);
    float rawWeights[11];
    float weightSum = 0.0f;
    const float3 luminance = float3(0.2126f, 0.7152f, 0.0722f);
    [loop]
    for (uint component = 0u; component < 11u; ++component)
    {
        float2 baseAlpha = float2(1.0f);
        float baseRotation = 0.0f;
        float activity = 1.0f;
        float clue = 0.05f + nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_TAIL_BOUNDS + 1u);
        if (component < 6u)
        {
            const uint offset = 9u * component;
            baseAlpha = float2(result.core[offset + 3u], result.core[offset + 4u]);
            if (component == 4u || component == 5u) baseAlpha = sqrt(baseAlpha);
            baseRotation = 3.141592653589793f * result.core[offset + 5u];
            activity = result.core[offset + 7u];
            clue = dot(float3(result.core[offset], result.core[offset + 1u], result.core[offset + 2u]), luminance)
                * result.core[offset + 6u] * activity;
        }
        else if (component < 10u)
        {
            const uint offset = 7u * (component - 6u);
            const float skew = result.residual[offset + 6u];
            baseAlpha = float2(result.residual[offset + 3u] * exp(0.35f * skew),
                result.residual[offset + 4u] * exp(-0.35f * skew));
            baseRotation = 1.570796326794897f * skew;
            activity = result.residual[offset + 5u];
            clue = dot(float3(result.residual[offset], result.residual[offset + 1u], result.residual[offset + 2u]), luminance) * activity;
        }
        const uint proposalOffset = 4u * component;
        const float2 modulation = float2(
            nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_PROPOSAL_MODULATION + 3u * component),
            nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_PROPOSAL_MODULATION + 3u * component + 1u))
            + float2(output[proposalOffset + 1u], output[proposalOffset + 2u]);
        const float rotationModulation = nclsMetalCompiledFloat(base,
            NCLS_METAL_COMPILED_PROPOSAL_MODULATION + 3u * component + 2u) + output[proposalOffset + 3u];
        const float2 alpha = clamp(baseAlpha * exp(0.5f * tanh(modulation)), 0.01f, 1.0f);
        const float rotation = baseRotation + 3.141592653589793f * tanh(rotationModulation);
        const float logit = nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_PROPOSAL_LOGITS + component)
            + output[proposalOffset];
        rawWeights[component] = activity * max(clue, 1e-6f) * exp(clamp(logit, -8.0f, 8.0f));
        weightSum += rawWeights[component];
        result.proposal.state[component][1] = alpha.x;
        result.proposal.state[component][2] = alpha.y;
        result.proposal.state[component][3] = rotation;
        result.proposal.state[component][4] = activity > 0.0f ? 1.0f : 0.0f;
        result.proposal.state[component][5] = float(nclsMetalProposalFrameIndex(component));
        result.proposal.state[component][6] = float(nclsMetalProposalDistribution(component));
        result.proposal.state[component][7] = clue;
    }
    [loop]
    for (uint component = 0u; component < 11u; ++component)
        result.proposal.state[component][0] = 0.98f * rawWeights[component] / max(weightSum, 1e-12f)
            + (component == 10u ? 0.02f : 0.0f);
    [loop]
    for (uint frame = 0u; frame < 4u; ++frame)
    {
        result.proposal.frames[frame][0] = result.frames[frame].tangent;
        result.proposal.frames[frame][1] = result.frames[frame].bitangent;
        result.proposal.frames[frame][2] = result.frames[frame].normal;
    }
    [loop] for (uint index = 0u; index < 32u; ++index)
        result.blockCondition[index] = nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_BLOCK + index);
    result.correctionBound = nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_TAIL_BOUNDS);
    result.tailScale = nclsMetalCompiledFloat(base, NCLS_METAL_COMPILED_TAIL_BOUNDS + 1u);
    result.valid = domainCount > 0u && frameValid != 0u && wo.z > 0.0f
        && all(isfinite(uv)) && isfinite(lod) ? 1u : 0u;
    result.proposal.valid = result.valid;
    return result;
}

#endif
