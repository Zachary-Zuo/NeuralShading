#ifndef NCLS_METAL_BUDGETED_ASSET_SLANG
#define NCLS_METAL_BUDGETED_ASSET_SLANG

#include "metal_budgeted_common.slang"

void nclsMetalBudgetedModulateLobe(
    uint baseWord,
    uint lobeIndex,
    float semantic[24],
    out float result[8])
{
    const uint baseHalf = 8u + 8u * lobeIndex;
    const uint delta = 8u + 8u * lobeIndex;
    [unroll]
    for (uint channel = 0u; channel < 3u; ++channel)
    {
        result[channel] = clamp(
            nclsMetalBudgetedCompiledHalf(baseWord, baseHalf + channel)
                + 0.25f * tanh(semantic[delta + channel]),
            0.0f,
            1.0f);
    }
    [unroll]
    for (uint channel = 3u; channel < 5u; ++channel)
    {
        result[channel] = clamp(
            nclsMetalBudgetedCompiledHalf(baseWord, baseHalf + channel)
                * exp(0.75f * tanh(semantic[delta + channel])),
            0.015f,
            1.0f);
    }
    result[5] = nclsMetalBudgetedCompiledHalf(baseWord, baseHalf + 5u)
        * nclsMetalBudgetedSoftplus(semantic[delta + 5u] + 1.0f);
    result[6] = nclsMetalBudgetedCompiledHalf(baseWord, baseHalf + 6u)
        + 0.5f * 3.14159265358979323846f * tanh(semantic[delta + 6u]);
    result[7] = clamp(
        nclsMetalBudgetedCompiledHalf(baseWord, baseHalf + 7u)
            * (2.0f / (1.0f + exp(-semantic[delta + 7u]))),
        0.0f,
        1.0f);
}

NclsMetalBudgetedPrepared nclsMetalBudgetedPrepare(
    NclsScatteringContext context,
    NclsMetalBudgetedCompiledMaterial material)
{
    NclsMetalBudgetedPrepared result = {};
    const uint baseWord = material.baseWord;
    if (nclsMetalBudgetedMetadata(baseWord, 0u)
        != NCLS_METAL_BUDGETED_COMPILED_LAYOUT_VERSION)
    {
        return result;
    }
    float access[8];
    [unroll]
    for (uint index = 0u; index < 8u; ++index)
        access[index] = nclsMetalBudgetedCompiledHalf(baseWord, 40u + index);
    float2 uv = context.surface.uv * float2(access[0], access[1]);
    uv = float2(
        access[4] * uv.x - access[5] * uv.y,
        access[5] * uv.x + access[4] * uv.y) + float2(access[2], access[3]);
    float2 dx = context.surface.uvDx * float2(access[0], access[1]);
    dx = float2(
        access[4] * dx.x - access[5] * dx.y,
        access[5] * dx.x + access[4] * dx.y);
    float2 dy = context.surface.uvDy * float2(access[0], access[1]);
    dy = float2(
        access[4] * dy.x - access[5] * dy.y,
        access[5] * dy.x + access[4] * dy.y);
    if (access[6] > 0.5f) uv = frac(uv);
    const uint detailWidth = nclsMetalBudgetedMetadata(baseWord, 1u);
    const uint detailHeight = nclsMetalBudgetedMetadata(baseWord, 2u);
    const uint detailMipCount = nclsMetalBudgetedMetadata(baseWord, 3u);
    const uint contextMipCount = nclsMetalBudgetedMetadata(baseWord, 6u);
    const float rho = max(
        length(dx * float2(detailWidth, detailHeight)),
        length(dy * float2(detailWidth, detailHeight)));
    const float lod = clamp(
        log2(max(rho, 1.0f)), 0.0f, float(detailMipCount - 1u));
    const uint lowerMip = uint(floor(lod));
    const uint selectedMip = min(
        lowerMip + (context.filterRandom < frac(lod) ? 1u : 0u),
        detailMipCount - 1u);
    const float4 rawDetail = gNclsMetalBudgetedDetail.SampleLevel(
        gNclsMetalBudgetedSampler, uv, float(selectedMip));
    const float4 rawContext = gNclsMetalBudgetedContext.SampleLevel(
        gNclsMetalBudgetedSampler, uv,
        float(min(selectedMip, contextMipCount - 1u)));
    const uint variant = nclsMetalBudgetedCompiledFlag(baseWord, 5u);
    float detail[4];
    float contextValue[4];
    [unroll]
    for (uint channel = 0u; channel < 4u; ++channel)
    {
        const float programScale = nclsMetalBudgetedCompiledHalf(
            baseWord, 24u + channel);
        const float programBias = nclsMetalBudgetedCompiledHalf(
            baseWord, 28u + channel);
        const uint variantBase =
            NCLS_METAL_BUDGETED_W_ASSET_VARIANT_SCALE_BIAS_WEIGHT
            + variant * 16u;
        const float detailScale = exp(
            0.25f * tanh(nclsMetalBudgetedWeight(variantBase + channel))
            + 0.25f * programScale);
        const float detailBias = 0.25f * tanh(
            nclsMetalBudgetedWeight(variantBase + 4u + channel))
            + 0.1f * programBias;
        const float contextScale = exp(
            0.25f * tanh(nclsMetalBudgetedWeight(variantBase + 8u + channel))
            + 0.25f * programScale);
        const float contextBias = 0.25f * tanh(
            nclsMetalBudgetedWeight(variantBase + 12u + channel))
            + 0.1f * programBias;
        detail[channel] = rawDetail[channel] * detailScale + detailBias;
        contextValue[channel] = rawContext[channel] * contextScale + contextBias;
    }
    const float3 woRaw = nclsFrameToLocal(
        context.surface.shadingFrame, context.woWorld);
    uint woValid;
    const float3 wo = nclsMetalBudgetedSafeNormalize(
        woRaw, float3(0.0f, 0.0f, 1.0f), woValid);
    float decoderInput[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    [unroll]
    for (uint index = 0u; index < 8u; ++index)
        decoderInput[index] = nclsMetalBudgetedCompiledHalf(baseWord, index);
    [unroll]
    for (uint index = 0u; index < 4u; ++index)
    {
        decoderInput[8u + index] = detail[index];
        decoderInput[12u + index] = contextValue[index];
    }
    const float view[8] = {
        wo.x, wo.y, wo.z,
        wo.x * wo.x, wo.y * wo.y, wo.z * wo.z,
        wo.z, 1.0f,
    };
    [unroll]
    for (uint index = 0u; index < 8u; ++index)
    {
        decoderInput[16u + index] = view[index];
        result.viewState[index] = view[index];
    }
    float hidden0[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    float hidden1[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    float semantic[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    nclsMetalBudgetedDense(
        NCLS_METAL_BUDGETED_W_PREPARED_MODEL_SEMANTIC_DECODER_0_WEIGHT,
        NCLS_METAL_BUDGETED_W_PREPARED_MODEL_SEMANTIC_DECODER_0_BIAS,
        24u, 32u, 1u, decoderInput, hidden0);
    nclsMetalBudgetedDense(
        NCLS_METAL_BUDGETED_W_PREPARED_MODEL_SEMANTIC_DECODER_2_WEIGHT,
        NCLS_METAL_BUDGETED_W_PREPARED_MODEL_SEMANTIC_DECODER_2_BIAS,
        32u, 32u, 1u, hidden0, hidden1);
    nclsMetalBudgetedDense(
        NCLS_METAL_BUDGETED_W_PREPARED_MODEL_SEMANTIC_DECODER_4_WEIGHT,
        NCLS_METAL_BUDGETED_W_PREPARED_MODEL_SEMANTIC_DECODER_4_BIAS,
        32u, 24u, 0u, hidden1, semantic);
    [unroll]
    for (uint index = 0u; index < 24u; ++index)
    {
        result.semantic[index] = semantic[index]
            + (index < 4u ? detail[index] : 0.0f);
    }
    float primary[8];
    float secondary[8];
    nclsMetalBudgetedModulateLobe(
        baseWord, 0u, result.semantic, primary);
    nclsMetalBudgetedModulateLobe(
        baseWord, 1u, result.semantic, secondary);
    [unroll]
    for (uint index = 0u; index < 8u; ++index)
    {
        result.lobes[index] = primary[index];
        result.lobes[8u + index] = secondary[index];
    }
    const float2 primarySlope = 0.75f * tanh(float2(
        result.semantic[0], result.semantic[1]));
    const float2 secondarySlope = 0.75f * tanh(float2(
        result.semantic[2], result.semantic[3]));
    result.frames[0] = nclsMetalBudgetedLocalFrame(primarySlope, primary[6]);
    result.frames[1] = nclsMetalBudgetedLocalFrame(secondarySlope, secondary[6]);
    result.compactFrame[0] = primarySlope.x;
    result.compactFrame[1] = primarySlope.y;
    result.compactFrame[2] = secondarySlope.x;
    result.compactFrame[3] = secondarySlope.y;
    result.compactFrame[4] = cos(primary[6]);
    result.compactFrame[5] = cos(secondary[6]);
    result.compactFrame[6] = sin(primary[6]);
    result.compactFrame[7] = sin(secondary[6]);
    float proposalInput[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    float proposalOutput[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    [unroll]
    for (uint i = 0u; i < 24u; ++i) proposalInput[i] = result.semantic[i];
    nclsMetalBudgetedDense(NCLS_METAL_BUDGETED_W_PREPARED_MODEL_PROPOSAL_ADAPTER_WEIGHT,
        NCLS_METAL_BUDGETED_W_PREPARED_MODEL_PROPOSAL_ADAPTER_BIAS,
        24u, 3u, 0u, proposalInput, proposalOutput);
    const float3 luminance = float3(0.2126f, 0.7152f, 0.0722f);
    float3 rawWeights;
    [unroll]
    for (uint i = 0u; i < 3u; ++i)
    {
        float clue = 1.0f;
        if (i < 2u)
            clue = max(dot(float3(result.lobes[8u*i], result.lobes[8u*i+1u], result.lobes[8u*i+2u]),
                luminance) * result.lobes[8u*i+5u] * result.lobes[8u*i+7u], 1e-5f);
        rawWeights[i] = nclsMetalBudgetedCompiledHalf(baseWord, 32u+i) * clue
            * exp(clamp(proposalOutput[i], -4.0f, 4.0f));
    }
    const float3 proposalWeights = 0.98f * rawWeights / max(rawWeights.x+rawWeights.y+rawWeights.z, 1e-12f)
        + float3(0.0f, 0.0f, 0.02f);
    [unroll]
    for (uint i = 0u; i < 3u; ++i)
    {
        result.proposal[4u*i] = proposalWeights[i];
        result.proposal[4u*i+1u] = i < 2u ? result.lobes[8u*i+3u] : 1.0f;
        result.proposal[4u*i+2u] = i < 2u ? result.lobes[8u*i+4u] : 1.0f;
        result.proposal[4u*i+3u] = i == 0u ? float(nclsMetalBudgetedCompiledFlag(baseWord, 6u)) : i == 1u ? 0.0f : 2.0f;
    }
    [unroll]
    for (uint index = 0u; index < 4u; ++index)
        result.accessState[index] = access[index];
    result.identityAndFlags[0] = variant;
    result.identityAndFlags[1] = selectedMip;
    result.identityAndFlags[2] = nclsMetalBudgetedCompiledFlag(baseWord, 6u);
    result.identityAndFlags[3] = (access[7] > 0.5f ? 1u : 0u)
        | (nclsMetalBudgetedMetadata(baseWord, 7u) << 16u);
    result.valid = woValid != 0u && wo.z > 0.0f
        && all(isfinite(rawDetail)) && all(isfinite(rawContext)) ? 1u : 0u;
    return result;
}

#endif
