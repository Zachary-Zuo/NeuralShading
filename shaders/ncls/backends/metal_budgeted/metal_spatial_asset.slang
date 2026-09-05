#ifndef NCLS_METAL_SPATIAL_ASSET_SLANG
#define NCLS_METAL_SPATIAL_ASSET_SLANG

float4 nclsMetalSpatialRead(uint group, bool detail, float2 uv, uint lod)
{
    switch (group)
    {
    case 0u: return detail ? gNclsMetalSpatialDetail0.SampleLevel(gNclsMetalSpatialSampler0, uv, lod) : gNclsMetalSpatialContext0.SampleLevel(gNclsMetalSpatialSampler0, uv, lod);
    case 1u: return detail ? gNclsMetalSpatialDetail1.SampleLevel(gNclsMetalSpatialSampler1, uv, lod) : gNclsMetalSpatialContext1.SampleLevel(gNclsMetalSpatialSampler1, uv, lod);
    case 2u: return detail ? gNclsMetalSpatialDetail2.SampleLevel(gNclsMetalSpatialSampler2, uv, lod) : gNclsMetalSpatialContext2.SampleLevel(gNclsMetalSpatialSampler2, uv, lod);
    case 3u: return detail ? gNclsMetalSpatialDetail3.SampleLevel(gNclsMetalSpatialSampler3, uv, lod) : gNclsMetalSpatialContext3.SampleLevel(gNclsMetalSpatialSampler3, uv, lod);
    case 4u: return detail ? gNclsMetalSpatialDetail4.SampleLevel(gNclsMetalSpatialSampler4, uv, lod) : gNclsMetalSpatialContext4.SampleLevel(gNclsMetalSpatialSampler4, uv, lod);
    case 5u: return detail ? gNclsMetalSpatialDetail5.SampleLevel(gNclsMetalSpatialSampler5, uv, lod) : gNclsMetalSpatialContext5.SampleLevel(gNclsMetalSpatialSampler5, uv, lod);
    case 6u: return detail ? gNclsMetalSpatialDetail6.SampleLevel(gNclsMetalSpatialSampler6, uv, lod) : gNclsMetalSpatialContext6.SampleLevel(gNclsMetalSpatialSampler6, uv, lod);
    case 7u: return detail ? gNclsMetalSpatialDetail7.SampleLevel(gNclsMetalSpatialSampler7, uv, lod) : gNclsMetalSpatialContext7.SampleLevel(gNclsMetalSpatialSampler7, uv, lod);
    case 8u: return detail ? gNclsMetalSpatialDetail8.SampleLevel(gNclsMetalSpatialSampler8, uv, lod) : gNclsMetalSpatialContext8.SampleLevel(gNclsMetalSpatialSampler8, uv, lod);
    }
    return float4(0.0f);
}

uint nclsMetalSpatialHash(uint x)
{
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}

float nclsMetalSpatialUintFloat(uint x)
{
    return float(x & 0x7fffffffu) + ((x & 0x80000000u) != 0u ? 2147483648.0f : 0.0f);
}

float2 nclsMetalSpatialHash22(int2 cell)
{
    const uint inside = nclsMetalSpatialHash(uint(cell.y));
    return float2(nclsMetalSpatialUintFloat(nclsMetalSpatialHash(uint(cell.x) + inside)),
        nclsMetalSpatialUintFloat(nclsMetalSpatialHash(uint(cell.x) + 32000u + inside))) / 4294967296.0f;
}

float nclsMetalSpatialHalf(float x) { return f16tof32(f32tof16(x)); }

NclsMetalBudgetedPrepared nclsMetalBudgetedPrepare(
    NclsScatteringContext context, NclsMetalBudgetedCompiledMaterial material)
{
    NclsMetalBudgetedPrepared result = {};
    const uint baseWord = material.baseWord;
    const uint groupCount = nclsMetalBudgetedMetadata(baseWord, 1u);
    if (nclsMetalBudgetedMetadata(baseWord, 0u) != 2u || groupCount == 0u || groupCount > 9u)
        return result;
    float decoderInput[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    float proposalInput[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    uint woValid;
    const float3 wo = nclsMetalBudgetedSafeNormalize(nclsFrameToLocal(context.surface.shadingFrame, context.woWorld),
        float3(0.0f, 0.0f, 1.0f), woValid);
    [unroll] for (uint i = 0u; i < 8u; ++i)
        decoderInput[i] = proposalInput[i] = nclsMetalBudgetedCompiledHalf(baseWord, i);
    decoderInput[8] = result.viewState[0] = wo.x;
    decoderInput[9] = result.viewState[1] = wo.y;
    decoderInput[10] = result.viewState[2] = wo.z;
    uint finite = woValid;
    [unroll] for (uint group = 0u; group < 9u; ++group)
    {
        if (group >= groupCount) continue;
        const uint offset = baseWord + 48u + 16u * group;
        const float3 a = float3(asfloat(gNclsCompiledMaterials[offset]), asfloat(gNclsCompiledMaterials[offset+1u]), asfloat(gNclsCompiledMaterials[offset+2u]));
        const float3 b = float3(asfloat(gNclsCompiledMaterials[offset+3u]), asfloat(gNclsCompiledMaterials[offset+4u]), asfloat(gNclsCompiledMaterials[offset+5u]));
        const float cellScale = asfloat(gNclsCompiledMaterials[offset+6u]);
        const float lookupScale = asfloat(gNclsCompiledMaterials[offset+7u]);
        const bool nonrepeat = gNclsCompiledMaterials[offset+8u] != 0u;
        const int hashOffset = int(gNclsCompiledMaterials[offset+9u]);
        const float2 extent = float2(gNclsCompiledMaterials[offset+10u], gNclsCompiledMaterials[offset+11u]);
        const uint detailMips = gNclsCompiledMaterials[offset+12u];
        const uint contextMips = gNclsCompiledMaterials[offset+15u];
        const float2 uv = float2(dot(a.xy, context.surface.uv)+a.z, dot(b.xy, context.surface.uv)+b.z);
        const float2 dx = float2(dot(a.xy, context.surface.uvDx), -dot(b.xy, context.surface.uvDx)) * lookupScale;
        const float2 dy = float2(dot(a.xy, context.surface.uvDy), -dot(b.xy, context.surface.uvDy)) * lookupScale;
        const float4 jacobian = float4(dx.x * extent.x, dy.x * extent.x, dx.y * extent.y, dy.y * extent.y);
        const float rho = max(length(dx * extent), length(dy * extent));
        const float lod = clamp(log2(max(rho, 1.0f)), 0.0f, float(detailMips-1u));
        const uint selected = uint(floor(lod)) + (context.filterRandom < frac(lod) ? 1u : 0u);
        const float4 boundedJ = jacobian / (1.0f + length(jacobian));
        const float2 tilted = float2(uv.x - uv.y / sqrt(3.0f), uv.y * (2.0f / sqrt(3.0f))) * cellScale;
        const int2 cell = int2(floor(tilted));
        const float2 fraction = frac(tilted);
        const float third = 1.0f - fraction.x - fraction.y;
        float3 weights = third > 0.0f ? float3(third, fraction.y, fraction.x)
            : float3(-third, 1.0f-fraction.y, 1.0f-fraction.x);
        weights /= length(weights);
        float4 detail = float4(0.0f), contextValue = float4(0.0f);
        [unroll] for (uint lookup = 0u; lookup < 3u; ++lookup)
        {
            if (!nonrepeat && lookup > 0u) continue;
            int2 delta;
            if (third > 0.0f)
                delta = lookup == 0u ? int2(0,0) : lookup == 1u ? int2(0,1) : int2(1,0);
            else
                delta = lookup == 0u ? int2(1,1) : lookup == 1u ? int2(1,0) : int2(0,1);
            float2 coordinate = nonrepeat ? uv * lookupScale - nclsMetalSpatialHash22(cell + delta + hashOffset) : uv;
            coordinate.y = 1.0f - coordinate.y;
            const float weight = nonrepeat ? weights[lookup] : 1.0f;
            detail += weight * nclsMetalSpatialRead(group, true, coordinate, selected);
            contextValue += weight * nclsMetalSpatialRead(group, false, coordinate, min(selected, contextMips-1u));
        }
        finite &= all(isfinite(detail)) && all(isfinite(contextValue)) ? 1u : 0u;
        const uint baseInput = 11u + 14u * group;
        [unroll] for (uint channel = 0u; channel < 4u; ++channel)
        {
            const float scale = exp(0.25f * nclsMetalBudgetedCompiledHalf(baseWord, 24u+channel));
            const float bias = 0.1f * nclsMetalBudgetedCompiledHalf(baseWord, 28u+channel);
            decoderInput[baseInput+channel] = proposalInput[8u+8u*group+channel] = detail[channel]*scale+bias;
            decoderInput[baseInput+4u+channel] = proposalInput[12u+8u*group+channel] = contextValue[channel]*scale+bias;
            decoderInput[baseInput+8u+channel] = boundedJ[channel];
            if (group == 0u) result.viewState[3u+channel] = boundedJ[channel];
        }
        decoderInput[baseInput+12u] = frac(lod);
        decoderInput[baseInput+13u] = 1.0f;
        if (group == 0u) result.viewState[7] = frac(lod);
    }
    float hidden0[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    float hidden1[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    float semantic[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    nclsMetalBudgetedDense(NCLS_METAL_BUDGETED_W_PREPARED_MODEL_SEMANTIC_DECODER_0_WEIGHT,
        NCLS_METAL_BUDGETED_W_PREPARED_MODEL_SEMANTIC_DECODER_0_BIAS, 137u, 32u, 1u, decoderInput, hidden0);
    nclsMetalBudgetedDense(NCLS_METAL_BUDGETED_W_PREPARED_MODEL_SEMANTIC_DECODER_2_WEIGHT,
        NCLS_METAL_BUDGETED_W_PREPARED_MODEL_SEMANTIC_DECODER_2_BIAS, 32u, 32u, 1u, hidden0, hidden1);
    nclsMetalBudgetedDense(NCLS_METAL_BUDGETED_W_PREPARED_MODEL_SEMANTIC_DECODER_4_WEIGHT,
        NCLS_METAL_BUDGETED_W_PREPARED_MODEL_SEMANTIC_DECODER_4_BIAS, 32u, 24u, 0u, hidden1, semantic);
    [unroll] for (uint i = 0u; i < 24u; ++i)
        result.semantic[i] = semantic[i] + (i < 4u ? decoderInput[11u+i] : 0.0f);
    [unroll] for (uint lobe = 0u; lobe < 2u; ++lobe)
    {
        float value[8];
        nclsMetalBudgetedModulateLobe(baseWord, lobe, result.semantic, value);
        [unroll] for (uint i = 0u; i < 8u; ++i) result.lobes[8u*lobe+i] = value[i];
        result.compactFrame[2u*lobe] = 0.75f*tanh(result.semantic[2u*lobe]);
        result.compactFrame[2u*lobe+1u] = 0.75f*tanh(result.semantic[2u*lobe+1u]);
        result.compactFrame[4u+lobe] = cos(value[6]);
        result.compactFrame[6u+lobe] = sin(value[6]);
    }
    float proposal[NCLS_METAL_BUDGETED_MAX_WIDTH] = {};
    nclsMetalBudgetedDense(NCLS_METAL_BUDGETED_W_PREPARED_MODEL_PROPOSAL_ADAPTER_0_WEIGHT,
        NCLS_METAL_BUDGETED_W_PREPARED_MODEL_PROPOSAL_ADAPTER_0_BIAS, 80u, 16u, 1u, proposalInput, hidden0);
    nclsMetalBudgetedDense(NCLS_METAL_BUDGETED_W_PREPARED_MODEL_PROPOSAL_ADAPTER_2_WEIGHT,
        NCLS_METAL_BUDGETED_W_PREPARED_MODEL_PROPOSAL_ADAPTER_2_BIAS, 16u, 13u, 0u, hidden0, proposal);
    float3 rawWeights;
    [unroll] for (uint i = 0u; i < 3u; ++i)
    {
        float clue = 1.0f;
        if (i < 2u)
        {
            const uint lobe = 8u+8u*i;
            const float3 color = float3(nclsMetalBudgetedCompiledHalf(baseWord,lobe), nclsMetalBudgetedCompiledHalf(baseWord,lobe+1u), nclsMetalBudgetedCompiledHalf(baseWord,lobe+2u));
            clue = max(dot(color, float3(0.2126f,0.7152f,0.0722f)) * nclsMetalBudgetedCompiledHalf(baseWord,lobe+5u) * nclsMetalBudgetedCompiledHalf(baseWord,lobe+7u), 1e-5f);
            const float angle = nclsMetalBudgetedCompiledHalf(baseWord,lobe+6u) + 0.5f*3.14159265358979323846f*tanh(proposal[4u+i]);
            result.compactProposalFrame[2u*i] = 0.75f*tanh(proposal[2u*i]);
            result.compactProposalFrame[2u*i+1u] = 0.75f*tanh(proposal[2u*i+1u]);
            result.compactProposalFrame[4u+i] = cos(angle);
            result.compactProposalFrame[6u+i] = sin(angle);
        }
        rawWeights[i] = nclsMetalBudgetedCompiledHalf(baseWord,32u+i)*clue*exp(clamp(proposal[10u+i],-4.0f,4.0f));
        [unroll] for (uint axis = 0u; axis < 2u; ++axis)
            result.proposal[4u*i+1u+axis] = i < 2u ? clamp(nclsMetalBudgetedCompiledHalf(baseWord,11u+8u*i+axis)*exp(0.75f*tanh(proposal[6u+2u*i+axis])),0.015f,1.0f) : 1.0f;
        result.proposal[4u*i+3u] = i == 0u ? float(nclsMetalBudgetedCompiledFlag(baseWord,6u)) : i == 1u ? 0.0f : 2.0f;
    }
    const float3 weights = 0.98f*rawWeights/max(rawWeights.x+rawWeights.y+rawWeights.z,1e-12f) + float3(0.0f,0.0f,0.02f);
    [unroll] for (uint i = 0u; i < 3u; ++i) result.proposal[4u*i] = weights[i];
    [unroll] for (uint i = 0u; i < 24u; ++i)
    {
        result.semantic[i] = nclsMetalSpatialHalf(result.semantic[i]);
        finite &= isfinite(result.semantic[i]) ? 1u : 0u;
    }
    [unroll] for (uint i = 0u; i < 16u; ++i)
    {
        result.lobes[i] = nclsMetalSpatialHalf(result.lobes[i]);
        finite &= isfinite(result.lobes[i]) ? 1u : 0u;
    }
    [unroll] for (uint i = 0u; i < 12u; ++i)
    {
        result.proposal[i] = nclsMetalSpatialHalf(result.proposal[i]);
        if (i % 4u == 1u || i % 4u == 2u) result.proposal[i] = max(result.proposal[i], 0.015f);
        finite &= isfinite(result.proposal[i]) ? 1u : 0u;
    }
    [unroll] for (uint i = 0u; i < 8u; ++i)
    {
        result.viewState[i] = nclsMetalSpatialHalf(result.viewState[i]);
        result.compactFrame[i] = nclsMetalSpatialHalf(result.compactFrame[i]);
        result.compactProposalFrame[i] = nclsMetalSpatialHalf(result.compactProposalFrame[i]);
        finite &= isfinite(result.viewState[i]) && isfinite(result.compactFrame[i])
            && isfinite(result.compactProposalFrame[i]) ? 1u : 0u;
    }
    [unroll] for (uint i = 0u; i < 2u; ++i)
    {
        result.frames[i] = nclsMetalBudgetedLocalFrame(float2(result.compactFrame[2u*i],result.compactFrame[2u*i+1u]), result.lobes[8u*i+6u]);
        result.proposalFrames[i] = nclsMetalBudgetedLocalFrame(float2(result.compactProposalFrame[2u*i],result.compactProposalFrame[2u*i+1u]), atan2(result.compactProposalFrame[6u+i],result.compactProposalFrame[4u+i]));
    }
    [unroll] for (uint i = 0u; i < 4u; ++i) result.accessState[i] = nclsMetalBudgetedCompiledHalf(baseWord,40u+i);
    result.identityAndFlags[0] = nclsMetalBudgetedCompiledFlag(baseWord,5u);
    result.identityAndFlags[2] = nclsMetalBudgetedCompiledFlag(baseWord,6u);
    result.identityAndFlags[3] = nclsMetalBudgetedCompiledHalf(baseWord,47u) > 0.5f ? 1u : 0u;
    result.valid = finite != 0u && wo.z > 0.0f ? 1u : 0u;
    return result;
}

#endif
