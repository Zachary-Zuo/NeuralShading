#ifndef NCLS_METAL_FUSED_COMPILER_HEADS_SLANG
#define NCLS_METAL_FUSED_COMPILER_HEADS_SLANG

void nclsMetalWriteCompilerHeads(float latent[NCLS_METAL_MAX_WIDTH])
{
    [unroll] for (uint index = 0u; index < 64u; ++index)
        gNclsCompiledMaterials[NCLS_METAL_COMPILED_COMPILER + index] = asuint(latent[index]);
    float head[NCLS_METAL_MAX_WIDTH] = {};
    nclsMetalDense(NCLS_METAL_W_TYPED_COMPILER_SPATIAL_HEAD_WEIGHT,
        NCLS_METAL_W_TYPED_COMPILER_SPATIAL_HEAD_BIAS, 64u, 64u, 0u, latent, head);
    [unroll] for (uint index = 0u; index < 64u; ++index)
        gNclsCompiledMaterials[NCLS_METAL_COMPILED_SPATIAL + index] = asuint(tanh(head[index]));
    nclsMetalDense(NCLS_METAL_W_TYPED_COMPILER_CORE_HEAD_WEIGHT,
        NCLS_METAL_W_TYPED_COMPILER_CORE_HEAD_BIAS, 64u, 54u, 0u, latent, head);
    [unroll]
    for (uint lobe = 0u; lobe < 6u; ++lobe)
    {
        const uint offset = 9u * lobe;
        [unroll] for (uint channel = 0u; channel < 3u; ++channel)
            gNclsCompiledMaterials[NCLS_METAL_COMPILED_CORE + offset + channel]
                = asuint(1.0f / (1.0f + exp(-head[offset + channel])));
        [unroll] for (uint channel = 3u; channel < 5u; ++channel)
            gNclsCompiledMaterials[NCLS_METAL_COMPILED_CORE + offset + channel]
                = asuint(0.015f + 0.985f / (1.0f + exp(-head[offset + channel])));
        gNclsCompiledMaterials[NCLS_METAL_COMPILED_CORE + offset + 5u] = asuint(tanh(head[offset + 5u]));
        gNclsCompiledMaterials[NCLS_METAL_COMPILED_CORE + offset + 6u] = asuint(nclsMetalSoftplus(head[offset + 6u]));
        gNclsCompiledMaterials[NCLS_METAL_COMPILED_CORE + offset + 7u] = asuint(1.0f / (1.0f + exp(-head[offset + 7u])));
        gNclsCompiledMaterials[NCLS_METAL_COMPILED_CORE + offset + 8u] = asuint(1.0f / (1.0f + exp(-head[offset + 8u])));
    }
    nclsMetalDense(NCLS_METAL_W_TYPED_COMPILER_RESIDUAL_HEAD_WEIGHT,
        NCLS_METAL_W_TYPED_COMPILER_RESIDUAL_HEAD_BIAS, 64u, 28u, 0u, latent, head);
    [unroll]
    for (uint lobe = 0u; lobe < 4u; ++lobe)
    {
        const uint offset = 7u * lobe;
        [unroll] for (uint channel = 0u; channel < 3u; ++channel)
            gNclsCompiledMaterials[NCLS_METAL_COMPILED_RESIDUAL + offset + channel]
                = asuint(nclsMetalSoftplus(head[offset + channel]));
        [unroll] for (uint channel = 3u; channel < 5u; ++channel)
            gNclsCompiledMaterials[NCLS_METAL_COMPILED_RESIDUAL + offset + channel]
                = asuint(0.02f + 0.98f / (1.0f + exp(-head[offset + channel])));
        gNclsCompiledMaterials[NCLS_METAL_COMPILED_RESIDUAL + offset + 5u]
            = asuint(1.0f / (1.0f + exp(-head[offset + 5u])));
        gNclsCompiledMaterials[NCLS_METAL_COMPILED_RESIDUAL + offset + 6u] = asuint(tanh(head[offset + 6u]));
    }
    nclsMetalDense(NCLS_METAL_W_TYPED_COMPILER_BLOCK_HEAD_WEIGHT,
        NCLS_METAL_W_TYPED_COMPILER_BLOCK_HEAD_BIAS, 64u, 32u, 0u, latent, head);
    [unroll] for (uint index = 0u; index < 32u; ++index)
        gNclsCompiledMaterials[NCLS_METAL_COMPILED_BLOCK + index] = asuint(tanh(head[index]));
    nclsMetalDense(NCLS_METAL_W_TYPED_COMPILER_PROPOSAL_HEAD_WEIGHT,
        NCLS_METAL_W_TYPED_COMPILER_PROPOSAL_HEAD_BIAS, 64u, 44u, 0u, latent, head);
    [unroll]
    for (uint component = 0u; component < 11u; ++component)
    {
        gNclsCompiledMaterials[NCLS_METAL_COMPILED_PROPOSAL_LOGITS + component]
            = asuint(head[4u * component]);
        [unroll] for (uint channel = 0u; channel < 3u; ++channel)
            gNclsCompiledMaterials[NCLS_METAL_COMPILED_PROPOSAL_MODULATION + 3u * component + channel]
                = asuint(tanh(head[4u * component + 1u + channel]));
    }
    nclsMetalDense(NCLS_METAL_W_TYPED_COMPILER_BOUNDS_HEAD_WEIGHT,
        NCLS_METAL_W_TYPED_COMPILER_BOUNDS_HEAD_BIAS, 64u, 3u, 0u, latent, head);
    gNclsCompiledMaterials[NCLS_METAL_COMPILED_TAIL_BOUNDS]
        = asuint(0.25f + 2.75f / (1.0f + exp(-head[0])));
    gNclsCompiledMaterials[NCLS_METAL_COMPILED_TAIL_BOUNDS + 1u]
        = asuint(0.01f + 0.49f / (1.0f + exp(-head[1])));
    gNclsCompiledMaterials[NCLS_METAL_COMPILED_TAIL_BOUNDS + 2u]
        = asuint(1.0f / (1.0f + exp(-head[2])));
    [unroll] for (uint index = 0u; index < 16u; ++index)
        gNclsCompiledMaterials[NCLS_METAL_COMPILED_ACCESS + index]
            = gNclsMetalRawParameters[NCLS_METAL_RAW_ACCESS + index];
    [unroll] for (uint index = 0u; index < 8u; ++index)
        gNclsCompiledMaterials[NCLS_METAL_COMPILED_FRAME + index]
            = gNclsMetalRawParameters[NCLS_METAL_RAW_FRAME + index];
}

#endif
