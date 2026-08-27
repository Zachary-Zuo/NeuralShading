#include "../../../shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_pack.slang"

Texture2D<float4> gLatent0;
Texture2D<float4> gLatent1;
SamplerState gLatentSampler;
RWStructuredBuffer<float4> gOutput;

[numthreads(3, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    NclsNvidiaNeuralCompiledMaterial material = {};
    material.width = 4u;
    material.height = 4u;
    material.mipCount = 2u;
    material.layoutVersion = 2u;

    float2 uv = float2(0.375f, 0.375f);
    float2 uvDx = float2(exp2(0.25f) / 4.0f, 0.0f);
    float2 uvDy = float2(0.0f, exp2(0.25f) / 4.0f);
    float filterRandom = index == 0u ? 0.20f : 0.30f;
    if (index == 2u)
    {
        uv = float2(0.0f);
        uvDx = float2(0.0f);
        uvDy = float2(0.0f);
        filterRandom = 0.5f;
    }

    float latent[NCLS_NVIDIA_NEURAL_LATENT_COUNT];
    nclsNvidiaNeuralFetchLatent(
        material, gLatent0, gLatent1, gLatentSampler,
        uv, uvDx, uvDy, filterRandom, latent);
    gOutput[2u * index + 0u] = float4(latent[0], latent[1], latent[2], latent[3]);
    gOutput[2u * index + 1u] = float4(latent[4], latent[5], latent[6], latent[7]);
}
