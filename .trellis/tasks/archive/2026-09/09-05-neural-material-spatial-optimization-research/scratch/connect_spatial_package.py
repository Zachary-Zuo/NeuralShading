from pathlib import Path

path = Path('src/ncls/learning/methods/metal/method.py')
text = path.read_text(encoding='utf-8')
start = text.index('        detail_height, detail_width = asset.detail_levels[0].shape[:2]', text.index('    def compile_asset('))
end = text.index('    def compile_instance(', start)
text = text[:start] + '\n' + text[end:]
text = text.replace('from ncls.learning.methods.metal.asset_cook import MetalBudgetedAssetCompiler',
                    'from ncls.learning.methods.metal.spatial_runtime import SPATIAL_COMPILED_WORD_COUNT')
text = text.replace('    METAL_BUDGETED_COMPILED_WORD_COUNT,\n', '')
text = text.replace('METAL_BUDGETED_COMPILED_WORD_COUNT', 'SPATIAL_COMPILED_WORD_COUNT')
text = text.replace('"dtype": "ncls-metal-budgeted-compiled-material@1"', '"dtype": "ncls-metal-spatial-compiled-material@1"')
text = text.replace('"prepared_state_bytes": 160,\n                "asset_reads": 2,',
                    '"prepared_state_bytes": model.profile.prepared_state_bytes,\n                "asset_reads": asset.texture_reads,')
text = text.replace('"B_asset": sum(\n                    level.nbytes\n                    for levels in (asset.detail_levels, asset.context_levels)\n                    for level in levels\n                ),', '"B_asset": asset.latent_bytes,')
path.write_text(text, encoding='utf-8', newline='\n')

path = Path('shaders/ncls/backends/metal_budgeted/metal_budgeted_common.slang')
text = path.read_text(encoding='utf-8').replace('static const uint NCLS_METAL_BUDGETED_MAX_WIDTH = 64u;', '''#ifdef NCLS_METAL_SPATIAL
static const uint NCLS_METAL_BUDGETED_MAX_WIDTH = 137u;
static const uint NCLS_METAL_BUDGETED_COMPILED_WORD_COUNT = 192u;
static const uint NCLS_METAL_BUDGETED_COMPILED_LAYOUT_VERSION = 2u;
#else
static const uint NCLS_METAL_BUDGETED_MAX_WIDTH = 64u;''')
text = text.replace('static const uint NCLS_METAL_BUDGETED_COMPILED_LAYOUT_VERSION = 1u;',
                    'static const uint NCLS_METAL_BUDGETED_COMPILED_LAYOUT_VERSION = 1u;\n#endif')
text = text.replace('    float accessState[4];', '''    float accessState[4];
#ifdef NCLS_METAL_SPATIAL
    float compactProposalFrame[8];
    NclsMetalBudgetedFrame proposalFrames[2];
#endif''')
text = text.replace('    uint words[40];', '''#ifdef NCLS_METAL_SPATIAL
    uint words[44];
#else
    uint words[40];
#endif''')
path.write_text(text, encoding='utf-8', newline='\n')

path = Path('shaders/ncls/backends/metal_budgeted/metal_budgeted.slang')
text = path.read_text(encoding='utf-8')
begin = text.index('Texture2D<float4> gNclsMetalBudgetedDetail;')
end = text.index('\n#include "metal_budgeted_asset.slang"', begin)
text = text[:begin] + '#ifdef NCLS_METAL_SPATIAL\n' + '\n'.join(
    f'Texture2D<float4> gNclsMetalSpatialDetail{i};\nTexture2D<float4> gNclsMetalSpatialContext{i};\nSamplerState gNclsMetalSpatialSampler{i};' for i in range(9)
) + '\n#else\n' + text[begin:end] + '\n#endif\n' + text[end:]
text = text.replace('    [unroll]\n    for (uint index = 0u; index < 4u; ++index)\n        packed.words[36u + index]', '''#ifdef NCLS_METAL_SPATIAL
    [unroll] for (uint index = 0u; index < 8u; ++index)
        values[index] = state.prepared.compactProposalFrame[index];
    nclsMetalBudgetedPackHalfArray(packed, 36u, 8u, values);
    const uint flagsOffset = 40u;
#else
    const uint flagsOffset = 36u;
#endif
    [unroll]
    for (uint index = 0u; index < 4u; ++index)
        packed.words[flagsOffset + index]''')
text = text.replace('packed.words[39]', 'packed.words[flagsOffset + 3u]')
text = text.replace('    [unroll]\n    for (uint index = 0u; index < 4u; ++index)\n        state.prepared.identityAndFlags[index] = packed.words[36u + index];', '''#ifdef NCLS_METAL_SPATIAL
    [unroll] for (uint index = 0u; index < 8u; ++index)
        state.prepared.compactProposalFrame[index] = nclsMetalBudgetedUnpackHalf(packed, 72u + index);
    [unroll] for (uint frame = 0u; frame < 2u; ++frame)
        state.prepared.proposalFrames[frame] = nclsMetalBudgetedLocalFrame(
            float2(state.prepared.compactProposalFrame[2u*frame], state.prepared.compactProposalFrame[2u*frame+1u]),
            atan2(state.prepared.compactProposalFrame[6u+frame], state.prepared.compactProposalFrame[4u+frame]));
    const uint flagsOffset = 40u;
#else
    const uint flagsOffset = 36u;
#endif
    [unroll]
    for (uint index = 0u; index < 4u; ++index)
        state.prepared.identityAndFlags[index] = packed.words[flagsOffset + index];''')
path.write_text(text, encoding='utf-8', newline='\n')

path = Path('shaders/ncls/backends/metal_budgeted/metal_budgeted_sampler.slang')
text = path.read_text(encoding='utf-8').replace('    const NclsMetalBudgetedFrame frame = state.frames[component == 1u ? 1u : 0u];', '''#ifdef NCLS_METAL_SPATIAL
    const NclsMetalBudgetedFrame frame = state.proposalFrames[component == 1u ? 1u : 0u];
#else
    const NclsMetalBudgetedFrame frame = state.frames[component == 1u ? 1u : 0u];
#endif''')
path.write_text(text, encoding='utf-8', newline='\n')

path = Path('shaders/ncls/backends/metal_budgeted/metal_budgeted_asset.slang')
text = path.read_text(encoding='utf-8').replace('NclsMetalBudgetedPrepared nclsMetalBudgetedPrepare(', '#ifdef NCLS_METAL_SPATIAL\n#include "metal_spatial_asset.slang"\n#else\nNclsMetalBudgetedPrepared nclsMetalBudgetedPrepare(')
text = text[:text.rindex('#endif')] + '#endif\n\n' + text[text.rindex('#endif'):]
path.write_text(text, encoding='utf-8', newline='\n')
