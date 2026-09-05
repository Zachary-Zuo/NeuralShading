from pathlib import Path
import ast
import json

root = Path(__file__).resolve().parents[4]

def text(name):
    return (root / name).read_text(encoding='utf-8')

def write(name, value):
    (root / name).write_text(value, encoding='utf-8', newline='\n')

def function(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno]) + '\n'

write('src/ncls/learning/quantization.py', 'import torch\n\n\n' + function(text('src/ncls/learning/metal_runtime.py'), 'fake_quantize_fp16_ste'))
write('src/ncls/learning/texture_roles.py', '"""源纹理通道按当前编码器消费方式分类。"""\n\n' + function(text('src/ncls/learning/models/metal_texture_codec.py'), 'semantic_role_class'))
for name, previous, new in [
    ('src/ncls/learning/methods/metal_budgeted.py', 'from ncls.learning.metal_runtime import fake_quantize_fp16_ste', 'from ncls.learning.quantization import fake_quantize_fp16_ste'),
    ('src/ncls/learning/mdl_metal_assets.py', 'from ncls.learning.models.metal_texture_codec import semantic_role_class', 'from ncls.learning.texture_roles import semantic_role_class'),
]:
    write(name, text(name).replace(previous, new))
write('src/ncls/learning/models/__init__.py', '"""当前方法模型；具体方法显式导入所需实现。"""\n')

name = 'src/ncls/learning/source_adapters.py'
value = text(name)
nodes = {node.name: node for node in ast.parse(value).body if isinstance(node, ast.ClassDef)}
base, derived = nodes['MetalFusedMdlSourceAdapter'], nodes['MetalBudgetedMdlSourceAdapter']
lines = value.splitlines()
base_text = '\n'.join(lines[base.lineno-1:base.end_lineno])
derived_text = '\n'.join(lines[derived.lineno-1:derived.end_lineno])
sample = derived_text[derived_text.index('    def sample_tensors('):]
sample = sample.replace('super().sample_tensors(', 'self._sample_tensors(')
base_text = base_text.replace('class MetalFusedMdlSourceAdapter', 'class MetalBudgetedMdlSourceAdapter').replace('metal-fused-neural-material', 'metal-budgeted-neural-material').replace('metal-fused.mdl-vmaterials2-metal@1', 'metal-budgeted.mdl-vmaterials2-metal@1').replace('    def sample_tensors(', '    def _sample_tensors(')
base_text = base_text.replace('        if len(semantic_indices) != 154:\n            raise ValueError("Metal typed semantic table drifted from the opaque audit")\n', '')
lines[derived.lineno-1:derived.end_lineno] = []
lines[base.lineno-1:base.end_lineno] = [base_text + '\n\n' + sample]
write(name, '\n'.join(lines).replace('    "MetalFusedMdlSourceAdapter",\n', '') + '\n')

layout = json.loads(text('src/ncls/learning/abi/metal_fused_layout_v1.json'))
proposal = layout['proposal_reservation']
write('src/ncls/learning/abi/metal_proposal.json', json.dumps(proposal, indent=2) + '\n')
name = 'src/ncls/learning/models/metal_sampler.py'
value = text(name)
value += '''\n\ndef load_proposal_layout():
    import json
    from ncls.paths import PROJECT_ROOT

    return json.loads((PROJECT_ROOT / "src/ncls/learning/abi/metal_proposal.json").read_text(encoding="utf-8"))
'''
write(name, value)
name = 'tools/learning/generate_metal_sampler_layout.py'
value = text(name)
start = value.index('from ncls.learning.models.metal_fused_profile import (')
end = value.index('\n)', start) + 2
value = value[:start] + 'from ncls.learning.models.metal_sampler import load_proposal_layout\nfrom ncls.core.identity import sha256_json' + value[end:]
value = value.replace('shaders/ncls/backends/metal_fused/metal_fused_layout.generated.slang', 'shaders/ncls/scattering/metal_proposal_layout.generated.slang')
value = value.replace('    layout = load_metal_fused_layout(METAL_FUSED_LAYOUT_PATH)\n    proposal = layout["proposal_reservation"]', '    proposal = load_proposal_layout()')
value = value.replace("{layout['identity']}", '{sha256_json(proposal)}').replace('NCLS_METAL_FUSED_LAYOUT_GENERATED_SLANG', 'NCLS_METAL_PROPOSAL_LAYOUT_GENERATED_SLANG')
write(name, value)
for source, target in [
    ('shaders/ncls/scattering/metal_fused_proposal.slang', 'shaders/ncls/scattering/metal_proposal.slang'),
    ('tests/gpu/kernels/metal_fused_proposal.cs.slang', 'tests/gpu/kernels/metal_proposal.cs.slang'),
    ('tests/gpu/test_metal_fused_sampler.py', 'tests/gpu/test_metal_proposal.py'),
]:
    value = text(source).replace('metal_fused_proposal', 'metal_proposal').replace('MetalFusedProposal', 'MetalProposal').replace('NCLS_METAL_FUSED_PROPOSAL', 'NCLS_METAL_PROPOSAL')
    value = value.replace('../backends/metal_fused/metal_fused_layout.generated.slang', 'metal_proposal_layout.generated.slang')
    write(target, value)
name = 'tests/unit/test_metal_sampler.py'
value = text(name).replace('from ncls.learning.models.metal_fused_profile import load_metal_fused_layout', 'from ncls.learning.models.metal_sampler import load_proposal_layout').replace('load_metal_fused_layout()["proposal_reservation"]', 'load_proposal_layout()')
write(name, value)
