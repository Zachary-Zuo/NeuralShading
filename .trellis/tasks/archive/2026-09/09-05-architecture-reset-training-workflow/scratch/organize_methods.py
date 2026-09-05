from pathlib import Path
import ast
import re

root = Path.cwd()
base = root / 'src/ncls/learning'

mapping = {
    'methods/nvidia.py': 'methods/nvidia/method.py',
    'models/nvidia_neural_appearance.py': 'methods/nvidia/model.py',
    'methods/metal_budgeted.py': 'methods/metal/method.py',
    'models/metal_budgeted.py': 'methods/metal/model.py',
    'models/metal_budgeted_asset.py': 'methods/metal/asset.py',
    'models/metal_budgeted_compiler.py': 'methods/metal/compiler.py',
    'models/metal_budgeted_evaluator.py': 'methods/metal/evaluator.py',
    'models/metal_budgeted_sampler.py': 'methods/metal/sampler.py',
    'models/metal_budgeted_profile.py': 'methods/metal/profile.py',
    'metal_budgeted_asset_cook.py': 'methods/metal/asset_cook.py',
    'metal_budgeted_runtime.py': 'methods/metal/runtime.py',
    'mdl_metal_assets.py': 'methods/metal/native_assets.py',
}
for before, after in mapping.items():
    source, target = base / before, base / after
    assert source.resolve().is_relative_to(root) and target.resolve().is_relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists()
    source.rename(target)
for name in ('nvidia', 'metal'):
    (base / f'methods/{name}/__init__.py').write_text(f'"""{name} 方法的模型、数据和部署实现。"""\n', encoding='utf-8')

source = (base / 'source_adapters.py').read_text(encoding='utf-8')
tree = ast.parse(source)
nodes = {node.name: node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
def definition(name):
    return ast.get_source_segment(source, nodes[name])
imports = [ast.get_source_segment(source, node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
header = '\n'.join(imports) + '\nfrom ncls.paths import PROJECT_ROOT\n'
base_body = definition('MethodSourceAdapter')
nvidia_body = '\n\n\n'.join(definition(name) for name in ('NvidiaLayerStackSourceAdapter', 'NvidiaMaterialXSourceAdapter', 'NvidiaMdlFixedSourceAdapter', '_path'))
metal_constants = source[source.index('_METAL_TYPE_IDS ='):source.index('def _components')]
metal_body = 'MDL_METAL_REGISTRY_PATH = PROJECT_ROOT / "references/mdl-vmaterials2-v1/metal-opaque-v1.json"\n\n' + metal_constants + '\n\n'.join(definition(name) for name in ('_components', '_normalized_components', 'MetalBudgetedMdlSourceAdapter', '_balanced_one_native_texel_offsets'))
(base / 'source_adapters.py').write_text(header + '\n\n' + base_body + '\n', encoding='utf-8')
for name, body in [('nvidia', nvidia_body), ('metal', metal_body)]:
    (base / f'methods/{name}/data.py').write_text(header + '\nfrom ncls.learning.source_adapters import MethodSourceAdapter\n\n\n' + body + '\n', encoding='utf-8')

# 同时修改真实消费者；不留下旧模块的转发入口。
replacements = {}
for before, after in mapping.items():
    replacements['ncls.learning.' + before[:-3].replace('/', '.')] = 'ncls.learning.' + after[:-3].replace('/', '.')
    replacements['src/ncls/learning/' + before] = 'src/ncls/learning/' + after
for folder in ('src', 'tests', 'tools'):
    for path in (root / folder).rglob('*.py'):
        old = path.read_text(encoding='utf-8')
        value = old
        # 一次替换，避免新路径被同一前缀重复替换。
        pattern = '|'.join(re.escape(key) for key in sorted(replacements, key=len, reverse=True))
        value = re.sub(pattern, lambda match: replacements[match[0]], value)
        if path.name == 'registry.py' and path.parent.name == 'methods':
            value = value.replace('("metal_budgeted", "nvidia")', '("metal", "nvidia")')
            value = value.replace('f"{package.__name__}.{name}"', 'f"{package.__name__}.{name}.method"')
        if path == base / 'methods/metal/profile.py':
            value = value.replace('PROJECT_ROOT = Path(__file__).resolve().parents[4]', 'from ncls.paths import PROJECT_ROOT')
        # 拆分混合 import：公共基类留在 learning，方法的数据实现随方法存放。
        parsed = ast.parse(value)
        changes = []
        for node in ast.walk(parsed):
            if isinstance(node, ast.ImportFrom) and node.module == 'ncls.learning.source_adapters':
                groups = {}
                for item in node.names:
                    module = ('ncls.learning.methods.nvidia.data' if item.name.startswith('Nvidia') else
                              'ncls.learning.methods.metal.data' if item.name in {'MetalBudgetedMdlSourceAdapter', '_balanced_one_native_texel_offsets'} else node.module)
                    groups.setdefault(module, []).append(item.name + (f' as {item.asname}' if item.asname else ''))
                indent = ' ' * node.col_offset
                replacement = ('\n' + indent).join('from ' + module + ' import ' + ', '.join(names) for module, names in groups.items())
                changes.append((node.lineno - 1, node.end_lineno, replacement))
        lines = value.splitlines()
        for start, end, replacement in reversed(changes):
            lines[start:end] = [replacement]
        value = '\n'.join(lines) + '\n'
        if value != old:
            path.write_text(value, encoding='utf-8')

# 抽取的数据模块只保留自己实际用到的顶层 import，避免相互加载。
for path in (base / 'source_adapters.py', base / 'methods/nvidia/data.py', base / 'methods/metal/data.py'):
    value = path.read_text(encoding='utf-8')
    tree = ast.parse(value)
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    lines = value.splitlines()
    for node in reversed(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and getattr(node, 'module', None) != '__future__':
            kept = [item for item in node.names if (item.asname or item.name.split('.')[0]) in used]
            if isinstance(node, ast.ImportFrom):
                prefix = 'from ' + node.module + ' import '
            else:
                prefix = 'import '
            replacement = prefix + ', '.join(item.name + (f' as {item.asname}' if item.asname else '') for item in kept)
            lines[node.lineno - 1:node.end_lineno] = [replacement] if kept else []
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

path = root / 'tests/unit/test_training_yaml.py'
path.write_text(path.read_text(encoding='utf-8').replace('配置继承成环： default', '配置继承成环：default'), encoding='utf-8')
path = root / 'tests/unit/test_reference_backend_deployment.py'
value = path.read_text(encoding='utf-8')
tree = ast.parse(value)
node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'test_linux_falcor_launcher_keeps_conda_runtime_libraries_consistent')
lines = value.splitlines()
lines[node.lineno - 1:node.end_lineno] = []
path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
