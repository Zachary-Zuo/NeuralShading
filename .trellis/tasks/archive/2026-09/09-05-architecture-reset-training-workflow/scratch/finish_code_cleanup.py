from pathlib import Path
import ast
import re

root = Path.cwd()
def modify(path, function):
    old = path.read_text(encoding='utf-8')
    new = function(old)
    if old != new:
        path.write_text(new, encoding='utf-8')
def remove_function(value, name):
    tree = ast.parse(value)
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    lines = value.splitlines()
    lines[node.lineno - 1:node.end_lineno] = []
    return '\n'.join(lines) + '\n'
modify(root / 'src/ncls/learning/training/engine.py', lambda s: remove_function(s, '_component_manifest'))
modify(root / 'src/ncls/learning/training/launch.py', lambda s: s.replace('    declared_world = environment.get("NCLS_DDP_WORLD_SIZE")\n    if declared_world is not None and int(declared_world) != world:\n        raise RuntimeError("distributed worker declared world size disagrees")\n', ''))
modify(root / 'src/ncls/learning/training/plan.py', lambda s: remove_function(s, 'to_runtime_config'))

names = {'get_method_plugin': 'get_method', 'method_plugins': 'registered_methods', 'public_method_keys': 'method_keys'}
for folder in ('src', 'tests', 'tools', 'docs', '.trellis/spec'):
    for path in (root / folder).rglob('*'):
        if path.suffix not in {'.py', '.md'}:
            continue
        if folder == 'docs' and 'research' in path.parts:
            continue
        def replace(s):
            for before, after in names.items():
                s = re.sub(r'\b' + before + r'\b', after, s)
            return s.replace('.to_runtime_config()', '.training')
        modify(path, replace)

path = root / 'src/ncls/learning/methods/registry.py'
modify(path, lambda s: s.replace('_PLUGINS', '_METHODS').replace('method plugin', 'method'))
path = root / 'src/ncls/learning/methods/__init__.py'
path.write_text('''from ncls.learning.method import Method
from .registry import get_method, registered_methods, method_keys, reset_method_registry_for_test

__all__ = ["Method", "get_method", "registered_methods", "method_keys", "reset_method_registry_for_test"]
''', encoding='utf-8')

# 单一方法消费者的短 helper 归回其所有者，不为退役实现留下共享模块。
for file, owner, name in [('quantization.py', 'methods/metal/method.py', 'fake_quantize_fp16_ste'),
                          ('texture_roles.py', 'methods/metal/native_assets.py', 'semantic_role_class')]:
    source = root / 'src/ncls/learning' / file
    text = source.read_text(encoding='utf-8')
    node = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == name)
    function = ast.get_source_segment(text, node)
    target = root / 'src/ncls/learning' / owner
    value = target.read_text(encoding='utf-8').replace(f'from ncls.learning.{source.stem} import {name}\n', '')
    tree = ast.parse(value)
    end = max(node.end_lineno for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)))
    lines = value.splitlines()
    lines[end:end] = ['\n\n' + function + '\n']
    target.write_text('\n'.join(lines) + '\n', encoding='utf-8')

# 清理由重组产生的重复 import 和多余空行，仅限本任务触及的 learning 文件。
for path in (root / 'src/ncls/learning').rglob('*.py'):
    def clean(value):
        seen = set()
        lines = []
        for line in value.splitlines():
            if line.startswith('from ') and ' import ' in line and not line.endswith('('):
                if line in seen:
                    continue
                seen.add(line)
            lines.append(line)
        return re.sub(r'\n{4,}', '\n\n\n', '\n'.join(lines) + '\n')
    modify(path, clean)
