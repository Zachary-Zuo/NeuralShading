from pathlib import Path
import ast

root = Path(__file__).resolve().parents[4]

def read(name):
    return (root / name).read_text(encoding='utf-8')

def write(name, text):
    (root / name).write_text(text, encoding='utf-8', newline='\n')

def remove_functions(text, names):
    lines = text.splitlines(keepends=True)
    nodes = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in reversed(nodes):
        first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        del lines[first-1:node.end_lineno]
    return ''.join(lines)

name = 'src/ncls/learning/method.py'
value = read(name)
start = value.index('@dataclass(frozen=True)\nclass MethodReadinessPolicy:')
end = value.index('@dataclass(frozen=True)\nclass MethodDescriptor:', start)
value = value[:start] + value[end:]
value = value.replace('    readiness_policies: Mapping[str, MethodReadinessPolicy] = field(default_factory=dict)\n', '')
start = value.index('        policies = {str(name): value for name, value in self.readiness_policies.items()}')
end = value.index('        object.__setattr__(self, "supported_sources"', start)
value = value[:start] + value[end:]
value = value.replace('        object.__setattr__(self, "readiness_policies", policies)\n','')
start = value.index('        if self.readiness_policies:')
end = value.index('        return result', start)
value = value[:start] + value[end:]
write(name, value)
name = 'src/ncls/learning/methods/metal_budgeted.py'
value = read(name).replace('    MethodReadinessPolicy,\n', '')
start = value.index('        readiness_policies={')
end = value.index('    )\n', start)
value = value[:start] + value[end:]
write(name, value)
name = 'src/ncls/learning/methods/nvidia.py'
value = read(name)
start = value.index('_FORMAL_RECIPE_ID = ')
end = value.index('def _fp16_fma_dense(', start)
value = value[:start] + value[end:]
write(name, value)

name = 'tests/unit/test_methods.py'
value = remove_functions(read(name), {'test_method_readiness_policy_is_owned_and_validated_by_the_descriptor'})
value = value.replace('MethodReadinessPolicy, ', '').replace(', MethodReadinessPolicy', '')
write(name, value)
name = 'tests/unit/test_metal_budgeted_method.py'
value = read(name)
start = value.index('    readiness = descriptor.readiness_policies')
end = value.index('\ndef ', start)
value = value[:start] + '\n' + value[end:]
write(name, value)
name = 'tests/unit/test_method_plugin.py'
value = remove_functions(read(name), {'test_method_plugin_rejects_data_requirement_drift'})
start = value.index('        assert set(plugin.facet_identities)')
end = value.index('        assert {', start)
value = value[:start] + value[end:]
write(name, value.replace('test_product_method_plugins_have_short_keys_and_complete_facets', 'test_registered_methods_directly_provide_their_data_requirements'))
name = 'tests/unit/test_nvidia_faithful_contract.py'
value = read(name)
node = next(n for n in ast.parse(value).body if isinstance(n, ast.FunctionDef) and n.name == 'test_nvidia_formal_recipe_rejects_budget_adaptations')
lines = value.splitlines(keepends=True)
lines[node.lineno-1:node.end_lineno] = ['''def test_nvidia_yaml_controls_training_budget_without_recipe_gates():
    config = TrainingPlanResolver(Path(__file__).resolve().parents[2]).resolve(
        "configs/training/runs/nvidia-materialx-formal.yaml"
    ).training.to_dict()
    config["seed"] += 1
    config["correspondence_id"] = "new experiment"
    config["validation"] = {"interval": 7, "batches": 2}
    config["phases"][0]["routes"][0]["batch_size"] = 8
    METHOD.validate_training_config(config)
''']
write(name, ''.join(lines))
name = 'tests/unit/test_training_distributed.py'
value = read(name).replace('    def compute(', '    def training_objective(')
value = value.replace('    plugin = replace(\n        _plugin(_PhaseMethod()),\n        checkpoint=_ForbiddenCheckpointCodec(),\n    )', '''    class MethodWithoutCheckpoint(_PhaseMethod):
        def export_training_state(self, model):
            raise AssertionError("non-rank0 must not encode weights")

    plugin = MethodWithoutCheckpoint()''')
write(name, value)
name = 'tests/unit/test_training_yaml.py'
value = read(name).replace('format_name: ncls.training-run\nformat_version: 1\n','')
start = value.index('format_name: ncls.training-fragment')
end = value.index('"""', start)
value = value[:start] + 'extends: {parent}\n' + value[end:]
value = value.replace('training fragment inheritance cycle:', '配置继承成环：')
write(name, value)
name = 'tests/unit/test_prepare_metal_catalog.py'
value = read(name).replace('public_method_key=', 'method_key=').replace('deployment_payload=', 'model_payload=')
value = value.replace('    calls: list[str] = []\n','').replace('        require_ready=lambda mode: calls.append(mode) or {"ready": True},\n        readiness_calls=calls,\n', '')
start = value.index('    assert hybrid.readiness_calls')
end = value.index('\ndef ', start)
value = value[:start] + '\n' + value[end:]
write(name, value)
name = 'tests/unit/test_viewer_slots.py'
value = read(name).replace('    assert \'"checkpoint_compatibility"\' in exporter\n', '    assert \'"training_diagnostics"\' in exporter\n')
write(name, value)

# 现有 GPU 工具共用 Python runtime；不保留 shell 训练入口。
name = 'scripts/build_reference_backend.ps1'
write(name, read(name).replace('& (Join-Path $PSScriptRoot "run_falcor_python.ps1") -m ncls.cli reference probe', '& conda run --no-capture-output -n neural-shading python -m ncls reference probe'))
name = 'scripts/deploy_reference_linux.sh'
write(name, read(name).replace('bash "${project_root}/scripts/run_falcor_python.sh" -m ncls.cli reference probe', 'conda run --no-capture-output -n neural-shading python -m ncls reference probe --device "${CUDA_VISIBLE_DEVICES:-0}"'))
for name in ('scripts/run_mdl_reference_parity.ps1', 'scripts/run_mdl_native_parity.ps1'):
    write(name, read(name).replace('& (Join-Path $projectRoot "scripts\\run_falcor_python.ps1")', '& conda run --no-capture-output -n neural-shading python -m ncls.runtime --'))
name = 'scripts/prepare_metal_viewer.ps1'
value = read(name).replace('[string]$OutputRoot = "artifacts\\viewer\\metal-budgeted-pair"', '[string]$OutputRoot')
value = value.replace('$ErrorActionPreference = "Stop"', '''if (-not $OutputRoot) {
    $checkpointDirectory = Split-Path -Parent (Resolve-Path -LiteralPath $HybridCheckpoint).Path
    $OutputRoot = Join-Path (Split-Path -Parent $checkpointDirectory) "exports\\viewer"
}

$ErrorActionPreference = "Stop"''')
value = value.replace('& (Join-Path $PSScriptRoot "run_falcor_python.ps1") @prepareArguments', '& conda run --no-capture-output -n neural-shading python -m ncls.runtime -- @prepareArguments')
write(name, value)
name = 'tools/reference/reference_backend_deploy.py'
write(name, read(name).replace('scripts/run_falcor_python.sh -m ncls.cli reference probe', 'python -m ncls reference probe'))
name = 'src/ncls/commands.py'
write(name, read(name).replace('    reference_commands.add_parser(\n        "probe", help="用仓库 fixture 验证 device、LayerStack 与 MDL compile/query"\n    )', '    probe = reference_commands.add_parser(\n        "probe", help="用仓库 fixture 验证 device、LayerStack 与 MDL compile/query"\n    )\n    probe.add_argument("--device", type=int, default=0)'))
