from pathlib import Path

root = Path(__file__).resolve().parents[4]
for filename in ('nvidia', 'metal_budgeted'):
    path = root / f'src/ncls/learning/methods/{filename}.py'
    value = path.read_text(encoding='utf-8')
    start = value.index('    def validate_training_config(')
    end = value.index('    def configure_phase(', start)
    if filename == 'nvidia':
        validation = '''    def validate_training_config(self, config: Mapping[str, Any]) -> None:
        for phase in config["phases"]:
            routes = {route["name"]: route["kind"] for route in phase["routes"]}
            if routes != {"evaluator": "reference-evaluator", "sampler": "method-sampler"}:
                raise ValueError("NVIDIA objective 需要 evaluator 和 sampler 两条 route")

'''
    else:
        validation = '''    def validate_training_config(self, config: Mapping[str, Any]) -> None:
        for phase in config["phases"]:
            self._calibration_recipe(phase)
            self._proposal_weight(phase)
            routes = {route["name"]: route for route in phase["routes"]}
            if set(routes) != {"evaluator", "sampler"}:
                raise ValueError("Metal objective 需要 evaluator 和 sampler 两条 route")
            if not routes["evaluator"]["options"].get("paired_uv", False):
                raise ValueError("当前 Metal 空间差分 objective 需要 paired_uv")

'''
    value = value[:start] + validation + value[end:]
    if filename == 'nvidia':
        value = value.replace('        name = str(phase["name"])\n        if name not in {"bootstrap", "finetune"}:\n            raise ValueError(f"unsupported NVIDIA training phase {name!r}")\n        model.set_training_phase(name)', '        model.set_training_phase("bootstrap" if "encoder" in phase["parameter_groups"] else "finetune")')
        start = value.index('        if phase.get("loss_terms") !=')
        end = value.index('        evaluator_values = ', start)
        value = value[:start] + value[end:]
    path.write_text(value, encoding='utf-8', newline='\n')

path = root / 'apps/viewer/NclsViewer.cpp'
value = path.read_text(encoding='utf-8')
start = value.index('                    if (options.requestedSlotPackages[slotIndex] == "source-reference"\n')
end = value.index('                    options.captureTargetSpp[slotIndex] = targetSpp;', start)
value = value[:start] + value[end:]
path.write_text(value, encoding='utf-8', newline='\n')

# 多卡启动只有 launcher，worker 只负责进入一个物理设备域。
path = root / 'src/ncls/ddp_worker.py'
value = path.read_text(encoding='utf-8')
value = value.replace('    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_indices[local_rank])', '    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_indices[local_rank])\n    os.environ["NCLS_FALCOR_GPU_INDEX"] = str(gpu_indices[local_rank])')
path.write_text(value, encoding='utf-8', newline='\n')
path = root / 'src/ncls/learning/training/launch.py'
value = path.read_text(encoding='utf-8')
start = value.index('def distributed_command(')
value = value[:start]
path.write_text(value, encoding='utf-8', newline='\n')
path = root / 'src/ncls/learning/training/__init__.py'
value = path.read_text(encoding='utf-8')
for name in ('distributed_command', 'launch_distributed'):
    value = value.replace(f'    {name},\n','').replace(f'    "{name}",\n','')
path.write_text(value, encoding='utf-8', newline='\n')
