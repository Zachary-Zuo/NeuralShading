from pathlib import Path
import yaml

root = Path.cwd()
path = root / 'src/ncls/learning/training/distributed.py'
value = path.read_text(encoding='utf-8')
start, end = value.index('\ndef configure_distributed_debug_environment('), value.index('\n\n__all__', value.index('\ndef configure_distributed_debug_environment('))
function = value[start:end]
value = value[:start] + value[end:]
value = value.replace('    "configure_distributed_debug_environment",\n', '')
path.write_text(value, encoding='utf-8')
path = root / 'src/ncls/runtime.py'
value = path.read_text(encoding='utf-8')
value = value.replace('\ndef process_environment(', function + '\n\n\ndef process_environment(')
value = value.replace('    return result\n\n\ndef main', '    configure_distributed_debug_environment(result)\n    return result\n\n\ndef main')
value = value.replace('        result["NCLS_FALCOR_GPU_INDEX"] = str(devices[0])', '        result.pop("NCLS_DDP_GPU_LIST", None)\n        result["NCLS_FALCOR_GPU_INDEX"] = str(devices[0])')
path.write_text(value, encoding='utf-8')
path = root / 'tests/unit/test_training_distributed.py'
value = path.read_text(encoding='utf-8').replace('    configure_distributed_debug_environment,\n', '')
value = value.replace('import pytest', 'from ncls.runtime import configure_distributed_debug_environment\n\nimport pytest')
path.write_text(value, encoding='utf-8')
path = root / 'src/ncls/viewer/export.py'
value = path.read_text(encoding='utf-8').replace('import json\n', 'from __future__ import annotations\n\nimport json\n', 1)
path.write_text(value, encoding='utf-8')
scratch = root / '.trellis/tasks/09-05-architecture-reset-training-workflow/scratch'
value = yaml.safe_load((root / 'configs/training/runs/nvidia-layer-stack-visual-smoke.yaml').read_text(encoding='utf-8'))
value['hooks']['visual_eval'].update(reference_spp=33, width=128, height=64)
(scratch / 'visual-33.yaml').write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding='utf-8')
path = scratch / 'check_visual.py'
value = path.read_text(encoding='utf-8').replace("for name in ('reference', 'neural', 'difference'):", "for name in ('slot_0_linear', 'slot_1_linear', 'difference_linear'):")
value = value.replace("run = Path('outputs/nvidia-layer-stack-smoke/260905-172009-9d5b6c')", "import sys\nrun = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('outputs/nvidia-layer-stack-smoke/260905-172009-9d5b6c')")
path.write_text(value, encoding='utf-8')
