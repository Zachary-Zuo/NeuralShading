import json
import time
import torch

from ncls.data import PipelineTrace
from ncls.core.source import create_source_family
from ncls.learning.methods.metal.data import MetalBudgetedMdlSourceAdapter
from ncls.learning.methods.metal.model import MetalBudgetedModel, METAL_BUDGETED_REQUIRED_CONTEXT
from ncls.learning.training import TrainingPlanResolver
from ncls.paths import PROJECT_ROOT


def main():
    config = TrainingPlanResolver(PROJECT_ROOT).resolve('configs/training/runs/metal-budgeted-hybrid-probe-bronze-scratched.yaml').training
    family = create_source_family('mdl.program@1')
    snapshot = family.load_snapshot(config.source['materials'][0]['locator'])
    adapter = MetalBudgetedMdlSourceAdapter((snapshot,), torch.device('cuda:0'))
    adapter.native_assets().enable_gpu_sampling(torch.device('cuda:0'), budget_bytes=8*1024**3, trace=PipelineTrace())
    try:
        asset, slots, groups = adapter.spatial_contract_for_source()
        print(json.dumps({'slots': [s.__dict__ for s in slots], 'groups': [g.mapping.__dict__ for g in groups]}, ensure_ascii=False), flush=True)
        t = time.perf_counter()
        data = adapter.sample_tensors(torch.zeros(8, dtype=torch.int64, device='cuda:0'), torch.Generator(device='cuda:0').manual_seed(93),
            {'paired_uv': True, 'paired_uv_recipe': 'one-native-texel-axis-balanced@1', 'spatial_core_texels': 128}, execution_source_indices=(0,))
        try:
            print('raw resource', len(data.resources.entries[0].tensors), 'parts', len(data.resources.entries[0].metadata['bundle'].parts), 'seconds', time.perf_counter()-t, flush=True)
            model = MetalBudgetedModel.from_context(METAL_BUDGETED_REQUIRED_CONTEXT).cuda()
            values = {**data.tensors, 'wo': torch.tensor([[0.,0.,1.]], device='cuda:0').expand(8,-1)}
            program = model.compile_program_state(values)
            encoded = model.asset.encode_resources(data.resources)
            sample = model.sample_asset(values, program, resources=data.resources, binding=data.bindings['metal_spatial'], encoded=encoded)
            prepared = model.prepare_from_components(program, sample, values['wo'])
            f = model.evaluate_prepared(prepared, values['wo'], values['wo'][:, None]).f
            f.sum().backward()
            torch.cuda.synchronize()
            print('response', f.detach().mean().item(), 'seconds', time.perf_counter()-t,
                  'peak_allocated', torch.cuda.max_memory_allocated(), flush=True)
        finally:
            data.resources.release()
    finally:
        adapter.close()


if __name__ == '__main__':
    main()
