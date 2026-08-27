from __future__ import annotations

from pathlib import Path

import torch

from ncls.cli import _batch_source
from ncls.learning.methods import get_method
from ncls.learning.training import TrainingConfig, TrainingRunner


config = TrainingConfig.load(Path("configs/learning/nvidia-rta2024-materialx-formal.json"))
source = _batch_source(config)
runner = TrainingRunner(get_method(config.method_key), source, config)
request = runner._request(config.routes[1], 20_000)
try:
    warm = source.next_batch(request)
    warm.release()
    torch.cuda.synchronize(source.device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profile:
        batch = source.next_batch(request)
        torch.cuda.synchronize(source.device)
        batch.release()
    print(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
    print(profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
finally:
    source.close()
