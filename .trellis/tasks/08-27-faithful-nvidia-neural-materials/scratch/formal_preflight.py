from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from ncls.cli import _batch_source
from ncls.learning.methods import get_method
from ncls.learning.training import TrainingConfig, TrainingRunner


config = TrainingConfig.load(Path("configs/learning/nvidia-rta2024-materialx-formal.json"))
definition = get_method(config.method_key)
source = _batch_source(config)
torch.empty(1, device=source.device)
torch.cuda.reset_peak_memory_stats(source.device)
started = time.perf_counter()
try:
    result = TrainingRunner(definition, source, config).run(stop_at_step=1)
    torch.cuda.synchronize(source.device)
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "kind": "diagnostic-preflight",
        "config_sha256": config.sha256,
        "step": result.checkpoint.step,
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(source.device),
        "metrics": list(result.metrics),
    }, ensure_ascii=False, indent=2))
finally:
    source.close()
