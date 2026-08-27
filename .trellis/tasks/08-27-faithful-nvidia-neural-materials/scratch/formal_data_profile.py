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
runner = TrainingRunner(definition, source, config)
runner._seed()
rows: list[dict[str, float | int | str]] = []


def synchronize() -> float:
    torch.cuda.synchronize(source.device)
    return time.perf_counter()


try:
    for step in (1, 2, 20_000, 20_001):
        for route in config.routes:
            started = synchronize()
            batch = source.next_batch(runner._request(route, step))
            returned = time.perf_counter()
            ended = synchronize()
            try:
                rows.append(
                    {
                        "step": step,
                        "route": route.name,
                        "seconds": ended - started,
                        "host_seconds": returned - started,
                        "sync_seconds": ended - returned,
                        "sample_count": int(batch.tensors["sample_count"].sum()),
                    }
                )
            finally:
                batch.release()
    route = config.routes[1]
    request = runner._request(route, 20_002)
    rng = source._rng(request)
    generator, _ = source._request_generator(rng, source.device)
    uv = torch.rand(
        (route.batch_size, 2), device=source.device, generator=generator
    )
    mip = torch.clamp(
        torch.floor(
            -torch.log1p(
                -torch.rand(route.batch_size, device=source.device, generator=generator)
            )
        ),
        max=len(source._feature_pyramid.level_shapes) - 1,
    )
    for index in range(3):
        started = synchronize()
        _, _, groups, _, _ = source._spatial_samples(uv, mip, request, generator)
        ended = synchronize()
        rows.append(
            {
                "step": 20_002,
                "route": f"spatial-only-{index}",
                "seconds": ended - started,
                "sample_count": len(groups),
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
finally:
    source.close()
