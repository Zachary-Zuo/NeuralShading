from __future__ import annotations

import argparse

import torch

from ncls.data import CollectionConfig
from ncls.data.providers import MaterialXGpuQueryRuntime, MaterialXProvider, MaterialXProviderConfig


parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=2000)
parser.add_argument("--query-count", type=int, default=262144)
args = parser.parse_args()

provider = MaterialXProvider(
    CollectionConfig(
        name="materialx-interop-soak",
        view_count=1,
        light_count=1,
        spatial_sample_count=1,
        proposal="uniform",
        seed=20260827,
    ),
    MaterialXProviderConfig(asset_ids=("american_walnut_veneer",)),
)
runtime = MaterialXGpuQueryRuntime(
    provider,
    tuple(provider.source_states())[0],
    query_capacity=args.query_count,
    slot_count=2,
)
device = torch.device("cuda:0")
generator = torch.Generator(device=device).manual_seed(20260827)
try:
    for iteration in range(args.iterations):
        views = torch.randn(
            (args.query_count, 3), device=device, dtype=torch.float32, generator=generator
        )
        views[:, 2].abs_().add_(0.01)
        views = torch.nn.functional.normalize(views, dim=1)
        lights = torch.randn(
            (args.query_count, 3), device=device, dtype=torch.float32, generator=generator
        )
        lights[:, 2].abs_().add_(0.01)
        lights = torch.nn.functional.normalize(lights, dim=1)
        uv = torch.rand(
            (args.query_count, 2), device=device, dtype=torch.float32, generator=generator
        )
        gradients = torch.zeros((args.query_count, 4), device=device, dtype=torch.float32)
        gradients[:, 0] = 1.0 / 4096.0
        gradients[:, 3] = 1.0 / 4096.0
        values = runtime.evaluate_torch(iteration % 2, views, lights, uv, gradients)
        checksum = values.square().mean()
        if (iteration + 1) % 100 == 0:
            print(iteration + 1, float(checksum))
    torch.cuda.synchronize()
finally:
    runtime.close()
