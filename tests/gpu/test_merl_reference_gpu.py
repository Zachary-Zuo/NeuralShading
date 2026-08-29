from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ncls.references import deterministic_directional_metrics, load_reference_acceptance
from ncls.references.backend import create_reference_backend
from ncls.source_materials import MerlBrdfReference, MerlMaterial


falcor = pytest.importorskip("falcor")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _hemisphere_directions(generator: np.random.Generator, count: int) -> np.ndarray:
    xy = generator.normal(size=(count, 2))
    azimuth = np.arctan2(xy[:, 1], xy[:, 0])
    cosine = generator.uniform(0.08, 1.0, size=count)
    sine = np.sqrt(1.0 - cosine * cosine)
    result = np.stack((sine * np.cos(azimuth), sine * np.sin(azimuth), cosine), axis=1)
    return result.astype(np.float32)


@pytest.mark.falcor
def test_merl_falcor_runtime_matches_native_table_reference() -> None:
    asset_root = PROJECT_ROOT / "assets" / "source-materials" / "merl-brdf" / "v1"
    if not (asset_root / "complete.json").is_file():
        pytest.skip("MERL source material asset is not downloaded")
    marker = json.loads((asset_root / "complete.json").read_text(encoding="utf-8"))
    material_index = json.loads((PROJECT_ROOT / marker["material_index"]).read_text(encoding="utf-8"))
    records = {record["material_id"]: record for record in material_index["materials"]}
    material_ids = ("alum-bronze", "beige-fabric", "blue-metallic-paint", "red-plastic")
    generator = np.random.default_rng(20260823)
    query_count = 4096
    views = _hemisphere_directions(generator, query_count)
    lights = _hemisphere_directions(generator, query_count)
    acceptance = load_reference_acceptance(PROJECT_ROOT / "references" / "acceptance.json")
    device = create_reference_backend()._create_device(falcor)  # noqa: SLF001
    compute = falcor.ComputePass(
        device,
        file=PROJECT_ROOT / "tests" / "gpu" / "kernels" / "merl_reference.cs.slang",
        cs_entry="evaluateMerlReference",
    )

    direction_flags = falcor.ResourceBindFlags.ShaderResource
    views4 = np.pad(views, ((0, 0), (0, 1)))
    lights4 = np.pad(lights, ((0, 0), (0, 1)))
    view_buffer = device.create_structured_buffer(struct_size=16, element_count=query_count, bind_flags=direction_flags)
    light_buffer = device.create_structured_buffer(struct_size=16, element_count=query_count, bind_flags=direction_flags)
    view_buffer.from_numpy(views4)
    light_buffer.from_numpy(lights4)
    output = device.create_structured_buffer(
        struct_size=16,
        element_count=query_count,
        bind_flags=direction_flags | falcor.ResourceBindFlags.UnorderedAccess,
    )
    compute.globals.gViews = view_buffer
    compute.globals.gLights = light_buffer
    compute.globals.gOutput = output
    compute.globals.gQueryCount = query_count

    for material_id in material_ids:
        record = records[material_id]
        material = MerlMaterial(material_id, record["table_uri"])
        native_reference = MerlBrdfReference(material, asset_root)
        table = native_reference.gpu_table()
        table_buffer = device.create_structured_buffer(
            struct_size=12,
            element_count=table.shape[0],
            bind_flags=direction_flags,
        )
        table_buffer.from_numpy(table)
        compute.globals.gBrdfTable = table_buffer
        compute.execute(threads_x=query_count)
        candidate = output.to_numpy().view(np.float32).reshape(query_count, 4)[:, :3]
        native = native_reference.evaluate(views, lights).response_cos
        metrics = deterministic_directional_metrics(native, candidate, acceptance.deterministic_directional)
        print(
            f"{material_id}: median={metrics.median_relative_l1:.9g}, "
            f"p99={metrics.p99_relative_l1:.9g}, max={metrics.max_relative_l1:.9g}, "
            f"max_abs={metrics.max_absolute_error:.9g}, scaled_abs={metrics.max_scaled_absolute_error:.9g}"
        )
        assert metrics.passed, f"{material_id}: {metrics}"
