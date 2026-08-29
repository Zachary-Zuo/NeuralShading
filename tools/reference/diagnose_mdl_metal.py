from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ncls.core.identity import sha256_json, write_json_atomic
from ncls.source_materials.mdl_metal import MdlMetalRegistry


_PIXEL_BYTES = {
    "Sint8": (1, 1),
    "Rgb": (1, 3),
    "Rgba": (1, 4),
    "Rgb_16": (2, 3),
    "Rgba_16": (2, 4),
    "Float32": (4, 1),
    "Float32<2>": (4, 2),
    "Float32<3>": (4, 3),
    "Float32<4>": (4, 4),
    "Rgb_fp": (4, 3),
    "Color": (4, 4),
}


def _static_ledger(registry: MdlMetalRegistry, module_root: Path) -> dict[str, Any]:
    unique_slots = {
        (slot["source_sha256"], slot["source_path"]): slot
        for texture_set in registry.texture_sets.values()
        for slot in texture_set["slots"]
    }
    decoded_bytes = 0
    authored_bytes = 0
    sdk_table_bytes = 0
    for slot in unique_slots.values():
        width, height, depth = map(int, slot["dimensions"])
        scalar_bytes, channels = _PIXEL_BYTES[str(slot["pixel_type"])]
        size = width * height * depth * scalar_bytes * channels
        decoded_bytes += size
        if slot["provenance_kind"] == "authored-file":
            authored_bytes += (module_root / Path(str(slot["source_path"]))).stat().st_size
        else:
            sdk_table_bytes += size
    opaque_payloads = registry.payload["opaque_exports"]
    group_candidates = {
        (record["graph_id"], record["texture_set_id"])
        for record in opaque_payloads
    }
    return {
        "schema": "ncls.mdl-metal-diagnostic-ledger@1",
        "status": "report-only",
        "registry_identity": registry.identity,
        "counts": dict(registry.payload["counts"]),
        "static_execution_groups": {
            "graph_texture_candidates": len(group_candidates),
            "opaque_graphs": len(registry.graphs),
            "recipes": len(registry.recipes),
        },
        "compiled_material_inventory": {
            "argument_block_bytes_sum": sum(
                int(record["compiled_layout"]["argument_block_bytes"])
                for record in opaque_payloads
            ),
            "read_only_data_bytes_sum": sum(
                int(record["compiled_layout"]["ro_data_bytes"])
                for record in opaque_payloads
            ),
            "generated_code_bytes_sum_without_group_dedup": sum(
                int(record["compiled_layout"]["generated_code_bytes"])
                for record in opaque_payloads
            ),
        },
        "storage": {
            "registry_bytes": registry.path.stat().st_size if registry.path else None,
            "source_closure_file_count": len(registry.payload["source"]["source_closure"]),
            "authored_compressed_texture_bytes": authored_bytes,
            "decoded_unique_texture_and_table_bytes": decoded_bytes,
            "sdk_static_table_bytes": sdk_table_bytes,
            "plan_resource_policy": "content-addressed-file-lazy-per-resident-group",
        },
        "source_operation_accounting": {
            "authoritative@1": {
                "prepare_calls": "footprint_samples * evaluation_samples",
                "evaluate_calls": "footprint_samples * evaluation_samples",
                "pdf_source": "each evaluate result; checked invariant within one footprint point",
            },
            "prepare-hoisted-pdf-reuse@1": {
                "prepare_calls": "footprint_samples",
                "evaluate_calls": "footprint_samples * evaluation_samples",
                "pdf_source": "first stochastic evaluate per footprint point",
            },
            "footprint_contract": "full linear f and pdf are averaged; invalid/event mismatch fails the query",
        },
    }


def _probe(
    registry: MdlMetalRegistry,
    module_root: Path,
    export_count: int,
    batch_size: int,
    footprint_samples: int,
    evaluation_samples: int,
) -> dict[str, Any]:
    import torch

    from ncls.references.backend import create_reference_backend
    from ncls.references.plan import compile_single_program_plan
    from ncls.references.programs import get_reference_program_for_source
    from ncls.references.query import ScatteringQuery
    from ncls.source_materials.families.mdl import MdlFamilyDefinition

    selected = []
    seen = set()
    for record in registry.exports:
        key = (record.graph_id, record.texture_set_id)
        if key in seen:
            continue
        seen.add(key)
        selected.append(record)
        if len(selected) == export_count:
            break
    family = MdlFamilyDefinition()
    start = perf_counter()
    snapshots = tuple(
        family.load_snapshot({**record.exact_locator, "module_root": str(module_root)})
        for record in selected
    )
    locator_seconds = perf_counter() - start
    definition = get_reference_program_for_source("mdl.program@1", 1)
    start = perf_counter()
    plan = compile_single_program_plan(
        definition,
        snapshots,
        query_recipe={
            "recipe_id": "mdl-metal-diagnostic@1",
            "registry_identity": registry.identity,
            "footprint_samples": footprint_samples,
            "evaluation_samples": evaluation_samples,
        },
    )
    plan_seconds = perf_counter() - start
    session = create_reference_backend().open(
        plan,
        query_capacity=batch_size,
        device="cuda:0",
        max_resident_groups=min(2, len(plan.groups)),
    )
    group_timings = []
    try:
        for group in plan.groups:
            source_index = torch.tensor(
                [group.global_source_indices[index % len(group.records)] for index in range(batch_size)],
                dtype=torch.int64,
                device="cuda:0",
            )
            wo = torch.tensor([[0.0, 0.0, 1.0]], device="cuda:0").expand(batch_size, 3)
            uv = torch.tensor([[0.37, 0.63]], device="cuda:0").expand(batch_size, 2)
            extent = 1.0 / 256.0
            query = ScatteringQuery(
                source_index,
                wo,
                group.group_id,
                uv=uv,
                uv_dx=torch.tensor([[extent, 0.0]], device="cuda:0").expand(batch_size, 2),
                uv_dy=torch.tensor([[0.0, extent]], device="cuda:0").expand(batch_size, 2),
            )
            wi = torch.tensor(
                [[[0.2, 0.1, math.sqrt(0.95)]]], device="cuda:0"
            ).expand(batch_size, 1, 3)
            seeds = torch.arange(batch_size, dtype=torch.int64, device="cuda:0")[:, None]
            torch.cuda.synchronize()
            materialization_start = perf_counter()
            warmup = session.evaluate(
                query,
                wi,
                seeds,
                evaluation_samples=1,
                footprint_samples=1,
                source_execution_mode="authoritative@1",
            )
            warmup.lease.release()
            torch.cuda.synchronize()
            first_materialization_seconds = perf_counter() - materialization_start
            session.end_iteration()
            modes = {}
            reference_values = None
            for mode in ("authoritative@1", "prepare-hoisted-pdf-reuse@1"):
                torch.cuda.synchronize()
                query_start = perf_counter()
                result = session.evaluate(
                    query,
                    wi,
                    seeds,
                    evaluation_samples=evaluation_samples,
                    footprint_samples=footprint_samples,
                    source_execution_mode=mode,
                )
                try:
                    current = result.f.clone()
                    valid_fraction = float(result.valid.float().mean())
                finally:
                    result.lease.release()
                torch.cuda.synchronize()
                elapsed = perf_counter() - query_start
                session.end_iteration()
                modes[mode] = {
                    "seconds": elapsed,
                    "query_count": batch_size,
                    "valid_fraction": valid_fraction,
                }
                if reference_values is None:
                    reference_values = current
                else:
                    modes[mode]["max_abs_difference_from_authoritative"] = float(
                        torch.max(torch.abs(current - reference_values))
                    )
            group_timings.append(
                {
                    "group_id": group.group_id,
                    "material_count": len(group.records),
                    "first_materialization_and_query_seconds": first_materialization_seconds,
                    "modes": modes,
                    "resident_group_ids_after_query": list(session.resident_group_ids),
                }
            )
    finally:
        session.close()
    return {
        "selected_export_count": len(selected),
        "locator_seconds": locator_seconds,
        "plan_compile_seconds": plan_seconds,
        "plan_identity": plan.identity,
        "plan_group_count": len(plan.groups),
        "batch_size": batch_size,
        "footprint_samples": footprint_samples,
        "evaluation_samples": evaluation_samples,
        "groups": group_timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="输出 vMaterials 2 Metal report-only 诊断账本。")
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "references/mdl-vmaterials2-v1/metal-opaque-v1.json",
    )
    parser.add_argument(
        "--module-root",
        type=Path,
        default=PROJECT_ROOT / "assets/source-materials/mdl-vmaterials2/2.4.0/Materials",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-exports", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--footprint-samples", type=int, default=8)
    parser.add_argument("--evaluation-samples", type=int, default=1)
    arguments = parser.parse_args()
    if min(
        arguments.probe_exports if arguments.probe_exports else 1,
        arguments.batch_size,
        arguments.footprint_samples,
        arguments.evaluation_samples,
    ) < 1:
        parser.error("diagnostic counts must be positive")
    registry = MdlMetalRegistry.load(arguments.registry)
    ledger = _static_ledger(registry, arguments.module_root.resolve())
    if arguments.probe_exports:
        ledger["observed_probe"] = _probe(
            registry,
            arguments.module_root.resolve(),
            arguments.probe_exports,
            arguments.batch_size,
            arguments.footprint_samples,
            arguments.evaluation_samples,
        )
    ledger["ledger_identity"] = sha256_json(ledger)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(arguments.output, ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
