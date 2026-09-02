from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist

from ncls.bundle import ScatteringPackage, write_scattering_package
from ncls.core.material import (
    BINARY_SIZE,
    MaterialProgram,
    canonicalize_layer_stack,
    pack_layer_stack,
    physical_material_hash,
    validate_material_program,
)
from ncls.core.source import create_source_family
from ncls.learning.batches import TrainingRouteRequest
from ncls.learning.conformance import MethodArtifactInventory, validate_artifact_coverage
from ncls.learning.methods import get_method, method_descriptors
from ncls.learning.producer import OnlineTrainingProducer
from ncls.learning.training import (
    TrainingConfig,
    TrainingRunner,
    build_training_review,
    load_checkpoint,
    load_metric_rows,
    save_checkpoint,
    write_training_review,
)
from ncls.references.backend import create_reference_backend


def _load_program(path: Path) -> MaterialProgram:
    return MaterialProgram.from_json(path.read_text(encoding="utf-8"))


def _material_validate(path: Path) -> int:
    program = _load_program(path)
    validate_material_program(program)
    stack = canonicalize_layer_stack(program)
    print(f"MaterialProgram OK: {physical_material_hash(program)}")
    print(
        f"LayerStackIR: {len(stack.interfaces)} interfaces, "
        f"{len(stack.media)} media, {BINARY_SIZE} bytes"
    )
    return 0


def _material_normalize(path: Path, output: Path) -> int:
    program = _load_program(path)
    normalized = MaterialProgram(
        tuple(sorted(program.nodes, key=lambda node: node.node_id)),
        program.outputs,
        tuple(sorted(program.resources, key=lambda resource: resource.resource_id)),
        program.metadata,
        program.color_model,
        program.schema_name,
        program.schema_version,
    )
    validate_material_program(normalized)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(normalized.to_json(), encoding="utf-8")
    print(f"Wrote {output}: {physical_material_hash(normalized)}")
    return 0


def _material_pack(path: Path, output: Path) -> int:
    payload = pack_layer_stack(canonicalize_layer_stack(_load_program(path)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(f"Wrote {output}: {len(payload)} bytes")
    return 0


def _learn_list() -> int:
    for descriptor in method_descriptors():
        print(
            f"{descriptor.method_key}\t{descriptor.display_name}\t"
            f"{descriptor.descriptor_sha256}"
        )
    return 0


def _learn_train(
    config_path: Path,
    output: Path,
    resume_path: Path | None,
    stop_at_step: int | None,
) -> int:
    ddp_rank, ddp_world = _setup_ddp()
    is_rank0 = ddp_rank == 0
    if ddp_world > 1 and output.is_absolute() is False:
        output = output.resolve()
    config = TrainingConfig.load(config_path)
    definition = get_method(config.method_key)
    producer = OnlineTrainingProducer(definition, config)
    gpu_indices = list(getattr(producer, "ddp_gpu_indices", ()))
    metrics_path = output.with_name(f"{output.stem}.metrics.jsonl")
    summary_path = output.with_name(f"{output.stem}.summary.json")
    review_path = output.with_name(f"{output.stem}.review.json")
    metric_count = 0
    checkpoint_write_seconds: list[float] = []
    started = time.perf_counter()
    ddp_completed = False
    try:
        resume = (
            load_checkpoint(
                resume_path,
                descriptor=definition.descriptor,
                map_location="cpu" if ddp_world > 1 else config.device,
            )
            if resume_path is not None
            else None
        )
        retained_metric_lines: list[str] = []
        if resume is not None and metrics_path.is_file():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if float(value.get("step", -1)) <= resume.global_step:
                    retained_metric_lines.append(
                        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    )
        output.parent.mkdir(parents=True, exist_ok=True)
        metric_stream = (
            metrics_path.open("w", encoding="utf-8", newline="\n")
            if is_rank0
            else None
        )
        if metric_stream is not None:
            for line in retained_metric_lines:
                metric_stream.write(line + "\n")
        metric_count = len(retained_metric_lines)

        def record_metric(row: dict[str, float]) -> None:
            nonlocal metric_count
            payload = {
                "record_kind": (
                    "validation" if "validation/loss" in row else "training"
                ),
                "training_config_sha256": config.sha256,
                **row,
            }
            if metric_stream is None:
                return
            metric_stream.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            metric_count += 1
            if metric_count % 256 == 0:
                metric_stream.flush()

        def save_periodic(checkpoint) -> None:
            if not is_rank0:
                return
            assert metric_stream is not None
            metric_stream.flush()
            path = output.with_name(
                f"{output.stem}.step{checkpoint.global_step:08d}{output.suffix}"
            )
            checkpoint_started = time.perf_counter()
            save_checkpoint(path, checkpoint)
            checkpoint_write_seconds.append(time.perf_counter() - checkpoint_started)

        result = TrainingRunner(
            definition,
            producer,
            config,
            checkpoint_callback=save_periodic,
            metric_callback=record_metric,
        ).run(resume=resume, stop_at_step=stop_at_step)
        digest = ""
        elapsed_seconds = time.perf_counter() - started
        if is_rank0:
            assert metric_stream is not None
            checkpoint_started = time.perf_counter()
            digest = save_checkpoint(output, result.checkpoint)
            checkpoint_write_seconds.append(time.perf_counter() - checkpoint_started)
            metric_stream.flush()
            metric_stream.close()
            elapsed_seconds = time.perf_counter() - started
            summary = {
                "schema_name": "ncls.training-run-summary",
                "schema_version": 2,
                "training_config_sha256": config.sha256,
                "checkpoint_sha256": digest,
                "checkpoint": output.name,
                "metrics": metrics_path.name,
                "review": review_path.name,
                "metric_records": metric_count,
                "final_step": result.checkpoint.global_step,
                "planned_final_step": config.total_steps,
                "complete": result.checkpoint.global_step == config.total_steps,
                "distributed": ddp_world > 1,
                "world_size": ddp_world,
                "gpu_indices": gpu_indices,
                "effective_global_batch": {
                    route.name: route.batch_size * route.direction_count * ddp_world
                    for phase in config.phases
                    for route in phase.routes
                },
                "elapsed_seconds": elapsed_seconds,
                "checkpoint_write_seconds": checkpoint_write_seconds,
            }
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            review = build_training_review(
                config,
                definition.descriptor,
                result.checkpoint,
                checkpoint_sha256=digest,
                checkpoint_bytes=output.stat().st_size,
                metric_rows=load_metric_rows(metrics_path, config_sha256=config.sha256),
                metrics_bytes=metrics_path.stat().st_size,
                elapsed_seconds=elapsed_seconds,
                checkpoint_write_seconds=checkpoint_write_seconds,
            )
            write_training_review(review_path, review)
        ddp_completed = True
    finally:
        if "metric_stream" in locals() and metric_stream is not None and not metric_stream.closed:
            metric_stream.close()
        producer.close()
        if ddp_world > 1 and dist.is_initialized():
            dist.destroy_process_group()
    if is_rank0:
        print(f"TrainingCheckpoint@4 {digest}: {output}")
    return 0


def _setup_ddp() -> tuple[int, int]:
    """Initialize torchrun process group and map each rank to one visible GPU."""
    world_raw = os.environ.get("WORLD_SIZE")
    rank_raw = os.environ.get("RANK")
    local_raw = os.environ.get("LOCAL_RANK")
    if world_raw is None and rank_raw is None and local_raw is None:
        return 0, 1
    if world_raw is None or rank_raw is None or local_raw is None:
        raise RuntimeError("DDP requires WORLD_SIZE, RANK and LOCAL_RANK")
    try:
        world = int(world_raw); rank = int(rank_raw); local = int(local_raw)
    except ValueError as error:
        raise RuntimeError("DDP rank environment values must be integers") from error
    if world < 2 or rank < 0 or rank >= world or local < 0 or local >= world:
        raise RuntimeError("DDP rank environment is invalid")
    if int(os.environ.get("NCLS_DDP_WORLD_SIZE", str(world))) != world:
        raise RuntimeError("NCLS_DDP_WORLD_SIZE disagrees with WORLD_SIZE")
    gpu_list = os.environ.get("NCLS_DDP_GPU_LIST", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu_list or visible != gpu_list:
        raise RuntimeError("DDP requires CUDA_VISIBLE_DEVICES to match NCLS_DDP_GPU_LIST")
    try:
        physical = [int(value) for value in gpu_list.split(",")]
    except ValueError as error:
        raise RuntimeError("NCLS_DDP_GPU_LIST must be comma-separated GPU indices") from error
    if len(physical) != world or len(set(physical)) != world:
        raise RuntimeError("DDP GPU list length must equal WORLD_SIZE and be unique")
    os.environ["NCLS_DDP_LOCAL_RANK"] = str(local)
    os.environ["NCLS_FALCOR_GPU_INDEX"] = str(physical[local])
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA")
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    return rank, world


def _learn_evaluate(config_path: Path, checkpoint_path: Path, batches: int) -> int:
    if batches < 1:
        raise ValueError("evaluation batch count must be positive")
    config = TrainingConfig.load(config_path)
    definition = get_method(config.method_key)
    checkpoint = load_checkpoint(
        checkpoint_path,
        descriptor=definition.descriptor,
        map_location=config.device,
    )
    if checkpoint.training_config_sha256 != config.sha256:
        raise ValueError("evaluation config disagrees with the checkpoint")
    producer = OnlineTrainingProducer(definition, config)
    if (
        checkpoint.reference_program_identity != producer.reference_program_identity
        or checkpoint.reference_execution_plan_identity
        != producer.reference_execution_plan_identity
        or checkpoint.native_asset_collection_identity
        != producer.native_asset_collection_identity
        or checkpoint.query_stream_identity != producer.query_stream_identity
        or checkpoint.source_snapshot_ids != producer.source_snapshot_ids
    ):
        producer.close()
        raise ValueError("evaluation producer identity disagrees with the checkpoint")
    losses: list[float] = []
    try:
        model = definition.create_trainable(config.model_context).to(producer.device)
        definition.restore_training_state(model, checkpoint.model_state)
        phase_index = min(checkpoint.phase_index, len(config.phases) - 1)
        phase = config.phases[phase_index]
        definition.configure_phase(model, phase.to_dict())
        model.eval()
        with torch.no_grad():
            for index in range(batches):
                route_batches = {}
                try:
                    for route in phase.routes:
                        request = TrainingRouteRequest(
                            f"evaluation:{route.name}",
                            route.kind,
                            route.batch_size,
                            route.direction_count,
                            checkpoint.global_step,
                            config.seed + route.seed_offset + index,
                            {
                                **dict(route.options),
                                "recipes": dict(phase.recipes),
                                "validation": True,
                            },
                        )
                        route_batches[route.name] = producer.next_batch(request)
                    loss, _ = definition.training_objective(
                        model,
                        route_batches,
                        {
                            "name": phase.name,
                            "phase_index": phase_index,
                            "phase_step": checkpoint.phase_step,
                            "phase_steps": phase.steps,
                            "global_step": checkpoint.global_step,
                            "total_steps": config.total_steps,
                            "parameter_groups": list(phase.parameter_groups),
                            "loss_terms": list(phase.loss_terms),
                            "recipes": dict(phase.recipes),
                            "validation": True,
                        },
                    )
                    losses.append(float(loss))
                finally:
                    for batch in reversed(tuple(route_batches.values())):
                        batch.release()
                    producer.end_iteration()
    finally:
        producer.close()
    print(f"Evaluation batches={batches} mean_loss={sum(losses) / len(losses):.9g}")
    return 0


def _learn_export(
    checkpoint_path: Path, output: Path, material_index: int
) -> int:
    checkpoint = load_checkpoint(checkpoint_path)
    definition = get_method(checkpoint.method_key)
    checkpoint.validate_method(definition.descriptor)
    config = TrainingConfig.from_dict(checkpoint.training_config)
    family = create_source_family(str(config.source["family_id"]))
    materials = config.source["materials"]
    if not 0 <= material_index < len(materials):
        raise ValueError("export material index is outside the checkpoint source list")
    snapshot = family.load_snapshot(materials[material_index]["locator"])
    family.validate_snapshot(snapshot)
    if snapshot.snapshot_id not in checkpoint.source_snapshot_ids:
        raise ValueError("export source snapshot does not occur in the checkpoint")
    payload = checkpoint.to_payload()
    runtime = definition.compile_program(payload)
    material = definition.compile_asset(snapshot, payload)
    instance = definition.compile_instance(snapshot, payload)
    validate_artifact_coverage(
        definition.descriptor,
        MethodArtifactInventory.from_payloads(
            runtime, material, checkpoint_model_state=bool(checkpoint.model_state)
        ),
    )
    validation = dict(definition.package_validation(snapshot, payload))
    validation["checkpoint_step"] = checkpoint.global_step
    manifest = write_scattering_package(
        output,
        program_kind="method",
        program_key=definition.descriptor.method_key,
        program_version=definition.descriptor.version,
        program_descriptor_sha256=definition.descriptor.descriptor_sha256,
        runtime_abi=definition.descriptor.runtime_abi,
        source=snapshot,
        program_payload=runtime,
        asset_payload=material,
        validation=validation,
        instance_payload=instance,
        provenance={
            "checkpoint_sha256": checkpoint_path.with_suffix(
                checkpoint_path.suffix + ".sha256"
            ).read_text(encoding="ascii").strip()
        },
    )
    print(f"ScatteringPackage@2 {manifest.package_id}: {output}")
    return 0


def _package_validate(path: Path) -> int:
    package = ScatteringPackage.open(path)
    binding = package.create_binding()
    print(f"ScatteringPackage OK: {package.manifest.package_id}")
    print(
        f"program={binding.program.program_id} asset={binding.asset.asset_id} "
        f"instance={binding.instance.instance_id}"
    )
    return 0


def _reference_doctor(as_json: bool) -> int:
    report = create_reference_backend().doctor()
    descriptor = report.descriptor
    payload = {
        "schema_name": "ncls.reference-backend-report",
        "schema_version": 1,
        "ready": report.ready,
        "backend": {
            "backend_key": descriptor.backend_key,
            "version": descriptor.version,
            "platform_id": descriptor.platform_id,
            "falcor_revision": descriptor.falcor_revision,
            "slang_revision": descriptor.slang_revision,
            "device_api": descriptor.device_api,
            "semantic_identity": descriptor.semantic_identity,
            "build_identity": descriptor.build_identity,
            "identity": descriptor.identity,
        },
        "assets": "not-managed",
        "statuses": [
            {
                "requirement_id": value.requirement_id,
                "category": value.category,
                "status": value.status,
                "detail": value.detail,
            }
            for value in report.statuses
        ],
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            f"ReferenceBackend {descriptor.backend_key}@{descriptor.version} "
            f"platform={descriptor.platform_id} ready={str(report.ready).lower()}"
        )
        for value in report.statuses:
            print(
                f"{value.status}\t{value.category}\t"
                f"{value.requirement_id}\t{value.detail}"
            )
        print("assets=not-managed")
    return 0 if report.ready else 1


def _reference_probe() -> int:
    import math

    from ncls.core.material import DiffuseInterface, LayerStackIR
    from ncls.references.programs import (
        discover_reference_programs,
        get_reference_program_for_source,
    )
    from ncls.references.plan import compile_single_program_plan
    from ncls.references.query import ScatteringQuery
    from ncls.source_materials.families.layer_stack import snapshot_from_layer_stack

    backend = create_reference_backend()
    backend.doctor().require_ready()
    runtime_programs = []
    for definition in discover_reference_programs():
        runtime = definition.compile_runtime()
        runtime_programs.append(
            {
                "program_key": definition.descriptor.program_key,
                "program_version": definition.descriptor.version,
                "program_module": runtime.program_module,
                "capabilities": runtime.capabilities,
            }
        )
    layer_snapshot = snapshot_from_layer_stack(
        LayerStackIR((DiffuseInterface((0.6, 0.3, 0.1)),), ())
    )
    mdl_family = create_source_family("mdl.program@1")
    mdl_snapshot = mdl_family.load_snapshot(
        {
            "kind": "mdl-export",
            "module_root": str(Path(__file__).resolve().parents[2] / "tests/fixtures/mdl"),
            "module": "::constant_diffuse",
            "export": "constant_diffuse",
            "arguments": {"tint": [0.8, 0.2, 0.1]},
        }
    )
    results = []
    for snapshot in (layer_snapshot, mdl_snapshot):
        definition = get_reference_program_for_source(
            snapshot.family_id, snapshot.source_contract_version
        )
        plan = compile_single_program_plan(
            definition, (snapshot,), query_recipe={"recipe_id": "cli-probe@1"}
        )
        session = backend.open(plan, query_capacity=1, device="cuda:0")
        try:
            device = torch.device("cuda:0")
            result = session.evaluate(
                ScatteringQuery(
                    torch.zeros(1, dtype=torch.int64, device=device),
                    torch.tensor([[0.0, 0.0, 1.0]], device=device),
                    plan.groups[0].group_id,
                ),
                torch.tensor(
                    [[[0.3, 0.0, math.sqrt(0.91)]]], device=device
                ),
                torch.zeros((1, 1), dtype=torch.int64, device=device),
            )
            try:
                if not bool(result.valid.all()) or not bool(torch.isfinite(result.f).all()):
                    raise RuntimeError("reference backend probe returned an invalid result")
            finally:
                result.lease.release()
            session.end_iteration()
            results.append(
                {
                    "program_key": definition.descriptor.program_key,
                    "reference_program_identity": session.reference_program_identity,
                }
            )
        finally:
            session.close()
    print(
        json.dumps(
            {
                "schema_name": "ncls.reference-backend-probe",
                "schema_version": 1,
                "backend_identity": backend.descriptor.identity,
                "assets": "not-managed",
                "runtime_programs": runtime_programs,
                "query_programs": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ncls", description="NeuralShading 统一 pipeline 工具"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    material = commands.add_parser("material", help="MaterialProgram 工具")
    material_commands = material.add_subparsers(
        dest="material_command", required=True
    )
    validate = material_commands.add_parser("validate", help="验证 MaterialProgram")
    validate.add_argument("path", type=Path)
    normalize = material_commands.add_parser("normalize", help="规范化 MaterialProgram")
    normalize.add_argument("path", type=Path)
    normalize.add_argument("output", type=Path)
    pack = material_commands.add_parser("pack", help="写出 LayerStackIR packet")
    pack.add_argument("path", type=Path)
    pack.add_argument("output", type=Path)

    learn = commands.add_parser("learn", help="统一 online 方法训练、评测和导出")
    learn_commands = learn.add_subparsers(dest="learn_command", required=True)
    learn_commands.add_parser("list", help="列出产品 MethodDefinition")
    train = learn_commands.add_parser("train", help="运行 online TrainingRunner")
    train.add_argument("config", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument(
        "--stop-at-step",
        type=int,
        help="在给定global step写出可恢复checkpoint后正常退出",
    )
    evaluate = learn_commands.add_parser(
        "evaluate", help="评测 TrainingCheckpoint@4"
    )
    evaluate.add_argument("config", type=Path)
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--batches", type=int, default=8)
    export = learn_commands.add_parser(
        "export", help="从 checkpoint 的 source locator 编译 ScatteringPackage@2"
    )
    export.add_argument("checkpoint", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--material-index", type=int, default=0)

    package = commands.add_parser("package", help="ScatteringPackage@2 工具")
    package_commands = package.add_subparsers(
        dest="package_command", required=True
    )
    validate_package = package_commands.add_parser(
        "validate", help="验证 schema、URI 与内容 hash"
    )
    validate_package.add_argument("path", type=Path)

    reference = commands.add_parser("reference", help="统一 reference backend 工具")
    reference_commands = reference.add_subparsers(
        dest="reference_command", required=True
    )
    doctor = reference_commands.add_parser(
        "doctor", help="检查五个 canonical reference program 的底层能力"
    )
    doctor.add_argument("--json", action="store_true")
    reference_commands.add_parser(
        "probe", help="用仓库 fixture 验证 device、LayerStack 与 MDL compile/query"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dispatch: dict[tuple[str, str], Any] = {
        ("material", "validate"): lambda: _material_validate(args.path),
        ("material", "normalize"): lambda: _material_normalize(
            args.path, args.output
        ),
        ("material", "pack"): lambda: _material_pack(args.path, args.output),
        ("learn", "list"): _learn_list,
        ("learn", "train"): lambda: _learn_train(
            args.config, args.output, args.resume, args.stop_at_step
        ),
        ("learn", "evaluate"): lambda: _learn_evaluate(
            args.config, args.checkpoint, args.batches
        ),
        ("learn", "export"): lambda: _learn_export(
            args.checkpoint, args.output, args.material_index
        ),
        ("package", "validate"): lambda: _package_validate(args.path),
        ("reference", "doctor"): lambda: _reference_doctor(args.json),
        ("reference", "probe"): _reference_probe,
    }
    command = getattr(args, f"{args.command}_command")
    return int(dispatch[(args.command, command)]())


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
