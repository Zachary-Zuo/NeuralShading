from __future__ import annotations

import json
import os

import yaml
from pathlib import Path
import time
from typing import Any

import torch

from ncls.commands import build_parser

from ncls.bundle import ScatteringPackage
from ncls.core.material import (
    BINARY_SIZE,
    MaterialProgram,
    canonicalize_layer_stack,
    pack_layer_stack,
    physical_material_hash,
    validate_material_program,
)
from ncls.core.source import create_source_family
from ncls.data import DataExecutionPlan, PipelineOnlineDataSession
from ncls.learning.batches import TrainingRouteRequest
from ncls.learning.evaluation_package import compile_evaluation_package
from ncls.learning.methods import get_method
from ncls.learning.producer import OnlineTrainingProducer
from ncls.learning.training import (
    TrainingCheckpoint,
    DistributedContext,
    TrainingEngine,
    TrainingPlanResolver,
    ResolvedTrainingPlan,
    HookBinding,
    TrainingEventBus,
    build_training_review,
    load_checkpoint,
    load_metric_rows,
    save_checkpoint,
    preflight_topology,
    write_training_review,
    worker_execution_context,
)
from ncls.learning.training.hooks import TensorBoardHook, VisualEvalHook
from ncls.paths import PROJECT_ROOT
from ncls.runs import RunPaths
from ncls.visual_eval import create_visual_evaluator
from ncls.visual_eval.evaluator import NoVisualEvaluation, VisualContext
from ncls.references.backend import (
    close_reference_backend_devices,
    create_reference_backend,
)


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


def _run_training(
    resolved_plan: ResolvedTrainingPlan,
    execution_context: Any,
    paths: RunPaths,
    resume_path: Path | None,
    stop_at_step: int | None,
) -> int:
    distributed = _setup_ddp(execution_context)
    config = resolved_plan.training
    plugin = get_method(resolved_plan.selection.method)
    data_session = None
    producer = None
    event_bus = None
    metric_stream = None
    started = time.perf_counter()
    checkpoint_seconds = []

    def setup_data():
        nonlocal producer, data_session
        data_plan = DataExecutionPlan.build(
            data_key=resolved_plan.selection.data,
            source_family_id=str(config.source["family_id"]),
            routes=[route.to_dict() for route in config.all_routes],
            requirements=plugin.requirements(),
            execution=resolved_plan.execution.to_dict(),
            rank=execution_context.rank, world_size=execution_context.world_size,
        )
        producer = OnlineTrainingProducer(plugin, config, execution_context=execution_context, data_execution_plan=data_plan)
        data_session = PipelineOnlineDataSession(
            producer, execution_plan_identity=data_plan.session_identity,
            ready_capacity=data_plan.ready_batches, production_batch_steps=data_plan.reference_batch_steps,
        )

    try:
        distributed.run_all_ranks("training data setup", setup_data)
        resume = distributed.run_all_ranks(
            "checkpoint load", lambda: None if resume_path is None else load_checkpoint(resume_path),
        )

        def setup_outputs():
            nonlocal event_bus, metric_stream
            bindings = []
            if distributed.is_rank_zero:
                paths.logs.mkdir(parents=True, exist_ok=True)
                (paths.root / "resolved.yaml").write_text(
                    yaml.safe_dump(resolved_plan.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8",
                )
                retained = []
                if resume is not None and paths.metrics.exists():
                    retained = [line for line in paths.metrics.read_text(encoding="utf-8").splitlines()
                                if line.strip() and float(json.loads(line).get("step", -1)) <= resume.global_step]
                metric_stream = paths.metrics.open("w", encoding="utf-8", newline="\n")
                for line in retained:
                    metric_stream.write(line + "\n")
                settings = resolved_plan.hooks.tensorboard
                if settings.enabled:
                    writer = TensorBoardHook(
                        paths.tensorboard, flush_seconds=settings.flush_seconds, queue_capacity=settings.queue_capacity,
                        resume_step=None if resume is None else resume.global_step,
                    )
                    bindings.append(HookBinding("tensorboard", writer, "fatal", True))
            event_bus = TrainingEventBus(tuple(bindings))

        distributed.run_all_ranks("training output setup", setup_outputs)
        context = VisualContext(
            0, plugin, config, producer.source_snapshot_ids,
            resolved_plan.hooks.visual_eval, paths.evaluation,
        )
        visual_hook = VisualEvalHook(create_visual_evaluator(context.settings), context, event_bus)

        def record_metric(row):
            if metric_stream is not None:
                metric_stream.write(json.dumps({
                    "record_kind": "validation" if "validation/loss" in row else "training",
                    "training_config_sha256": config.sha256, **row,
                }, ensure_ascii=False) + "\n")

        def persist(checkpoint, target):
            checkpoint.resolved_plan = resolved_plan.to_dict()
            tick = time.perf_counter()
            digest = save_checkpoint(target, checkpoint)
            checkpoint_seconds.append(time.perf_counter() - tick)
            metric_stream.flush()
            return digest

        result = TrainingEngine(
            plugin, data_session, config,
            checkpoint_callback=lambda checkpoint: persist(checkpoint, paths.step_checkpoint(checkpoint.global_step)),
            metric_callback=record_metric, event_bus=event_bus, distributed_context=distributed,
            visual_callback=visual_hook,
        ).run(resume=resume, stop_at_step=stop_at_step)

        def commit():
            checkpoint = result.checkpoint
            digest = persist(checkpoint, paths.checkpoint)
            event_bus.flush()
            elapsed = time.perf_counter() - started
            summary = {
                "checkpoint": str(paths.checkpoint.relative_to(paths.root)), "checkpoint_sha256": digest,
                "final_step": checkpoint.global_step, "planned_final_step": config.total_steps,
                "complete": checkpoint.global_step == config.total_steps,
                "world_size": distributed.world_size, "devices": list(resolved_plan.execution.devices),
                "elapsed_seconds": elapsed, "checkpoint_write_seconds": checkpoint_seconds,
            }
            (paths.logs / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            review = build_training_review(
                config, plugin.descriptor, checkpoint, checkpoint_sha256=digest,
                checkpoint_bytes=paths.checkpoint.stat().st_size,
                metric_rows=load_metric_rows(paths.metrics, allow_empty=True),
                metrics_bytes=paths.metrics.stat().st_size, elapsed_seconds=elapsed,
                checkpoint_write_seconds=checkpoint_seconds,
            )
            write_training_review(paths.logs / "review.json", review)
            print(f"Checkpoint: {paths.checkpoint}")

        distributed.run_rank_zero("final training output commit", commit)
    finally:
        try:
            if metric_stream is not None:
                metric_stream.close()
            if event_bus is not None:
                event_bus.close()
        finally:
            try:
                if data_session is not None:
                    data_session.close()
                elif producer is not None:
                    producer.close()
            finally:
                close_reference_backend_devices()
                distributed.close()
    return 0


def _train_yaml(config_path, devices, resume_path, stop_at_step) -> int:
    plan = TrainingPlanResolver(PROJECT_ROOT).resolve(config_path, devices=devices)
    topology = preflight_topology(devices)
    execution_context = worker_execution_context(topology)
    paths = RunPaths(Path(os.environ["NCLS_RUN_DIR"]))
    return _run_training(plan, execution_context, paths, resume_path, stop_at_step)


def _setup_ddp(execution_context: Any) -> DistributedContext:
    """Initialize NCCL from a launcher-validated execution context."""
    return DistributedContext.initialize(execution_context)


def _validate_checkpoint(checkpoint_path: Path, batches: int, device: int) -> int:
    if batches < 1:
        raise ValueError("validation batch count must be positive")
    if device < 0:
        raise ValueError("validation device must be nonnegative")
    topology = preflight_topology((device,))
    execution_context = worker_execution_context(topology)
    evaluation = load_checkpoint(checkpoint_path)
    plugin = get_method(evaluation.method_key)
    resolved_plan = ResolvedTrainingPlan.from_dict(evaluation.resolved_plan)
    config = resolved_plan.training
    data_execution_plan = DataExecutionPlan.build(
        data_key=resolved_plan.selection.data, source_family_id=str(config.source["family_id"]),
        routes=[route.to_dict() for route in config.all_routes], requirements=plugin.requirements(),
        execution=resolved_plan.execution.to_dict(), rank=0, world_size=1,
    )
    producer = OnlineTrainingProducer(
        plugin,
        config,
        execution_context=execution_context,
        data_execution_plan=data_execution_plan,
    )
    data_session = PipelineOnlineDataSession(
        producer,
        execution_plan_identity=data_execution_plan.session_identity,
        ready_capacity=data_execution_plan.ready_batches,
        production_batch_steps=data_execution_plan.reference_batch_steps,
    )
    try:
        if producer.source_snapshot_ids != evaluation.source_snapshot_ids:
            raise ValueError("validation source 与 checkpoint 不匹配")
        model = plugin.create_trainable(config.model_context).to(producer.device)
        plugin.restore_training_state(
            model, evaluation.model_state
        )
        phase_index = min(
            config.locate_step(evaluation.global_step)[0], len(config.phases) - 1
        )
        phase = config.phases[phase_index]
        plugin.configure_phase(model, phase.to_dict())
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for index in range(batches):
                requests = {
                    route.name: TrainingRouteRequest(
                        f"validation:{route.name}",
                        route.kind,
                        route.batch_size,
                        route.direction_count,
                        evaluation.global_step,
                        (
                            int(route.options["validation_seed"])
                            if "validation_seed" in route.options
                            else config.seed + route.seed_offset + index
                        ),
                        {
                            **dict(route.options),
                            "recipes": dict(phase.recipes),
                            "validation": True,
                            "validation_group_index": index,
                        },
                    )
                    for route in phase.routes
                }
                logical_id = data_session.submit_step(
                    requests,
                    boundary_id=f"validation:{phase.name}:{index}",
                )
                step_batch = data_session.acquire_step(logical_id)
                try:
                    loss, _ = plugin.training_objective(
                        model,
                        step_batch.batches,
                        {
                            "name": phase.name,
                            "phase_index": phase_index,
                            "phase_step": evaluation.global_step
                            - config.phase_start_step(phase_index),
                            "phase_steps": phase.steps,
                            "global_step": evaluation.global_step,
                            "total_steps": config.total_steps,
                            "parameter_groups": list(phase.parameter_groups),
                            "loss_terms": list(phase.loss_terms),
                            "recipes": dict(phase.recipes),
                            "validation": True,
                        },
                    )
                    losses.append(float(loss))
                finally:
                    step_batch.release()
    finally:
        try:
            data_session.close()
        finally:
            close_reference_backend_devices()
    print(
        f"Validation method={evaluation.method_key} step={evaluation.global_step} "
        f"batches={batches} mean_loss={sum(losses) / len(losses):.9g}"
    )
    return 0


def _export_checkpoint(
    checkpoint_path: Path,
    output: Path | None,
    material_index: int,
) -> int:
    evaluation = load_checkpoint(checkpoint_path)
    if output is None:
        output = RunPaths.from_checkpoint(checkpoint_path).exports / f"step-{evaluation.global_step:08d}" / f"material-{material_index}"
    compiled = compile_evaluation_package(
        evaluation,
        output,
        material_index=material_index,
    )
    from ncls.viewer.export import prepare_source_reference
    source = prepare_source_reference(compiled, compiled.root.with_name(compiled.root.name + "-source"),
        evaluation.source["materials"][material_index]["locator"])
    print(f"ScatteringPackage@2 {compiled.manifest.package_id}: {compiled.root}")
    print(f'Viewer: scripts/launch_viewer.ps1 -Package "{compiled.root}" -Material "{source}"')
    return 0


def _visual_eval(checkpoint_path: Path, config_path: Path | None) -> int:
    from ncls.learning.training.plan import VisualEvalSettings

    settings = TrainingPlanResolver(PROJECT_ROOT).resolve(config_path).hooks.visual_eval if config_path else VisualEvalSettings()
    evaluator = create_visual_evaluator(settings)
    if isinstance(evaluator, NoVisualEvaluation):
        return 0
    checkpoint = load_checkpoint(checkpoint_path)
    plan = ResolvedTrainingPlan.from_dict(checkpoint.resolved_plan)
    if config_path is None:
        settings = plan.hooks.visual_eval
        evaluator = create_visual_evaluator(settings)
        if isinstance(evaluator, NoVisualEvaluation):
            return 0
    method = get_method(checkpoint.method_key)
    model = method.create_trainable(plan.training.model_context)
    method.restore_training_state(model, checkpoint.model_state)
    paths = RunPaths.from_checkpoint(checkpoint_path)
    writer = TensorBoardHook(paths.tensorboard)
    events = TrainingEventBus((HookBinding("tensorboard", writer, "fatal", True),))
    try:
        context = VisualContext(checkpoint.global_step, method, plan.training, checkpoint.source_snapshot_ids, settings, paths.evaluation)
        result = evaluator.evaluate(model, context)
        if result is not None:
            from ncls.learning.training.events import TrainingEvent
            events.emit(TrainingEvent("visual-eval-completed", checkpoint.global_step, 0, 1,
                artifacts={name: str(path) for name, path in result.images.items()}))
    finally:
        events.close()
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        return _train_yaml(
            args.config,
            args.devices,
            args.resume,
            args.stop_at_step,
        )
    if args.command == "validate":
        return _validate_checkpoint(args.checkpoint, args.batches, args.device)
    if args.command == "export":
        return _export_checkpoint(
            args.checkpoint, args.output, args.material_index
        )
    if args.command == "eval":
        return _visual_eval(args.checkpoint, args.config)
    dispatch: dict[tuple[str, str], Any] = {
        ("material", "validate"): lambda: _material_validate(args.path),
        ("material", "normalize"): lambda: _material_normalize(
            args.path, args.output
        ),
        ("material", "pack"): lambda: _material_pack(args.path, args.output),
        ("package", "validate"): lambda: _package_validate(args.path),
        ("reference", "doctor"): lambda: _reference_doctor(args.json),
        ("reference", "probe"): _reference_probe,
    }
    command = getattr(args, f"{args.command}_command")
    return int(dispatch[(args.command, command)]())


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
