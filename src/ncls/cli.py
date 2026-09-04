from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

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
from ncls.learning.methods import get_method_plugin
from ncls.learning.producer import OnlineTrainingProducer
from ncls.learning.training import (
    TrainingCheckpointV1,
    DistributedContext,
    TrainingEngine,
    TrainingPlanResolver,
    ResolvedTrainingPlan,
    HookBinding,
    TrainingEventBus,
    build_training_review,
    load_evaluation_snapshot,
    load_training_checkpoint_v1,
    load_metric_rows,
    save_training_checkpoint_v1,
    launch_distributed,
    preflight_topology,
    prepare_process_environment,
    write_training_review,
    worker_execution_context,
)
from ncls.learning.training.config import TrainingConfig
from ncls.learning.training.hooks import TensorBoardHook, VisualEvalHook
from ncls.paths import PROJECT_ROOT
from ncls.references.backend import (
    close_reference_backend_devices,
    create_reference_backend,
)
from ncls.visual_eval import VisualEvalCollector, VisualEvalSpool
from ncls.visual_eval import VisualEvalWorker, WindowsViewerExecutor


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
    resolved_plan: Any,
    execution_context: Any,
    output: Path,
    resume_path: Path | None,
    stop_at_step: int | None,
) -> int:
    distributed = _setup_ddp(execution_context)
    ddp_rank = distributed.rank
    ddp_world = distributed.world_size
    is_rank0 = ddp_rank == 0
    if ddp_world > 1 and output.is_absolute() is False:
        output = output.resolve()
    setup_error: BaseException | None = None
    producer = None
    data_session = None
    try:
        config = resolved_plan.to_runtime_config()
        plugin = get_method_plugin(resolved_plan.selection.method)
        data_execution_plan = DataExecutionPlan.build(
            data_key=resolved_plan.selection.data,
            source_family_id=str(config.source["family_id"]),
            routes=[route.to_dict() for route in config.all_routes],
            requirements=plugin.data.requirements(),
            execution=resolved_plan.execution.to_dict(),
            rank=execution_context.rank,
            world_size=execution_context.world_size,
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
    except BaseException as error:
        setup_error = error
    try:
        distributed.synchronize_rank_errors("training data setup", setup_error)
    except BaseException as distributed_setup_error:
        rollback_error: BaseException | None = None
        try:
            if data_session is not None:
                data_session.close()
            elif producer is not None:
                producer.close()
        except BaseException as error:
            rollback_error = error
        finally:
            close_reference_backend_devices()
            distributed.close()
        if rollback_error is not None:
            raise distributed_setup_error from rollback_error
        raise
    gpu_indices = list(resolved_plan.execution.devices)
    metrics_path = output.with_name(f"{output.stem}.metrics.jsonl")
    summary_path = output.with_name(f"{output.stem}.summary.json")
    review_path = output.with_name(f"{output.stem}.review.json")
    metric_count = 0
    checkpoint_write_seconds: list[float] = []
    started = time.perf_counter()
    run_completed = False
    tensorboard_hook = None
    visual_eval_hook = None
    visual_eval_collector = None
    event_bus = None
    resume_v1 = None
    try:
        if resume_path is None:
            resume = None
        else:
            resume_v1 = load_training_checkpoint_v1(
                resume_path,
                map_location="cpu" if ddp_world > 1 else config.device,
            )
            if (
                resume_v1.data_identity["data_execution_plan_identity"]
                != producer.data_execution_plan_identity
            ):
                raise ValueError("resume checkpoint data execution plan identity mismatch")
            resume = resume_v1.to_runner_checkpoint(
                plan=resolved_plan, plugin=plugin
            )
        bindings = []
        if (
            is_rank0
            and resolved_plan.hooks.tensorboard.enabled
        ):
            tensorboard_hook = TensorBoardHook(
                output.with_name(f"{output.stem}.tensorboard"),
                rank=0,
                flush_seconds=resolved_plan.hooks.tensorboard.flush_seconds,
                queue_capacity=resolved_plan.hooks.tensorboard.queue_capacity,
            )
            bindings.append(
                HookBinding("tensorboard", tensorboard_hook, "diagnostic", True)
            )
        if (
            is_rank0
            and resolved_plan.hooks.visual_eval.enabled
        ):
            visual_spool = VisualEvalSpool(
                output.with_name(f"{output.stem}.visual-eval"),
                capacity=resolved_plan.hooks.visual_eval.queue_capacity,
            )
            visual_eval_hook = VisualEvalHook(
                resolved_plan,
                output,
                output.parent,
                visual_spool,
                rank=0,
            )
            visual_eval_collector = VisualEvalCollector(
                visual_spool,
                output.parent,
                rank=0,
                world_size=ddp_world,
            )
            if resume_v1 is not None:
                visual_state = resume_v1.hook_state.get("visual_eval")
                if isinstance(visual_state, dict):
                    visual_eval_hook.load_state_dict(visual_state)
                collector_state = resume_v1.hook_state.get("visual_eval_collector")
                if isinstance(collector_state, dict):
                    visual_eval_collector.load_state_dict(collector_state)
            bindings.append(
                HookBinding("visual-eval", visual_eval_hook, "diagnostic", True)
            )
        event_bus = TrainingEventBus(tuple(bindings))

        def current_hook_state() -> dict[str, Any]:
            state: dict[str, Any] = {}
            if visual_eval_hook is not None:
                state["visual_eval"] = dict(visual_eval_hook.state_dict())
            if visual_eval_collector is not None:
                state["visual_eval_collector"] = dict(
                    visual_eval_collector.state_dict()
                )
            return state

        def current_probe_ids() -> tuple[str, ...]:
            return () if visual_eval_hook is None else visual_eval_hook.probe_ids
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
            if visual_eval_collector is not None and event_bus is not None:
                visual_eval_collector.collect(event_bus)

        def save_periodic(checkpoint) -> None:
            if not is_rank0:
                return
            assert metric_stream is not None
            metric_stream.flush()
            path = output.with_name(
                f"{output.stem}.step{checkpoint.global_step:08d}{output.suffix}"
            )
            checkpoint_started = time.perf_counter()
            save_training_checkpoint_v1(
                path,
                TrainingCheckpointV1.from_runner_checkpoint(
                    checkpoint,
                    plan=resolved_plan,
                    plugin=plugin,
                    data_execution_plan_identity=data_execution_plan.identity,
                    hook_state=current_hook_state(),
                    visual_eval_probe_ids=current_probe_ids(),
                ),
            )
            checkpoint_write_seconds.append(time.perf_counter() - checkpoint_started)

        result = TrainingEngine(
            plugin,
            data_session,
            config,
            checkpoint_callback=save_periodic,
            metric_callback=record_metric,
            event_bus=event_bus,
            distributed_context=distributed,
        ).run(resume=resume, stop_at_step=stop_at_step)
        digest = ""
        elapsed_seconds = time.perf_counter() - started

        def commit_final_artifacts() -> None:
            nonlocal digest, elapsed_seconds
            if result.checkpoint is None:
                raise RuntimeError("rank0 training result has no checkpoint")
            if event_bus is not None:
                if visual_eval_collector is not None:
                    visual_eval_collector.collect(event_bus)
                event_bus.flush()
            assert metric_stream is not None
            checkpoint_started = time.perf_counter()
            digest = save_training_checkpoint_v1(
                output,
                TrainingCheckpointV1.from_runner_checkpoint(
                    result.checkpoint,
                    plan=resolved_plan,
                    plugin=plugin,
                    data_execution_plan_identity=data_execution_plan.identity,
                    hook_state=current_hook_state(),
                    visual_eval_probe_ids=current_probe_ids(),
                ),
            )
            checkpoint_write_seconds.append(time.perf_counter() - checkpoint_started)
            metric_stream.flush()
            metric_stream.close()
            elapsed_seconds = time.perf_counter() - started
            summary = {
                "format_name": "ncls.training-run-summary",
                "format_version": 1,
                "training_config_sha256": config.sha256,
                "resolved_plan_sha256": resolved_plan.sha256,
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
                "hook_failures": (
                    []
                    if event_bus is None
                    else [
                        {
                            "hook": item.hook_name,
                            "operation": item.operation,
                            "event_kind": item.event_kind,
                            "message": item.message,
                        }
                        for item in event_bus.failures
                    ]
                ),
            }
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            review = build_training_review(
                config,
                plugin.descriptor,
                result.checkpoint,
                checkpoint_sha256=digest,
                checkpoint_bytes=output.stat().st_size,
                metric_rows=load_metric_rows(metrics_path, config_sha256=config.sha256),
                metrics_bytes=metrics_path.stat().st_size,
                elapsed_seconds=elapsed_seconds,
                checkpoint_write_seconds=checkpoint_write_seconds,
            )
            write_training_review(review_path, review)
        distributed.run_rank_zero("final training artifact commit", commit_final_artifacts)
        run_completed = True
    finally:
        cleanup_error: BaseException | None = None
        try:
            if "metric_stream" in locals() and metric_stream is not None and not metric_stream.closed:
                metric_stream.close()
        except BaseException as error:
            cleanup_error = error
        try:
            data_session.close()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        try:
            if event_bus is not None:
                event_bus.close()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        close_reference_backend_devices()
        try:
            if run_completed:
                distributed.synchronize_rank_errors(
                    "teardown readiness",
                    cleanup_error,
                )
            elif cleanup_error is not None:
                raise cleanup_error
        finally:
            distributed.close()
    if is_rank0:
        print(
            f"TrainingCheckpoint v1 {digest}: {output}"
        )
    return 0


def _parse_devices(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    parts = value.split(",")
    if not parts or any(not part or not part.isdecimal() for part in parts):
        raise ValueError("--devices must be a comma-separated list of GPU indices")
    devices = tuple(int(part) for part in parts)
    if len(set(devices)) != len(devices):
        raise ValueError("--devices contains duplicate GPU indices")
    return devices


def _train_yaml(
    config_path: Path,
    output: Path | None,
    devices_text: str | None,
    resume_path: Path | None,
    stop_at_step: int | None,
) -> int:
    resolver = TrainingPlanResolver(PROJECT_ROOT)
    overrides = _parse_devices(devices_text)
    plan = resolver.resolve(config_path, devices=overrides)
    output = (
        PROJECT_ROOT
        / "artifacts"
        / "training"
        / config_path.stem
        / "checkpoint.pt"
        if output is None
        else output
    )
    topology = preflight_topology(plan.execution.devices)
    if topology.mode == "distributed-launch":
        extra = ["--devices", ",".join(str(item) for item in topology.devices)]
        if resume_path is not None:
            extra.extend(("--resume", str(resume_path.resolve())))
        if stop_at_step is not None:
            extra.extend(("--stop-at-step", str(stop_at_step)))
        return launch_distributed(
            topology,
            config=config_path.resolve(),
            output=output.resolve(),
            extra_arguments=extra,
        )
    prepare_process_environment(topology)
    execution_context = worker_execution_context(topology)
    return _run_training(
        plan,
        execution_context,
        output,
        resume_path,
        stop_at_step,
    )


def _setup_ddp(execution_context: Any) -> DistributedContext:
    """Initialize NCCL from a launcher-validated execution context."""
    return DistributedContext.initialize(execution_context)


def _validate_checkpoint(checkpoint_path: Path, batches: int, device: int) -> int:
    if batches < 1:
        raise ValueError("validation batch count must be positive")
    if device < 0:
        raise ValueError("validation device must be nonnegative")
    topology = preflight_topology((device,))
    prepare_process_environment(topology)
    execution_context = worker_execution_context(topology)
    evaluation = load_evaluation_snapshot(checkpoint_path, map_location="cpu")
    config = TrainingConfig.from_dict(
        evaluation.deployment_payload["training_config"]
    )
    plugin = get_method_plugin(evaluation.public_method_key)
    if evaluation.legacy_v4:
        # Legacy v4 did not persist the new data-plane policy. Evaluation is
        # read-only, so reconstruct the explicit synchronous baseline; its
        # identity is deliberately excluded from legacy compatibility checks.
        data_execution_plan = DataExecutionPlan.build(
            data_key="legacy",
            source_family_id=str(config.source["family_id"]),
            routes=[route.to_dict() for route in config.all_routes],
            requirements=plugin.data.requirements(),
            execution={
                "num_workers": 0,
                "host_prefetch": 1,
                "ready_batches": 1,
                "reference_batch_steps": 1,
                "reference_inflight": 1,
                "transfer_streams": 0,
                "residency": {"budget_mib": 4096},
            },
            rank=execution_context.rank,
            world_size=execution_context.world_size,
        )
    else:
        resolved_value = evaluation.deployment_payload.get("resolved_plan")
        if not isinstance(resolved_value, dict):
            raise ValueError("new evaluation checkpoint has no resolved training plan")
        resolved_plan = ResolvedTrainingPlan.from_dict(resolved_value)
        plugin = get_method_plugin(evaluation.public_method_key)
        data_execution_plan = DataExecutionPlan.build(
            data_key=resolved_plan.selection.data,
            source_family_id=str(config.source["family_id"]),
            routes=[route.to_dict() for route in config.all_routes],
            requirements=plugin.data.requirements(),
            execution=resolved_plan.execution.to_dict(),
            rank=execution_context.rank,
            world_size=execution_context.world_size,
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
        expected = evaluation.data_identity
        actual = {
            "reference_program_identity": producer.reference_program_identity,
            "reference_execution_plan_identity": producer.reference_execution_plan_identity,
            "native_asset_collection_identity": producer.native_asset_collection_identity,
            "query_stream_identity": producer.query_stream_identity,
        }
        if not evaluation.legacy_v4:
            actual["data_execution_plan_identity"] = str(
                producer.data_execution_plan_identity
            )
        else:
            expected = {
                name: value
                for name, value in expected.items()
                if name != "data_execution_plan_identity"
            }
        if actual != expected or producer.source_snapshot_ids != evaluation.source_snapshot_ids:
            raise ValueError("validation producer identity disagrees with the checkpoint")
        model = plugin.model_factory.create(config.model_context).to(producer.device)
        plugin.checkpoint.restore(
            model, evaluation.deployment_payload["model_state"]
        )
        phase_index = min(
            config.locate_step(evaluation.global_step)[0], len(config.phases) - 1
        )
        phase = config.phases[phase_index]
        plugin.lifecycle.configure_phase(model, phase.to_dict())
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
                        config.seed + route.seed_offset + index,
                        {
                            **dict(route.options),
                            "recipes": dict(phase.recipes),
                            "validation": True,
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
                    loss, _ = plugin.objective.compute(
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
        f"Validation method={evaluation.public_method_key} step={evaluation.global_step} "
        f"batches={batches} mean_loss={sum(losses) / len(losses):.9g}"
    )
    return 0


def _export_checkpoint(
    checkpoint_path: Path,
    output: Path,
    material_index: int,
) -> int:
    evaluation = load_evaluation_snapshot(checkpoint_path, map_location="cpu")
    compiled = compile_evaluation_package(
        evaluation,
        output,
        material_index=material_index,
        readiness_mode="formal",
    )
    print(f"ScatteringPackage@2 {compiled.manifest.package_id}: {compiled.root}")
    return 0


def _visual_eval_worker(
    spool_path: Path,
    artifact_root: Path,
    viewer: Path | None,
    worker_identity: str,
    capacity: int,
    max_jobs: int,
) -> int:
    if max_jobs < 1:
        raise ValueError("visual eval max jobs must be positive")
    worker = VisualEvalWorker(
        VisualEvalSpool(spool_path, capacity=capacity),
        artifact_root,
        WindowsViewerExecutor(viewer),
        worker_identity=worker_identity,
    )
    completed = 0
    failed = 0
    for _ in range(max_jobs):
        status = worker.run_once()
        if status is None:
            break
        if status.state == "completed":
            completed += 1
        else:
            failed += 1
        print(f"{status.request_id}\t{status.state}\t{status.message or ''}")
    print(f"Visual eval worker completed={completed} failed={failed}")
    return 1 if failed else 0


def _visual_eval_collect(
    spool_path: Path,
    artifact_root: Path,
    tensorboard_path: Path,
    capacity: int,
) -> int:
    spool = VisualEvalSpool(spool_path, capacity=capacity)
    collector = VisualEvalCollector(spool, artifact_root, rank=0, world_size=1)
    hook = TensorBoardHook(tensorboard_path, rank=0)
    bus = TrainingEventBus((HookBinding("tensorboard", hook, "fatal", True),))
    try:
        count = collector.collect(bus)
        bus.flush()
    finally:
        bus.close()
    print(f"Collected {count} visual eval result(s) into {tensorboard_path}")
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

    train_yaml = commands.add_parser(
        "train", help="从组合式 YAML 运行统一 online TrainingEngine"
    )
    train_yaml.add_argument("config", type=Path)
    train_yaml.add_argument(
        "--output",
        type=Path,
        help="checkpoint 路径；默认写入 artifacts/training/<run>/checkpoint.pt",
    )
    train_yaml.add_argument(
        "--devices",
        help="白名单执行覆盖，例如 0 或 0,1；多 GPU 仅支持 Linux/NCCL",
    )
    train_yaml.add_argument("--resume", type=Path)
    train_yaml.add_argument("--stop-at-step", type=int)

    validate_checkpoint = commands.add_parser(
        "validate", help="用 checkpoint 内嵌计划运行数值 batch 验证"
    )
    validate_checkpoint.add_argument("checkpoint", type=Path)
    validate_checkpoint.add_argument("--batches", type=int, default=8)
    validate_checkpoint.add_argument("--device", type=int, default=0)

    export_checkpoint = commands.add_parser(
        "export", help="从新 checkpoint 或只读 legacy v4 导出正式 ScatteringPackage"
    )
    export_checkpoint.add_argument("checkpoint", type=Path)
    export_checkpoint.add_argument("output", type=Path)
    export_checkpoint.add_argument("--material-index", type=int, default=0)

    visual_eval = commands.add_parser(
        "eval", help="Windows 1024 spp 可视化诊断 worker 与迟到结果收集"
    )
    visual_commands = visual_eval.add_subparsers(dest="eval_command", required=True)
    visual_worker = visual_commands.add_parser("worker", help="领取并执行可视化任务")
    visual_worker.add_argument("spool", type=Path)
    visual_worker.add_argument("artifact_root", type=Path)
    visual_worker.add_argument(
        "--viewer",
        type=Path,
        default=None,
        help="NclsViewer 路径；缺省由 Windows visual-eval capability 解析",
    )
    visual_worker.add_argument("--worker-id", default="windows-d3d12-worker")
    visual_worker.add_argument("--capacity", type=int, default=8)
    visual_worker.add_argument("--max-jobs", type=int, default=1)
    visual_collect = visual_commands.add_parser(
        "collect", help="把已完成或迟到的可视化结果写入 TensorBoard"
    )
    visual_collect.add_argument("spool", type=Path)
    visual_collect.add_argument("artifact_root", type=Path)
    visual_collect.add_argument("tensorboard", type=Path)
    visual_collect.add_argument("--capacity", type=int, default=8)

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
    if args.command == "train":
        return _train_yaml(
            args.config,
            args.output,
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
        if args.eval_command == "worker":
            return _visual_eval_worker(
                args.spool,
                args.artifact_root,
                args.viewer,
                args.worker_id,
                args.capacity,
                args.max_jobs,
            )
        return _visual_eval_collect(
            args.spool,
            args.artifact_root,
            args.tensorboard,
            args.capacity,
        )
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
