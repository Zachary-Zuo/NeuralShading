from pathlib import Path

root = Path(__file__).resolve().parents[4]
path = root / 'src/ncls/cli.py'
value = path.read_text(encoding='utf-8')
value = value.replace('import json\n', 'import json\nimport os\n\nimport yaml\n')
value = value.replace('    TrainingCheckpointV1,', '    TrainingCheckpoint,')
value = value.replace('    load_evaluation_snapshot,\n', '').replace('    load_training_checkpoint_v1,', '    load_checkpoint,').replace('    save_training_checkpoint_v1,', '    save_checkpoint,').replace('    launch_distributed,\n', '')
value = value.replace('from ncls.paths import PROJECT_ROOT', 'from ncls.paths import PROJECT_ROOT\nfrom ncls.runs import RunPaths\nfrom ncls.visual_eval import create_visual_evaluator\nfrom ncls.visual_eval.evaluator import VisualContext')
value = value.replace('from ncls.visual_eval import VisualEvalCollector, VisualEvalSpool\n', '').replace('from ncls.visual_eval import VisualEvalWorker, WindowsViewerExecutor\n','')
start = value.index('def _run_training(')
end = value.index('def _setup_ddp(', start)
value = value[:start] + '''def _run_training(
    resolved_plan: ResolvedTrainingPlan,
    execution_context: Any,
    paths: RunPaths,
    resume_path: Path | None,
    stop_at_step: int | None,
) -> int:
    distributed = _setup_ddp(execution_context)
    config = resolved_plan.to_runtime_config()
    plugin = get_method_plugin(resolved_plan.selection.method)
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
                metric_stream = paths.metrics.open("w", encoding="utf-8", newline="\\n")
                for line in retained:
                    metric_stream.write(line + "\\n")
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
                }, ensure_ascii=False) + "\\n")

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
            visual_callback=visual_hook, visual_interval=context.settings.interval_steps,
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
            (paths.logs / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
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


''' + value[end:]
value = value.replace('evaluation = load_evaluation_snapshot(checkpoint_path, map_location="cpu")', 'evaluation = load_checkpoint(checkpoint_path)')
value = value.replace('evaluation.deployment_payload["training_config"]', 'evaluation.training_config')
value = value.replace('evaluation.public_method_key', 'evaluation.method_key')
start = value.index('    if evaluation.legacy_v4:')
end = value.index('    producer = OnlineTrainingProducer(', start)
value = value[:start] + '''    resolved_plan = ResolvedTrainingPlan.from_dict(evaluation.resolved_plan)
    data_execution_plan = DataExecutionPlan.build(
        data_key=resolved_plan.selection.data, source_family_id=str(config.source["family_id"]),
        routes=[route.to_dict() for route in config.all_routes], requirements=plugin.requirements(),
        execution=resolved_plan.execution.to_dict(), rank=0, world_size=1,
    )
''' + value[end:]
start = value.index('        expected = evaluation.data_identity')
end = value.index('        model = ', start)
value = value[:start] + '''        if producer.source_snapshot_ids != evaluation.source_snapshot_ids:
            raise ValueError("validation source 与 checkpoint 不匹配")
''' + value[end:]
value = value.replace('evaluation.deployment_payload["model_state"]', 'evaluation.model_state')
value = value.replace('    output: Path,\n    material_index: int,', '    output: Path | None,\n    material_index: int,')
value = value.replace('    compiled = compile_evaluation_package(\n        evaluation,\n        output,', '    if output is None:\n        output = RunPaths.from_checkpoint(checkpoint_path).exports / f"step-{evaluation.global_step:08d}" / f"material-{material_index}"\n    compiled = compile_evaluation_package(\n        evaluation,\n        output,')
value = value.replace('        readiness_mode="formal",\n', '')
start = value.index('def _visual_eval_worker(')
end = value.index('def _package_validate(', start)
value = value[:start] + '''def _visual_eval(checkpoint_path: Path, config_path: Path | None) -> int:
    checkpoint = load_checkpoint(checkpoint_path)
    plan = ResolvedTrainingPlan.from_dict(checkpoint.resolved_plan)
    if config_path is not None:
        plan.hooks = TrainingPlanResolver(PROJECT_ROOT).resolve(config_path).hooks
    method = get_method_plugin(checkpoint.method_key)
    model = method.create_trainable(plan.training.model_context)
    method.restore_training_state(model, checkpoint.model_state)
    paths = RunPaths.from_checkpoint(checkpoint_path)
    writer = TensorBoardHook(paths.tensorboard)
    events = TrainingEventBus((HookBinding("tensorboard", writer, "fatal", True),))
    try:
        context = VisualContext(checkpoint.global_step, method, plan.training, checkpoint.source_snapshot_ids, plan.hooks.visual_eval, paths.evaluation)
        result = create_visual_evaluator(context.settings).evaluate(model, context)
        if result is not None:
            from ncls.learning.training.events import TrainingEvent
            events.emit(TrainingEvent("visual-eval-completed", checkpoint.global_step, 0, 1,
                artifacts={name: str(path) for name, path in result.images.items()}))
    finally:
        events.close()
    return 0


''' + value[end:]
value = value.replace('            args.output,\n            args.devices,', '            args.devices,')
start = value.index('    if args.command == "eval":')
end = value.index('    dispatch:', start)
value = value[:start] + '    if args.command == "eval":\n        return _visual_eval(args.checkpoint, args.config)\n' + value[end:]
path.write_text(value, encoding='utf-8', newline='\n')

path = root / 'src/ncls/learning/training/engine.py'
value = path.read_text(encoding='utf-8')
value = value.replace('        distributed_context: DistributedContext | None = None,', '        distributed_context: DistributedContext | None = None,\n        visual_callback: Callable[[nn.Module, int], None] | None = None,\n        visual_interval: int = 5000,')
value = value.replace('        self.event_bus = event_bus', '        self.event_bus = event_bus\n        self.visual_callback = visual_callback\n        self.visual_interval = visual_interval')
start = value.index('        plugin.validate_training_config(config.to_dict())')
end = value.index('        self.plugin = plugin', start)
value = value[:start] + value[end:]
value = value.replace('                barrier = min(target_step, phase_end, next_validation)', '''                next_checkpoint = (global_step // self.config.checkpoint_interval + 1) * self.config.checkpoint_interval
                next_visual = (global_step // self.visual_interval + 1) * self.visual_interval
                barrier = min(target_step, phase_end, next_validation, next_checkpoint, next_visual)''')
value = value.replace('                if needs_validation or checkpoint_boundary:', '                if global_step % self.config.checkpoint_interval == 0 or checkpoint_boundary:')
anchor = '                checkpoint_boundary = boundary and phase.checkpoint_boundary\n'
value = value.replace(anchor, '''                if self.visual_callback is not None and self.distributed.is_rank_zero:
                    self.visual_callback(model, global_step)

''' + anchor)
path.write_text(value, encoding='utf-8', newline='\n')

path = root / 'src/ncls/learning/training/distributed.py'
value = path.read_text(encoding='utf-8').replace('from ncls.learning.methods.contracts import ObjectiveFacet', 'from ncls.learning.method import Method').replace('ObjectiveFacet', 'Method').replace('self.objective.compute(', 'self.objective.training_objective(')
path.write_text(value, encoding='utf-8', newline='\n')
path = root / 'src/ncls/learning/training/review.py'
value = path.read_text(encoding='utf-8').replace('config_sha256: str,', 'config_sha256: str | None = None,')
value = value.replace('if value.get("training_config_sha256") != config_sha256:', 'if config_sha256 is not None and value.get("training_config_sha256") != config_sha256:')
path.write_text(value, encoding='utf-8', newline='\n')

path = root / 'src/ncls/visual_eval/__init__.py'
path.write_text('''from __future__ import annotations

import platform
from .evaluator import NoVisualEvaluation, VisualContext, VisualEvaluator, VisualResult


def create_visual_evaluator(settings, *, system: str | None = None) -> VisualEvaluator:
    system = platform.system() if system is None else system
    if not settings.enabled or system != "Windows":
        return NoVisualEvaluation()
    from .windows import WindowsVisualEvaluator

    return WindowsVisualEvaluator()
''', encoding='utf-8', newline='\n')
