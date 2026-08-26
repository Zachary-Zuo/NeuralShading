from __future__ import annotations

import argparse
from pathlib import Path

from .core.material import (
    BINARY_SIZE,
    MaterialProgram,
    canonicalize_layer_stack,
    pack_layer_stack,
    physical_material_hash,
    validate_material_program,
)


def _load_program(path: Path) -> MaterialProgram:
    return MaterialProgram.from_json(path.read_text(encoding="utf-8"))


def _validate_material(path: Path) -> int:
    program = _load_program(path)
    validate_material_program(program)
    stack = canonicalize_layer_stack(program)
    print(f"MaterialProgram OK: {physical_material_hash(program)}")
    print(f"LayerStackIR: {len(stack.interfaces)} interfaces, {len(stack.media)} media, {BINARY_SIZE} bytes")
    return 0


def _normalize_material(path: Path, output: Path) -> int:
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


def _pack_material(path: Path, output: Path) -> int:
    program = _load_program(path)
    payload = pack_layer_stack(canonicalize_layer_stack(program))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(f"Wrote {output}: {len(payload)} bytes")
    return 0


def _validate_dataset(path: Path, *, skip_hashes: bool) -> int:
    from .data import validate_reference_dataset

    dataset = validate_reference_dataset(path, verify_hashes=not skip_hashes)
    counts = dataset.manifest.counts
    print(f"ReferenceShard OK: {dataset.manifest.dataset_id}")
    print(
        f"{counts['state_count']} states, {counts['query_group_count']} query groups, "
        f"{counts['direction_count']} directions per group"
    )
    dataset.close()
    return 0


def _plan_corpus(args: argparse.Namespace) -> int:
    from .data import CorpusPlan, CorpusSelection, plan_layer_stack_corpus

    plan = CorpusPlan.load(args.config)
    selection = CorpusSelection.load(args.selection) if args.selection is not None else None
    manifest = plan_layer_stack_corpus(plan, args.shard_root, selection)
    manifest.write(args.output)
    print(f"Planned {len(manifest.shards)} shards for {manifest.name}: {manifest.corpus_id}")
    return 0


def _collect_corpus(args: argparse.Namespace) -> int:
    from .data import CorpusPlan, CorpusSelection, collect_layer_stack_corpus

    plan = CorpusPlan.load(args.config)
    selection = CorpusSelection.load(args.selection) if args.selection is not None else None
    manifest = collect_layer_stack_corpus(
        plan,
        args.shard_root,
        args.output,
        selection,
    )
    print(f"Collected ReferenceCorpus {manifest.corpus_id} to {args.output}")
    return 0


def _collect_state(args: argparse.Namespace) -> int:
    from .data import CorpusPlan, collect_layer_stack_state

    plan = CorpusPlan.load(args.config)
    state, manifest = collect_layer_stack_state(
        plan,
        args.structure_family,
        args.state_index,
        args.role,
        args.output,
    )
    counts = manifest.counts
    print(
        f"Collected {state.asset_id} (difficulty={state.difficulty_class}, "
        f"tags={'+'.join(state.difficulty_tags) or 'none'}) as {args.role}: "
        f"{manifest.dataset_id}"
    )
    print(
        f"{counts['query_group_count']} query groups x "
        f"{counts['direction_count']} directions -> {args.output}"
    )
    return 0


def _validate_corpus(args: argparse.Namespace) -> int:
    from .data import validate_reference_corpus

    manifest = validate_reference_corpus(args.path)
    print(f"ReferenceCorpus OK: {manifest.corpus_id} ({len(manifest.shards)} shards)")
    return 0


def _audit_dense(args: argparse.Namespace) -> int:
    from .data import audit_dense_slice_resolution

    report = audit_dense_slice_resolution(args.path, args.output)
    print(
        f"Dense slice audit {report['report_sha256']}: "
        f"{len(report['promote_state_ids'])} states require 16,384 directions"
    )
    return 0


def _mollification_freeze(args: argparse.Namespace) -> int:
    from .data import freeze_mollification_anchors

    lock = freeze_mollification_anchors(args.config, args.output)
    print(f"Mollification anchor lock: {lock['anchor_lock_sha256']} -> {args.output}")
    return 0


def _mollification_audit(args: argparse.Namespace) -> int:
    from .data import run_mollification_audit

    report = run_mollification_audit(args.config, args.anchor_lock, args.raw, args.output)
    print(
        f"Mollification adequacy {report['report_sha256']}: "
        f"decision={report['decision']}"
    )
    return 0


def _mollification_supplement_freeze(args: argparse.Namespace) -> int:
    from .data import freeze_mollification_supplement_anchors

    lock = freeze_mollification_supplement_anchors(
        args.config, args.anchor_lock, args.audit_report, args.output
    )
    print(
        "Mollification supplement anchor lock: "
        f"{lock['supplement_anchor_lock_sha256']} -> {args.output}"
    )
    return 0


def _mollification_collection_freeze(args: argparse.Namespace) -> int:
    from .data import freeze_mollification_supplement_collection

    lock = freeze_mollification_supplement_collection(
        args.config,
        args.anchor_lock,
        args.audit_report,
        args.supplement_anchor_lock,
        args.budget_plan,
        args.output,
    )
    print(
        "Mollification supplement collection lock: "
        f"{lock['collection_lock_sha256']} -> {args.output}"
    )
    return 0


def _collect_mollification_supplement(args: argparse.Namespace) -> int:
    from .data import collect_mollification_supplement

    manifest = collect_mollification_supplement(
        args.config,
        args.anchor_lock,
        args.audit_report,
        args.supplement_anchor_lock,
        args.budget_plan,
        args.collection_lock,
        args.shard_root,
        args.output,
    )
    print(
        f"Collected MollifiedReferenceCorpus {manifest['corpus_id']} "
        f"({manifest['totals']['state_count']} states) -> {args.output}"
    )
    return 0


def _collect_mollification_supplement_states(args: argparse.Namespace) -> int:
    from .data import collect_mollification_supplement_states

    entries = collect_mollification_supplement_states(
        args.config,
        args.anchor_lock,
        args.audit_report,
        args.supplement_anchor_lock,
        args.budget_plan,
        args.collection_lock,
        args.shard_root,
        args.state_id,
    )
    for entry in entries:
        print(
            f"Collected MollifiedReferenceShard {entry['dataset_id']} "
            f"(state={entry['state_id']}) -> {entry['uri']}"
        )
    return 0


def _validate_mollification_supplement(args: argparse.Namespace) -> int:
    from .data import validate_mollification_supplement

    manifest = validate_mollification_supplement(args.path)
    print(
        f"MollifiedReferenceCorpus OK: {manifest['corpus_id']} "
        f"({manifest['totals']['state_count']} states)"
    )
    return 0


def _write_mollification_data_entry(args: argparse.Namespace) -> int:
    from .data import write_mollification_training_data_entry

    entry = write_mollification_training_data_entry(
        args.config,
        args.anchor_lock,
        args.audit_report,
        args.output,
        supplement_manifest_path=args.supplement_manifest,
    )
    print(
        f"Mollification training data entry {entry['entry_id']} "
        f"(variant={entry['variant']}) -> {args.output}"
    )
    return 0


def _train_learning(args: argparse.Namespace) -> int:
    from .learning.training import TrainingConfig, train

    config = TrainingConfig.load(args.config)
    print(f"Starting training run at {args.run}", flush=True)
    manifest = train(
        args.data,
        args.run,
        config,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Completed training run {manifest['run_id']} at {args.run}")
    return 0


def _train_sampler(args: argparse.Namespace) -> int:
    from .learning.training import SamplerTrainingConfig, train_sampler

    config = SamplerTrainingConfig.load(args.config)
    print(f"Starting sampler training run at {args.run}", flush=True)
    manifest = train_sampler(
        args.data,
        args.run,
        config,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Completed sampler run {manifest['run_id']} at {args.run}")
    return 0


def _export_unified_compiled(args: argparse.Namespace) -> int:
    from .learning.unified_artifacts import export_unified_compiled_set

    manifest = export_unified_compiled_set(args.evaluator, args.sampler, args.output)
    print(f"Exported unified compiled set {manifest['compiled_set_id']} to {args.output}")
    return 0


def _export_nvidia_compiled(args: argparse.Namespace) -> int:
    from .learning.nvidia_neural_artifacts import export_nvidia_neural_compiled_set

    manifest = export_nvidia_neural_compiled_set(args.checkpoint, args.output)
    print(
        f"Exported NVIDIA original-scale compiled set "
        f"{manifest['compiled_set_id']} to {args.output}"
    )
    return 0


def _audit_convergence(args: argparse.Namespace) -> int:
    from .learning.evaluation.convergence import run_convergence_audit

    report = run_convergence_audit(
        args.data,
        args.run,
        args.protocol,
        args.output,
        device_name=args.device,
    )
    print(
        f"Validation-relative convergence {report['report_sha256']}: "
        f"passed={report['passed']}"
    )
    return 0


def _audit_sampler_convergence(args: argparse.Namespace) -> int:
    from .learning.evaluation.convergence import run_sampler_convergence_audit

    report = run_sampler_convergence_audit(
        args.data,
        args.run,
        args.protocol,
        args.output,
        device_name=args.device,
    )
    print(
        f"Sampler validation-relative convergence {report['report_sha256']}: "
        f"passed={report['passed']}"
    )
    return 0


def _audit_unified_sampler(args: argparse.Namespace) -> int:
    from .learning.evaluation.sampler_correctness import (
        run_unified_sampler_correctness,
    )

    report = run_unified_sampler_correctness(
        args.data,
        args.evaluator,
        args.sampler,
        args.output,
        device_name=args.device,
        progress=print,
    )
    print(
        f"Unified sampler correctness {report['report_sha256']}: "
        f"passed={report['passed']}"
    )
    return 0


def _audit_unified_offline_cook(args: argparse.Namespace) -> int:
    from .learning.evaluation.offline_cook import run_unified_offline_cook

    report = run_unified_offline_cook(
        args.data,
        args.checkpoint,
        args.config,
        args.output,
        device_name=args.device,
        progress=print,
    )
    print(f"Unified offline cook {report['cook_id']} completed at {args.output}")
    return 0


def _audit_unified_parity(args: argparse.Namespace) -> int:
    from .learning.evaluation.unified_parity import run_unified_checkpoint_parity

    report = run_unified_checkpoint_parity(
        args.data,
        args.evaluator,
        args.sampler,
        args.compiled,
        args.output,
        device_name=args.device,
        progress=print,
    )
    print(
        f"Unified checkpoint parity {report['report_sha256']}: "
        f"passed={report['passed']}"
    )
    return 0


def _select_unified_method(args: argparse.Namespace) -> int:
    from .learning.evaluation import build_unified_selection_from_artifacts

    manifest = build_unified_selection_from_artifacts(
        args.inputs,
        args.output,
        source_git_commit=args.source_git_commit,
    )
    print(
        f"Unified method selection {manifest['selection_id']}: "
        f"selected={manifest['selected_cell']}"
    )
    return 0


def _evaluate_learning(args: argparse.Namespace) -> int:
    from .learning.evaluation import evaluate_checkpoint

    result = evaluate_checkpoint(
        args.data,
        args.checkpoint,
        split=args.split,
        output_path=args.output,
        device_name=args.device,
        max_query_groups=args.max_query_groups,
    )
    relative = result["primary"]["directional_l1_by_state"]
    print(
        f"{args.split}: directional L1 state-median={relative['median']:.6f}, "
        f"state-p95={relative['p95']:.6f}; valid={result['valid']}"
    )
    return 0


def _list_learning_pipelines() -> int:
    from .learning.pipelines import pipeline_descriptors

    for descriptor in pipeline_descriptors():
        print(
            f"{descriptor.name}\t{descriptor.stage}\t{descriptor.scope}\t"
            f"{descriptor.sha256[:12]}"
        )
    return 0


def _compare_learning(args: argparse.Namespace) -> int:
    from .learning.evaluation import compare_quality_reports, write_comparison_report

    result = compare_quality_reports(
        args.baseline,
        args.candidate,
        iterations=args.iterations,
        seed=args.seed,
    )
    write_comparison_report(args.output, result)
    print(
        f"Compared {result['state_count']} matched states with "
        f"{result['iterations']} bootstrap samples: {result['report_sha256']}"
    )
    return 0


def _oracle_m3(args: argparse.Namespace) -> int:
    from .learning.direct_fit import run_response_dictionary_oracle

    result = run_response_dictionary_oracle(
        args.data,
        args.output,
        codebook_sizes=tuple(args.codebook_sizes),
        seed=args.seed,
        maximum_iterations=args.maximum_iterations,
    )
    units = result["units"]
    print(
        f"M3 response-space oracle {result['report_sha256']}: "
        f"{units['per_state']['unit_count']} state units, "
        f"{units['per_state_view']['unit_count']} state-view units"
    )
    return 0


def _benchmark_learning(args: argparse.Namespace) -> int:
    from .learning.evaluation import benchmark_checkpoint

    result = benchmark_checkpoint(
        args.data,
        args.checkpoint,
        args.output,
        device_name=args.device,
        packet_size=args.packet_size,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    single = result["single_query"].get(
        "device_execution_ms", result["single_query"]["synchronized_wall_ms"]
    )
    print(
        f"{result['pipeline']}: single-query median={single['median']:.6f} ms; "
        f"packet={result['coherent_packet']['median_microseconds_per_direction']:.3f} us/direction"
    )
    return 0


def _audit_p1_learning(args: argparse.Namespace) -> int:
    from .learning.evaluation.p1_audit import parse_checkpoint_specs, run_p1_audit

    result = run_p1_audit(
        args.data,
        parse_checkpoint_specs(args.checkpoint),
        args.output,
        roles=tuple(args.roles),
        device_name=args.device,
        bootstrap_iterations=args.iterations,
        bootstrap_seed=args.seed,
        progress=lambda message: print(message, flush=True),
    )
    print(f"P1 audit complete: {result['report_sha256']}")
    return 0


def _validate_bundle(path: Path) -> int:
    from .bundle import MethodBundle

    bundle = MethodBundle.open(path)
    print(
        f"MethodBundle OK: {bundle.manifest.method_id} "
        f"({bundle.manifest.backend_id}, {bundle.manifest.runtime_class})"
    )
    return 0


def _export_compiled_set_bundle(args: argparse.Namespace) -> int:
    from .bundle import export_compiled_set_bundle

    manifest = export_compiled_set_bundle(
        args.compiled_set,
        args.preview_material,
        args.parity,
        args.output,
        display_name=args.display_name,
        state_id=args.state_id,
    )
    print(f"Exported MethodBundle {manifest.method_id} to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ncls", description="NeuralShading 项目工具")
    commands = parser.add_subparsers(dest="command", required=True)
    material = commands.add_parser("material", help="MaterialProgram 工具")
    material_commands = material.add_subparsers(dest="material_command", required=True)

    validate = material_commands.add_parser("validate", help="验证并规范化检查 MaterialProgram")
    validate.add_argument("path", type=Path)

    normalize = material_commands.add_parser("normalize", help="写出节点顺序稳定的 MaterialProgram")
    normalize.add_argument("path", type=Path)
    normalize.add_argument("output", type=Path)

    pack = material_commands.add_parser("pack", help="写出固定布局 LayerStackIR")
    pack.add_argument("path", type=Path)
    pack.add_argument("output", type=Path)

    data = commands.add_parser("data", help="多源材质 reference 查询数据工具")
    data_commands = data.add_subparsers(dest="data_command", required=True)

    validate_data = data_commands.add_parser("validate", help="验证 HDF5 合同、查询结构和语义内容哈希")
    validate_data.add_argument("path", type=Path)
    validate_data.add_argument("--skip-hashes", action="store_true", help="仅用于快速诊断，不验证内容哈希")

    plan_corpus = data_commands.add_parser("plan-corpus", help="从 CorpusPlan 解析 state、密度与 shard 清单")
    plan_corpus.add_argument("--config", type=Path, required=True)
    plan_corpus.add_argument("--selection", type=Path)
    plan_corpus.add_argument("--shard-root", type=Path, required=True)
    plan_corpus.add_argument("--output", type=Path, required=True)
    collect_corpus = data_commands.add_parser("collect-corpus", help="按 CorpusPlan 采集或续采经过校验的 shard")
    collect_corpus.add_argument("--config", type=Path, required=True)
    collect_corpus.add_argument("--selection", type=Path)
    collect_corpus.add_argument("--shard-root", type=Path, required=True)
    collect_corpus.add_argument("--output", type=Path, required=True, help="写入 artifacts 的 corpus manifest")
    collect_state = data_commands.add_parser(
        "collect-state",
        help="按正式密度采集一个可读定位的 LayerStack state shard",
    )
    collect_state.add_argument("--config", type=Path, required=True)
    collect_state.add_argument("--structure-family", required=True)
    collect_state.add_argument("--state-index", type=int, required=True)
    collect_state.add_argument(
        "--role",
        choices=("train", "validation", "test", "adversarial_probe", "dense_slice"),
        required=True,
    )
    collect_state.add_argument("--output", type=Path, required=True)
    validate_corpus = data_commands.add_parser("validate-corpus", help="验证 corpus manifest 与全部 HDF5 hash")
    validate_corpus.add_argument("path", type=Path)
    audit_dense = data_commands.add_parser("audit-dense", help="审计 dense slice 峰邻域并给出 16,384 晋级清单")
    audit_dense.add_argument("path", type=Path)
    audit_dense.add_argument("--output", type=Path, required=True)
    mollification_freeze = data_commands.add_parser(
        "mollification-freeze",
        help="在 fresh reference 查询前锁定 directional mollification 协议与 anchor",
    )
    mollification_freeze.add_argument("--config", type=Path, required=True)
    mollification_freeze.add_argument("--output", type=Path, required=True)
    mollification_audit = data_commands.add_parser(
        "mollification-audit",
        help="执行或复用 frozen directional mollification matched audit",
    )
    mollification_audit.add_argument("--config", type=Path, required=True)
    mollification_audit.add_argument("--anchor-lock", type=Path, required=True)
    mollification_audit.add_argument("--raw", type=Path, required=True)
    mollification_audit.add_argument("--output", type=Path, required=True)
    supplement_freeze = data_commands.add_parser(
        "mollification-supplement-freeze",
        help="在 supplement reference 查询前锁定 30-state 的 8×64 train anchor",
    )
    supplement_freeze.add_argument("--config", type=Path, required=True)
    supplement_freeze.add_argument("--anchor-lock", type=Path, required=True)
    supplement_freeze.add_argument("--audit-report", type=Path, required=True)
    supplement_freeze.add_argument("--output", type=Path, required=True)
    collection_freeze = data_commands.add_parser(
        "mollification-collection-freeze",
        help="在 versioned supplement query 前锁定 budget plan 与 collection identity",
    )
    collection_freeze.add_argument("--config", type=Path, required=True)
    collection_freeze.add_argument("--anchor-lock", type=Path, required=True)
    collection_freeze.add_argument("--audit-report", type=Path, required=True)
    collection_freeze.add_argument("--supplement-anchor-lock", type=Path, required=True)
    collection_freeze.add_argument("--budget-plan", type=Path, required=True)
    collection_freeze.add_argument("--output", type=Path, required=True)
    collect_supplement = data_commands.add_parser(
        "collect-mollification-supplement",
        help="按 frozen 30-state anchor lock 采集 versioned mollification supplement",
    )
    collect_supplement.add_argument("--config", type=Path, required=True)
    collect_supplement.add_argument("--anchor-lock", type=Path, required=True)
    collect_supplement.add_argument("--audit-report", type=Path, required=True)
    collect_supplement.add_argument("--supplement-anchor-lock", type=Path, required=True)
    collect_supplement.add_argument("--budget-plan", type=Path, required=True)
    collect_supplement.add_argument("--collection-lock", type=Path, required=True)
    collect_supplement.add_argument("--shard-root", type=Path, required=True)
    collect_supplement.add_argument("--output", type=Path, required=True)
    collect_supplement_states = data_commands.add_parser(
        "collect-mollification-supplement-state",
        help="按 frozen collection lock 续采指定 state，且不提前发布 manifest",
    )
    collect_supplement_states.add_argument("--config", type=Path, required=True)
    collect_supplement_states.add_argument("--anchor-lock", type=Path, required=True)
    collect_supplement_states.add_argument("--audit-report", type=Path, required=True)
    collect_supplement_states.add_argument(
        "--supplement-anchor-lock", type=Path, required=True
    )
    collect_supplement_states.add_argument("--budget-plan", type=Path, required=True)
    collect_supplement_states.add_argument("--collection-lock", type=Path, required=True)
    collect_supplement_states.add_argument("--shard-root", type=Path, required=True)
    collect_supplement_states.add_argument("--state-id", action="append", required=True)
    validate_supplement = data_commands.add_parser(
        "validate-mollification",
        help="验证 mollified-reference-corpus 的 hash、measure、noise 与 provenance",
    )
    validate_supplement.add_argument("path", type=Path)
    mollification_data_entry = data_commands.add_parser(
        "mollification-data-entry",
        help="把 frozen adequacy 决定发布成 learning 可消费的唯一数据入口",
    )
    mollification_data_entry.add_argument("--config", type=Path, required=True)
    mollification_data_entry.add_argument("--anchor-lock", type=Path, required=True)
    mollification_data_entry.add_argument("--audit-report", type=Path, required=True)
    mollification_data_entry.add_argument("--supplement-manifest", type=Path)
    mollification_data_entry.add_argument("--output", type=Path, required=True)

    learn = commands.add_parser("learn", help="训练、validation 与 held-out test 工具")
    learn_commands = learn.add_subparsers(dest="learn_command", required=True)
    learn_commands.add_parser("list-pipelines", help="列出已注册的 learning pipeline")
    train_command = learn_commands.add_parser("train", help="训练明确命名的研究 baseline")
    train_command.add_argument("--data", type=Path, required=True, help="reference-corpus manifest 或单 shard")
    train_command.add_argument("--run", type=Path, required=True)
    train_command.add_argument("--config", type=Path, required=True)
    sampler_train = learn_commands.add_parser(
        "train-sampler", help="从冻结 evaluator checkpoint 训练 detached sampler head"
    )
    sampler_train.add_argument("--data", type=Path, required=True)
    sampler_train.add_argument("--run", type=Path, required=True)
    sampler_train.add_argument("--config", type=Path, required=True)
    export_unified = learn_commands.add_parser(
        "export-unified-compiled", help="导出03冻结的packed compiled-material set"
    )
    export_unified.add_argument("--evaluator", type=Path, required=True)
    export_unified.add_argument("--sampler", type=Path, required=True)
    export_unified.add_argument("--output", type=Path, required=True)
    export_nvidia = learn_commands.add_parser(
        "export-nvidia-compiled",
        help="从joint checkpoint导出NVIDIA原规模32 B/96 B私有ABI资产",
    )
    export_nvidia.add_argument("--checkpoint", type=Path, required=True)
    export_nvidia.add_argument("--output", type=Path, required=True)
    audit_convergence = learn_commands.add_parser(
        "audit-convergence",
        help="用validation相对改善、后期轨迹和checkpoint恢复审计多seed稳定收敛",
    )
    audit_convergence.add_argument("--data", type=Path, required=True)
    audit_convergence.add_argument("--run", type=Path, action="append", required=True)
    audit_convergence.add_argument("--protocol", type=Path, required=True)
    audit_convergence.add_argument("--output", type=Path, required=True)
    audit_convergence.add_argument("--device", default="cuda")
    audit_sampler_convergence = learn_commands.add_parser(
        "audit-sampler-convergence",
        help="审计frozen-evaluator sampler的多seed相对收敛与checkpoint恢复",
    )
    audit_sampler_convergence.add_argument("--data", type=Path, required=True)
    audit_sampler_convergence.add_argument(
        "--run", type=Path, action="append", required=True
    )
    audit_sampler_convergence.add_argument("--protocol", type=Path, required=True)
    audit_sampler_convergence.add_argument("--output", type=Path, required=True)
    audit_sampler_convergence.add_argument("--device", default="cuda")
    audit_unified_sampler = learn_commands.add_parser(
        "audit-unified-sampler",
        help="按冻结30x4协议审计03 learned sampler的PDF/null/histogram/MC无偏性",
    )
    audit_unified_sampler.add_argument("--data", type=Path, required=True)
    audit_unified_sampler.add_argument("--evaluator", type=Path, required=True)
    audit_unified_sampler.add_argument("--sampler", type=Path, required=True)
    audit_unified_sampler.add_argument("--output", type=Path, required=True)
    audit_unified_sampler.add_argument("--device", default="cuda")
    audit_unified_cook = learn_commands.add_parser(
        "audit-unified-offline-cook",
        help="冻结shared evaluator，仅从base-v5 train query重拟合latent并评测validation/dense",
    )
    audit_unified_cook.add_argument("--data", type=Path, required=True)
    audit_unified_cook.add_argument("--checkpoint", type=Path, required=True)
    audit_unified_cook.add_argument("--config", type=Path, required=True)
    audit_unified_cook.add_argument("--output", type=Path, required=True)
    audit_unified_cook.add_argument("--device", default="cuda")
    audit_unified_parity = learn_commands.add_parser(
        "audit-unified-parity",
        help="对03 checkpoint执行SlangPy/Falcor FP32与FP16-packed完整前向parity",
    )
    audit_unified_parity.add_argument("--data", type=Path, required=True)
    audit_unified_parity.add_argument("--evaluator", type=Path, required=True)
    audit_unified_parity.add_argument("--sampler", type=Path, required=True)
    audit_unified_parity.add_argument("--compiled", type=Path, required=True)
    audit_unified_parity.add_argument("--output", type=Path, required=True)
    audit_unified_parity.add_argument("--device", default="cuda")
    select_unified = learn_commands.add_parser(
        "select-unified-method",
        help="从正式四格实现、收敛、质量与成本证据生成03比较manifest",
    )
    select_unified.add_argument("--inputs", type=Path, required=True)
    select_unified.add_argument("--output", type=Path, required=True)
    select_unified.add_argument("--source-git-commit", required=True)

    evaluate = learn_commands.add_parser(
        "evaluate", help="显式评测 validation、held-out test 或 adversarial probe"
    )
    evaluate.add_argument("--data", type=Path, required=True, help="reference-corpus manifest 或单 shard")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument(
        "--split",
        choices=("train", "validation", "test", "adversarial_probe", "dense_slice"),
        required=True,
    )
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--device", type=str)
    evaluate.add_argument("--max-query-groups", type=int)

    compare = learn_commands.add_parser(
        "compare", help="在完全相同的 test state 上做 state-block 配对 bootstrap"
    )
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--iterations", type=int, default=1000)
    compare.add_argument("--seed", type=int, default=20260824)

    oracle_m3 = learn_commands.add_parser(
        "oracle-m3",
        help="在 canonical dense probes 上运行 M3 top-2 字典与 matched PCA 诊断",
    )
    oracle_m3.add_argument("--data", type=Path, required=True)
    oracle_m3.add_argument("--output", type=Path, required=True)
    oracle_m3.add_argument(
        "--codebook-sizes", type=int, nargs="+", default=(8, 16, 32, 64)
    )
    oracle_m3.add_argument("--seed", type=int, default=20260824)
    oracle_m3.add_argument("--maximum-iterations", type=int, default=30)

    benchmark = learn_commands.add_parser(
        "benchmark", help="测量 checkpoint 的单 query 延迟与 prepare 复用后的 coherent packet 成本"
    )
    benchmark.add_argument("--data", type=Path, required=True)
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", type=str)
    benchmark.add_argument("--packet-size", type=int, default=256)
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--iterations", type=int, default=50)

    audit_p1 = learn_commands.add_parser(
        "audit-p1",
        help="对冻结的 P1 checkpoint 运行独立死区、signed 能量、reference SE 与长尾诊断",
    )
    audit_p1.add_argument("--data", type=Path, required=True)
    audit_p1.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="可重复；格式为 label=path，省略 label 时从 run/checkpoint 路径推导",
    )
    audit_p1.add_argument("--output", type=Path, required=True)
    audit_p1.add_argument(
        "--roles",
        nargs="+",
        choices=("train", "validation", "test", "adversarial_probe", "dense_slice"),
        default=("train", "validation", "test", "adversarial_probe", "dense_slice"),
    )
    audit_p1.add_argument("--device", type=str)
    audit_p1.add_argument("--iterations", type=int, default=10000)
    audit_p1.add_argument("--seed", type=int, default=20260825)

    bundle = commands.add_parser("bundle", help="MethodBundle 导出与验证")
    bundle_commands = bundle.add_subparsers(dest="bundle_command", required=True)
    validate_bundle = bundle_commands.add_parser("validate", help="验证 manifest 与全部内容哈希")
    validate_bundle.add_argument("path", type=Path)
    export_compiled = bundle_commands.add_parser(
        "export-compiled-set",
        help="把带标准 runtime adapter 的 compiled set 导出为 MethodBundle",
    )
    export_compiled.add_argument("--compiled-set", type=Path, required=True)
    export_compiled.add_argument("--preview-material", type=Path, required=True)
    export_compiled.add_argument("--parity", type=Path, required=True)
    export_compiled.add_argument("--output", type=Path, required=True)
    export_compiled.add_argument("--display-name", required=True)
    export_compiled.add_argument("--state-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "material" and args.material_command == "validate":
        return _validate_material(args.path)
    if args.command == "material" and args.material_command == "normalize":
        return _normalize_material(args.path, args.output)
    if args.command == "material" and args.material_command == "pack":
        return _pack_material(args.path, args.output)
    if args.command == "data" and args.data_command == "validate":
        return _validate_dataset(args.path, skip_hashes=args.skip_hashes)
    if args.command == "data" and args.data_command == "plan-corpus":
        return _plan_corpus(args)
    if args.command == "data" and args.data_command == "collect-corpus":
        return _collect_corpus(args)
    if args.command == "data" and args.data_command == "collect-state":
        return _collect_state(args)
    if args.command == "data" and args.data_command == "validate-corpus":
        return _validate_corpus(args)
    if args.command == "data" and args.data_command == "audit-dense":
        return _audit_dense(args)
    if args.command == "data" and args.data_command == "mollification-freeze":
        return _mollification_freeze(args)
    if args.command == "data" and args.data_command == "mollification-audit":
        return _mollification_audit(args)
    if args.command == "data" and args.data_command == "mollification-supplement-freeze":
        return _mollification_supplement_freeze(args)
    if args.command == "data" and args.data_command == "mollification-collection-freeze":
        return _mollification_collection_freeze(args)
    if args.command == "data" and args.data_command == "collect-mollification-supplement":
        return _collect_mollification_supplement(args)
    if args.command == "data" and args.data_command == "collect-mollification-supplement-state":
        return _collect_mollification_supplement_states(args)
    if args.command == "data" and args.data_command == "validate-mollification":
        return _validate_mollification_supplement(args)
    if args.command == "data" and args.data_command == "mollification-data-entry":
        return _write_mollification_data_entry(args)
    if args.command == "learn" and args.learn_command == "train":
        return _train_learning(args)
    if args.command == "learn" and args.learn_command == "train-sampler":
        return _train_sampler(args)
    if args.command == "learn" and args.learn_command == "export-unified-compiled":
        return _export_unified_compiled(args)
    if args.command == "learn" and args.learn_command == "export-nvidia-compiled":
        return _export_nvidia_compiled(args)
    if args.command == "learn" and args.learn_command == "audit-convergence":
        return _audit_convergence(args)
    if args.command == "learn" and args.learn_command == "audit-sampler-convergence":
        return _audit_sampler_convergence(args)
    if args.command == "learn" and args.learn_command == "audit-unified-sampler":
        return _audit_unified_sampler(args)
    if args.command == "learn" and args.learn_command == "audit-unified-offline-cook":
        return _audit_unified_offline_cook(args)
    if args.command == "learn" and args.learn_command == "audit-unified-parity":
        return _audit_unified_parity(args)
    if args.command == "learn" and args.learn_command == "select-unified-method":
        return _select_unified_method(args)
    if args.command == "learn" and args.learn_command == "list-pipelines":
        return _list_learning_pipelines()
    if args.command == "learn" and args.learn_command == "evaluate":
        return _evaluate_learning(args)
    if args.command == "learn" and args.learn_command == "compare":
        return _compare_learning(args)
    if args.command == "learn" and args.learn_command == "oracle-m3":
        return _oracle_m3(args)
    if args.command == "learn" and args.learn_command == "benchmark":
        return _benchmark_learning(args)
    if args.command == "learn" and args.learn_command == "audit-p1":
        return _audit_p1_learning(args)
    if args.command == "bundle" and args.bundle_command == "validate":
        return _validate_bundle(args.path)
    if args.command == "bundle" and args.bundle_command == "export-compiled-set":
        return _export_compiled_set_bundle(args)
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
