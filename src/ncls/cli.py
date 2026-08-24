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
    from .data import CorpusPlan, plan_layer_stack_corpus

    plan = CorpusPlan.load(args.config)
    manifest = plan_layer_stack_corpus(plan, args.shard_root)
    manifest.write(args.output)
    print(f"Planned {len(manifest.shards)} shards for {manifest.name}: {manifest.corpus_id}")
    return 0


def _collect_corpus(args: argparse.Namespace) -> int:
    from .data import CorpusPlan, collect_layer_stack_corpus

    plan = CorpusPlan.load(args.config)
    manifest = collect_layer_stack_corpus(plan, args.shard_root, args.output)
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


def _train_learning(args: argparse.Namespace) -> int:
    from .learning.training import TrainingConfig, train

    config = TrainingConfig.load(args.config)
    manifest = train(args.data, args.run, config)
    print(f"Completed training run {manifest['run_id']} at {args.run}")
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


def _validate_bundle(path: Path) -> int:
    from .bundle import MethodBundle

    bundle = MethodBundle.open(path)
    print(
        f"MethodBundle OK: {bundle.manifest.method_id} "
        f"({bundle.manifest.backend_id}, {bundle.manifest.runtime_class})"
    )
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
    plan_corpus.add_argument("--shard-root", type=Path, required=True)
    plan_corpus.add_argument("--output", type=Path, required=True)
    collect_corpus = data_commands.add_parser("collect-corpus", help="按 CorpusPlan 采集或续采经过校验的 shard")
    collect_corpus.add_argument("--config", type=Path, required=True)
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

    learn = commands.add_parser("learn", help="训练、validation 与 held-out test 工具")
    learn_commands = learn.add_subparsers(dest="learn_command", required=True)
    learn_commands.add_parser("list-pipelines", help="列出已注册的 learning pipeline")
    train_command = learn_commands.add_parser("train", help="训练明确命名的研究 baseline")
    train_command.add_argument("--data", type=Path, required=True, help="reference-corpus manifest 或单 shard")
    train_command.add_argument("--run", type=Path, required=True)
    train_command.add_argument("--config", type=Path, required=True)

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

    bundle = commands.add_parser("bundle", help="MethodBundle 验证")
    bundle_commands = bundle.add_subparsers(dest="bundle_command", required=True)
    validate_bundle = bundle_commands.add_parser("validate", help="验证 manifest 与全部内容哈希")
    validate_bundle.add_argument("path", type=Path)
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
    if args.command == "learn" and args.learn_command == "train":
        return _train_learning(args)
    if args.command == "learn" and args.learn_command == "list-pipelines":
        return _list_learning_pipelines()
    if args.command == "learn" and args.learn_command == "evaluate":
        return _evaluate_learning(args)
    if args.command == "learn" and args.learn_command == "compare":
        return _compare_learning(args)
    if args.command == "bundle" and args.bundle_command == "validate":
        return _validate_bundle(args.path)
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
