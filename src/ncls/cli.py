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
    print(f"ReferenceDataset OK: {dataset.manifest.dataset_id}")
    print(
        f"{counts['state_count']} states, {counts['query_group_count']} query groups, "
        f"{counts['direction_count']} directions per group"
    )
    dataset.close()
    return 0


def _generate_dataset(args: argparse.Namespace) -> int:
    from .data import CollectionConfig
    from .data.generator import generate_reference_dataset
    from .data.providers import LayerStackProviderConfig
    from .paths import DATA_ROOT, REFERENCE_RESPONSE_ROOT

    output = args.output.resolve()
    if DATA_ROOT.resolve() in output.parents and REFERENCE_RESPONSE_ROOT.resolve() not in output.parents:
        raise ValueError("data/ only accepts ReferenceDataset HDF5 under data/reference-responses/")

    collection = CollectionConfig(
        view_count=args.views,
        validation_view_count=args.validation_views,
        test_view_count=args.test_views,
        adversarial_view_count=args.adversarial_views,
        light_count=args.lights,
        spatial_sample_count=args.spatial_samples,
        footprint_width=args.footprint_width,
        seed=args.seed,
        query_profile_id=args.query_profile,
    )
    layer_stack = LayerStackProviderConfig(
        family_count=args.families,
        local_state_count=args.local_states,
        samples_per_replica=args.samples_per_replica,
        query_group_batch=args.query_group_batch,
        max_depth=args.max_depth,
        adaptive=args.adaptive,
        batch_samples=args.batch_samples,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        relative_standard_error=args.relative_standard_error,
        state_profile_id=args.layer_stack_state_profile,
    )
    manifest = generate_reference_dataset(
        output,
        args.provider,
        collection,
        material_ids=args.material_id,
        layer_stack=layer_stack,
    )
    print(f"Wrote ReferenceDataset {manifest.dataset_id} to {output}")
    return 0


def _train_learning(args: argparse.Namespace) -> int:
    from .learning.training import TrainingConfig, train

    config = TrainingConfig.load(args.config) if args.config else TrainingConfig(
        model_parameters={"width": args.width},
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_interval=args.validation_interval,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        device=args.device,
    )
    manifest = train(args.dataset, args.run, config)
    print(f"Completed training run {manifest['run_id']} at {args.run}")
    return 0


def _evaluate_learning(args: argparse.Namespace) -> int:
    from .learning.evaluation import evaluate_checkpoint

    result = evaluate_checkpoint(
        args.dataset,
        args.checkpoint,
        split=args.split,
        output_path=args.output,
        device_name=args.device,
        max_query_groups=args.max_query_groups,
    )
    relative = result["metrics"]["relative_l1"]
    print(
        f"{args.split}: relative-L1 median={relative['median']:.6f}, "
        f"p90={relative['p90']:.6f}"
    )
    return 0


def _audit_learning(args: argparse.Namespace) -> int:
    from .learning.audit import audit_supervision

    result = audit_supervision(
        args.dataset,
        args.output,
        verify_hashes=not args.skip_hashes,
        max_distribution_query_groups=args.max_query_groups,
        gate_path=args.gate,
    )
    split = result["split_audit"]
    print(
        f"Supervision audit {result['audit_sha256']}: "
        f"split-group leaks={split['split_group_id']['leak_count']}, "
        f"source leaks={split['source_sha256']['leak_count']}"
    )
    return 0


def _list_learning_pipelines() -> int:
    from .learning.pipelines import pipeline_descriptors

    for descriptor in pipeline_descriptors():
        print(
            f"{descriptor.pipeline_id}\t{descriptor.candidate_id}\t"
            f"{descriptor.research_role}\t{descriptor.scope}"
        )
    return 0


def _direct_fit(args: argparse.Namespace) -> int:
    from .learning.direct_fit import DirectFitConfig, run_direct_fit

    result = run_direct_fit(
        args.dataset,
        args.output,
        split=args.split,
        config=DirectFitConfig(
            family=args.family,
            lobe_count=args.lobes,
            fit_batch=args.fit_batch,
            steps=args.steps,
            restarts=args.restarts,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
        ),
        max_query_groups=args.max_query_groups,
    )
    relative = result["relative_l1"]
    print(
        f"Representation ceiling: relative-L1 median={relative['median']:.6f}, "
        f"p90={relative['p90']:.6f}"
    )
    return 0


def _export_bundle(args: argparse.Namespace) -> int:
    from .bundle import export_legacy_ltc_k2_checkpoint

    bundle = export_legacy_ltc_k2_checkpoint(
        args.checkpoint,
        args.output,
        display_name=args.display_name,
        source_run_manifest=args.run_manifest,
    )
    print(f"Wrote MethodBundle {bundle.manifest.method_id} to {args.output}")
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

    generate = data_commands.add_parser("collect-reference", help="通过统一 Falcor provider 采集 HDF5 reference 数据")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument(
        "--provider",
        action="append",
        choices=("layer-stack", "merl", "openpbr", "materialx", "all"),
        required=True,
        help="可重复指定；all 导出当前全部正式 reference provider",
    )
    generate.add_argument("--material-id", action="append", default=[], help="单 provider 下只导出指定资产；可重复")
    generate.add_argument("--families", type=int, default=8)
    generate.add_argument("--local-states", type=int, default=4)
    generate.add_argument(
        "--layer-stack-state-profile",
        choices=("ncls.layer-stack-research-prior@1", "ncls.e0-layer-stack-boundary@1"),
        default="ncls.layer-stack-research-prior@1",
        help="LayerStack provider-local 的版本化状态分布",
    )
    generate.add_argument("--views", type=int, default=4)
    generate.add_argument("--validation-views", type=int, default=0)
    generate.add_argument("--test-views", type=int, default=0)
    generate.add_argument("--adversarial-views", type=int, default=0)
    generate.add_argument("--lights", type=int, default=128)
    generate.add_argument("--spatial-samples", type=int, default=1)
    generate.add_argument("--footprint-width", type=float, default=1.0 / 4096.0)
    generate.add_argument("--samples-per-replica", type=int, default=64)
    generate.add_argument("--query-group-batch", type=int, default=64)
    generate.add_argument("--seed", type=int, default=20260822)
    generate.add_argument(
        "--query-profile",
        choices=("ncls.uniform-split-independent@1", "ncls.e0-peak-grazing-mixture@2"),
        default="ncls.uniform-split-independent@1",
    )
    generate.add_argument("--max-depth", type=int, default=64)
    generate.add_argument("--adaptive", action="store_true")
    generate.add_argument("--batch-samples", type=int, default=256)
    generate.add_argument("--min-samples", type=int, default=512)
    generate.add_argument("--max-samples", type=int, default=16384)
    generate.add_argument("--relative-standard-error", type=float, default=0.03)

    learn = commands.add_parser("learn", help="训练、validation 与 held-out test 工具")
    learn_commands = learn.add_subparsers(dest="learn_command", required=True)
    audit = learn_commands.add_parser("audit", help="只读审计监督覆盖、split、reference noise 与 train-only transform 统计")
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--max-query-groups", type=int, default=8192)
    audit.add_argument("--gate", type=Path, help="可选的冻结 E0 gate 配置；结果写入 gate_result.json")
    audit.add_argument("--skip-hashes", action="store_true", help="仅用于局部诊断，不验证 HDF5 内容哈希")
    learn_commands.add_parser("list-pipelines", help="列出已注册且带版本的 learning pipeline")
    train_command = learn_commands.add_parser("train", help="训练明确命名的研究 baseline")
    train_command.add_argument("--dataset", type=Path, required=True)
    train_command.add_argument("--run", type=Path, required=True)
    train_command.add_argument("--config", type=Path, help="完整 TrainingConfig JSON；指定后忽略下列超参数")
    train_command.add_argument("--width", type=int, default=64)
    train_command.add_argument("--steps", type=int, default=10000)
    train_command.add_argument("--batch-size", type=int, default=256)
    train_command.add_argument("--learning-rate", type=float, default=3e-4)
    train_command.add_argument("--validation-interval", type=int, default=250)
    train_command.add_argument("--checkpoint-interval", type=int, default=250)
    train_command.add_argument("--seed", type=int, default=20260822)
    train_command.add_argument("--device", type=str)

    evaluate = learn_commands.add_parser("evaluate", help="显式评测 validation 或 held-out test")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "validation", "test"), required=True)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--device", type=str)
    evaluate.add_argument("--max-query-groups", type=int)

    direct_fit = learn_commands.add_parser("direct-fit", help="逐 query group 优化参数，测量方向切片上界")
    direct_fit.add_argument("--dataset", type=Path, required=True)
    direct_fit.add_argument("--output", type=Path, required=True)
    direct_fit.add_argument("--split", choices=("train", "validation", "test"), required=True)
    direct_fit.add_argument("--family", choices=("ggx", "ltc", "sg"), default="ltc")
    direct_fit.add_argument("--lobes", type=int, default=2)
    direct_fit.add_argument("--fit-batch", type=int, default=256)
    direct_fit.add_argument("--steps", type=int, default=800)
    direct_fit.add_argument("--restarts", type=int, default=3)
    direct_fit.add_argument("--learning-rate", type=float, default=0.03)
    direct_fit.add_argument("--seed", type=int, default=20260822)
    direct_fit.add_argument("--device", type=str)
    direct_fit.add_argument("--max-query-groups", type=int)

    bundle = commands.add_parser("bundle", help="MethodBundle 导出与验证")
    bundle_commands = bundle.add_subparsers(dest="bundle_command", required=True)
    export = bundle_commands.add_parser("export-legacy-ltc-k2", help="从显式 checkpoint 导出 realtime bundle")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--run-manifest", type=Path)
    export.add_argument("--display-name", default="Legacy LTC K2 P1")
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
    if args.command == "data" and args.data_command == "collect-reference":
        return _generate_dataset(args)
    if args.command == "learn" and args.learn_command == "train":
        return _train_learning(args)
    if args.command == "learn" and args.learn_command == "audit":
        return _audit_learning(args)
    if args.command == "learn" and args.learn_command == "list-pipelines":
        return _list_learning_pipelines()
    if args.command == "learn" and args.learn_command == "evaluate":
        return _evaluate_learning(args)
    if args.command == "learn" and args.learn_command == "direct-fit":
        return _direct_fit(args)
    if args.command == "bundle" and args.bundle_command == "export-legacy-ltc-k2":
        return _export_bundle(args)
    if args.command == "bundle" and args.bundle_command == "validate":
        return _validate_bundle(args.path)
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
