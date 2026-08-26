from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ncls.bundle import ScatteringPackage, write_scattering_package
from ncls.core.material import (
    BINARY_SIZE,
    MaterialProgram,
    canonicalize_layer_stack,
    pack_layer_stack,
    physical_material_hash,
    validate_material_program,
)
from ncls.data.batch_sources import LiveReferenceBatchSource, OfflineBatchSource
from ncls.learning.methods import get_method, method_descriptors
from ncls.learning.training import TrainingConfig, TrainingRunner, load_checkpoint, save_checkpoint
from ncls.source_materials.families.layer_stack import snapshot_from_layer_stack


def _load_program(path: Path) -> MaterialProgram:
    return MaterialProgram.from_json(path.read_text(encoding="utf-8"))


def _material_validate(path: Path) -> int:
    program = _load_program(path)
    validate_material_program(program)
    stack = canonicalize_layer_stack(program)
    print(f"MaterialProgram OK: {physical_material_hash(program)}")
    print(f"LayerStackIR: {len(stack.interfaces)} interfaces, {len(stack.media)} media, {BINARY_SIZE} bytes")
    return 0


def _material_normalize(path: Path, output: Path) -> int:
    program = _load_program(path)
    normalized = MaterialProgram(
        tuple(sorted(program.nodes, key=lambda node: node.node_id)), program.outputs,
        tuple(sorted(program.resources, key=lambda resource: resource.resource_id)),
        program.metadata, program.color_model, program.schema_name, program.schema_version,
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


def _data_validate(path: Path, skip_hashes: bool) -> int:
    from ncls.data import validate_reference_dataset

    dataset = validate_reference_dataset(path, verify_hashes=not skip_hashes)
    try:
        print(f"ReferenceShard OK: {dataset.manifest.dataset_id}")
    finally:
        dataset.close()
    return 0


def _data_plan(args: argparse.Namespace) -> int:
    from ncls.data import CorpusPlan, CorpusSelection, plan_layer_stack_corpus

    plan = CorpusPlan.load(args.config)
    selection = CorpusSelection.load(args.selection) if args.selection else None
    manifest = plan_layer_stack_corpus(plan, args.shard_root, selection)
    manifest.write(args.output)
    print(f"Planned {len(manifest.shards)} shards: {manifest.corpus_id}")
    return 0


def _data_collect(args: argparse.Namespace) -> int:
    from ncls.data import CorpusPlan, CorpusSelection, collect_layer_stack_corpus

    plan = CorpusPlan.load(args.config)
    selection = CorpusSelection.load(args.selection) if args.selection else None
    manifest = collect_layer_stack_corpus(plan, args.shard_root, args.output, selection)
    print(f"Collected ReferenceCorpus {manifest.corpus_id}: {args.output}")
    return 0


def _data_validate_corpus(path: Path) -> int:
    from ncls.data import validate_reference_corpus

    manifest = validate_reference_corpus(path)
    print(f"ReferenceCorpus OK: {manifest.corpus_id} ({len(manifest.shards)} shards)")
    return 0


def _data_audit_dense(path: Path, output: Path) -> int:
    from ncls.data import audit_dense_slice_resolution

    report = audit_dense_slice_resolution(path, output)
    print(f"Dense audit {report['report_sha256']}: {len(report['promote_state_ids'])} promoted states")
    return 0


def _batch_source(config: TrainingConfig):
    options = dict(config.batch_source["options"])
    if config.batch_source["kind"] == "offline":
        from ncls.data.stores import open_reference_store

        required = {"path", "partition_policy", "lifecycle_role"}
        if set(options) != required:
            raise ValueError(f"offline batch source options must be exactly {sorted(required)}")
        store = open_reference_store(Path(str(options["path"])))
        candidates = store.partition_indices(str(options["partition_policy"]), str(options["lifecycle_role"]))
        return OfflineBatchSource(store, candidates, device=config.device, seed=config.seed)
    required = {"material_programs", "light_count", "samples_per_replica", "max_depth"}
    if set(options) != required:
        raise ValueError(f"live batch source options must be exactly {sorted(required)}")
    programs = tuple(_load_program(Path(str(path))) for path in options["material_programs"])
    stacks = tuple(canonicalize_layer_stack(program) for program in programs)
    snapshots = tuple(snapshot_from_layer_stack(stack, metadata=program.metadata) for stack, program in zip(stacks, programs, strict=True))
    return LiveReferenceBatchSource(
        stacks, tuple(snapshot.snapshot_id for snapshot in snapshots),
        light_count=int(options["light_count"]), samples_per_replica=int(options["samples_per_replica"]),
        max_depth=int(options["max_depth"]), max_batch_size=config.batch_size,
        seed=config.seed, device=config.device,
    )


def _learn_list() -> int:
    for descriptor in method_descriptors():
        print(f"{descriptor.method_key}\t{descriptor.display_name}\t{descriptor.descriptor_sha256}")
    return 0


def _learn_train(config_path: Path, output: Path) -> int:
    config = TrainingConfig.load(config_path)
    definition = get_method(config.method_key)
    source = _batch_source(config)
    try:
        result = TrainingRunner(definition, source, config).run()
        digest = save_checkpoint(output, result.checkpoint)
    finally:
        source.close()
    print(f"TrainingCheckpoint@2 {digest}: {output}")
    return 0


def _learn_evaluate(config_path: Path, checkpoint_path: Path, batches: int) -> int:
    config = TrainingConfig.load(config_path)
    definition = get_method(config.method_key)
    checkpoint = load_checkpoint(checkpoint_path, descriptor=definition.descriptor, map_location=config.device)
    source = _batch_source(config)
    losses = []
    try:
        model = definition.create_trainable(config.model_context).to(source.device)
        definition.restore_training_state(model, checkpoint.model_state)
        definition.configure_phase(model, "evaluator")
        model.eval()
        with torch.no_grad():
            for _ in range(batches):
                batch = source.next_batch(config.batch_size)
                try:
                    loss, _ = definition.training_objective(model, batch, "evaluator")
                    losses.append(float(loss))
                finally:
                    batch.release()
    finally:
        source.close()
    print(f"Evaluation batches={batches} mean_loss={sum(losses) / len(losses):.9g}")
    return 0


def _learn_export(checkpoint_path: Path, source_path: Path, output: Path) -> int:
    checkpoint = load_checkpoint(checkpoint_path)
    definition = get_method(checkpoint.method_key)
    checkpoint.validate_method(definition.descriptor)
    program = _load_program(source_path)
    snapshot = snapshot_from_layer_stack(canonicalize_layer_stack(program), metadata=program.metadata)
    runtime = definition.compile_runtime(checkpoint.to_payload())
    material = definition.compile_material(snapshot, checkpoint.to_payload())
    manifest = write_scattering_package(
        output,
        program_kind="method", program_key=definition.descriptor.method_key,
        program_version=definition.descriptor.version,
        program_descriptor_sha256=definition.descriptor.descriptor_sha256,
        runtime_abi=definition.descriptor.runtime_abi,
        source=snapshot, runtime=runtime, material=material,
        validation={"status": "unverified", "checkpoint_step": checkpoint.step},
        provenance={"checkpoint_sha256": checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256").read_text(encoding="ascii").strip()},
    )
    print(f"ScatteringPackage@1 {manifest.package_id}: {output}")
    return 0


def _package_validate(path: Path) -> int:
    package = ScatteringPackage.open(path)
    binding = package.create_binding()
    print(f"ScatteringPackage OK: {package.manifest.package_id}")
    print(f"runtime={binding.program_runtime_id} material={binding.material_asset_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ncls", description="NeuralShading 统一 pipeline 工具")
    commands = parser.add_subparsers(dest="command", required=True)

    material = commands.add_parser("material", help="MaterialProgram 工具")
    material_commands = material.add_subparsers(dest="material_command", required=True)
    validate = material_commands.add_parser("validate", help="验证 MaterialProgram")
    validate.add_argument("path", type=Path)
    normalize = material_commands.add_parser("normalize", help="规范化 MaterialProgram")
    normalize.add_argument("path", type=Path); normalize.add_argument("output", type=Path)
    pack = material_commands.add_parser("pack", help="写出 LayerStackIR packet")
    pack.add_argument("path", type=Path); pack.add_argument("output", type=Path)

    data = commands.add_parser("data", help="reference query/corpus 工具")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    validate_data = data_commands.add_parser("validate", help="验证 reference shard")
    validate_data.add_argument("path", type=Path); validate_data.add_argument("--skip-hashes", action="store_true")
    plan = data_commands.add_parser("plan", help="从 CorpusPlan 生成 shard 计划")
    plan.add_argument("config", type=Path); plan.add_argument("shard_root", type=Path); plan.add_argument("output", type=Path); plan.add_argument("--selection", type=Path)
    collect = data_commands.add_parser("collect", help="按 CorpusPlan 采集 corpus")
    collect.add_argument("config", type=Path); collect.add_argument("shard_root", type=Path); collect.add_argument("output", type=Path); collect.add_argument("--selection", type=Path)
    validate_corpus = data_commands.add_parser("validate-corpus", help="验证 corpus")
    validate_corpus.add_argument("path", type=Path)
    audit = data_commands.add_parser("audit-dense", help="审计 dense slice")
    audit.add_argument("path", type=Path); audit.add_argument("output", type=Path)

    learn = commands.add_parser("learn", help="统一方法训练、评测和导出")
    learn_commands = learn.add_subparsers(dest="learn_command", required=True)
    learn_commands.add_parser("list", help="列出产品 MethodDefinition")
    train = learn_commands.add_parser("train", help="运行统一 TrainingRunner")
    train.add_argument("config", type=Path); train.add_argument("output", type=Path)
    evaluate = learn_commands.add_parser("evaluate", help="评测 TrainingCheckpoint@2")
    evaluate.add_argument("config", type=Path); evaluate.add_argument("checkpoint", type=Path); evaluate.add_argument("--batches", type=int, default=8)
    export = learn_commands.add_parser("export", help="把 checkpoint 编译为 ScatteringPackage@1")
    export.add_argument("checkpoint", type=Path); export.add_argument("source", type=Path); export.add_argument("output", type=Path)

    package = commands.add_parser("package", help="ScatteringPackage@1 工具")
    package_commands = package.add_subparsers(dest="package_command", required=True)
    validate_package = package_commands.add_parser("validate", help="验证 schema、URI 与内容 hash")
    validate_package.add_argument("path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dispatch: dict[tuple[str, str], Any] = {
        ("material", "validate"): lambda: _material_validate(args.path),
        ("material", "normalize"): lambda: _material_normalize(args.path, args.output),
        ("material", "pack"): lambda: _material_pack(args.path, args.output),
        ("data", "validate"): lambda: _data_validate(args.path, args.skip_hashes),
        ("data", "plan"): lambda: _data_plan(args), ("data", "collect"): lambda: _data_collect(args),
        ("data", "validate-corpus"): lambda: _data_validate_corpus(args.path),
        ("data", "audit-dense"): lambda: _data_audit_dense(args.path, args.output),
        ("learn", "list"): _learn_list,
        ("learn", "train"): lambda: _learn_train(args.config, args.output),
        ("learn", "evaluate"): lambda: _learn_evaluate(args.config, args.checkpoint, args.batches),
        ("learn", "export"): lambda: _learn_export(args.checkpoint, args.source, args.output),
        ("package", "validate"): lambda: _package_validate(args.path),
    }
    return int(dispatch[(args.command, getattr(args, f"{args.command}_command"))]())


__all__ = ["build_parser", "main"]
