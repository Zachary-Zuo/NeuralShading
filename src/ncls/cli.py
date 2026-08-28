from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
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
from ncls.core.source import create_source_family
from ncls.learning.batches import TrainingRouteRequest
from ncls.learning.methods import get_method, method_descriptors
from ncls.learning.producer import OnlineTrainingProducer
from ncls.learning.training import (
    TrainingConfig,
    TrainingRunner,
    load_checkpoint,
    save_checkpoint,
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


def _learn_list() -> int:
    for descriptor in method_descriptors():
        print(
            f"{descriptor.method_key}\t{descriptor.display_name}\t"
            f"{descriptor.descriptor_sha256}"
        )
    return 0


def _learn_train(config_path: Path, output: Path, resume_path: Path | None) -> int:
    config = TrainingConfig.load(config_path)
    definition = get_method(config.method_key)
    producer = OnlineTrainingProducer(definition, config)
    metrics_path = output.with_name(f"{output.stem}.metrics.jsonl")
    summary_path = output.with_name(f"{output.stem}.summary.json")
    metric_count = 0
    started = time.perf_counter()
    try:
        resume = (
            load_checkpoint(
                resume_path,
                descriptor=definition.descriptor,
                map_location=config.device,
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
                if float(value.get("step", -1)) <= resume.step:
                    retained_metric_lines.append(
                        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    )
        output.parent.mkdir(parents=True, exist_ok=True)
        metric_stream = metrics_path.open("w", encoding="utf-8", newline="\n")
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
            metric_stream.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            metric_count += 1
            if metric_count % 256 == 0:
                metric_stream.flush()

        def save_periodic(checkpoint) -> None:
            metric_stream.flush()
            path = output.with_name(
                f"{output.stem}.step{checkpoint.step:08d}{output.suffix}"
            )
            save_checkpoint(path, checkpoint)

        result = TrainingRunner(
            definition,
            producer,
            config,
            checkpoint_callback=save_periodic,
            metric_callback=record_metric,
        ).run(resume=resume)
        digest = save_checkpoint(output, result.checkpoint)
        metric_stream.flush()
        metric_stream.close()
        summary_path.write_text(
            json.dumps(
                {
                    "schema_name": "ncls.training-run-summary",
                    "schema_version": 2,
                    "training_config_sha256": config.sha256,
                    "checkpoint_sha256": digest,
                    "checkpoint": output.name,
                    "metrics": metrics_path.name,
                    "metric_records": metric_count,
                    "final_step": result.checkpoint.step,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        if "metric_stream" in locals() and not metric_stream.closed:
            metric_stream.close()
        producer.close()
    print(f"TrainingCheckpoint@3 {digest}: {output}")
    return 0


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
        or checkpoint.query_stream_identity != producer.query_stream_identity
        or checkpoint.source_snapshot_ids != producer.source_snapshot_ids
    ):
        producer.close()
        raise ValueError("evaluation producer identity disagrees with the checkpoint")
    losses: list[float] = []
    try:
        model = definition.create_trainable(config.model_context).to(producer.device)
        definition.restore_training_state(model, checkpoint.model_state)
        definition.configure_lifecycle(model, checkpoint.lifecycle_state)
        model.eval()
        with torch.no_grad():
            for index in range(batches):
                route_batches = {}
                try:
                    for route in config.routes:
                        request = TrainingRouteRequest(
                            f"evaluation:{route.name}",
                            route.kind,
                            route.batch_size,
                            route.direction_count,
                            checkpoint.step,
                            config.seed + route.seed_offset + index,
                            {
                                **dict(route.options),
                                "filtering": dict(config.filtering),
                                "mollification": dict(config.mollification),
                                "validation": True,
                            },
                        )
                        route_batches[route.name] = producer.next_batch(request)
                    loss, _ = definition.training_objective(
                        model, route_batches, checkpoint.lifecycle_state
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
    runtime = definition.compile_runtime(payload)
    material = definition.compile_material(snapshot, payload)
    validation = dict(definition.package_validation(snapshot, payload))
    validation["checkpoint_step"] = checkpoint.step
    manifest = write_scattering_package(
        output,
        program_kind="method",
        program_key=definition.descriptor.method_key,
        program_version=definition.descriptor.version,
        program_descriptor_sha256=definition.descriptor.descriptor_sha256,
        runtime_abi=definition.descriptor.runtime_abi,
        source=snapshot,
        runtime=runtime,
        material=material,
        validation=validation,
        provenance={
            "checkpoint_sha256": checkpoint_path.with_suffix(
                checkpoint_path.suffix + ".sha256"
            ).read_text(encoding="ascii").strip()
        },
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
    evaluate = learn_commands.add_parser(
        "evaluate", help="评测 TrainingCheckpoint@3"
    )
    evaluate.add_argument("config", type=Path)
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--batches", type=int, default=8)
    export = learn_commands.add_parser(
        "export", help="从 checkpoint 的 source locator 编译 ScatteringPackage@1"
    )
    export.add_argument("checkpoint", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--material-index", type=int, default=0)

    package = commands.add_parser("package", help="ScatteringPackage@1 工具")
    package_commands = package.add_subparsers(
        dest="package_command", required=True
    )
    validate_package = package_commands.add_parser(
        "validate", help="验证 schema、URI 与内容 hash"
    )
    validate_package.add_argument("path", type=Path)
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
            args.config, args.output, args.resume
        ),
        ("learn", "evaluate"): lambda: _learn_evaluate(
            args.config, args.checkpoint, args.batches
        ),
        ("learn", "export"): lambda: _learn_export(
            args.checkpoint, args.output, args.material_index
        ),
        ("package", "validate"): lambda: _package_validate(args.path),
    }
    command = getattr(args, f"{args.command}_command")
    return int(dispatch[(args.command, command)]())


__all__ = ["build_parser", "main"]
