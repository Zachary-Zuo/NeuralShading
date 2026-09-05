from __future__ import annotations

import argparse
from pathlib import Path
from .runtime import parse_devices


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
    train_yaml.add_argument("devices", type=parse_devices, help="物理 GPU 编号，例如 0 或 0,1")
    train_yaml.add_argument("--config", type=Path, required=True)
    train_yaml.add_argument("--resume", type=Path)
    train_yaml.add_argument("--stop-at-step", type=int)

    validate_checkpoint = commands.add_parser(
        "validate", help="用 checkpoint 内嵌计划运行数值 batch 验证"
    )
    validate_checkpoint.add_argument("checkpoint", type=Path)
    validate_checkpoint.add_argument("--batches", type=int, default=8)
    validate_checkpoint.add_argument("--device", type=int, default=0)

    export_checkpoint = commands.add_parser(
        "export", help="从当前 checkpoint 导出 ScatteringPackage"
    )
    export_checkpoint.add_argument("checkpoint", type=Path)
    export_checkpoint.add_argument("--output", type=Path, help="默认写入 run/exports/step-N/")
    export_checkpoint.add_argument("--material-index", type=int, default=0)

    visual_eval = commands.add_parser("eval", help="使用 checkpoint 的图像设置评估当前模型")
    visual_eval.add_argument("checkpoint", type=Path)
    visual_eval.add_argument("--config", type=Path, help="可覆盖图像设置的 YAML")
    visual_eval.add_argument("--device", type=int, default=0)

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
    probe = reference_commands.add_parser(
        "probe", help="用仓库 fixture 验证 device、LayerStack 与 MDL compile/query"
    )
    probe.add_argument("--device", type=int, default=0)
    return parser
