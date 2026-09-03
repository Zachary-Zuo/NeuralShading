from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from ncls.core.identity import write_json_atomic
from ncls.learning.methods.metal_fused import METHOD_DEFINITION
from ncls.learning.training import TrainingConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 canonical Windows smoke 生成固定预算的 Metal 学习探针"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--joint-steps", type=int, default=512)
    parser.add_argument("--qat-steps", type=int, default=32)
    parser.add_argument(
        "--route-geometry-source",
        type=Path,
        help="仅复制各 route 的 batch_size/direction_count，用于硬件匹配的结构探针",
    )
    args = parser.parse_args()
    if min(args.joint_steps, args.qat_steps) < 1:
        raise ValueError("probe phase steps must be positive")

    source = TrainingConfig.load(args.source).to_dict()
    payload = deepcopy(source)
    payload["run_class"] = "profile"
    payload["recipe_id"] = (
        f"metal-fused-windows-learning-probe-"
        f"{args.joint_steps + args.qat_steps}step@1"
    )
    if args.route_geometry_source is not None:
        geometry_config = TrainingConfig.load(args.route_geometry_source)
        geometry = {
            route.name: (route.batch_size, route.direction_count)
            for route in geometry_config.phases[0].routes
        }
        if set(geometry) != {"asset", "evaluator", "sampler"}:
            raise ValueError("route geometry source omits a Metal typed route")
        for phase in payload["phases"]:
            for route in phase["routes"]:
                route["batch_size"], route["direction_count"] = geometry[route["name"]]
    total_steps = args.joint_steps + args.qat_steps
    phase_steps = (args.joint_steps, args.qat_steps)
    offset = 0
    for phase, steps in zip(payload["phases"], phase_steps, strict=True):
        phase["steps"] = steps
        phase["schedule"]["total_steps"] = total_steps
        phase["schedule"]["offset"] = offset
        phase["log_interval"] = min(16, steps)
        phase["gradient_audit_interval"] = min(32, steps)
        offset += steps
    payload["validation"] = {
        "interval": min(64, total_steps),
        "batches": 4,
    }
    config = TrainingConfig.from_dict(payload)
    METHOD_DEFINITION.validate_training_config(config.to_dict())
    write_json_atomic(args.output, config.to_dict())
    print(f"config_sha256={config.sha256} steps={config.total_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
