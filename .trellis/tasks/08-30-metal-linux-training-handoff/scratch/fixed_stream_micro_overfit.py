from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import torch

from ncls.core.identity import sha256_json, write_json_atomic
from ncls.learning.batches import TrainingRouteRequest
from ncls.learning.methods import get_method
from ncls.learning.producer import OnlineTrainingProducer
from ncls.learning.training import TrainingConfig


def _autocast(phase, device: torch.device):
    name = str(phase.precision["autocast"])
    if name == "fp32":
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if name == "float16" else torch.bfloat16,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="以相同query cursor重复运行authoritative reference并验证四phase局部下降"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=12)
    arguments = parser.parse_args()
    if arguments.repetitions < 4:
        raise ValueError("fixed-stream micro-overfit requires at least four repetitions")

    config = TrainingConfig.load(arguments.config)
    definition = get_method(config.method_key)
    producer = OnlineTrainingProducer(definition, config)
    model = definition.create_trainable(config.model_context).to(producer.device)
    frozen_query_state = producer.state_dict()
    phase_reports = []
    try:
        for phase_index, phase in enumerate(config.phases):
            definition.configure_phase(model, phase.to_dict())
            registry = definition.parameter_registry(model)
            active = tuple(
                parameter
                for group in phase.parameter_groups
                for parameter in registry[group]
            )
            optimizer = torch.optim.Adam(
                active,
                lr=float(phase.schedule["start"]),
                betas=tuple(float(value) for value in phase.optimizer["betas"]),
                eps=float(phase.optimizer["epsilon"]),
                weight_decay=float(phase.optimizer["weight_decay"]),
                fused=True,
            )
            losses = []
            elapsed = 0.0
            target_devices = set()
            for repetition in range(arguments.repetitions):
                producer.load_state_dict(frozen_query_state)
                batches = {}
                started = time.perf_counter()
                try:
                    for route in phase.routes:
                        request = TrainingRouteRequest(
                            f"fixed:{phase.name}:{route.name}",
                            route.kind,
                            route.batch_size,
                            route.direction_count,
                            0,
                            config.seed + route.seed_offset,
                            {
                                **route.options,
                                "recipes": dict(phase.recipes),
                                "validation": False,
                            },
                        )
                        batches[route.name] = producer.next_batch(request)
                    evaluator = batches.get("evaluator")
                    if evaluator is not None:
                        target_devices.add(str(evaluator.tensors["target_f"].device))
                        if evaluator.provenance.get("host_response_readback") is not False:
                            raise RuntimeError("fixed stream observed host response readback")
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast(phase, producer.device):
                        loss, _ = definition.training_objective(
                            model,
                            batches,
                            {
                                "name": phase.name,
                                "phase_index": phase_index,
                                "phase_step": repetition,
                                "global_step": repetition,
                                "parameter_groups": list(phase.parameter_groups),
                                "loss_terms": list(phase.loss_terms),
                                "recipes": dict(phase.recipes),
                                "validation": False,
                            },
                        )
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("fixed-stream objective produced non-finite loss")
                    loss.backward()
                    gradients = [parameter.grad for parameter in active if parameter.grad is not None]
                    if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
                        raise RuntimeError("fixed-stream objective produced invalid gradients")
                    optimizer.step()
                    losses.append(float(loss.detach()))
                finally:
                    for batch in reversed(tuple(batches.values())):
                        batch.release()
                    producer.end_iteration()
                    elapsed += time.perf_counter() - started
            window = max(1, arguments.repetitions // 4)
            initial = sum(losses[:window]) / window
            final = sum(losses[-window:]) / window
            phase_reports.append(
                {
                    "name": phase.name,
                    "repetitions": arguments.repetitions,
                    "initial_window_mean": initial,
                    "final_window_mean": final,
                    "final_minus_initial": final - initial,
                    "minimum": min(losses),
                    "losses": losses,
                    "target_devices": sorted(target_devices),
                    "authoritative_reference_reexecuted": "evaluator" in batches,
                    "persistent_response_batch": False,
                    "elapsed_seconds": elapsed,
                }
            )
        body = {
            "schema": "ncls.fixed-online-query-micro-overfit@1",
            "training_config_sha256": config.sha256,
            "method_descriptor_sha256": definition.descriptor.descriptor_sha256,
            "query_stream_identity": producer.query_stream_identity,
            "frozen_query_state_sha256": sha256_json(frozen_query_state),
            "phases": phase_reports,
        }
        report = {**body, "identity": sha256_json(body)}
        write_json_atomic(arguments.output, report)
        print(report["identity"])
        for phase in phase_reports:
            print(
                f"{phase['name']}: initial={phase['initial_window_mean']:.8g} "
                f"final={phase['final_window_mean']:.8g} "
                f"delta={phase['final_minus_initial']:.8g}"
            )
    finally:
        producer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
