from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from ncls.core.scattering import MaterialPayload, RuntimePayload
from ncls.core.source import SourceSnapshot
from ncls.data.training_batch import TrainingBatch, TrainingRouteRequest
from ncls.data.native_features import DenseNativeFeaturePyramid, NativeFeaturePyramid
from ncls.learning.method import (
    MethodDefinition,
    MethodDescriptor,
    SourceAdaptationContract,
    TensorField,
)
from ncls.learning.training import TrainingConfig, TrainingRoute, TrainingRunner


class _LifecycleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Parameter(torch.tensor([0.25]))
        self.latent = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.evaluator = torch.nn.Parameter(torch.tensor([0.5]))
        self.sampler = torch.nn.Parameter(torch.tensor([0.75]))
        self.lifecycle_stage = "bootstrap"


class _LifecycleMethod(MethodDefinition):
    descriptor = MethodDescriptor(
        "lifecycle-fixture", 1, "Lifecycle fixture", "e" * 64,
        (SourceAdaptationContract("fixture.family", 1, ("/",), "recompile"),),
        ("wo", "wi", "target"),
        tuple(TensorField(name, dtype, (1,)) for name, dtype in (
            ("encoder", "float32"), ("latent", "float32"),
            ("evaluator", "float32"), ("sampler", "float32"),
            ("lifecycle_stage", "int64"),
        )),
        "fixture@1", 3,
        {"maximum_prepare_steps": 1, "maximum_evaluate_steps": 1,
         "maximum_state_bytes": 4, "maximum_reads": 1},
        {"fixture": True},
    )

    def create_trainable(self, context: Mapping[str, Any]) -> torch.nn.Module:
        del context
        return _LifecycleModel()

    def configure_lifecycle(self, model: torch.nn.Module, lifecycle: Mapping[str, Any]) -> None:
        assert isinstance(model, _LifecycleModel)
        stage = str(lifecycle["stage"])
        model.lifecycle_stage = stage
        model.encoder.requires_grad_(stage == "bootstrap")
        model.latent.requires_grad_(stage == "finetune")
        model.evaluator.requires_grad_(True)
        model.sampler.requires_grad_(True)

    def materialize_latent(
        self, model: torch.nn.Module, native_feature_pyramid: NativeFeaturePyramid
    ) -> None:
        del native_feature_pyramid
        assert isinstance(model, _LifecycleModel)
        with torch.no_grad():
            model.latent.copy_(model.encoder)
        self.configure_lifecycle(model, {"stage": "finetune"})

    def training_objective(
        self,
        model: torch.nn.Module,
        batches: Mapping[str, TrainingBatch],
        lifecycle: Mapping[str, Any],
    ):
        assert isinstance(model, _LifecycleModel)
        shared = model.encoder if lifecycle["stage"] == "bootstrap" else model.latent
        evaluator_target = batches["evaluator"].tensors["target"].mean()
        sampler_target = batches["sampler"].tensors["target"].mean()
        evaluator_loss = (model.evaluator * shared - evaluator_target).square().mean()
        sampler_loss = (model.sampler * shared.detach() - sampler_target).square().mean()
        return evaluator_loss + sampler_loss, {
            "evaluator": evaluator_loss.detach(), "sampler": sampler_loss.detach()
        }

    def export_training_state(self, model: torch.nn.Module):
        assert isinstance(model, _LifecycleModel)
        return {
            "encoder": model.encoder.detach().clone(),
            "latent": model.latent.detach().clone(),
            "evaluator": model.evaluator.detach().clone(),
            "sampler": model.sampler.detach().clone(),
            "lifecycle_stage": torch.tensor(
                [0 if model.lifecycle_stage == "bootstrap" else 1], dtype=torch.int64
            ),
        }

    def restore_training_state(
        self, model: torch.nn.Module, state: Mapping[str, torch.Tensor]
    ) -> None:
        assert isinstance(model, _LifecycleModel)
        with torch.no_grad():
            for name in ("encoder", "latent", "evaluator", "sampler"):
                getattr(model, name).copy_(state[name])
        stage = "bootstrap" if int(state["lifecycle_stage"].item()) == 0 else "finetune"
        self.configure_lifecycle(model, {"stage": stage})

    def compile_runtime(self, checkpoint: Mapping[str, Any]) -> RuntimePayload:
        del checkpoint
        return RuntimePayload("fixture.slang", {"fixture.slang": b""}, {}, {}, 3)

    def compile_material(
        self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]
    ) -> MaterialPayload:
        del checkpoint
        return MaterialPayload(snapshot.snapshot_id, {"fixture": b"\0"}, {
            "fixture": {"dtype": "uint8", "shape": [1], "stride": 1,
                        "alignment": 1, "usage": "fixture"}
        })


class _RouteSource:
    kind = "live"
    identity = "route-source"
    source_contracts = ({"family_id": "fixture.family", "source_contract_version": 1},)
    source_state_ids = ("a" * 64,)
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.generators: dict[str, np.random.Generator] = {}
        self.counts: dict[str, int] = {}

    def next_batch(self, request: TrainingRouteRequest) -> TrainingBatch:
        generator = self.generators.setdefault(
            request.name, np.random.default_rng(request.seed)
        )
        target_value = float(generator.random())
        self.counts[request.name] = self.counts.get(request.name, 0) + 1
        batch = request.batch_size
        directions = request.direction_count
        return TrainingBatch(
            "fixture.family", tuple("a" * 64 for _ in range(batch)), "linear-response",
            {
                "source_index": torch.zeros(batch, dtype=torch.int64),
                "wo": torch.tensor([[0.0, 0.0, 1.0]]).expand(batch, 3).clone(),
                "wi": torch.tensor([[[0.0, 0.0, 1.0]]]).expand(
                    batch, directions, 3
                ).clone(),
                "target": torch.full((batch, directions, 3), target_value),
                "solid_angle_weight": torch.ones(batch, directions),
                "reference_pdf": torch.ones(batch, directions),
                "sample_count": torch.ones(batch, directions, dtype=torch.int64),
                "rng_seed": torch.full(
                    (batch, directions), request.seed, dtype=torch.int64
                ),
                "query_role": torch.full((batch,), request.query_role, dtype=torch.int64),
            },
            {"route_name": request.name, "request_index": self.counts[request.name] - 1},
        )

    def materialization_features(self) -> NativeFeaturePyramid:
        return DenseNativeFeaturePyramid((torch.zeros(1, 1, 1),))

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "rng": {name: generator.bit_generator.state
                    for name, generator in self.generators.items()},
            "counts": dict(self.counts),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.generators = {}
        for name, rng_state in state["rng"].items():
            generator = np.random.default_rng()
            generator.bit_generator.state = rng_state
            self.generators[name] = generator
        self.counts = dict(state["counts"])

    def close(self) -> None:
        pass


def _config() -> TrainingConfig:
    return TrainingConfig(
        "lifecycle-fixture", "smoke", "fixture-correspondence@1", "fixture-recipe@1",
        "fixture-adapter@1",
        {"kind": "live", "options": {}}, {"fixture": True},
        {"total_steps": 4, "materialization_step": 2},
        (
            TrainingRoute("evaluator", 2, 1, 0, 0, {}),
            TrainingRoute("sampler", 2, 1, 1, 1, {}),
        ),
        11, "cpu",
        {"kind": "adam", "betas": [0.9, 0.999], "epsilon": 1e-7,
         "weight_decay": 0.0},
        {"kind": "cosine", "start": 1e-3, "end": 1e-4, "total_steps": 4},
        {"steps": 0, "start_degrees": 0.0, "samples": 1},
        {"fixture": True}, {"fixture": True},
        {"interval": 2, "batches": 1}, "tail_guard",
    )


def test_runner_resume_matches_uninterrupted_two_route_lifecycle() -> None:
    definition = _LifecycleMethod()
    full = TrainingRunner(definition, _RouteSource(), _config()).run().checkpoint
    partial_source = _RouteSource()
    partial = TrainingRunner(definition, partial_source, _config()).run(stop_at_step=2).checkpoint
    assert partial.phase == "finetune"
    assert partial.lifecycle_state["stage"] == "finetune"
    resumed = TrainingRunner(definition, _RouteSource(), _config()).run(resume=partial).checkpoint
    assert resumed.phase == "complete"
    assert set(resumed.batch_source_state["rng"]) == {
        "evaluator", "sampler", "validation:evaluator", "validation:sampler"
    }
    assert all(
        torch.equal(full.model_state[name], resumed.model_state[name])
        for name in full.model_state
    )
    assert full.scheduler_state == resumed.scheduler_state
