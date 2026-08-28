from __future__ import annotations

from typing import Any, Mapping

import torch

from ncls.core.scattering import MaterialPayload, RuntimePayload
from ncls.core.source import SourceSnapshot
from ncls.learning.batches import (
    EvaluatorBatch,
    MethodSamplerBatch,
    OnlineTrainingBatch,
    TrainingConditioning,
    TrainingRouteRequest,
)
from ncls.learning.source_adaptation import DenseNativeFeaturePyramid, NativeFeaturePyramid
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
        {
            "reference-evaluator": ("wo", "wi", "target_f"),
            "method-sampler": ("wo", "sample_u"),
        },
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
        batches: Mapping[str, OnlineTrainingBatch],
        lifecycle: Mapping[str, Any],
    ):
        assert isinstance(model, _LifecycleModel)
        evaluator_batch = batches["evaluator"]
        sampler_batch = batches["sampler"]
        assert isinstance(evaluator_batch, EvaluatorBatch)
        assert isinstance(sampler_batch, MethodSamplerBatch)
        shared = model.encoder if lifecycle["stage"] == "bootstrap" else model.latent
        evaluator_target = evaluator_batch.target_f.mean()
        sampler_target = sampler_batch.sample_u.mean()
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


class _RouteProducer:
    reference_program_identity = "reference-program:fixture"
    query_stream_identity = "query-stream:fixture"
    source_contracts = (
        {"family_id": "fixture.family", "source_contract_version": 1},
    )
    source_snapshot_ids = ("a" * 64,)
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.generators: dict[str, torch.Generator] = {}
        self.counts: dict[str, int] = {}

    def _generator(self, request: TrainingRouteRequest) -> torch.Generator:
        generator = self.generators.get(request.name)
        if generator is None:
            generator = torch.Generator().manual_seed(request.seed)
            self.generators[request.name] = generator
        return generator

    def next_batch(self, request: TrainingRouteRequest) -> OnlineTrainingBatch:
        generator = self._generator(request)
        batch = request.batch_size
        conditioning = TrainingConditioning(
            "fixture.family",
            self.source_snapshot_ids,
            {
                "source_index": torch.zeros(batch, dtype=torch.int64),
                "wo": torch.tensor([[0.0, 0.0, 1.0]]).expand(batch, 3).clone(),
            },
            {
                "route_name": request.name,
                "request_index": self.counts.get(request.name, 0),
            },
        )
        self.counts[request.name] = self.counts.get(request.name, 0) + 1
        if request.kind == "reference-evaluator":
            wi = torch.tensor([[[0.0, 0.0, 1.0]]]).expand(
                batch, request.direction_count, 3
            ).clone()
            target_f = torch.rand(
                (batch, request.direction_count, 3), generator=generator
            )
            return EvaluatorBatch(conditioning, wi, target_f)
        return MethodSamplerBatch(
            conditioning, torch.rand((batch, 2), generator=generator)
        )

    def materialization_features(self) -> NativeFeaturePyramid:
        return DenseNativeFeaturePyramid((torch.zeros(1, 1, 1),))

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "query_stream_identity": self.query_stream_identity,
            "generator_states": {
                name: generator.get_state() for name, generator in self.generators.items()
            },
            "request_count": dict(self.counts),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        assert state["query_stream_identity"] == self.query_stream_identity
        self.generators = {}
        for name, generator_state in state["generator_states"].items():
            generator = torch.Generator()
            generator.set_state(generator_state)
            self.generators[name] = generator
        self.counts = dict(state["request_count"])

    def end_iteration(self) -> None:
        pass

    def close(self) -> None:
        pass


def _config() -> TrainingConfig:
    return TrainingConfig(
        method_key="lifecycle-fixture",
        run_class="smoke",
        correspondence_id="fixture-correspondence@1",
        recipe_id="fixture-recipe@1",
        source_adaptation_id="fixture-adapter@1",
        source={
            "family_id": "fixture.family",
            "materials": [{"locator": {"kind": "fixture"}}],
        },
        online_query={"recipe_id": "fixture-online-query@1"},
        model_context={"fixture": True},
        lifecycle={"total_steps": 4, "materialization_step": 2},
        routes=(
            TrainingRoute("evaluator", "reference-evaluator", 2, 1, 0, {}),
            TrainingRoute("sampler", "method-sampler", 2, 1, 1, {}),
        ),
        seed=11,
        device="cpu",
        optimizer={"kind": "adam", "betas": [0.9, 0.999], "epsilon": 1e-7,
                   "weight_decay": 0.0},
        schedule={"kind": "cosine", "start": 1e-3, "end": 1e-4, "total_steps": 4},
        mollification={"steps": 0, "start_degrees": 0.0, "samples": 1},
        filtering={"fixture": True},
        loss={"fixture": True},
        validation={"interval": 2, "batches": 1},
        checkpoint_selection="tail_guard",
    )


def test_runner_resume_matches_uninterrupted_two_route_lifecycle() -> None:
    definition = _LifecycleMethod()
    full = TrainingRunner(definition, _RouteProducer(), _config()).run().checkpoint
    partial = TrainingRunner(definition, _RouteProducer(), _config()).run(
        stop_at_step=2
    ).checkpoint
    assert partial.phase == "finetune"
    assert partial.lifecycle_state["stage"] == "finetune"
    resumed = TrainingRunner(definition, _RouteProducer(), _config()).run(
        resume=partial
    ).checkpoint
    assert resumed.phase == "complete"
    assert set(resumed.query_stream_state["generator_states"]) == {
        "evaluator", "sampler", "validation:evaluator", "validation:sampler"
    }
    assert all(
        torch.equal(full.model_state[name], resumed.model_state[name])
        for name in full.model_state
    )
    assert full.scheduler_state == resumed.scheduler_state
