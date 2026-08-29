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
from ncls.learning.method import (
    ComponentContract,
    MethodDefinition,
    MethodDescriptor,
    SourceAdaptationContract,
    TensorField,
)
from ncls.learning.source_adaptation import (
    DenseNativeAssetCollection,
    NativeAssetCollection,
    NativeAssetRole,
)
from ncls.learning.training import TrainingConfig, TrainingPhase, TrainingRoute, TrainingRunner


class _PhaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Parameter(torch.tensor([0.25]))
        self.latent = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.evaluator = torch.nn.Parameter(torch.tensor([0.5]))
        self.sampler = torch.nn.Parameter(torch.tensor([0.75]))
        self.phase_name = "bootstrap"


class _PhaseMethod(MethodDefinition):
    descriptor = MethodDescriptor(
        "phase-fixture", 1, "Phase fixture", "e" * 64,
        (SourceAdaptationContract("fixture.family", 1, ("/",), "recompile"),),
        {
            "reference-evaluator": ("wo", "wi", "target_f"),
            "method-sampler": ("wo", "sample_u"),
        },
        tuple(TensorField(name, dtype, (1,)) for name, dtype in (
            ("encoder", "float32"), ("latent", "float32"),
            ("evaluator", "float32"), ("sampler", "float32"),
            ("phase_code", "int64"),
        )),
        "fixture@1", 3,
        {"maximum_prepare_steps": 1, "maximum_evaluate_steps": 1,
         "maximum_state_bytes": 4, "maximum_reads": 1},
        {"fixture": True},
        {
            "encoder": ("encoder",),
            "asset": ("latent",),
            "evaluator": ("evaluator",),
            "sampler": ("sampler",),
        },
        (
            ComponentContract("encoder", True, ("encoder",), ("bootstrap",),
                              ("reference-evaluator",), ("evaluator",), (), ()),
            ComponentContract("asset", True, ("asset",), ("finetune",),
                              ("reference-evaluator",), ("evaluator",), ("asset:latent",), ()),
            ComponentContract("evaluator", True, ("evaluator",), ("bootstrap", "finetune"),
                              ("reference-evaluator",), ("evaluator",), ("program:evaluator",), ()),
            ComponentContract("sampler", True, ("sampler",), ("bootstrap", "finetune"),
                              ("method-sampler",), ("sampler",), ("program:sampler",), ()),
        ),
    )

    def create_trainable(self, context: Mapping[str, Any]) -> torch.nn.Module:
        del context
        return _PhaseModel()

    def configure_phase(self, model: torch.nn.Module, phase: Mapping[str, Any]) -> None:
        assert isinstance(model, _PhaseModel)
        super().configure_phase(model, phase)
        model.phase_name = str(phase["name"])

    def apply_phase_transition(
        self, model: torch.nn.Module, transition: str, native_assets: NativeAssetCollection
    ) -> None:
        del native_assets
        assert isinstance(model, _PhaseModel)
        assert transition == "materialize-assets"
        with torch.no_grad():
            model.latent.copy_(model.encoder)

    def training_objective(
        self,
        model: torch.nn.Module,
        batches: Mapping[str, OnlineTrainingBatch],
        phase: Mapping[str, Any],
    ):
        assert isinstance(model, _PhaseModel)
        evaluator_batch = batches["evaluator"]
        sampler_batch = batches["sampler"]
        assert isinstance(evaluator_batch, EvaluatorBatch)
        assert isinstance(sampler_batch, MethodSamplerBatch)
        shared = model.encoder if phase["name"] == "bootstrap" else model.latent
        evaluator_target = evaluator_batch.target_f.mean()
        sampler_target = sampler_batch.sample_u.mean()
        evaluator_loss = (model.evaluator * shared - evaluator_target).square().mean()
        sampler_loss = (model.sampler * shared.detach() - sampler_target).square().mean()
        return evaluator_loss + sampler_loss, {
            "evaluator": evaluator_loss.detach(), "sampler": sampler_loss.detach()
        }

    def export_training_state(self, model: torch.nn.Module):
        assert isinstance(model, _PhaseModel)
        return {
            "encoder": model.encoder.detach().clone(),
            "latent": model.latent.detach().clone(),
            "evaluator": model.evaluator.detach().clone(),
            "sampler": model.sampler.detach().clone(),
            "phase_code": torch.tensor(
                [0 if model.phase_name == "bootstrap" else 1], dtype=torch.int64
            ),
        }

    def restore_training_state(
        self, model: torch.nn.Module, state: Mapping[str, torch.Tensor]
    ) -> None:
        assert isinstance(model, _PhaseModel)
        with torch.no_grad():
            for name in ("encoder", "latent", "evaluator", "sampler"):
                getattr(model, name).copy_(state[name])
        model.phase_name = "bootstrap" if int(state["phase_code"].item()) == 0 else "finetune"

    def compile_program(self, checkpoint: Mapping[str, Any]) -> RuntimePayload:
        del checkpoint
        return RuntimePayload("fixture.slang", {"fixture.slang": b""}, {}, {}, 3)

    def compile_asset(
        self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]
    ) -> MaterialPayload:
        del checkpoint
        return MaterialPayload(snapshot.snapshot_id, {"fixture": b"\0"}, {
            "fixture": {"dtype": "uint8", "shape": [1], "stride": 1,
                        "alignment": 1, "usage": "fixture"}
        })


class _RouteProducer:
    reference_program_identity = "1" * 64
    reference_execution_plan_identity = "2" * 64
    native_asset_collection_identity = "3" * 64
    query_stream_identity = "4" * 64
    source_contracts = (
        {"family_id": "fixture.family", "source_contract_version": 1},
    )
    source_snapshot_ids = ("a" * 64,)
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.generators: dict[str, torch.Generator] = {}
        self.counts: dict[str, int] = {}
        self._assets = DenseNativeAssetCollection(
            ((torch.zeros(1, 1, 1),),),
            ("fixture",),
            "fixture-layout",
            "constant",
            "constant",
            "clamp",
            (NativeAssetRole("value", "fixture", 0, 1, "linear", "constant"),),
        )
        self.native_asset_collection_identity = self._assets.collection_id

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
            target_f = torch.rand((batch, request.direction_count, 3), generator=generator)
            return EvaluatorBatch(conditioning, wi, target_f)
        return MethodSamplerBatch(
            conditioning, torch.rand((batch, 2), generator=generator)
        )

    def native_assets(self) -> NativeAssetCollection:
        return self._assets

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


def _phase(
    name: str,
    groups: tuple[str, ...],
    offset: int,
    transition: str | None,
    policy: str,
) -> TrainingPhase:
    return TrainingPhase(
        name,
        2,
        (
            TrainingRoute("evaluator", "reference-evaluator", 2, 1, 0, {}),
            TrainingRoute("sampler", "method-sampler", 2, 1, 1, {}),
        ),
        groups,
        ("evaluator", "sampler"),
        {"fixture": True},
        {"kind": "adam", "betas": [0.9, 0.999], "epsilon": 1e-7,
         "weight_decay": 0.0},
        policy,
        {"kind": "cosine", "start": 1e-3, "end": 1e-4,
         "total_steps": 4, "offset": offset},
        {"autocast": "fp32", "gradient_scaler": False},
        True,
        transition,
        1,
        1,
        2,
    )


def _config() -> TrainingConfig:
    return TrainingConfig(
        method_key="phase-fixture",
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
        phases=(
            _phase("bootstrap", ("encoder", "evaluator", "sampler"), 0,
                   "materialize-assets", "reset"),
            _phase("finetune", ("asset", "evaluator", "sampler"), 2, None,
                   "carry-overlap"),
        ),
        seed=11,
        device="cpu",
        validation={"interval": 2, "batches": 1},
        checkpoint_selection="tail_guard",
    )


def test_runner_resume_matches_uninterrupted_phase_graph() -> None:
    definition = _PhaseMethod()
    full = TrainingRunner(definition, _RouteProducer(), _config()).run().checkpoint
    partial = TrainingRunner(definition, _RouteProducer(), _config()).run(
        stop_at_step=2
    ).checkpoint
    assert partial.phase_name == "finetune" and partial.phase_step == 0
    resumed = TrainingRunner(definition, _RouteProducer(), _config()).run(
        resume=partial
    ).checkpoint
    assert resumed.phase_name == "complete"
    assert set(resumed.query_stream_state["generator_states"]) == {
        "bootstrap:evaluator", "bootstrap:sampler",
        "finetune:evaluator", "finetune:sampler",
        "validation:finetune:evaluator", "validation:finetune:sampler",
    }
    assert all(
        torch.equal(full.model_state[name], resumed.model_state[name])
        for name in full.model_state
    )
    assert all(
        value["parameter_update_observed"]
        for value in resumed.gradient_coverage.values()
    )
