from __future__ import annotations

from typing import Any, Mapping

import torch

from ncls.core.scattering import MaterialPayload, RuntimePayload
from ncls.core.source import SourceSnapshot
from ncls.data.training_batch import TrainingBatch
from ncls.learning.method import MethodDefinition, MethodDescriptor, SourceAdaptationContract, TensorField


class ContractFixtureMethod(MethodDefinition):
    descriptor = MethodDescriptor(
        "contract-fixture", 1, "Contract fixture", "f" * 64,
        (
            SourceAdaptationContract("openpbr.material@1.1.1", 1, ("/inputs",), "runtime-patch"),
            SourceAdaptationContract("ncls.layer-stack@1", 1, ("/interfaces", "/slabs"), "recompile"),
        ),
        ("wo", "wi", "target"),
        (TensorField("fixture.scale", "float32", (3,)), TensorField("fixture.bias", "float32", (3,))),
        "ncls.scattering-backend@1", 3,
        {"maximum_prepare_steps": 1, "maximum_evaluate_steps": 2, "maximum_state_bytes": 16, "maximum_reads": 2},
        {"fixture": True},
    )

    def create_trainable(self, context: Mapping[str, Any]) -> torch.nn.Module:
        del context
        return torch.nn.Linear(3, 3)

    def training_objective(
        self,
        model: torch.nn.Module,
        batches: Mapping[str, TrainingBatch],
        lifecycle: Mapping[str, Any],
    ):
        del lifecycle
        batch = next(iter(batches.values()))
        prediction = model(batch.tensors["wi"])
        loss = torch.mean(torch.abs(prediction - batch.tensors["target"]))
        return loss, {"l1": loss.detach()}

    def export_training_state(self, model: torch.nn.Module):
        return {"fixture.scale": model.weight.diagonal().detach(), "fixture.bias": model.bias.detach()}

    def restore_training_state(self, model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
        with torch.no_grad():
            model.weight.zero_()
            model.weight.diagonal().copy_(state["fixture.scale"])
            model.bias.copy_(state["fixture.bias"])

    def compile_runtime(self, checkpoint: Mapping[str, Any]) -> RuntimePayload:
        del checkpoint
        module = b"struct NclsPackageBackend {};\n"
        return RuntimePayload("fixture.slang", {"fixture.slang": module}, {}, {}, 3)

    def compile_material(self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]) -> MaterialPayload:
        del checkpoint
        return MaterialPayload(snapshot.snapshot_id, {"fixture": b"\0" * 16}, {
            "fixture": {"dtype": "uint8", "shape": [16], "stride": 1, "alignment": 16, "usage": "fixture"}
        })


METHOD_DEFINITION = ContractFixtureMethod()
