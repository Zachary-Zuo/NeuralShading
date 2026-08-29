from types import SimpleNamespace

import pytest
import torch

from ncls.core.identity import sha256_bytes
from ncls.core.material import DiffuseInterface, LayerStackIR
from ncls.core.scattering import (
    BackendCapability,
    FileResourcePayload,
    MaterialPayload,
    ReferenceProgramDefinition,
    ReferenceProgramDescriptor,
    RuntimePayload,
    read_resource_payload,
    resource_payload_sha256,
)
from ncls.core.source import SourceSnapshot
from ncls.references import query as query_module
from ncls.references.plan import (
    compile_reference_execution_plan,
    compile_single_program_plan,
)
from ncls.references.programs import get_reference_program_for_source
from ncls.references.query import ReferenceBackendSession
from ncls.source_materials.families.layer_stack import snapshot_from_layer_stack


def test_reference_execution_plan_has_dense_global_and_group_local_indices() -> None:
    snapshots = tuple(
        snapshot_from_layer_stack(LayerStackIR((DiffuseInterface(color),), ()))
        for color in ((0.6, 0.3, 0.1), (0.2, 0.5, 0.8))
    )
    definition = get_reference_program_for_source(
        snapshots[0].family_id, snapshots[0].source_contract_version
    )
    plan = compile_single_program_plan(
        definition,
        snapshots,
        query_recipe={"recipe_id": "unit-plan@1"},
    )
    assert plan.schema_version == 1 and len(plan.identity) == 64
    assert len(plan.groups) == 1
    assert plan.groups[0].global_source_indices == (0, 1)
    assert tuple(record.local_material_index for record in plan.records) == (0, 1)
    assert plan.source_snapshot_ids == tuple(value.snapshot_id for value in snapshots)


def test_reference_execution_plan_rejects_duplicate_snapshot_identity() -> None:
    snapshot = snapshot_from_layer_stack(
        LayerStackIR((DiffuseInterface((0.6, 0.3, 0.1)),), ())
    )
    definition = get_reference_program_for_source(
        snapshot.family_id, snapshot.source_contract_version
    )
    with pytest.raises(ValueError, match="unique"):
        compile_single_program_plan(
            definition,
            (snapshot, snapshot),
            query_recipe={"recipe_id": "unit-plan@1"},
        )


class _SplitGroupDefinition(ReferenceProgramDefinition):
    descriptor = ReferenceProgramDescriptor(
        "unit.split-groups",
        1,
        "Unit split groups",
        "unit.family@1",
        1,
        "a" * 64,
        "ncls.scattering-backend@1",
        int(
            BackendCapability.PREPARE
            | BackendCapability.EVALUATE
            | BackendCapability.SAMPLE
            | BackendCapability.PDF
        ),
        {
            "maximum_prepare_steps": 1,
            "maximum_evaluate_steps": 1,
            "maximum_state_bytes": 16,
            "maximum_reads": 1,
        },
    )

    def execution_group_key(self, snapshot, material) -> str:
        self.validate_snapshot(snapshot)
        assert material.source_snapshot_id == snapshot.snapshot_id
        return sha256_bytes(snapshot.native_payload)

    def compile_runtime(self) -> RuntimePayload:
        return RuntimePayload("unused.slang", {"unused.slang": b"unused"}, {}, {}, 15)

    def compile_material(self, snapshot: SourceSnapshot) -> MaterialPayload:
        return MaterialPayload(snapshot.snapshot_id, {}, {})


class _FakeGroupSession:
    def __init__(self, group, **kwargs) -> None:
        del kwargs
        self.group = group
        self.device = torch.device("cpu")
        self.active_lease_count = 0
        self.closed = False

    def assert_idle(self) -> None:
        if self.active_lease_count:
            raise RuntimeError("reference execution group has active query leases")

    def close(self) -> None:
        self.assert_idle()
        self.closed = True


class _FakeDevice:
    def __init__(self) -> None:
        self.end_frame_count = 0

    def end_frame(self) -> None:
        self.end_frame_count += 1


def test_multi_group_session_ends_one_shared_frame_and_closes_atomically(monkeypatch) -> None:
    definition = _SplitGroupDefinition()
    snapshots = tuple(
        SourceSnapshot(
            "unit.family@1",
            1,
            "unit.schema@1",
            str(index) * 64,
            payload,
        )
        for index, payload in ((1, b"first"), (2, b"second"))
    )
    plan = compile_reference_execution_plan(
        ((definition, snapshot) for snapshot in snapshots),
        query_recipe={"recipe_id": "unit-multi-group@1"},
    )
    assert len(plan.groups) == 2
    monkeypatch.setattr(query_module, "_ReferenceExecutionGroupSession", _FakeGroupSession)
    device = _FakeDevice()
    session = ReferenceBackendSession(
        plan,
        backend_descriptor=SimpleNamespace(identity="b" * 64),
        falcor=object(),
        device_handle=device,
        query_capacity=1,
        device="cpu",
    )
    groups = tuple(session._session(group.group_id) for group in plan.groups)

    session.end_iteration()
    assert device.end_frame_count == 1
    groups[1].active_lease_count = 1
    with pytest.raises(RuntimeError, match="active query leases"):
        session.close()
    assert not any(group.closed for group in groups)

    groups[1].active_lease_count = 0
    session.close()
    assert all(group.closed for group in groups)


def test_multi_group_session_materializes_lazily_and_evicts_only_idle_lru(monkeypatch) -> None:
    definition = _SplitGroupDefinition()
    snapshots = tuple(
        SourceSnapshot(
            "unit.family@1",
            1,
            "unit.schema@1",
            str(index) * 64,
            payload,
        )
        for index, payload in ((1, b"first"), (2, b"second"))
    )
    plan = compile_reference_execution_plan(
        ((definition, snapshot) for snapshot in snapshots),
        query_recipe={"recipe_id": "unit-lazy-group@1"},
    )
    monkeypatch.setattr(query_module, "_ReferenceExecutionGroupSession", _FakeGroupSession)
    session = ReferenceBackendSession(
        plan,
        backend_descriptor=SimpleNamespace(identity="b" * 64),
        falcor=object(),
        device_handle=_FakeDevice(),
        query_capacity=1,
        device="cpu",
        max_resident_groups=1,
    )
    assert session.resident_group_ids == ()
    first = session._session(plan.groups[0].group_id)
    assert session.resident_group_ids == (plan.groups[0].group_id,)
    first.active_lease_count = 1
    with pytest.raises(RuntimeError, match="active leases"):
        session._session(plan.groups[1].group_id)
    first.active_lease_count = 0
    second = session._session(plan.groups[1].group_id)
    assert first.closed
    assert not second.closed
    assert session.resident_group_ids == (plan.groups[1].group_id,)
    session.close()


def test_file_resource_payload_is_content_addressed_lazy_and_detects_tamper(tmp_path) -> None:
    first_path = tmp_path / "first.bin"
    second_path = tmp_path / "second.bin"
    first_path.write_bytes(b"large-resource")
    second_path.write_bytes(b"large-resource")
    first = FileResourcePayload.from_path(first_path)
    second = FileResourcePayload.from_path(second_path)
    assert first == second
    assert resource_payload_sha256(first) == resource_payload_sha256(b"large-resource")
    assert read_resource_payload(first) == b"large-resource"
    first_path.write_bytes(b"changed-resourc")
    with pytest.raises(ValueError, match="changed after plan compilation"):
        read_resource_payload(first)
