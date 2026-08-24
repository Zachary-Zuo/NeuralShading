from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from ncls.cli import main as cli_main
from ncls.data import (
    FORMAT_NAME,
    FORMAT_VERSION,
    QUERY_ROLE_NAMES,
    CollectionConfig,
    E0_FOOTPRINT_PROFILE_ID,
    EvaluatedBlock,
    QueryPlan,
    ReferenceDataset,
    ReferenceDescriptor,
    ReplicaMoments,
    SourceState,
    SurfaceSample,
    collect_reference_dataset,
    combine_replica_moments,
    equal_area_hemisphere,
    grazing_anchored_view_directions,
    peak_grazing_mixture_pdf,
    peak_grazing_mixture_query,
    make_state_id,
    stratified_view_directions,
    uv_surface_samples,
)
from ncls.data.providers import LayerStackProvider, LayerStackProviderConfig
from ncls.data.directions import _folded_vmf_pdf


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakeProvider:
    descriptor = ReferenceDescriptor(
        "test.surface@1",
        "test.reference@1",
        "test.native-json@1",
        implementation_sha256="1" * 64,
    )

    def __init__(self) -> None:
        states = []
        for index, split in enumerate((0, 1, 2)):
            payload = json.dumps({"value": index + 1}, sort_keys=True).encode("utf-8")
            source_hash = hashlib.sha256(payload).hexdigest()
            states.append(SourceState(
                make_state_id(self.descriptor.family_id, self.descriptor.native_schema_id, payload, source_hash),
                self.descriptor.family_id,
                self.descriptor.reference_id,
                f"asset-{index}",
                f"group-{index}",
                self.descriptor.native_schema_id,
                payload,
                f"test://asset-{index}",
                source_hash,
                split,
                index + 1,
            ))
        self.states = tuple(states)

    def source_states(self):
        return self.states

    def surface_samples(self, state):
        return (SurfaceSample(),)

    def query_plan(self, state, surfaces=()):
        lights, weights = equal_area_hemisphere(4)
        per_view_lights = np.stack((lights, np.roll(lights, 1, axis=0)))
        return QueryPlan(
            stratified_view_directions(2),
            per_view_lights,
            weights,
            np.full(4, 1.0 / (2.0 * np.pi), dtype=np.float32),
            ("train-uniform@1", "test-uniform@1"),
            17,
            (0, 2),
        )

    def evaluate(self, state, surfaces, plan):
        shape = (len(surfaces), len(plan.view_directions), plan.direction_count, 3)
        value = np.full(shape, state.runtime_state, dtype=np.float32)
        return EvaluatedBlock.deterministic(
            value,
            rng_seed=np.full(shape[:-1], 100 + state.runtime_state, dtype=np.uint64),
        )

    def metadata(self):
        return {"family_id": self.descriptor.family_id, "reference_id": self.descriptor.reference_id}

    def close(self):
        pass


class _SeedRecordingEvaluator:
    def __init__(self, light_count: int) -> None:
        self.light_count = light_count
        self.query_group_seeds = np.empty(0, dtype=np.uint32)

    def evaluate_query_groups(
        self,
        materials,
        view_directions,
        *,
        sample_count_per_replica: int,
        query_group_seeds: np.ndarray,
        light_directions: np.ndarray | None = None,
        sample_offset: int = 0,
    ):
        self.query_group_seeds = np.asarray(query_group_seeds, dtype=np.uint32).copy()
        shape = (len(materials), self.light_count, 3)
        mean_a = np.full(shape, 0.2, dtype=np.float32)
        mean_b = np.full(shape, 0.22, dtype=np.float32)
        second_a = mean_a * mean_a
        second_b = mean_b * mean_b
        return mean_a, second_a, mean_b, second_b


def _collect(path: Path):
    return collect_reference_dataset(
        path,
        [_FakeProvider()],
        CollectionConfig(view_count=2, light_count=4),
        created_at="2026-08-24T00:00:00+00:00",
        generator_git_commit="test",
    )


def test_replica_moment_merge_uses_stable_parallel_variance() -> None:
    shape = (1, 2, 3)
    a = ReplicaMoments(np.zeros(shape), np.ones(shape), np.asarray([2], dtype=np.uint32))
    b = ReplicaMoments(np.full(shape, 2.0), np.ones(shape), np.asarray([2], dtype=np.uint32))
    combined = combine_replica_moments(a, b)
    assert combined.mean == pytest.approx(1.0)
    assert combined.variance == pytest.approx(2.0)
    assert combined.sample_count.tolist() == [4]
    assert combined.standard_error == pytest.approx(np.sqrt(0.5))


def test_equal_area_directions_integrate_lambert_response() -> None:
    directions, weights = equal_area_hemisphere(128)
    albedo = np.asarray([0.2, 0.5, 0.9], dtype=np.float64)
    response_cos = albedo[None, :] * directions[:, 2:3] / np.pi
    integrated = np.sum(response_cos * weights[:, None], axis=0)
    assert integrated == pytest.approx(albedo, rel=1e-6, abs=1e-6)


def test_peak_grazing_mixture_is_per_view_and_has_normalized_pdf() -> None:
    views = stratified_view_directions(3)
    directions, weights, pdf = peak_grazing_mixture_query(
        views,
        128,
        full_sphere=False,
        seed=59,
    )
    assert directions.shape == (3, 128, 3)
    assert weights.shape == pdf.shape == (3, 128)
    assert not np.array_equal(directions[0], directions[1])
    np.testing.assert_allclose(weights * pdf, 1.0 / 128.0, rtol=1e-6, atol=1e-7)

    quadrature, solid_angle = equal_area_hemisphere(65536)
    integrated_pdf = np.sum(
        peak_grazing_mixture_pdf(
            quadrature,
            views[1],
            full_sphere=False,
            component_weights=(0.99, 0.01, 0.0),
        ) * solid_angle
    )
    assert integrated_pdf == pytest.approx(1.0, rel=0.02, abs=0.02)

    center = np.asarray((0.2, -0.3, 0.9327379053), dtype=np.float64)
    center /= np.linalg.norm(center)
    folded_integral = np.sum(_folded_vmf_pdf(quadrature, center, 8.0, 1.0) * solid_angle)
    assert folded_integral == pytest.approx(1.0, rel=2e-3, abs=2e-3)


def test_grazing_anchored_views_include_one_boundary_probe() -> None:
    views = grazing_anchored_view_directions(4, max_theta_degrees=89.0, azimuth_offset=0.3)
    assert views.shape == (4, 3)
    assert np.sum(views[:, 2] < np.sin(np.deg2rad(5.0))) == 1
    assert views[-1, 2] == pytest.approx(np.cos(np.deg2rad(89.0)), abs=1e-7)


def test_peak_mixture_accepts_explicit_surface_conditioned_reflection_centers() -> None:
    views = stratified_view_directions(2)
    centers = np.asarray(((0.3, -0.2, 0.9327379), (-0.25, 0.35, 0.9027735)), dtype=np.float64)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    directions, weights, pdf = peak_grazing_mixture_query(
        views,
        1024,
        full_sphere=False,
        seed=73,
        reflection_centers=centers,
    )
    nearest = np.degrees(np.arccos(np.clip(np.max(np.sum(directions * centers[:, None], axis=-1), axis=1), -1.0, 1.0)))
    assert np.all(nearest < 0.5)
    np.testing.assert_allclose(weights * pdf, 1.0 / 1024.0, rtol=1e-6, atol=1e-7)


def test_query_plan_accepts_surface_dependent_direction_tables() -> None:
    views = stratified_view_directions(2)
    lights, weights = equal_area_hemisphere(4)
    per_view = np.stack((lights, np.roll(lights, 1, axis=0)))
    per_surface = np.stack((per_view, np.roll(per_view, 1, axis=2)))
    plan = QueryPlan(
        views,
        per_surface,
        np.broadcast_to(weights, (2, 2, 4)),
        np.full((2, 2, 4), 1.0 / (2.0 * np.pi), dtype=np.float32),
        ("first@1", "second@1"),
        79,
    )
    assert plan.light_directions.shape == (2, 2, 4, 3)
    assert plan.solid_angle_weights.shape == (2, 2, 4)
    assert plan.direction_count == 4


def test_full_sphere_mixture_includes_reflection_transmission_and_grazing() -> None:
    view = stratified_view_directions(1)
    directions, _, pdf = peak_grazing_mixture_query(
        view,
        128,
        full_sphere=True,
        seed=61,
    )
    assert np.any(directions[..., 2] > 0.0)
    assert np.any(directions[..., 2] < 0.0)
    assert np.any(np.abs(directions[..., 2]) < np.sin(np.deg2rad(5.0)))
    assert np.all(pdf > 0.0)


def test_e0_surface_profile_covers_scales_rotations_and_both_uv_seams() -> None:
    samples = uv_surface_samples(20, 1.0 / 4096.0, 71, E0_FOOTPRINT_PROFILE_ID)
    assert len(samples) == 20
    dx = np.asarray([sample.uv_dx for sample in samples])
    dy = np.asarray([sample.uv_dy for sample in samples])
    scales = {(round(float(np.linalg.norm(x)), 12), round(float(np.linalg.norm(y)), 12)) for x, y in zip(dx, dy, strict=True)}
    rotations = {round(float(np.mod(np.arctan2(x[1], x[0]), np.pi)), 10) for x in dx}
    assert len(scales) >= 4
    assert len(rotations) >= 4
    np.testing.assert_allclose(np.sum(dx * dy, axis=1), 0.0, atol=1e-12)
    uv = np.asarray([sample.uv for sample in samples])
    assert np.any(uv[:, 0] < 0.01) and np.any(uv[:, 0] > 0.99)
    assert np.any(uv[:, 1] < 0.01) and np.any(uv[:, 1] > 0.99)


def test_e0_surface_profile_requires_complete_coverage() -> None:
    with pytest.raises(ValueError, match="at least 20 spatial samples"):
        CollectionConfig(
            spatial_sample_count=19,
            surface_profile_id=E0_FOOTPRINT_PROFILE_ID,
        )


def test_e0_query_suite_separates_train_validation_test_and_adversarial_roles() -> None:
    collection = CollectionConfig(
        view_count=2,
        validation_view_count=1,
        test_view_count=1,
        adversarial_view_count=2,
        light_count=32,
        seed=67,
        query_profile_id="ncls.e0-peak-grazing-mixture@2",
    )
    provider = LayerStackProvider(
        collection,
        LayerStackProviderConfig(family_count=3, local_state_count=1),
    )
    plan = provider.query_plan(provider.source_states()[0])
    role_names = [QUERY_ROLE_NAMES[int(role)] for role in plan.query_roles]
    assert role_names == ["train", "train", "validation", "test", "adversarial_probe", "adversarial_probe"]
    assert all("peak" in plan.proposal_id[index] for index in (0, 1, 4, 5))
    assert all("peak" not in plan.proposal_id[index] for index in (2, 3))
    assert len(set(plan.proposal_id)) == 4
    np.testing.assert_allclose(
        plan.proposal_pdf * plan.solid_angle_weights,
        1.0 / collection.light_count,
        rtol=1e-6,
        atol=1e-7,
    )
    all_plans = [provider.query_plan(state) for state in provider.source_states()]
    assert {state.split for state in provider.source_states()} == {0, 1, 2}
    assert len({direction.tobytes() for item in all_plans for direction in item.view_directions}) == 18


def test_layer_stack_provider_wraps_per_view_seeds_to_uint32() -> None:
    collection = CollectionConfig(view_count=8, light_count=4, seed=20260824)
    evaluator = _SeedRecordingEvaluator(collection.light_count)
    provider = LayerStackProvider(
        collection,
        LayerStackProviderConfig(
            family_count=1,
            local_state_count=1,
            samples_per_replica=4,
            query_group_batch=8,
            max_depth=4,
        ),
        evaluator=evaluator,
    )
    state = provider.source_states()[0]
    plan = provider.query_plan(state)
    provider.evaluate(state, provider.surface_samples(state), plan)

    expected = np.asarray(
        [
            (collection.seed ^ int(state.state_id[:8], 16) ^ (index * 0x9E3779B1)) & 0xFFFFFFFF
            for index in range(collection.view_count)
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(evaluator.query_group_seeds, expected)


def test_hdf5_contract_preserves_native_state_queries_and_responses(tmp_path: Path) -> None:
    path = tmp_path / "reference.h5"
    manifest = _collect(path)
    assert manifest.format_name == FORMAT_NAME
    assert manifest.format_version == FORMAT_VERSION
    assert manifest.counts == {"state_count": 3, "query_group_count": 6, "direction_count": 4}

    with ReferenceDataset.open(path) as dataset:
        assert json.loads(dataset.state_payload(1)) == {"value": 2}
        assert {name: len(dataset.group_indices(source_split=name)) for name in ("train", "validation", "test")} == {
            "train": 2,
            "validation": 2,
            "test": 2,
        }
        assert len(dataset.group_indices(query_role="train")) == 3
        assert len(dataset.group_indices(query_role="test")) == 3
        batch = dataset.group_batch((0, 3, 5))
        assert batch["wo"].shape == (3, 3)
        assert batch["wi"].shape == (3, 4, 3)
        assert batch["proposal_pdf"].shape == (3, 4)
        assert batch["mean"].shape == (3, 4, 3)
        np.testing.assert_allclose(batch["standard_error"], 0.0)
        np.testing.assert_allclose(batch["mean"][:, 0, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(batch["rng_seed"][:, 0], [101, 102, 103])
        assert not np.array_equal(dataset.group_batch((0,))["wi"], dataset.group_batch((1,))["wi"])


def test_dataset_cli_and_semantic_hash_validation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "reference.h5"
    _collect(path)
    assert cli_main(["data", "validate", str(path)]) == 0
    assert "ReferenceDataset OK" in capsys.readouterr().out
    with h5py.File(path, "r+") as stream:
        stream["responses/mean"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="semantic content hash mismatch"):
        ReferenceDataset.open(path)


def test_reference_dataset_layout_tracks_runtime_constants() -> None:
    layout_path = PROJECT_ROOT / "src" / "ncls" / "data" / "schemas" / "reference_dataset_v4.layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    assert layout["format_name"] == FORMAT_NAME
    assert layout["format_version"] == FORMAT_VERSION
    assert "payload_blob" in layout["required_groups"]["states"]
    assert layout["legacy_reader"] is False
