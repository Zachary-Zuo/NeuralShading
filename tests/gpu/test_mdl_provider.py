from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ncls.data.collector import CollectionConfig, collect_reference_dataset
from ncls.data.dataset import ReferenceDataset
from ncls.data.providers.mdl import MdlAssetSpec, MdlProvider, MdlProviderConfig


@pytest.mark.falcor
def test_mdl_provider_uses_current_falcor_and_preserves_parameter_edit() -> None:
    tint = np.asarray([0.1, 0.4, 0.7], dtype=np.float32)
    provider = MdlProvider(
        CollectionConfig(
            name="mdl-provider-smoke",
            view_count=3,
            light_count=7,
            spatial_sample_count=2,
            proposal="uniform",
        ),
        MdlProviderConfig(
            assets=(
                MdlAssetSpec(
                    "edited-diffuse",
                    "::constant_diffuse",
                    "constant_diffuse",
                    {"tint": tint.tolist()},
                ),
            ),
        ),
    )
    try:
        state = provider.source_states()[0]
        payload = json.loads(state.native_payload.decode("utf-8"))
        np.testing.assert_allclose(payload["arguments"]["tint"]["value"], tint)
        surfaces = provider.surface_samples(state)
        plan = provider.query_plan(state, surfaces)
        result = provider.evaluate(state, surfaces, plan)
        expected = tint * np.maximum(plan.light_directions[..., 2:3], 0.0) / np.pi
        expected = np.broadcast_to(expected[None, ...], result.mean.shape)
        np.testing.assert_allclose(result.mean, expected, rtol=3e-6, atol=3e-7)
        expected_pdf = np.maximum(plan.light_directions[..., 2], 0.0) / np.pi
        expected_pdf = np.broadcast_to(expected_pdf[None, ...], result.reference_pdf.shape)
        np.testing.assert_allclose(result.reference_pdf, expected_pdf, rtol=3e-6, atol=3e-7)
        assert provider.metadata()["formal_executor"].startswith("Falcor 8.0")
        assert provider.metadata()["falcor2_role"] == "external validation oracle only"
    finally:
        provider.close()


@pytest.mark.falcor
def test_mdl_provider_round_trips_through_unified_reference_dataset(tmp_path: Path) -> None:
    collection = CollectionConfig(
        name="mdl-reference-shard-smoke",
        query_role="test",
        view_count=2,
        light_count=4,
        spatial_sample_count=2,
        proposal="uniform",
        seed=29,
    )
    path = tmp_path / "mdl-reference.h5"
    manifest = collect_reference_dataset(
        path,
        (MdlProvider(collection),),
        collection,
        created_at="2026-08-27T00:00:00+00:00",
        generator_git_commit="test",
    )
    assert manifest.format_name == "reference-shard"
    with ReferenceDataset.open(path) as dataset:
        assert dataset.state_count == 1
        assert dataset.direction_count == 4
        assert dataset.manifest.provider_metadata[0]["formal_executor"].startswith("Falcor 8.0")
        assert np.all(np.isfinite(dataset.stream["responses/mean"][...]))
