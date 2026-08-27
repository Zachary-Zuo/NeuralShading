from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from ncls.data.batch_sources import MaterialXLiveReferenceBatchSource
from ncls.data.native_features import (
    MaterialXNativeFeaturePyramid,
    materialx_native_feature_layout,
)
from ncls.learning.objectives import sampler_forward_kl_score


def _inputs() -> np.ndarray:
    values = np.zeros(24, dtype=np.float32)
    values[1:4] = (0.25, 0.5, 0.75)
    values[5] = 0.2
    values[12] = 0.4
    values[17] = 1.0
    return values


def test_materialx_native_feature_pyramid_keeps_constants_compact() -> None:
    pyramid = MaterialXNativeFeaturePyramid.from_textures(
        _inputs(), base_color=None, roughness=None, metalness=None, normal=None
    )
    assert pyramid.feature_count == materialx_native_feature_layout().channel_count == 38
    assert pyramid.level_shapes == ((1, 1),)
    sampled = pyramid.sample(
        np.asarray([[0.1, 0.9]], dtype=np.float32), np.asarray([0.0], dtype=np.float32)
    )
    assert np.allclose(sampled[0, 24:27], (0.25, 0.5, 0.75), atol=2e-4)
    assert np.allclose(sampled[0, 29:32], (0.0, 0.0, 1.0), atol=2e-4)
    assert np.allclose(sampled[0, 32:38], (0.0, 0.0, 0.0, 0.0, 0.0, 1.0), atol=2e-4)


def test_materialx_base_color_is_decoded_before_filtered_mips(tmp_path) -> None:
    pixels = np.asarray(
        [[[0, 0, 0], [255, 255, 255]], [[255, 255, 255], [0, 0, 0]]],
        dtype=np.uint8,
    )
    path = tmp_path / "base.png"
    Image.fromarray(pixels, "RGB").save(path)
    pyramid = MaterialXNativeFeaturePyramid.from_textures(
        _inputs(), base_color=path, roughness=None, metalness=None, normal=None
    )
    assert pyramid.level_shapes == ((2, 2), (1, 1))
    coarse = pyramid.sample(
        np.asarray([[0.5, 0.5]], dtype=np.float32), np.asarray([1.0], dtype=np.float32)
    )
    assert np.allclose(coarse[0, 24:27], 0.5, atol=2e-3)


def test_uniform_half_difference_mapping_and_jacobian_are_finite() -> None:
    rng = np.random.default_rng(29)
    views, lights, pdf, weight = (
        MaterialXLiveReferenceBatchSource._half_difference_directions(4096, rng)
    )
    assert np.all(views[:, 2] > 0.0) and np.all(lights[:, 2] > 0.0)
    half = views + lights
    half /= np.linalg.norm(half, axis=1, keepdims=True)
    assert np.all(half[:, 2] > 0.0)
    assert np.all(np.isfinite(pdf)) and np.all(pdf > 0.0)
    assert np.allclose(pdf * weight, 1.0, rtol=2e-6, atol=2e-6)


def test_sampler_forward_kl_uses_learned_sampler_score_estimator() -> None:
    evaluator = torch.tensor([[[100.0, 100.0, 100.0]], [[1.0, 1.0, 1.0]]])
    valid = torch.ones(2, 1, dtype=torch.bool)
    preferred, preferred_valid = sampler_forward_kl_score(
        evaluator, torch.tensor([[0.9], [0.1]]), valid
    )
    reversed_pdf, _ = sampler_forward_kl_score(
        evaluator, torch.tensor([[0.1], [0.9]]), valid
    )
    assert preferred < reversed_pdf
    assert preferred_valid == 1.0


def test_materialx_live_source_closes_one_falcor_frame_after_all_routes_release() -> None:
    frame_count = 0

    def end_frame() -> None:
        nonlocal frame_count
        frame_count += 1

    source = MaterialXLiveReferenceBatchSource.__new__(
        MaterialXLiveReferenceBatchSource
    )
    evaluator = SimpleNamespace(route_name="evaluator", slot_index=0)
    sampler = SimpleNamespace(route_name="sampler", slot_index=-1)
    source._active_leases = {"evaluator": evaluator, "sampler": sampler}
    source._free_slots = []
    source._runtime = SimpleNamespace(device=SimpleNamespace(end_frame=end_frame))

    source._release(evaluator)
    assert frame_count == 0
    assert source._free_slots == [0]
    source._release(sampler)
    assert frame_count == 1
