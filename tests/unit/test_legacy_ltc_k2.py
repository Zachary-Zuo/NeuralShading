from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ncls.core.material import RoughDielectricInterface
from ncls.core.representations.legacy_ltc_k2 import (
    BINARY_SIZE,
    DESCRIPTOR,
    LegacyLtcK2Lobe,
    LegacyLtcK2State,
    backend_descriptor,
    pack_state,
    unpack_state,
)
from ncls.core.representations.legacy_ltc_k2.torch_eval import (
    evaluate_state_bsdf,
    evaluate_state_response_cos,
    states_to_tensors,
)


def make_state() -> LegacyLtcK2State:
    return LegacyLtcK2State(
        RoughDielectricInterface(0.12, 0.08, 1.5, 0.25),
        (
            LegacyLtcK2Lobe((0.2, 0.4, 0.6), (1.2, 0.8), (0.1, -0.2, 0.3), 0.4),
            LegacyLtcK2Lobe((0.1, 0.05, 0.2), (0.7, 1.4), (-0.1, 0.2, -0.3), -0.6),
        ),
    )


def test_legacy_ltc_k2_state_round_trip_is_backend_private() -> None:
    state = make_state()
    payload = pack_state(state)
    assert len(payload) == BINARY_SIZE == DESCRIPTOR.state_bytes == 176
    restored = unpack_state(payload)
    assert pack_state(restored) == payload
    assert restored.direct_top.relative_ior == pytest.approx(1.5)  # type: ignore[union-attr]


def test_legacy_ltc_k2_descriptor_implements_complete_scattering_contract() -> None:
    descriptor = backend_descriptor()
    assert descriptor.backend_id == "legacy-ltc-k2"
    assert descriptor.is_complete_realtime_backend
    assert descriptor.state_stride == BINARY_SIZE


def test_response_and_pure_bsdf_differ_by_exactly_one_light_cosine() -> None:
    tensors = states_to_tensors([make_state()])
    view = torch.tensor([[0.2, 0.1, math.sqrt(0.95)]], dtype=torch.float32)
    lights = torch.tensor([[0.6, 0.0, 0.8], [0.0, 0.0, 1.0]], dtype=torch.float32)
    response = evaluate_state_response_cos(tensors, view, lights)
    bsdf = evaluate_state_bsdf(tensors, view, lights)
    np.testing.assert_allclose(
        response.numpy(),
        (bsdf * lights[None, :, 2:3]).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
    assert torch.all(response >= 0.0)
