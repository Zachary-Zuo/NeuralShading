import json

import numpy as np
import torch

from closures.torch_eval import decode_ltc_residual
from datagen.priors import sample_stacks
from model.compiler import RecurrentCompilerBaseline
from model.features import CONTINUOUS_FEATURE_COUNT, encode_stack, load_stack_feature_table
from model.train import response_loss
from schema import BINARY_SIZE, MAX_LAYERS, pack_stack


def test_stack_features_preserve_type_order_and_medium_presence(tmp_path) -> None:
    stacks = sample_stacks(3, seed=601)
    (tmp_path / "stacks.bin").write_bytes(b"".join(pack_stack(stack) for stack in stacks))
    (tmp_path / "metadata.json").write_text(
        json.dumps({"state_count": len(stacks)}), encoding="utf-8"
    )
    table = load_stack_feature_table(tmp_path)
    assert table.layer_types.shape == (3, MAX_LAYERS)
    assert table.continuous.shape == (3, MAX_LAYERS, CONTINUOUS_FEATURE_COUNT)
    for index, stack in enumerate(stacks):
        expected_types, expected_features, expected_count = encode_stack(stack)
        np.testing.assert_array_equal(table.layer_types[index], expected_types)
        np.testing.assert_allclose(table.continuous[index], expected_features, rtol=1e-6, atol=1e-7)
        assert table.layer_counts[index] == expected_count
    assert (tmp_path / "stacks.bin").stat().st_size == 3 * BINARY_SIZE


def test_compiler_is_order_sensitive_and_outputs_two_ltc_lobes() -> None:
    model = RecurrentCompilerBaseline(width=32)
    with torch.no_grad():
        final = model.head[-1]
        final.weight.normal_(0.0, 0.05)
    types = torch.tensor([[0, 2, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0]])
    features = torch.zeros((2, MAX_LAYERS, CONTINUOUS_FEATURE_COUNT))
    features[0, 0, 0] = features[1, 1, 0] = 0.2
    features[0, 1, 0] = features[1, 0, 0] = 0.8
    count = torch.tensor([2, 2])
    view = torch.tensor([[0.2, 0.0, np.sqrt(0.96)]]).repeat(2, 1).float()
    raw = model(types, features, count, view)
    assert raw.shape == (2, 18)
    assert not torch.allclose(raw[0], raw[1])
    amplitude, inverse_scale, shear, angle = decode_ltc_residual(raw)
    assert amplitude.shape == (2, 2, 3)
    assert inverse_scale.shape == (2, 2, 2)
    assert shear.shape == (2, 2, 3)
    assert angle.shape == (2, 2)


def test_average_vs_average_response_loss_is_finite_and_differentiable() -> None:
    prediction = torch.full((3, 16, 3), 0.2, requires_grad=True)
    mean_a = torch.full_like(prediction, 0.18)
    mean_b = torch.full_like(prediction, 0.22)
    loss = response_loss(prediction, mean_a, mean_b)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.all(torch.isfinite(prediction.grad))
