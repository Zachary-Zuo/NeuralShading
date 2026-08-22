import numpy as np
import torch

from baselines.closure_families import evaluate_exported_parameters
from closures.oracle import load_oracle_packets
from closures.packet import BINARY_SIZE, ClosurePacket, LtcResidualLobe, pack_packet, unpack_packet
from closures.torch_eval import (
    decode_ltc_residual,
    eval_direct_top,
    eval_ltc_residual,
    evaluate_closure_packet,
    packets_to_tensors,
)
from datagen.directions import equal_area_hemisphere
from schema import LayerInterface, LayerMedium, LayerStack, LayerType, pack_stack


def _packet(layer: LayerInterface) -> ClosurePacket:
    return ClosurePacket(
        layer,
        (
            LtcResidualLobe((0.7, 0.3, 0.1), (1.2, 0.8), (0.1, -0.2, 0.3), 0.4),
            LtcResidualLobe((0.2, 0.4, 0.6), (0.5, 1.7), (-0.3, 0.2, -0.1), -0.7),
        ),
    )


def test_packet_round_trip_has_fixed_176_byte_layout() -> None:
    original = _packet(
        LayerInterface(
            LayerType.ROUGH_CONDUCTOR,
            0.23,
            0.41,
            eta=(0.2, 0.8, 1.1),
            k=(3.7, 2.4, 1.9),
            tangent_rotation=0.37,
        )
    )
    payload = pack_packet(original)
    restored = unpack_packet(payload)
    assert BINARY_SIZE == len(payload) == 176
    assert restored.direct_top.layer_type == original.direct_top.layer_type
    np.testing.assert_allclose(restored.direct_top.eta, original.direct_top.eta, rtol=1e-6)
    np.testing.assert_allclose(
        restored.residual_lobes[1].shear, original.residual_lobes[1].shear, rtol=1e-6
    )


def test_exported_ltc_semantics_match_oracle_baseline() -> None:
    lights, _ = equal_area_hemisphere(32)
    packet = _packet(LayerInterface(LayerType.DIFFUSE, 1.0, 1.0, albedo=(0.4, 0.2, 0.1)))
    tensors = packets_to_tensors([packet])
    actual = eval_ltc_residual(
        torch.from_numpy(lights[:, :3]),
        tensors.amplitude,
        tensors.inverse_scale,
        tensors.shear,
        tensors.angle,
    ).numpy()
    expected = evaluate_exported_parameters(
        "ltc",
        {
            "amplitude": tensors.amplitude.numpy(),
            "inverse_scale": tensors.inverse_scale.numpy(),
            "shear": tensors.shear.numpy(),
            "angle": tensors.angle.numpy(),
        },
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        lights[:, :3],
        device="cpu",
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


def test_packet_is_exact_direct_top_plus_two_ltc_residuals() -> None:
    lights, _ = equal_area_hemisphere(24)
    packet = _packet(LayerInterface(LayerType.DIFFUSE, 1.0, 1.0, albedo=(0.3, 0.5, 0.7)))
    tensors = packets_to_tensors([packet])
    views = torch.tensor([[0.2, -0.1, np.sqrt(0.95)]], dtype=torch.float32)
    light_tensor = torch.from_numpy(lights[:, :3])
    total = evaluate_closure_packet(tensors, views, light_tensor)
    direct = eval_direct_top(tensors, views, light_tensor)
    residual = eval_ltc_residual(
        light_tensor, tensors.amplitude, tensors.inverse_scale, tensors.shear, tensors.angle
    )
    torch.testing.assert_close(total, direct + residual)
    expected_direct = light_tensor[:, 2:3] * tensors.albedo[0] / np.pi
    torch.testing.assert_close(direct[0], expected_direct, rtol=1e-6, atol=1e-7)


def test_network_decoder_produces_bounded_differentiable_ltc_parameters() -> None:
    raw = torch.linspace(-8.0, 8.0, 36, dtype=torch.float32, requires_grad=True).reshape(2, 18)
    amplitude, inverse_scale, shear, angle = decode_ltc_residual(raw)
    assert amplitude.shape == (2, 2, 3)
    assert inverse_scale.shape == (2, 2, 2)
    assert shear.shape == (2, 2, 3)
    assert angle.shape == (2, 2)
    assert torch.all(amplitude >= 0.0)
    assert torch.all((inverse_scale >= np.exp(-3.0)) & (inverse_scale <= np.exp(3.0)))
    assert torch.all(torch.abs(shear) <= 3.0)
    assert torch.all(torch.abs(angle) <= np.pi)
    (amplitude.sum() + inverse_scale.sum() + shear.sum() + angle.sum()).backward()
    assert raw.grad_fn is not None


def test_oracle_loader_uses_archive_tile_to_state_mapping(tmp_path) -> None:
    first = LayerStack((LayerInterface(LayerType.DIFFUSE, 1.0, 1.0, albedo=(0.1, 0.2, 0.3)),), ())
    second = LayerStack(
        (
            LayerInterface(LayerType.ROUGH_DIELECTRIC, 0.2, 0.3, eta=(1.45, 1.45, 1.45)),
            LayerInterface(LayerType.DIFFUSE, 1.0, 1.0, albedo=(0.8, 0.7, 0.6)),
        ),
        (LayerMedium(),),
    )
    (tmp_path / "stacks.bin").write_bytes(pack_stack(first) + pack_stack(second))
    archive_path = tmp_path / "oracle.npz"
    np.savez(
        archive_path,
        state_indices=np.asarray([1, 0], dtype=np.uint32),
        amplitude=np.ones((2, 2, 3), dtype=np.float32),
        inverse_scale=np.ones((2, 2, 2), dtype=np.float32),
        shear=np.zeros((2, 2, 3), dtype=np.float32),
        angle=np.zeros((2, 2), dtype=np.float32),
    )
    packets = load_oracle_packets(tmp_path, archive_path)
    assert packets[0].direct_top.layer_type == LayerType.ROUGH_DIELECTRIC
    assert packets[1].direct_top.layer_type == LayerType.DIFFUSE
