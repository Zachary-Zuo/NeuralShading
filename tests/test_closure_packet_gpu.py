import numpy as np
import pytest
import torch


falcor = pytest.importorskip("falcor")

from closures.packet import ClosurePacket, LtcResidualLobe
from closures.torch_eval import evaluate_closure_packet, packets_to_tensors
from datagen.directions import equal_area_hemisphere
from schema import LayerInterface, LayerType
from viewer.oracle_lookup import FalcorOracleLookup


pytestmark = pytest.mark.falcor


def _lobe(scale: float, angle: float) -> LtcResidualLobe:
    return LtcResidualLobe(
        (0.3 * scale, 0.2 * scale, 0.1 * scale),
        (0.7 + scale, 1.3 - 0.2 * scale),
        (0.15, -0.12, 0.08),
        angle,
    )


def test_falcor_packet_layout_and_eval_match_pytorch() -> None:
    layers = [
        LayerInterface(LayerType.ROUGH_DIELECTRIC, 0.18, 0.31, eta=(1.52, 1.52, 1.52), tangent_rotation=0.2),
        LayerInterface(LayerType.ROUGH_CONDUCTOR, 0.24, 0.11, eta=(0.2, 0.9, 1.1), k=(3.9, 2.5, 2.1), tangent_rotation=-0.4),
        LayerInterface(LayerType.DIFFUSE, 1.0, 1.0, albedo=(0.7, 0.3, 0.1)),
        LayerInterface(LayerType.SHEEN, 0.45, 0.45, albedo=(0.8, 0.2, 0.1), tangent_rotation=0.7),
    ]
    packets = [ClosurePacket(layer, (_lobe(0.5, 0.3), _lobe(0.9, -0.8))) for layer in layers]
    views = np.asarray(
        [
            [0.2, 0.1, np.sqrt(0.95)],
            [-0.35, 0.05, np.sqrt(0.875)],
            [0.0, 0.0, 1.0],
            [0.45, -0.1, np.sqrt(0.7875)],
        ],
        dtype=np.float32,
    )
    lights, _ = equal_area_hemisphere(32)
    expected = evaluate_closure_packet(
        packets_to_tensors(packets),
        torch.from_numpy(views),
        torch.from_numpy(lights[:, :3]),
    ).numpy()
    evaluator = FalcorOracleLookup(lights, max_packet_batch=len(packets))
    actual = evaluator.evaluate(packets, views)
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-6)
