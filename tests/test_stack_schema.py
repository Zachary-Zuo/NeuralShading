import math

import pytest

from schema import (
    BINARY_SIZE,
    LayerInterface,
    LayerMedium,
    LayerStack,
    LayerType,
    pack_stack,
    unpack_stack,
)


def make_stack() -> LayerStack:
    return LayerStack(
        layers=(
            LayerInterface(
                layer_type=LayerType.ROUGH_DIELECTRIC,
                roughness_x=0.08,
                roughness_y=0.15,
                eta=(1.5, 1.5, 1.5),
            ),
            LayerInterface(
                layer_type=LayerType.ROUGH_CONDUCTOR,
                roughness_x=0.25,
                roughness_y=0.4,
                eta=(0.2, 0.9, 1.1),
                k=(3.9, 2.5, 2.1),
                albedo=(0.95, 0.7, 0.3),
                tangent_rotation=0.3,
            ),
        ),
        media=(
            LayerMedium(
                sigma_a=(0.1, 0.2, 0.4),
                sigma_s=(0.0, 0.0, 0.0),
                thickness=0.2,
            ),
        ),
    )


def test_binary_round_trip() -> None:
    stack = make_stack()
    payload = pack_stack(stack)
    restored = unpack_stack(payload)

    assert len(payload) == BINARY_SIZE == 752
    assert [layer.layer_type for layer in restored.layers] == [
        LayerType.ROUGH_DIELECTRIC,
        LayerType.ROUGH_CONDUCTOR,
    ]
    assert math.isclose(restored.layers[1].roughness_y, 0.4, rel_tol=1e-6)
    assert restored.media[0].sigma_a == pytest.approx((0.1, 0.2, 0.4))


def test_binary_pack_is_deterministic() -> None:
    stack = make_stack()
    assert pack_stack(stack) == pack_stack(stack)


def test_requires_one_medium_between_each_interface() -> None:
    with pytest.raises(ValueError, match="N-1 media"):
        LayerStack(
            layers=(
                LayerInterface(LayerType.DIFFUSE, 1.0, 1.0),
                LayerInterface(LayerType.ROUGH_DIELECTRIC, 0.2, 0.2),
            ),
            media=(),
        )


def test_rejects_invalid_roughness() -> None:
    with pytest.raises(ValueError, match="roughness"):
        LayerInterface(LayerType.ROUGH_DIELECTRIC, -0.1, 0.2)

