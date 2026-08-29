from __future__ import annotations

import pytest

from ncls.references.query import _texture_extent


@pytest.mark.parametrize(
    ("kind", "shape", "expected"),
    (
        ("texture2d", (64, 65), {"width": 65, "height": 64}),
        ("texture2d", (64, 65, 4), {"width": 65, "height": 64}),
        (
            "texture3d",
            (33, 64, 65),
            {"width": 65, "height": 64, "depth": 33},
        ),
        (
            "texture3d",
            (33, 64, 65, 4),
            {"width": 65, "height": 64, "depth": 33},
        ),
    ),
)
def test_typed_texture_extent_uses_leading_spatial_axes(
    kind: str, shape: tuple[int, ...], expected: dict[str, int]
) -> None:
    assert _texture_extent(kind, shape) == expected


def test_typed_texture_extent_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="invalid shape"):
        _texture_extent("texture3d", (64, 65))
