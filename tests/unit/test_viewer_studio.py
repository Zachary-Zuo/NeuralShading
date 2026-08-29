from __future__ import annotations

import json
from copy import deepcopy

import pytest

from ncls.viewer import validate_studio


def test_studio_has_two_symmetric_slots_and_no_split():
    value = json.loads(open("configs/viewer-studio-v2.json", encoding="utf-8").read())
    assert len(validate_studio(value)["slots"]) == 2
    assert "split" not in json.dumps(value)


def test_studio_rejects_legacy_shape():
    with pytest.raises(ValueError):
        validate_studio({"format_name": "ncls.viewer-studio", "format_version": 1})


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("scene", "geometry_sha256"), "not-a-hash"),
        (("slots", 0, "package"), "not-an-id"),
        (("camera", "distance"), 0.0),
        (("camera", "vertical_fov_degrees"), float("nan")),
        (("lighting", "environment_intensity"), -1.0),
        (("display", "divider_color"), [0.8, -0.1, 0.8]),
    ),
)
def test_studio_rejects_invalid_runtime_values(path, value):
    document = json.loads(open("configs/viewer-studio-v2.json", encoding="utf-8").read())
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_studio(deepcopy(document))
