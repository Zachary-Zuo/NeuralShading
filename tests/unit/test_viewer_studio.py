from __future__ import annotations

import json

import pytest

from ncls.viewer import validate_studio


def test_studio_has_two_symmetric_slots_and_no_split():
    value = json.loads(open("configs/viewer-studio-v2.json", encoding="utf-8").read())
    assert len(validate_studio(value)["slots"]) == 2
    assert "split" not in json.dumps(value)


def test_studio_rejects_legacy_shape():
    with pytest.raises(ValueError):
        validate_studio({"format_name": "ncls.viewer-studio", "format_version": 1})
