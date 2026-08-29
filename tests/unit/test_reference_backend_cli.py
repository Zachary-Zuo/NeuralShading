from __future__ import annotations

import json

import ncls.cli as cli
from ncls.references.backend import (
    ReferenceBackendDescriptor,
    ReferenceBackendReport,
    ReferenceCapabilityStatus,
)


class _Backend:
    def doctor(self) -> ReferenceBackendReport:
        descriptor = ReferenceBackendDescriptor(
            "ncls.fixture-reference-backend",
            1,
            "windows-x86_64@1",
            "f" * 40,
            "fixture-slang",
            "d3d12",
            cli.Path("build"),
            cli.Path("python"),
            cli.Path("runtime"),
            "a" * 64,
            "b" * 64,
        )
        return ReferenceBackendReport(
            descriptor,
            (
                ReferenceCapabilityStatus(
                    "fixture", "execution", "ready", "fixture ready"
                ),
            ),
        )


def test_reference_doctor_json_is_generic_and_assets_are_not_managed(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "create_reference_backend", lambda: _Backend())
    assert cli.main(["reference", "doctor", "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["schema_name"] == "ncls.reference-backend-report"
    assert value["ready"] is True
    assert value["assets"] == "not-managed"
    assert value["backend"]["device_api"] == "d3d12"
    assert value["statuses"][0]["requirement_id"] == "fixture"
