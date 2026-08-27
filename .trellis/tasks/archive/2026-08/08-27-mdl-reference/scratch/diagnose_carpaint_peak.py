from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from ncls.data.collector import CollectionConfig
from ncls.data.providers.mdl import MdlAssetSpec, MdlProvider, MdlProviderConfig
from ncls.paths import PROJECT_ROOT


sys.path.insert(0, str(PROJECT_ROOT / "tools/reference"))
from mdl_oracle.protocol import canonical_json  # noqa: E402
from mdl_parity import _formal_result, _make_request, _run_oracle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--roughness-variation", type=float)
    parser.add_argument("--flake-layer-visibility", type=float)
    arguments = parser.parse_args()

    base = MdlProviderConfig.from_vmaterials2(("carpaint-shifting-flakes",))
    original = base.assets[0]
    overrides = {
        name: value
        for name, value in {
            "roughness_variation": arguments.roughness_variation,
            "flake_layer_visibility": arguments.flake_layer_visibility,
        }.items()
        if value is not None
    }
    config = MdlProviderConfig(
        module_root=base.module_root,
        assets=(
            MdlAssetSpec(
                original.asset_id,
                original.module,
                original.material,
                overrides,
                original.pack_id,
                original.pack_version,
            ),
        ),
    )
    provider = MdlProvider(
        CollectionConfig(
            name="mdl-carpaint-peak-diagnostic",
            view_count=1,
            light_count=1,
            spatial_sample_count=1,
            footprint_width=0.0,
            seed=0x4D444C32,
        ),
        config,
    )
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        request = _make_request(provider, config, "formal")
        query = request["query"]
        query["query_set"] = "carpaint-peak-diagnostic"
        query["surfaces"] = [query["surfaces"][2]]
        query["view_directions"] = [query["view_directions"][0]]
        query["light_directions"] = [query["light_directions"][0]]
        request_path = output / "request.json"
        request_path.write_bytes(canonical_json(request))
        formal_value, formal_pdf = _formal_result(provider, request)
        np.savez(output / "formal.npz", value=formal_value, pdf=formal_pdf)
        _run_oracle(request_path, output / "oracle")
        oracle = np.load(output / "oracle/result.npz")
        value = np.asarray(oracle["value"], dtype=np.float32)
        pdf = np.asarray(oracle["pdf"], dtype=np.float32)
        report = {
            "overrides": overrides,
            "formal_value": formal_value.tolist(),
            "oracle_value": value.tolist(),
            "formal_pdf": formal_pdf.tolist(),
            "oracle_pdf": pdf.tolist(),
            "max_absolute_value_error": float(np.max(np.abs(formal_value - value))),
            "max_absolute_pdf_error": float(np.max(np.abs(formal_pdf - pdf))),
        }
        (output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    finally:
        provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
