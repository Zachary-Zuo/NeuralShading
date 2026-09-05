import json
from pathlib import Path

from ncls.core.identity import sha256_json
from ncls.paths import PROJECT_ROOT

path = PROJECT_ROOT / "src/ncls/learning/abi/metal_budgeted_layout_v1.json"
value = json.loads(path.read_text(encoding="utf-8"))
value.pop("identity")
value["schema"] = "ncls.metal-budgeted-layout@2"
value["correspondence"] = "metal-spatial-native-uv-encoder@1"
value["profiles"] = {
    name: {"evaluator_mode": "hybrid", "asset_detail_aggregation": "uv-group-fusion@1",
           "asset_detail_center": "full-native-grid@1", "asset_spatial_features": features,
           "asset_context_resolution_divisor": 4}
    for name, features in (("metal_spatial_hybrid_v1", "semantic-cnn@1"),
                            ("metal_spatial_summary_control_v1", "native-summary-control@1"))
}
value["shape"].update(maximum_uv_groups=9, maximum_native_lookups=3,
                     semantic_decoder_layers=[137, 32, 32, 24], semantic_decoder_macs=6176,
                     proposal_decoder_layers=[80, 16, 13], proposal_decoder_macs=1488)
value["bounded_execution"].update(maximum_prepare_steps=5, maximum_texture_reads=54)
state = value["prepared_state"]
state["total_bytes"] = 176
state["fields"][-1]["offset"] = 160
state["fields"].insert(-1, {"name": "proposal_frame_state", "dtype": "float16", "shape": [8], "offset": 144})
value["asset_packing"]["schema"] = "ncls.metal-spatial-uv-group-asset@1"
value["asset_packing"]["variant_selection"] = "native-uv-group-bindings@1"
value["asset_packing"].pop("scale_bias")
value["directional_features"]["schema"] = "ncls.metal-positive-z-continuous-half-difference@2"
value["proposal"]["schema"] = "ncls.metal-view-independent-proposal-parameters@2"
value["precision"]["asset_planes"] = "rgba8-snorm-per-texel-before-bilinear"
value["identity"] = sha256_json(value)
path.with_name("metal_budgeted_layout_v2.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
