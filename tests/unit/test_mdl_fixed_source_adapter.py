from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ncls.core.identity import sha256_file
from ncls.core.source import SourceSnapshot
from ncls.learning.methods.nvidia import METHOD_DEFINITION
from ncls.learning.source_adaptation import (
    MDL_FIXED_PARAMETER_SLOTS,
    encode_mdl_fixed_native_features,
    mdl_fixed_native_feature_layout,
)
from ncls.learning.source_adapters import create_method_source_adapter
from ncls.learning.training.config import TrainingConfig
from ncls.source_materials.mdl import (
    MDL_FAMILY_ID,
    MDL_NATIVE_SCHEMA,
    MdlMaterialSource,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _snapshot(arguments) -> SourceSnapshot:
    module_root = PROJECT_ROOT / "tests/fixtures/mdl"
    module_path = module_root / "constant_diffuse.mdl"
    source = MdlMaterialSource(
        module_root,
        "project.fixtures",
        "1",
        "::constant_diffuse",
        "::constant_diffuse::constant_diffuse(color)",
        arguments,
        "1.7",
    )
    return SourceSnapshot(
        MDL_FAMILY_ID,
        1,
        MDL_NATIVE_SCHEMA,
        sha256_file(module_path),
        source.to_payload(),
        {"constant_diffuse.mdl": sha256_file(module_path)},
        {"module_root": str(module_root)},
        source,
    )


def _arguments():
    return {
        "enabled": {"mdl_type": "bool", "value": True, "editable": True},
        "count": {
            "mdl_type": "int",
            "value": 2,
            "minimum": 0,
            "maximum": 4,
            "editable": True,
        },
        "roughness": {
            "mdl_type": "float",
            "value": 0.25,
            "minimum": 0.0,
            "maximum": 1.0,
            "editable": True,
        },
        "tint": {
            "mdl_type": "color",
            "value": [0.2, 0.4, 0.8],
            "editable": True,
        },
        "scale": {
            "mdl_type": "float3",
            "value": [1.0, 2.0, 3.0],
            "editable": True,
        },
        "mode": {
            "mdl_type": "enum",
            "value": {"name": "second", "value": 1},
            "choices": [
                {"name": "first", "value": 0},
                {"name": "second", "value": 1},
            ],
            "editable": True,
        },
    }


def test_mdl_fixed_uniform_adapter_is_bounded_finite_and_one_by_one() -> None:
    snapshot = _snapshot(_arguments())
    adapter = create_method_source_adapter(
        "nvidia-neural-appearance", (snapshot,), torch.device("cpu")
    )
    tensors, provenance = adapter.sample_tensors(
        torch.zeros(3, dtype=torch.int64), torch.Generator().manual_seed(7)
    )
    layout = mdl_fixed_native_feature_layout()
    assert layout.channel_count == 896
    assert tensors["native_features"].shape == (3, layout.channel_count)
    assert bool(torch.isfinite(tensors["native_features"]).all())
    assert provenance["native_feature_layout_id"] == layout.layout_id
    assert len(provenance["mdl_parameter_schema_identity"]) == 64
    assets = adapter.native_assets()
    domain = assets.descriptors[0].domain("constant")
    assert domain.level_shapes == ((1, 1),)
    assert domain.channel_count == layout.channel_count


def test_mdl_fixed_uniform_adapter_rejects_spatial_or_unbounded_inputs() -> None:
    texture = _arguments()
    texture["image"] = {
        "mdl_type": "texture_2d",
        "value": {"path": "checker.ppm", "effective_gamma": 1.0},
        "editable": True,
    }
    with pytest.raises(ValueError, match="does not support"):
        encode_mdl_fixed_native_features(texture)
    nonfinite = _arguments()
    nonfinite["roughness"] = {
        "mdl_type": "float",
        "value": float("nan"),
        "editable": True,
    }
    with pytest.raises(ValueError, match="finite"):
        encode_mdl_fixed_native_features(nonfinite)
    too_many = {
        f"value-{index}": {"mdl_type": "float", "value": 0.0}
        for index in range(MDL_FIXED_PARAMETER_SLOTS + 1)
    }
    with pytest.raises(ValueError, match="at most"):
        encode_mdl_fixed_native_features(too_many)


def test_mdl_fixed_uniform_adapter_rejects_multiple_snapshots() -> None:
    snapshot = _snapshot(_arguments())
    with pytest.raises(RuntimeError, match="one snapshot"):
        create_method_source_adapter(
            "nvidia-neural-appearance",
            (snapshot, snapshot),
            torch.device("cpu"),
        )


def test_mdl_effect_pigment_smoke_config_matches_adapter_contract() -> None:
    config = TrainingConfig.load(
        PROJECT_ROOT
        / "configs/learning/nvidia-rta2024-mdl-effect-pigment-smoke.json"
    )
    assert config.source_adaptation_id == "nvidia.mdl-fixed-uniform@1"
    assert config.model_context["native_feature_count"] == (
        mdl_fixed_native_feature_layout().channel_count
    )
    assert config.source["family_id"] == "mdl.program@1"
    assert config.source["materials"][0]["locator"]["module"].endswith(
        "::Effect_Pigment_Metallic"
    )
    assert any(
        value.family_id == "mdl.program@1"
        for value in METHOD_DEFINITION.descriptor.supported_sources
    )
