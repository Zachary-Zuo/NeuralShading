from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from ncls.learning.methods.nvidia import METHOD_DEFINITION
from ncls.data.native_features import DenseNativeFeaturePyramid
from ncls.bundle.typed_texture import inspect_rgba16f_dds
from ncls.core.source import SourceSnapshot


def _formal_config() -> dict:
    return {
        "format_name": "ncls.training-config",
        "format_version": 2,
        "method_key": "nvidia-neural-appearance",
        "run_class": "formal",
        "correspondence_id": "nvidia-rta2024-functional@1",
        "recipe_id": "nvidia-rta2024-materialx-formal-300k-stage100k@1",
        "source_adaptation_id": "materialx-standard-surface-spatial@1",
        "batch_source": {"kind": "live", "options": {
            "materialx_asset_id": "american_walnut_veneer", "query_tile_size": 262_144,
        }},
        "model_context": {
            "native_feature_count": 38,
            "latent_width": 4096,
            "latent_height": 4096,
            "latent_mip_count": 13,
        },
        "lifecycle": {"total_steps": 300_000, "materialization_step": 100_000},
        "routes": [
            {
                "name": "evaluator", "batch_size": 65_000, "direction_count": 1,
                "query_role": 0, "seed_offset": 0,
                "options": {
                    "direction_proposal": "uniform-half-difference@1",
                    "target_estimator": "reference",
                    "mip_exponential_scale": 1.0,
                    "spatial_samples_per_texel_area": 1.0,
                    "maximum_spatial_samples": 64,
                },
            },
            {
                "name": "sampler", "batch_size": 65_000, "direction_count": 1,
                "query_role": 1, "seed_offset": 1,
                "options": {
                    "direction_proposal": "uniform-hemisphere-conditioning@1",
                    "target_estimator": "learned-sampler",
                    "mip_exponential_scale": 1.0,
                    "spatial_samples_per_texel_area": 1.0,
                    "maximum_spatial_samples": 64,
                },
            },
        ],
        "seed": 20260827,
        "device": "cuda:0",
        "optimizer": {
            "kind": "adam", "betas": [0.9, 0.999], "epsilon": 1e-7,
            "weight_decay": 0.0,
        },
        "schedule": {
            "kind": "cosine", "start": 1e-3, "end": 1e-4,
            "total_steps": 300_000,
        },
        "mollification": {"steps": 20_000, "start_degrees": 10.0, "samples": 256},
        "filtering": {
            "latent_fetch": "discrete-mip-bilinear-wrap@1",
            "mip_distribution": "truncated-exponential-scale-1@1",
            "spatial_filter": "gaussian-area-proportional-cap-64@1",
            "normal_filter": "lean-first-second-moments@1",
        },
        "loss": {
            "evaluator": "log1p-l1@1",
            "sampler": "learned-sampler-forward-kl-score@1",
        },
        "validation": {"interval": 50_000, "batches": 1},
        "checkpoint_selection": "tail_guard",
    }


def test_nvidia_formal_recipe_rejects_budget_adaptations() -> None:
    config = _formal_config()
    METHOD_DEFINITION.validate_training_config(config)
    for path, replacement in (
        (("lifecycle", "total_steps"), 25_000),
        (("routes", 0, "batch_size"), 16),
        (("optimizer", "epsilon"), 1e-8),
        (("mollification", "samples"), 1),
    ):
        changed = deepcopy(config)
        parent = changed
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = replacement
        with pytest.raises(ValueError):
            METHOD_DEFINITION.validate_training_config(changed)


def test_nvidia_encoder_materialization_and_checkpoint_state_roundtrip() -> None:
    context = {
        "native_feature_count": 5,
        "latent_width": 2,
        "latent_height": 2,
        "latent_mip_count": 2,
    }
    model = METHOD_DEFINITION.create_trainable(context)
    features = (torch.randn(2, 2, 5), torch.randn(1, 1, 5))
    expected = tuple(
        model.encode(level.reshape(-1, 5)).reshape(*level.shape[:2], 8)
        for level in features
    )
    METHOD_DEFINITION.materialize_latent(model, DenseNativeFeaturePyramid(features))
    assert model.lifecycle_stage == "finetune"
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters()
               if name.startswith("encoder_"))
    assert all(parameter.requires_grad for parameter in model.latent_levels)
    for actual, encoded in zip(model.latent_levels, expected, strict=True):
        assert torch.equal(actual.permute(1, 2, 0), encoded)

    state = METHOD_DEFINITION.export_training_state(model)
    assert state["latent_mip_offsets"].tolist() == [0, 4, 5]
    restored = METHOD_DEFINITION.create_trainable(context)
    METHOD_DEFINITION.restore_training_state(restored, state)
    restored_state = METHOD_DEFINITION.export_training_state(restored)
    assert set(restored_state) == set(state)
    assert all(torch.equal(restored_state[name], state[name]) for name in state)


def test_formal_sampler_shader_has_only_two_lobes_and_2d_random() -> None:
    source = Path("shaders/ncls/scattering/nvidia_proposal.slang")
    text = source.read_text(encoding="utf-8")
    assert "SAFETY_WEIGHT" not in text
    assert "float3 u" not in text
    assert "float2 u" in text
    assert "dot(params.learnedWeights, componentPdfs)" in text
    assert "u.x / weightSpecular" in text
    assert "(u.x - weightSpecular) / weightDiffuse" in text


def test_nvidia_material_compiler_emits_two_full_rgba16f_mip_chains() -> None:
    context = {
        "native_feature_count": 2,
        "latent_width": 2,
        "latent_height": 2,
        "latent_mip_count": 2,
    }
    model = METHOD_DEFINITION.create_trainable(context)
    METHOD_DEFINITION.materialize_latent(
        model,
        DenseNativeFeaturePyramid((torch.randn(2, 2, 2), torch.randn(1, 1, 2))),
    )
    snapshot = SourceSnapshot("ncls.layer-stack@1", 1, "fixture", "a" * 64, b"{}")
    material = METHOD_DEFINITION.compile_material(
        snapshot,
        {
            "source_state_ids": [snapshot.snapshot_id],
            "model_state": METHOD_DEFINITION.export_training_state(model),
        },
    )
    assert set(material.resources) == {
        "latent0.dds", "latent1.dds", "latent-sampler.json"
    }
    assert all(inspect_rgba16f_dds(material.resources[name]) == (2, 2, 2)
               for name in ("latent0.dds", "latent1.dds"))
    assert len(material.blobs["compiled-material"]) == 32


def test_nvidia_package_freezes_deterministic_fp16_runtime_parity() -> None:
    torch.manual_seed(17)
    context = {
        "native_feature_count": 2,
        "latent_width": 2,
        "latent_height": 2,
        "latent_mip_count": 2,
    }
    model = METHOD_DEFINITION.create_trainable(context)
    METHOD_DEFINITION.materialize_latent(
        model,
        DenseNativeFeaturePyramid((torch.randn(2, 2, 2), torch.randn(1, 1, 2))),
    )
    snapshot = SourceSnapshot("ncls.layer-stack@1", 1, "fixture", "a" * 64, b"{}")
    checkpoint = {
        "source_state_ids": [snapshot.snapshot_id],
        "model_state": METHOD_DEFINITION.export_training_state(model),
    }
    first = METHOD_DEFINITION.package_validation(snapshot, checkpoint)
    second = METHOD_DEFINITION.package_validation(snapshot, checkpoint)

    assert first == second
    assert first["status"] == "gpu-parity-required"
    parity = first["parity"]
    assert parity["oracle"] == "nvidia-rta2024-packed-fp16-cpu-emulation@1"
    assert parity["view"] == [0.0, 0.0, 1.0]
    assert len(parity["lights"]) == len(parity["expected_response_cos"]) == 4
    assert parity["relative_tolerance"] == 2e-2
    assert parity["absolute_tolerance"] == 2e-4
    values = torch.tensor(parity["expected_response_cos"])
    assert values.shape == (4, 3)
    assert torch.isfinite(values).all() and torch.all(values >= 0.0)


def test_nvidia_deployment_shader_uses_regular_fp16_mlp_path() -> None:
    wrapper = Path(
        "shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance.slang"
    ).read_text(encoding="utf-8")
    fp16 = Path(
        "shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_fp16.slang"
    ).read_text(encoding="utf-8")
    assert "nclsNvidiaNeuralPrepareFp16" in wrapper
    assert "nclsNvidiaNeuralEvaluateFFp16" in wrapper
    assert "half value" in fp16
    assert "half input[NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH]" in fp16
    assert "half(exp(float(output[0]) - 3.0f))" in fp16
