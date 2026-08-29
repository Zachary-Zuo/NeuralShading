from dataclasses import replace

import pytest
import torch

from ncls.learning.batches import (
    AssetTileBatch,
    EvaluatorBatch,
    MethodSamplerBatch,
    TrainingConditioning,
    TrainingRouteRequest,
)
from ncls.learning.source_adaptation import DenseNativeAssetCollection, NativeAssetRole


def _conditioning(device: str = "cpu") -> TrainingConditioning:
    return TrainingConditioning(
        "fixture.family@1",
        ("a" * 64, "b" * 64),
        {
            "source_index": torch.zeros(2, dtype=torch.int64, device=device),
            "wo": torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device
            ),
        },
        {"source": "test"},
    )


def test_typed_batches_only_require_semantic_route_tensors() -> None:
    conditioning = _conditioning()
    evaluator = EvaluatorBatch(
        conditioning,
        torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]]),
        torch.ones((2, 1, 3)),
    )
    sampler = MethodSamplerBatch(conditioning, torch.full((2, 2), 0.5))

    assert set(evaluator.tensors) == {"source_index", "wo", "wi", "target_f"}
    assert set(sampler.tensors) == {"source_index", "wo", "sample_u"}
    assert "target" not in sampler.tensors
    assert "wi" not in sampler.tensors


def test_evaluator_batch_rejects_non_f_target_shape() -> None:
    conditioning = _conditioning()
    wi = torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]])
    with pytest.raises(ValueError, match="target_f"):
        EvaluatorBatch(conditioning, wi, torch.ones((2, 2, 3)))


def test_method_sampler_batch_rejects_dummy_target_by_construction() -> None:
    conditioning = _conditioning()
    with pytest.raises(TypeError):
        MethodSamplerBatch(
            conditioning,
            torch.ones((2, 2)),
            target_f=torch.zeros(2, 1, 3),  # type: ignore[call-arg]
        )


def test_training_route_request_uses_explicit_kind() -> None:
    request = TrainingRouteRequest(
        "evaluator", "reference-evaluator", 2, 1, 0, 17, {}
    )
    assert request.kind == "reference-evaluator"
    with pytest.raises(ValueError, match="kind"):
        TrainingRouteRequest("legacy", "unknown", 2, 1, 0, 17, {})  # type: ignore[arg-type]


def test_native_asset_collection_tiles_multiple_assets_with_halo() -> None:
    collection = DenseNativeAssetCollection(
        (
            (torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),),
            (torch.ones(1, 2, 4),),
        ),
        ("asset-a", "asset-b"),
        "fixture-layout",
        "surface",
        "uv0",
        "wrap",
        (
            NativeAssetRole("color", "base-color", 0, 3, "linear", "box"),
            NativeAssetRole("mask", "coverage", 3, 1, "linear", "box"),
        ),
    )
    requests = tuple(collection.iter_tile_requests(0, "surface", 4, 1))
    tiles = tuple(collection.acquire_tile(request, torch.device("cpu")) for request in requests)
    assert len(tiles) == 2
    assert all(tile.values.shape[2] == 4 and tile.halo == 1 for tile in tiles)
    assert all(tile.role_values("color").shape[2] == 3 for tile in tiles)
    batch = AssetTileBatch(
        collection.descriptors,
        tiles,
        {"native_asset_collection_identity": collection.collection_id},
    )
    assert batch.device.type == "cpu" and len(batch.tensors) == 4
    batch.release()


def test_native_asset_collection_identity_covers_schema_domain_and_roles() -> None:
    assets = ((torch.zeros(1, 1, 2),),)
    roles = (NativeAssetRole("value", "fixture", 0, 2, "linear", "box"),)
    baseline = DenseNativeAssetCollection(
        assets, ("asset",), "schema-a", "surface", "uv0", "wrap", roles
    )
    schema_variant = DenseNativeAssetCollection(
        assets, ("asset",), "schema-b", "surface", "uv0", "wrap", roles
    )
    domain_variant = DenseNativeAssetCollection(
        assets, ("asset",), "schema-a", "volume", "object", "clamp", roles
    )
    role_variant = DenseNativeAssetCollection(
        assets,
        ("asset",),
        "schema-a",
        "surface",
        "uv0",
        "wrap",
        (NativeAssetRole("value", "another-semantic", 0, 2, "linear", "box"),),
    )

    assert len(
        {
            baseline.collection_id,
            schema_variant.collection_id,
            domain_variant.collection_id,
            role_variant.collection_id,
        }
    ) == 4


def test_native_asset_working_set_requires_explicit_release_before_eviction() -> None:
    collection = DenseNativeAssetCollection(
        (
            (torch.zeros(1, 1, 1),),
            (torch.ones(1, 1, 1),),
        ),
        ("asset-a", "asset-b"),
        "fixture-schema",
        "constant",
        "constant",
        "clamp",
        (NativeAssetRole("value", "fixture", 0, 1, "linear", "constant"),),
        working_set_capacity=1,
    )
    first_request = next(collection.iter_tile_requests(0, "constant", 1, 0))
    second_request = next(collection.iter_tile_requests(1, "constant", 1, 0))
    first_tile = collection.acquire_tile(first_request, torch.device("cpu"))

    with pytest.raises(RuntimeError, match="fully leased"):
        collection.acquire_tile(second_request, torch.device("cpu"))

    first_tile.release()
    second_tile = collection.acquire_tile(second_request, torch.device("cpu"))
    assert torch.equal(second_tile.core, torch.ones(1, 1, 1))
    second_tile.release()


def test_native_asset_collection_rejects_tile_outside_declared_mip() -> None:
    collection = DenseNativeAssetCollection(
        ((torch.zeros(2, 2, 1),),),
        ("asset",),
        "fixture-schema",
        "surface",
        "uv0",
        "wrap",
        (NativeAssetRole("value", "fixture", 0, 1, "linear", "box"),),
    )
    request = next(collection.iter_tile_requests(0, "surface", 4, 0))
    with pytest.raises(ValueError, match="exceeds its mip extent"):
        collection.acquire_tile(
            replace(request, core_shape=(3, 2)), torch.device("cpu")
        )
