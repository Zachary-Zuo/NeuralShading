from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .collector import CollectionConfig, collect_reference_dataset
from .providers import (
    LayerStackProvider,
    LayerStackProviderConfig,
    MaterialXProvider,
    MaterialXProviderConfig,
    MerlProvider,
    MerlProviderConfig,
    OpenPBRProvider,
    OpenPBRProviderConfig,
)


PROVIDER_IDS = ("layer-stack", "merl", "openpbr", "materialx")


def create_providers(
    provider_ids: Sequence[str],
    collection: CollectionConfig,
    *,
    material_ids: Sequence[str] = (),
    layer_stack: LayerStackProviderConfig = LayerStackProviderConfig(),
):
    requested = tuple(PROVIDER_IDS if "all" in provider_ids else provider_ids)
    if not requested:
        raise ValueError("at least one provider ID is required")
    invalid = sorted(set(requested) - set(PROVIDER_IDS))
    if invalid:
        raise ValueError(f"unknown reference providers: {invalid}")
    selected = tuple(material_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("reference provider IDs must be unique")
    if selected and len(requested) != 1:
        raise ValueError("--material-id can only be used when collecting exactly one provider")
    if selected and requested == ("layer-stack",):
        raise ValueError("--material-id is not defined for the sampled LayerStack provider")
    providers = []
    for provider_id in requested:
        if provider_id == "layer-stack":
            providers.append(LayerStackProvider(collection, layer_stack))
        elif provider_id == "merl":
            providers.append(MerlProvider(collection, MerlProviderConfig(selected)))
        elif provider_id == "openpbr":
            providers.append(OpenPBRProvider(collection, OpenPBRProviderConfig(selected)))
        elif provider_id == "materialx":
            providers.append(MaterialXProvider(collection, MaterialXProviderConfig(selected)))
    return tuple(providers)


def generate_reference_dataset(
    output: Path | str,
    provider_ids: Sequence[str],
    collection: CollectionConfig,
    *,
    material_ids: Sequence[str] = (),
    layer_stack: LayerStackProviderConfig = LayerStackProviderConfig(),
):
    """稳定入口：所有材质族都通过 provider protocol 写入同一 HDF5 合同。"""

    providers = create_providers(
        provider_ids,
        collection,
        material_ids=material_ids,
        layer_stack=layer_stack,
    )
    return collect_reference_dataset(output, providers, collection)
