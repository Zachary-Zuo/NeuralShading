from __future__ import annotations

from typing import Mapping

import torch

from ncls.core.material import MAX_INTERFACES
from ncls.learning.features import CONTINUOUS_FEATURE_COUNT


ADAPTER_ID = "ncls.layer-stack-source-token-adapter@1"
FEATURE_CONTRACT_ID = "ncls.layer-stack-native-token-source-compiler-input@1"
FEATURE_CONTRACT = {
    "format_name": "ncls.source-feature-contract",
    "format_version": 1,
    "feature_contract_id": FEATURE_CONTRACT_ID,
    "source_adapter_id": ADAPTER_ID,
    "source_family_id": "ncls.layer-stack@1",
    "native_schema_id": "ncls.material-program@1",
    "token_order": "top-interface-to-opaque-base",
    "padded_interface_count": MAX_INTERFACES,
    "inputs": ["interface_kind", "continuous_native_parameters", "interface_count"],
    "continuous_feature_count": CONTINUOUS_FEATURE_COUNT,
    "runtime_role": "offline-source-compilation-only",
}


def layer_stack_source_tensors(
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从 family-specific batch 提取完整 LayerStack 原生 token，不读取 response。"""

    try:
        interface_kinds = batch["interface_kinds"].long()
        continuous = batch["continuous"].float()
        interface_counts = batch["interface_counts"].long()
    except KeyError as error:
        raise ValueError("LayerStack source compiler batch is missing native tokens") from error
    group_count = len(interface_counts)
    if (
        interface_kinds.shape != (group_count, MAX_INTERFACES)
        or continuous.shape
        != (group_count, MAX_INTERFACES, CONTINUOUS_FEATURE_COUNT)
        or torch.any(interface_kinds < 0)
        or torch.any(interface_kinds > 3)
        or torch.any(interface_counts < 1)
        or torch.any(interface_counts > MAX_INTERFACES)
        or not torch.all(torch.isfinite(continuous))
    ):
        raise ValueError("LayerStack source compiler tokens violate the feature contract")
    return interface_kinds, continuous, interface_counts


def repeat_layer_stack_source_tensors(
    batch: Mapping[str, torch.Tensor],
    repeat_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if repeat_count < 1:
        raise ValueError("LayerStack source token repeat_count must be positive")
    values = layer_stack_source_tensors(batch)
    return tuple(value.repeat_interleave(repeat_count, dim=0) for value in values)  # type: ignore[return-value]
