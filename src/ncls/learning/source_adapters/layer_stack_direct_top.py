from __future__ import annotations

from typing import Any, Mapping

import torch

from ncls.core.material import (
    DiffuseInterface,
    MaterialProgram,
    RoughConductorInterface,
    RoughDielectricInterface,
    SheenInterface,
    canonicalize_layer_stack,
)
from ncls.core.representations.legacy_ltc_k2.torch_eval import (
    LegacyLtcK2Tensors,
    eval_direct_top_bsdf,
)


def _top_fields(interface: Any) -> dict[str, Any]:
    alpha = (1.0, 1.0)
    relative_ior = 1.0
    eta = (0.0, 0.0, 0.0)
    k = (0.0, 0.0, 0.0)
    color = (0.0, 0.0, 0.0)
    rotation = 0.0
    if isinstance(interface, RoughDielectricInterface):
        alpha = (interface.alpha_x, interface.alpha_y)
        relative_ior = interface.relative_ior
        rotation = interface.tangent_rotation
    elif isinstance(interface, RoughConductorInterface):
        alpha = (interface.alpha_x, interface.alpha_y)
        eta = interface.eta
        k = interface.k
        rotation = interface.tangent_rotation
    elif isinstance(interface, DiffuseInterface):
        color = interface.color
    elif isinstance(interface, SheenInterface):
        alpha = (interface.roughness, interface.roughness)
        color = interface.color
    else:
        raise TypeError(f"unsupported top interface {type(interface)!r}")
    return {
        "interface_kind": int(interface.kind),
        "alpha": list(alpha),
        "relative_ior": float(relative_ior),
        "eta": list(eta),
        "k": list(k),
        "color": list(color),
        "tangent_rotation": float(rotation),
    }


def fit_direct_top_state(store: Any) -> dict[str, Any]:
    rows = []
    for index in range(store.state_count):
        program = MaterialProgram.from_json(store.state_payload(index).decode("utf-8"))
        stack = canonicalize_layer_stack(program)
        rows.append(_top_fields(stack.interfaces[0]))
    return {
        "contract": "ncls.layer-stack-direct-top@1",
        "state_ids": list(map(str, store.state_strings("state_id").tolist())),
        "rows": rows,
    }


def direct_top_bsdf(
    fitted_state: Mapping[str, Any],
    state_index: torch.Tensor,
    wo: torch.Tensor,
    wi: torch.Tensor,
) -> torch.Tensor:
    rows = fitted_state.get("rows")
    if fitted_state.get("contract") != "ncls.layer-stack-direct-top@1" or not isinstance(rows, list):
        raise ValueError("invalid LayerStack direct-top fitted state")
    device = wo.device
    indices = state_index.long()

    def values(name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(
            [row[name] for row in rows],
            dtype=dtype,
            device=device,
        )[indices]

    group_count = len(indices)
    tensors = LegacyLtcK2Tensors(
        interface_kind=values("interface_kind", torch.long),
        alpha=values("alpha"),
        relative_ior=values("relative_ior"),
        eta=values("eta"),
        k=values("k"),
        color=values("color"),
        tangent_rotation=values("tangent_rotation"),
        amplitude=torch.zeros((group_count, 2, 3), dtype=torch.float32, device=device),
        inverse_scale=torch.ones((group_count, 2, 2), dtype=torch.float32, device=device),
        shear=torch.zeros((group_count, 2, 3), dtype=torch.float32, device=device),
        angle=torch.zeros((group_count, 2), dtype=torch.float32, device=device),
    )
    return eval_direct_top_bsdf(tensors, wo.float(), wi.float())
