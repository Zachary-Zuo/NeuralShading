from __future__ import annotations

from pathlib import Path

import torch

from ncls.learning.methods import get_method
from ncls.learning.training import load_checkpoint


ROOT = Path("artifacts/metal-linux-training/long")
EARLY_PATH = ROOT / "checkpoint.step00005000.pt"
PREVIEW_PATH = ROOT / "checkpoint.step00020000.pt"


def main() -> None:
    early = load_checkpoint(EARLY_PATH)
    preview = load_checkpoint(PREVIEW_PATH)
    definition = get_method(preview.method_key)
    print(
        "early_cursor",
        early.global_step,
        early.phase_index,
        early.phase_name,
        early.phase_step,
    )
    print(
        "preview_cursor",
        preview.global_step,
        preview.phase_index,
        preview.phase_name,
        preview.phase_step,
    )
    print("preview_selection_evidence", preview.selection_evidence)
    print("preview_validation_state_keys", sorted(preview.validation_state))
    print("checkpoint_descriptor", preview.method_descriptor_sha256)
    print("checkpoint_implementation", preview.implementation_identity)
    print("runtime_descriptor", definition.descriptor.descriptor_sha256)
    print("runtime_implementation", definition.descriptor.implementation_sha256)

    groups = preview.component_manifest["parameter_groups"]
    for group_name, tensor_names in groups.items():
        changed = 0
        max_abs_delta = 0.0
        missing = []
        for tensor_name in tensor_names:
            if tensor_name not in early.model_state or tensor_name not in preview.model_state:
                missing.append(tensor_name)
                continue
            early_tensor = early.model_state[tensor_name]
            preview_tensor = preview.model_state[tensor_name]
            if not torch.equal(early_tensor, preview_tensor):
                changed += 1
                if early_tensor.is_floating_point():
                    delta = float((preview_tensor - early_tensor).abs().max())
                    max_abs_delta = max(max_abs_delta, delta)
        print(
            "group",
            group_name,
            "tensors",
            len(tensor_names),
            "changed_5k_to_20k",
            changed,
            "max_abs_delta",
            max_abs_delta,
            "coverage",
            preview.gradient_coverage[group_name],
            "missing",
            missing,
        )


if __name__ == "__main__":
    main()
