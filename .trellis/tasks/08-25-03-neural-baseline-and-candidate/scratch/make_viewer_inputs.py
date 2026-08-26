from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ncls.learning.data import ReferenceCorpusStore
from ncls.learning.pipelines import create_pipeline
from ncls.learning.training.checkpoint import load_checkpoint
from ncls.learning.training.config import TrainingConfig


ROOT = Path(__file__).resolve().parents[4]
STATE_ID = "0d98a60a182c6901b89cf6a4b230e9f70a66db92bdc91d09250c241d49f3b8fd"
OUTPUT = ROOT / "artifacts/exports/unified-scattering-03-viewer-inputs"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ReferenceCorpusStore(ROOT / "artifacts/corpus/layer-stack-p1-v1.json") as store:
        state_ids = tuple(map(str, store.state_strings("state_id")))
        (OUTPUT / "preview-material.json").write_bytes(
            store.state_payload(state_ids.index(STATE_ID))
        )

    work = ROOT / "artifacts/reports/unified-scattering-03/core-frame-native-parity-work"
    values = np.load(work / "fp16_packed_input.npz", allow_pickle=False)
    falcor = np.load(work / "fp16_packed_falcor.npy", allow_pickle=False)
    count = 8
    if not np.all(values["state_index"][:count] == 0):
        raise ValueError("candidate parity prefix does not belong to state zero")
    wo = values["wo"][:count]
    if not np.allclose(wo, wo[0], rtol=0.0, atol=0.0):
        raise ValueError("candidate parity prefix does not share one view")
    wi = values["wi"][:count]
    expected = falcor[:count, :3] * np.maximum(wi[:, 2:3], 0.0)
    manifest = json.loads((
        ROOT / "artifacts/compiled-materials/unified-scattering-03-core-frame-viewer-v1/manifest.json"
    ).read_text(encoding="utf-8"))
    parity = {
        "format_name": "ncls.backend-parity-probe",
        "format_version": 1,
        "architecture_id": "core-frame-neural-v1",
        "compiled_set_id": manifest["compiled_set_id"],
        "compiled_state_id": STATE_ID,
        "view_direction_local": wo[0].tolist(),
        "light_directions_local": wi.tolist(),
        "expected_response_cos": expected.tolist(),
        "tolerance": {"rtol": 2e-5, "atol": 1e-7},
        "precision": "falcor-fp16-packed-core-vs-viewer-contract-wrapper",
    }
    (OUTPUT / "core-frame-parity.json").write_text(
        json.dumps(parity, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )

    baseline_root = ROOT / "artifacts/compiled-materials/unified-scattering-03-nvidia-original-viewer-v1"
    baseline_manifest = json.loads((baseline_root / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(
        ROOT / "artifacts/runs/unified-scattering-03/formal-nvidia-original-seed-20260824/checkpoints/best.pt"
    )
    pipeline = create_pipeline(str(checkpoint["pipeline"]))
    pipeline.load_training_state(checkpoint["fitted_training_state"])
    model = pipeline.create_model(TrainingConfig.from_dict(checkpoint["training_config"]).model)
    model.load_state_dict(checkpoint["model_state"])
    model = model.cuda().eval()
    packed_weights = np.frombuffer(
        (baseline_root / "shared_weights_fp16.bin").read_bytes(), dtype="<f2"
    ).astype(np.float32)
    packed_records = (baseline_root / "compiled_materials.bin").read_bytes()
    latents = np.stack([
        np.frombuffer(packed_records[index * 32 : index * 32 + 16], dtype="<f2").astype(np.float32)
        for index in range(int(baseline_manifest["state_count"]))
    ])
    named_parameters = dict(model.named_parameters())
    with torch.no_grad():
        named_parameters["latent"].copy_(torch.as_tensor(latents, device="cuda"))
        for name, layout in baseline_manifest["parameter_layout"].items():
            offset = int(layout["offset_elements"])
            count_value = int(layout["element_count"])
            value = packed_weights[offset : offset + count_value].reshape(layout["shape"])
            named_parameters[name].copy_(torch.as_tensor(value, device="cuda"))
        state = torch.zeros(1, dtype=torch.long, device="cuda")
        wo_tensor = torch.as_tensor(wo[0:1], dtype=torch.float32, device="cuda")
        wi_tensor = torch.as_tensor(wi[None, :, :], dtype=torch.float32, device="cuda")
        baseline_f = model(state, wo_tensor, wi_tensor)[0].cpu().numpy()
    baseline_expected = baseline_f * np.maximum(wi[:, 2:3], 0.0)
    baseline_parity = {
        "format_name": "ncls.backend-parity-probe",
        "format_version": 1,
        "architecture_id": "nvidia-frame-two-lobe-paper-v1",
        "compiled_set_id": baseline_manifest["compiled_set_id"],
        "compiled_state_id": STATE_ID,
        "view_direction_local": wo[0].tolist(),
        "light_directions_local": wi.tolist(),
        "expected_response_cos": baseline_expected.tolist(),
        # 由 SlangPy FP16 与 Falcor FP16 的实测 envelope 决定；这是实现 parity 容差，不是质量门。
        "tolerance": {"rtol": 3e-3, "atol": 3e-7},
        "precision": "slangpy-fp16-packed-core-vs-viewer-contract-wrapper",
    }
    (OUTPUT / "nvidia-original-parity.json").write_text(
        json.dumps(baseline_parity, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
