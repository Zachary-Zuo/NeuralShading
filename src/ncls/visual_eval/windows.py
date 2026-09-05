from __future__ import annotations

import json
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from ncls.core.identity import sha256_file
from ncls.viewer.export import prepare_source_reference
from ncls.learning.evaluation_package import compile_evaluation_package
from ncls.learning.training.checkpoint import TrainingCheckpoint
from ncls.paths import PROJECT_ROOT
from .evaluator import VisualContext, VisualResult




class WindowsVisualEvaluator:
    def __init__(self, viewer: Path | None = None) -> None:
        self.viewer = viewer or PROJECT_ROOT / "external/Falcor/build/windows-vs2022/bin/Release/NclsViewer.exe"

    def evaluate(self, model, context: VisualContext) -> VisualResult:
        import subprocess

        if not self.viewer.is_file():
            raise FileNotFoundError(f"缺少 viewer 构建：{self.viewer}")
        started = time.perf_counter()
        output = context.output / f"step-{context.step:08d}-{uuid.uuid4().hex[:6]}"
        output.mkdir(parents=True, exist_ok=True)
        settings = context.settings
        generator = random.Random(settings.seed)
        material_index = generator.randrange(len(context.config.source["materials"]))
        numpy_state, python_state = np.random.get_state(), random.getstate()
        # compiler 可以创建临时模型；恢复全局 RNG，训练数据游标不参与出图。
        try:
            with torch.random.fork_rng():
                # step 是已完成更新数；恰在 phase 边界时归属刚完成的 phase。
                phase_index, phase_step = context.config.locate_step(max(0, context.step - 1))
                checkpoint = TrainingCheckpoint(
                    context.method.key, context.config.to_dict(), context.method.export_training_state(model),
                    global_step=context.step, source_snapshot_ids=context.source_snapshot_ids,
                    phase_index=phase_index,
                    phase_name=context.config.phases[phase_index].name if context.step else "initialization",
                    phase_step=phase_step + 1 if context.step else 0,
                )
                compiled = compile_evaluation_package(checkpoint, output / "package", material_index=material_index)
                source = prepare_source_reference(compiled, output, context.config.source["materials"][material_index]["locator"])
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        geometry = PROJECT_ROOT / "assets/viewer/scenes/studio-v1/shaderball.glb"
        replay = {
            "format_name": "ncls.viewer-capture", "format_version": 4,
            "reference_integrator": "ncls.scene-path-tracer@1",
            "bundle_root": str(compiled.root), "source_material": str(source),
            "reference_geometry": str(geometry), "reference_geometry_sha256": sha256_file(geometry),
            "slots": [
                {"package_id": "source-reference", "mode": "path-tracing", "target_spp": settings.reference_spp},
                {"package_id": compiled.manifest.package_id, "mode": settings.neural_mode, "target_spp": settings.neural_spp},
            ],
            "comparison_purpose": "training-diagnostic",
            "resolution": [settings.width, settings.height], "reference_spp": settings.reference_spp,
            "reference_samples_per_frame": 16,
            "camera": {"target": [-0.05485052, 1.04786098, -0.06448951], "yaw": -0.65, "pitch": 0.25,
                       "distance": 4.2, "vertical_fov_degrees": 38.0},
            "display": {"comparison_mode": 0, "exposure_ev": 0.0, "difference_scale": 8.0},
            "lighting": {
                "use_environment": True, "environment_rotation": 0.0, "environment_intensity": 1.0,
                "use_sun": True, "sun_direction": [0.36514837, 0.54772256, 0.73029673],
                "sun_intensity": 1.0, "sun_color": [1.0, 1.0, 1.0],
                "use_point": False, "point_position": [2.0, 3.0, 2.0],
                "point_intensity": 1.0, "point_color": [1.0, 1.0, 1.0],
                "use_rectangle": False, "rectangle_center": [0.0, 3.0, 0.0],
                "rectangle_axis_u": [1.0, 0.0, 0.0], "rectangle_axis_v": [0.0, 0.0, 1.0],
                "rectangle_intensity": 1.0, "rectangle_color": [1.0, 1.0, 1.0],
            },
        }
        replay_path, capture = output / "replay.json", output / "capture.json"
        replay_path.write_text(json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")
        subprocess.run(
            [str(self.viewer), "--replay", str(replay_path), "--headless", "--capture", str(capture)],
            cwd=self.viewer.parent, check=True,
        )
        document = json.loads(capture.read_text(encoding="utf-8"))
        for slot in document["slots"]:
            if slot["status"] != "ready":
                raise RuntimeError(f"图像评估 slot {slot['slot_index']}：{slot['diagnostic']}")
        files = document["files"]
        return VisualResult({"comparison": output / files["display"], "difference": output / files["difference_display"]}, time.perf_counter() - started)
