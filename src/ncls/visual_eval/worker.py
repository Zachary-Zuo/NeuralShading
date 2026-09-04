from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Callable, Mapping, Protocol

from ncls.core.identity import sha256_file
from ncls.learning.evaluation_package import compile_evaluation_package
from ncls.learning.training import EvaluationSnapshot, load_evaluation_snapshot
from ncls.paths import PROJECT_ROOT

from .contracts import VisualArtifact, VisualEvalRequest, VisualEvalResult, VisualEvalStatus
from .spool import VisualEvalSpool


@dataclass(frozen=True)
class ViewerPackage:
    bundle_root: Path
    package_id: str
    source_material: Path


@dataclass(frozen=True)
class ViewerCapture:
    reference: Path
    neural: Path
    difference: Path
    display: Path
    manifest: Path


class VisualEvalExecutor(Protocol):
    def __call__(
        self,
        request: VisualEvalRequest,
        snapshot: EvaluationSnapshot,
        output_root: Path,
    ) -> ViewerCapture: ...


def default_windows_viewer_path() -> Path:
    return (
        PROJECT_ROOT
        / "external/Falcor/build/windows-vs2022/bin/Release/NclsViewer.exe"
    ).resolve()


class WindowsViewerExecutor:
    """Thin Windows/D3D12 executor; package creation is method deployment work."""

    def __init__(
        self,
        viewer: Path | str | None = None,
        package_builder: Callable[
            [VisualEvalRequest, EvaluationSnapshot, Path], ViewerPackage
        ] | None = None,
    ) -> None:
        self.viewer = (
            default_windows_viewer_path()
            if viewer is None
            else Path(viewer).resolve()
        )
        self.package_builder = package_builder or build_viewer_package

    @staticmethod
    def _write_atomic(path: Path, value: Mapping[str, object]) -> None:
        descriptor, name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _lighting(value: Mapping[str, object]) -> dict[str, object]:
        return {
            "use_environment": True,
            "environment_rotation": float(value.get("environment_rotation", 0.0)),
            "environment_intensity": float(value.get("environment_intensity", 1.0)),
            "use_sun": True,
            "sun_direction": [0.36514837, 0.54772256, 0.73029673],
            "sun_intensity": 1.0,
            "sun_color": [1.0, 1.0, 1.0],
            "use_point": False,
            "point_position": [0.0, 0.0, 0.0],
            "point_intensity": 0.0,
            "point_color": [1.0, 1.0, 1.0],
            "use_rectangle": False,
            "rectangle_center": [0.0, 0.0, 0.0],
            "rectangle_axis_u": [1.0, 0.0, 0.0],
            "rectangle_axis_v": [0.0, 0.0, 1.0],
            "rectangle_intensity": 0.0,
            "rectangle_color": [1.0, 1.0, 1.0],
        }

    def __call__(
        self,
        request: VisualEvalRequest,
        snapshot: EvaluationSnapshot,
        output_root: Path,
    ) -> ViewerCapture:
        if os.name != "nt":
            raise RuntimeError("visual eval viewer execution is Windows/D3D12 only")
        if not self.viewer.is_file():
            raise FileNotFoundError(f"visual eval viewer executable is missing: {self.viewer}")
        package = self.package_builder(request, snapshot, output_root / "deployment")
        output_root.mkdir(parents=True, exist_ok=True)
        resolution = request.renderer.get("resolution")
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            raise ValueError("visual eval renderer resolution is invalid")
        replay = output_root / "replay.json"
        geometry = (
            PROJECT_ROOT / "assets/viewer/scenes/studio-v1/shaderball.glb"
        ).resolve()
        if not geometry.is_file():
            raise FileNotFoundError(
                f"visual eval reference geometry is missing: {geometry}"
            )
        self._write_atomic(
            replay,
            {
                "format_name": "ncls.viewer-capture",
                "format_version": 4,
                "reference_integrator": "ncls.scene-path-tracer@1",
                "bundle_root": str(package.bundle_root.resolve()),
                "source_material": str(package.source_material.resolve()),
                "reference_geometry": str(geometry),
                "reference_geometry_sha256": sha256_file(geometry),
                "slots": [
                    {
                        "package_id": "source-reference",
                        "mode": "path-tracing",
                        "target_spp": request.reference_spp,
                    },
                    {
                        "package_id": package.package_id,
                        "mode": request.neural_mode,
                        **(
                            {"target_spp": request.neural_spp}
                            if request.neural_mode == "path-tracing"
                            else {}
                        ),
                    },
                ],
                "comparison_purpose": "training-diagnostic",
                "resolution": [int(resolution[0]), int(resolution[1])],
                "reference_spp": request.reference_spp,
                "reference_samples_per_frame": int(
                    request.renderer.get("reference_samples_per_frame", 16)
                ),
                "camera": dict(request.camera),
                "display": {
                    "comparison_mode": 0,
                    "exposure_ev": 0.0,
                    "difference_scale": 8.0,
                },
                "lighting": self._lighting(request.lighting),
            },
        )
        manifest = output_root / "capture.json"
        subprocess.run(
            [
                str(self.viewer),
                "--replay",
                str(replay),
                "--headless",
                "--capture",
                str(manifest),
            ],
            check=True,
            cwd=self.viewer.parent,
        )
        document = json.loads(manifest.read_text(encoding="utf-8"))
        files = document.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("viewer capture manifest has no files table")
        return ViewerCapture(
            manifest.parent / str(files["slot_0_linear"]),
            manifest.parent / str(files["slot_1_linear"]),
            manifest.parent / str(files["difference_linear"]),
            manifest.parent / str(files["display"]),
            manifest,
        )


def build_viewer_package(
    request: VisualEvalRequest,
    snapshot: EvaluationSnapshot,
    output_root: Path,
) -> ViewerPackage:
    material_index = int(request.source.get("material_index", -1))
    compiled = compile_evaluation_package(
        snapshot,
        output_root / "package",
        material_index=material_index,
        readiness_mode="visual-diagnostic",
    )
    if compiled.source_material_path is None:
        raise ValueError(
            "visual eval source needs a viewer-loadable source artifact; "
            "MDL sources require a linked one-entry catalog"
        )
    return ViewerPackage(
        compiled.root,
        compiled.manifest.package_id,
        compiled.source_material_path,
    )


class VisualEvalWorker:
    def __init__(
        self,
        spool: VisualEvalSpool,
        artifact_root: Path | str,
        executor: VisualEvalExecutor,
        *,
        worker_identity: str,
    ) -> None:
        if not worker_identity:
            raise ValueError("visual eval worker identity is required")
        self.spool = spool
        self.artifact_root = Path(artifact_root).resolve()
        self.executor = executor
        self.worker_identity = worker_identity

    def _resolve(self, uri: str) -> Path:
        path = (self.artifact_root / Path(*uri.split("/"))).resolve()
        try:
            path.relative_to(self.artifact_root)
        except ValueError as error:
            raise ValueError("visual eval URI resolves outside artifact root") from error
        return path

    @staticmethod
    def _verify_capture(request: VisualEvalRequest, capture: ViewerCapture) -> None:
        document = json.loads(capture.manifest.read_text(encoding="utf-8"))
        if (
            document.get("format_name") != "ncls.viewer-capture"
            or int(document.get("format_version", -1)) != 4
            or int(document.get("reference_spp", -1)) != request.reference_spp
            or document.get("comparison_purpose") != "training-diagnostic"
        ):
            raise ValueError("visual eval viewer capture contract mismatch")
        slots = document.get("slots")
        if not isinstance(slots, list) or len(slots) != 2:
            raise ValueError("visual eval capture requires exactly two slots")
        expected_modes = ("path-tracing", request.neural_mode)
        expected_spp = (
            request.reference_spp,
            request.neural_spp if request.neural_mode == "path-tracing" else 0,
        )
        for index, slot in enumerate(slots):
            if (
                not isinstance(slot, Mapping)
                or slot.get("status") != "ready"
                or slot.get("mode") != expected_modes[index]
                or int(slot.get("spp", -1)) != expected_spp[index]
                or int(slot.get("target_spp", -1)) != expected_spp[index]
            ):
                raise ValueError(
                    "visual eval capture slot "
                    f"{index} did not reach ready {expected_spp[index]} spp; "
                    f"status={slot.get('status')!r}, diagnostic={slot.get('diagnostic')!r}"
                )

    def run_once(self) -> VisualEvalStatus | None:
        claim = self.spool.claim_next(self.worker_identity)
        if claim is None:
            return None
        request = claim.request
        try:
            snapshot_path = self._resolve(request.snapshot.uri)
            if not snapshot_path.is_file() or sha256_file(snapshot_path) != request.snapshot.sha256:
                raise ValueError("visual eval diagnostic snapshot is missing or tampered")
            snapshot = load_evaluation_snapshot(snapshot_path)
            if snapshot.public_method_key != request.method_key:
                raise ValueError("visual eval method disagrees with diagnostic snapshot")
            snapshot.require_ready("visual-diagnostic")
            # Slang module closures are deep. Keep the Windows filesystem path
            # bounded while retaining the full request ID in every manifest.
            capture_root = self.artifact_root / "v" / request.request_id[:24]
            capture = self.executor(request, snapshot, capture_root)
            paths = {
                "reference": capture.reference,
                "neural": capture.neural,
                "difference": capture.difference,
                "display": capture.display,
            }
            # Validate slot state before checking its advertised files so an
            # unsupported/error slot reports the viewer diagnostic instead of
            # the secondary symptom of an empty output URI.
            self._verify_capture(request, capture)
            for path in (*paths.values(), capture.manifest):
                resolved = path.resolve()
                resolved.relative_to(self.artifact_root)
                if not resolved.is_file():
                    raise FileNotFoundError(f"visual eval output is missing: {resolved}")
            result = VisualEvalResult(
                request.request_id,
                request.probe_id,
                self.worker_identity,
                {
                    name: VisualArtifact(
                        path.resolve().relative_to(self.artifact_root).as_posix(),
                        sha256_file(path),
                        "srgb" if name == "display" else "linear-srgb",
                    )
                    for name, path in paths.items()
                },
                VisualArtifact(
                    capture.manifest.resolve().relative_to(self.artifact_root).as_posix(),
                    sha256_file(capture.manifest),
                    "data",
                ),
            )
            return self.spool.complete(result, worker_identity=self.worker_identity)
        except Exception as error:
            return self.spool.fail(
                request.request_id,
                worker_identity=self.worker_identity,
                message=f"{type(error).__name__}: {error}",
            )


__all__ = [
    "ViewerCapture",
    "ViewerPackage",
    "build_viewer_package",
    "default_windows_viewer_path",
    "VisualEvalExecutor",
    "VisualEvalWorker",
    "WindowsViewerExecutor",
]
