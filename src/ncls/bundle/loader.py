from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .manifest import MethodBundleManifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MethodBundle:
    root: Path
    manifest: MethodBundleManifest

    @classmethod
    def open(cls, root: Path | str, *, verify_hashes: bool = True) -> MethodBundle:
        path = Path(root)
        manifest = MethodBundleManifest.from_json((path / "manifest.json").read_text(encoding="utf-8"))
        if verify_hashes:
            for uri, expected in manifest.content_hashes.items():
                target = path / uri
                if not target.is_file() or sha256_file(target) != expected:
                    raise ValueError(f"MethodBundle content hash mismatch: {uri}")
        return cls(path, manifest)

    def file(self, logical_name: str) -> Path:
        try:
            return self.root / self.manifest.files[logical_name]
        except KeyError as exc:
            raise KeyError(f"MethodBundle has no file {logical_name!r}") from exc
