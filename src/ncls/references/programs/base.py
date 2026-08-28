from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

from ncls.core.identity import sha256_file, sha256_json
from ncls.core.scattering import ReferenceProgramDefinition, RuntimePayload


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SHADER_ROOT = PROJECT_ROOT / "shaders"
_INCLUDE = re.compile(rb'^\s*#include\s+"([^"]+)"', re.MULTILINE)


def slang_module_closure(entry: Path, *, root: Path = PROJECT_ROOT) -> dict[str, bytes]:
    """收集 package-private quoted include closure；Falcor import 仍由 host 提供。"""

    result: dict[str, bytes] = {}

    def visit(path: Path) -> None:
        resolved = path.resolve()
        try:
            name = resolved.relative_to(root.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(f"Slang module escapes package closure root: {resolved}") from error
        if name in result:
            return
        payload = resolved.read_bytes()
        result[name] = payload
        for match in _INCLUDE.finditer(payload):
            include = match.group(1).decode("utf-8")
            candidate = (resolved.parent / include).resolve()
            if candidate.is_file():
                visit(candidate)

    visit(entry)
    return result


def implementation_identity(paths: Iterable[Path]) -> str:
    return sha256_json({path.resolve().relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in paths})


class FileReferenceProgram(ReferenceProgramDefinition):
    shader: Path

    def runtime_blobs(self) -> tuple[dict[str, bytes], dict[str, dict]]:
        return {}, {}

    def runtime_samplers(self) -> dict[str, dict]:
        return {}

    def compile_runtime(self) -> RuntimePayload:
        closure = slang_module_closure(self.shader)
        module = self.shader.resolve().relative_to(PROJECT_ROOT).as_posix()
        blobs, descriptors = self.runtime_blobs()
        return RuntimePayload(
            module,
            closure,
            blobs,
            descriptors,
            self.descriptor.capabilities,
            sampler_descriptors=self.runtime_samplers(),
        )
