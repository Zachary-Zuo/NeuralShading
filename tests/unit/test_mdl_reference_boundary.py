from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts))


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append(node.module)
    return tuple(result)


def test_formal_source_tree_does_not_import_or_launch_falcor2_oracle() -> None:
    violations: list[str] = []
    for path in _python_files(PROJECT_ROOT / "src/ncls"):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        for module in _imports(path):
            if module == "falcor2" or module.startswith("falcor2.") or "mdl_oracle" in module:
                violations.append(f"{relative}: import {module}")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and (
                "run_falcor2" in node.value or "run_falcor2_mdl_oracle" in node.value
            ):
                violations.append(f"{relative}: oracle launch string")
    assert not violations, "formal MDL dependency boundary violations:\n" + "\n".join(violations)


def test_oracle_tree_does_not_import_formal_mdl_provider() -> None:
    violations = []
    roots = (
        PROJECT_ROOT / "tools/reference/mdl_oracle",
        PROJECT_ROOT / "tools/reference/run_falcor2_mdl_oracle.py",
        PROJECT_ROOT / "tools/reference/build_falcor2_oracle.py",
    )
    files = tuple(path for root in roots for path in (_python_files(root) if root.is_dir() else (root,)))
    for path in files:
        for module in _imports(path):
            if module == "ncls.data.providers.mdl" or module.startswith("ncls.data.providers.mdl."):
                violations.append(f"{path.relative_to(PROJECT_ROOT).as_posix()}: import {module}")
    assert not violations, "oracle imported the formal provider:\n" + "\n".join(violations)
