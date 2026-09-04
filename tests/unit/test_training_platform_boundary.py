from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_training_and_data_plane_do_not_branch_on_platform_or_physical_gpu() -> None:
    roots = (
        PROJECT_ROOT / "src/ncls/data",
        PROJECT_ROOT / "src/ncls/learning/methods",
    )
    files = [
        PROJECT_ROOT / "src/ncls/learning/training/engine.py",
        PROJECT_ROOT / "src/ncls/learning/batches.py",
        PROJECT_ROOT / "src/ncls/learning/method.py",
        PROJECT_ROOT / "src/ncls/learning/source_adapters.py",
    ]
    for root in roots:
        files.extend(root.rglob("*.py"))

    forbidden = (
        "platform.system",
        "CUDA_VISIBLE_DEVICES",
        "NCLS_FALCOR_GPU_INDEX",
        "physical_device",
        "backend_key",
        "import falcor",
    )
    violations = {
        path.relative_to(PROJECT_ROOT).as_posix(): token
        for path in files
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    }
    assert violations == {}


def test_production_has_one_pipeline_session_without_sync_compatibility_alias() -> None:
    files = tuple((PROJECT_ROOT / "src/ncls").rglob("*.py"))
    matches = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in files
        if "SynchronousOnlineDataSession" in path.read_text(encoding="utf-8")
    ]
    assert matches == []
