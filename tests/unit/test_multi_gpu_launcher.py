from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_linux_launcher_exposes_explicit_multi_gpu_fanout() -> None:
    launcher = (PROJECT_ROOT / "scripts/run_falcor_python.sh").read_text(
        encoding="utf-8"
    )
    multi = (PROJECT_ROOT / "scripts/run_falcor_python_multi.sh").read_text(
        encoding="utf-8"
    )
    assert '"--gpus"' in launcher
    assert "run_falcor_python_multi.sh" in launcher
    assert "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index." in launcher
    assert "--gpus must be a comma-separated list" in multi
    assert "(0|[1-9][0-9]*)" in multi
    assert 'CUDA_VISIBLE_DEVICES="${gpu}"' in multi
    assert "{gpu}" in multi
    assert "declare -A seen" in multi


def test_multi_gpu_launcher_keeps_single_gpu_children() -> None:
    multi = (PROJECT_ROOT / "scripts/run_falcor_python_multi.sh").read_text(
        encoding="utf-8"
    )
    assert "run_falcor_python.sh" in multi
    assert "torchrun" not in multi
    assert "DistributedDataParallel" not in multi
