from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_linux_launcher_exposes_explicit_multi_gpu_ddp() -> None:
    launcher = (PROJECT_ROOT / "scripts/run_falcor_python.sh").read_text(
        encoding="utf-8"
    )
    multi = (PROJECT_ROOT / "scripts/run_falcor_python_multi.sh").read_text(
        encoding="utf-8"
    )
    assert '"--gpus"' in launcher
    assert "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index." in launcher
    assert "--gpus must be a comma-separated list" in launcher
    assert "(0|[1-9][0-9]*)" in launcher
    assert "{gpu}" in launcher
    assert "declare -A seen" in launcher
    assert "torchrun" in launcher
    assert "NCLS_DDP_GPU_LIST" in launcher
    assert "backend=NCCL" in launcher


def test_multi_gpu_launcher_uses_shared_ddp_job() -> None:
    multi = (PROJECT_ROOT / "scripts/run_falcor_python_multi.sh").read_text(
        encoding="utf-8"
    )
    assert "run_falcor_python.sh" in multi
    assert "torchrun" not in multi
    assert "Compatibility-only thin forwarder" in multi
    assert "independent single-GPU" not in multi


def test_ddp_worker_narrows_cuda_visibility_before_module_execution() -> None:
    worker = (PROJECT_ROOT / "src/ncls/ddp_worker.py").read_text(encoding="utf-8")
    launcher = (PROJECT_ROOT / "scripts/run_falcor_python.sh").read_text(
        encoding="utf-8"
    )
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_indices[local_rank])' in worker
    assert 'os.environ["NCLS_DDP_DEVICE_INDEX"] = "0"' in worker
    assert 'runpy.run_module(module, run_name="__main__")' in worker
    assert "-m ncls.ddp_worker" in launcher
