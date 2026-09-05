import sys
from pathlib import Path

import pytest

from ncls.commands import build_parser
from ncls.launcher import worker_command
from ncls.runs import RunPaths
from ncls.runtime import parse_devices, process_environment
from ncls.learning.training.launch import preflight_topology, worker_execution_context


def test_train_command_selects_one_gpu_or_automatic_torchrun():
    for text, count in [("3", 1), ("2,5", 2)]:
        args = build_parser().parse_args(["train", text, "--config", "experiment.yaml"])
        command = worker_command(["train", text, "--config", str(args.config)], args.devices)
        assert command[0] == sys.executable
        assert ("torch.distributed.run" in command) == (count > 1)
        if count > 1:
            assert f"--nproc-per-node={count}" in command


def test_rank_selects_physical_device_from_single_list():
    env = {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1", "NCLS_DDP_GPU_LIST": "2,5"}
    topology = preflight_topology((2, 5), platform_name="Linux", environment=env)
    context = worker_execution_context(topology, environment=env)
    assert context.physical_device == 5
    assert context.torch_device == "cuda:0"
    assert not context.is_rank_zero


def test_windows_multi_gpu_rejected_before_environment_probe(tmp_path):
    with pytest.raises(RuntimeError, match="Linux/NCCL"):
        process_environment((0, 1), project_root=tmp_path, system="Windows")


@pytest.mark.parametrize("separator", [[], ["--"]])
def test_runtime_forwards_python_module_arguments(monkeypatch, separator):
    from ncls import runtime

    monkeypatch.setattr(runtime, "process_environment", lambda devices: {"GPU": str(devices[0])})
    calls = []
    monkeypatch.setattr(runtime.subprocess, "call", lambda command, *, env: calls.append((command, env)) or 0)
    assert runtime.main(["--device", "3", *separator, "-m", "pytest", "tests/gpu", "-q"]) == 0
    assert calls == [([sys.executable, "-m", "pytest", "tests/gpu", "-q"], {"GPU": "3"})]


def test_new_runs_are_isolated_and_resume_resolves_original_run(tmp_path):
    config = tmp_path / "experiment.yaml"
    config.write_text("compose: {}", encoding="utf-8")
    first = RunPaths.create(config, output_root=tmp_path / "outputs")
    second = RunPaths.create(config, output_root=tmp_path / "outputs")
    assert first.root != second.root
    assert first.root.parent == second.root.parent
    assert first.metrics.parent == first.logs
    first.checkpoints.mkdir()
    first.checkpoint.write_bytes(b"checkpoint")
    assert RunPaths.from_checkpoint(first.checkpoint) == first


def test_ddp_worker_sets_rank_device_before_importing_training(monkeypatch):
    from ncls import ddp_worker

    monkeypatch.setenv("NCLS_DDP_GPU_LIST", "2,5")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(sys, "argv", ["worker", "-m", "ncls.cli", "train", "2,5"])
    calls = []

    def run(module, *, run_name):
        import os
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
        assert os.environ["NCLS_FALCOR_GPU_INDEX"] == "5"
        assert os.environ["NCLS_DDP_DEVICE_INDEX"] == "0"
        calls.append((module, run_name))

    monkeypatch.setattr(ddp_worker.runpy, "run_module", run)
    assert ddp_worker.main() == 0
    assert calls == [("ncls.cli", "__main__")]


@pytest.mark.parametrize("driver,uses_compat", [("550.54.15", True), ("570.86.16", False)])
def test_linux_runtime_maps_devices_and_selects_current_conda_libraries(tmp_path, monkeypatch, driver, uses_compat):
    import json
    from ncls import runtime

    root = tmp_path / "project"
    (root / "references").mkdir(parents=True)
    binary = root / "external/falcor/bin"
    module = binary / "python"
    module.mkdir(parents=True)
    (module / "falcor_ext.so").touch()
    (root / "references/reference-backend-toolchains.json").write_text(json.dumps({
        "platforms": [{"platform_id": "linux-x86_64@1", "falcor": {
            "runtime_library_root": "external/falcor/bin", "python_module_root": "external/falcor/bin/python",
            "python_extension": "falcor_ext*.so",
        }}],
    }), encoding="utf-8")
    prefix = tmp_path / "environment"
    (prefix / "cuda-compat").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *args, **kwargs: driver + "\n")
    env = process_environment((2, 5), project_root=root, system="Linux", environment={"PATH": "/bin"})
    assert env["CUDA_VISIBLE_DEVICES"] == env["NCLS_DDP_GPU_LIST"] == "2,5"
    assert "NCLS_FALCOR_GPU_INDEX" not in env
    assert (str(prefix / "cuda-compat") in env["LD_LIBRARY_PATH"]) == uses_compat
    assert env["LD_LIBRARY_PATH"].endswith(str(prefix / "lib"))
