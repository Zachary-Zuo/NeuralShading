# 测试说明

所有 Python、pytest 和 pip 命令使用唯一 Conda 环境 `neural-shading`。需要导入 Falcor Python 模块时，Windows 使用 `scripts/run_falcor_python.ps1`，Linux 使用 `scripts/run_falcor_python.sh`。

## 单元与静态检查

```powershell
conda run -n neural-shading python -m pytest tests\unit -q
conda run -n neural-shading python -m compileall -q src tests tools
git diff --check
```

训练架构的快速定向集合：

```powershell
conda run -n neural-shading python -m pytest `
  tests\unit\test_training_yaml.py `
  tests\unit\test_training_plan.py `
  tests\unit\test_method_plugin.py `
  tests\unit\test_data_plan.py `
  tests\unit\test_online_data_session.py `
  tests\unit\test_data_pipeline.py `
  tests\unit\test_gpu_residency.py `
  tests\unit\test_reference_scheduler.py `
  tests\unit\test_training_engine.py `
  tests\unit\test_training_events.py `
  tests\unit\test_training_checkpoint_new.py `
  tests\unit\test_training_launcher.py `
  tests\unit\test_tensorboard_hook.py `
  tests\unit\test_visual_eval_spool.py `
  tests\unit\test_visual_eval_worker.py `
  tests\unit\test_legacy_checkpoint_v4.py -q
```

## Windows GPU、Falcor 与 viewer

```powershell
.\scripts\build_reference_backend.ps1 -Configuration Release
.\scripts\run_falcor_python.ps1 -m ncls reference doctor
.\scripts\run_falcor_python.ps1 -m ncls reference probe
.\scripts\run_falcor_python.ps1 -m pytest tests\gpu tests\integration\reference -q
.\scripts\build_viewer.ps1 -Configuration Release
```

最小新架构训练与恢复：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls train `
  configs/training/runs/nvidia-layer-stack-smoke.yaml `
  --devices 0 --output artifacts/training/windows-smoke/checkpoint.pt `
  --stop-at-step 1
.\scripts\run_falcor_python.ps1 -m ncls train `
  configs/training/runs/nvidia-layer-stack-smoke.yaml `
  --devices 0 --output artifacts/training/windows-smoke/checkpoint.pt `
  --resume artifacts/training/windows-smoke/checkpoint.pt --stop-at-step 2
.\scripts\run_falcor_python.ps1 -m ncls validate `
  artifacts/training/windows-smoke/checkpoint.pt --batches 1 --device 0
```

visual eval 默认让 reference 跑 1024 spp path tracing、neural 跑 deterministic deferred，不要求 neural 再做 1024 spp。worker/collector：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls eval worker `
  artifacts/training/windows-smoke/checkpoint.visual-eval `
  artifacts/training/windows-smoke --max-jobs 1
.\scripts\run_falcor_python.ps1 -m ncls eval collect `
  artifacts/training/windows-smoke/checkpoint.visual-eval `
  artifacts/training/windows-smoke `
  artifacts/training/windows-smoke/checkpoint.tensorboard
```

验收 capture manifest 时检查 `comparison_purpose=training-diagnostic`、reference `mode=path-tracing` 且 target/actual 为 1024、neural `mode=deferred` 且 spp 为 0；reference/neural/difference EXR 必须 finite。低 spp neural path tracing 或双侧 1024 spp 仅为手工深度检查。

## Linux 单卡、多卡与性能

目标 Linux 必须先具备锁定的 Falcor/MDL 构建和所需 source assets：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/deploy_reference_linux.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls train \
  configs/training/runs/metal-linux-smoke.yaml --devices 0 \
  --output artifacts/metal-linux-training/smoke/checkpoint.pt

bash scripts/run_falcor_python.sh -m ncls train \
  configs/training/runs/metal-linux-smoke.yaml --devices 0,1 \
  --output artifacts/metal-linux-training/ddp/checkpoint.pt
```

Linux 结果必须由原生目标机回填，Windows 不替代：

- 单卡和 DDP checkpoint 均能完成/恢复，DDP 只由 rank 0 写输出；
- 每 rank 的 Falcor 物理卡与 Torch `cuda:0` 映射一致；
- Metal 热路径 request metadata 的 GPU→CPU readback bytes 为 0；
- host/data、reference、model、barrier、cache/queue、吞吐与峰值显存分别记录；
- residency/ready ring 不超过配置预算，active lease 不被驱逐；
- full-cohort smoke 后再估计 120000 步 long run 的 ETA，不沿用旧约 52 小时估计。

## Reference 与材质族

OpenPBR、MERL、MaterialX、MDL 和 LayerStack 的固定 source/reference gate 继续使用各自测试与工具。完整 GPU 集合：

```powershell
.\scripts\run_falcor_python.ps1 -m pytest `
  tests\gpu\test_reference_query_dispatcher.py `
  tests\gpu\test_reference_backend_contracts.py `
  tests\gpu\test_mdl_native_crosscheck.py `
  tests\gpu\test_mdl_hlsl_feasibility.py `
  tests\gpu\test_merl_reference_gpu.py `
  tests\gpu\test_openpbr_reference_gpu.py `
  tests\gpu\test_layer_stack_ir_gpu.py -q
```

MaterialX viewer/parity、pbrt probe 和 OpenPBR probe 的详细命令分别见对应 `docs/` 文档。它们验证原生 source 语义，不把其他材质族归约成层模型。

## 仓库边界

```powershell
git -C external\Falcor status --short
git -C external\pbrt-v4 status --short
git -C external\OpenPBR status --short
git -C external\openpbr-bsdf status --short
git -C external\glm status --short
git -C external\MaterialX status --short
```

所有上游工作树必须为空。`build/`、`data/`、`artifacts/`、`external/` 和缓存不得进入根仓库。
