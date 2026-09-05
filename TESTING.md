# 验证与交接

所有 Python/pytest 命令使用 `neural-shading`。先按 `.trellis/spec/project/dev-environment.md` 判定机器能力，再运行对应检查。

## 原生多 UV 实施检查点

本机仍为完整 Windows。实现任务 `09-05-neural-material-spatial-optimization-research` 已按用户要求归档，命令、真实 run 和边界见 [实施验证记录](.trellis/tasks/archive/2026-09/09-05-neural-material-spatial-optimization-research/research/implementation-validation.md)。全量 unit 366 passed；之后新增 C3/visual phase 回归 11 passed；多 UV GPU parity、独立 reference footprint witness、实际 train 0→2/validate/export/eval 和 Release viewer 均已执行。剩余实验移交 [服务器 24 小时任务](.trellis/tasks/09-05-neural-material-24h-server-research/server-handoff.md)，尚无 matched 质量结论。

```powershell
conda run -n neural-shading python -m pytest tests/unit -q
conda run -n neural-shading python -m tools.learning.generate_metal_budgeted_layout --check
conda run -n neural-shading python -m ncls.runtime --device 0 -- -m pytest tests/gpu/test_metal_spatial_runtime.py tests/gpu/test_metal_spatial_reference.py tests/gpu/test_metal_model_correctness.py -q
conda run -n neural-shading python -m ncls train 0 --config configs/training/runs/metal-spatial-probe-bronze-scratched.yaml --stop-at-step 0
```

后续 resume/validate/export/eval 使用命令返回的实际 checkpoint；已执行 smoke 为 `outputs/metal-spatial-probe-bronze-scratched/260905-220316-6185cc/checkpoints/latest.pt`。图像必须显式使用 `configs/training/runs/metal-spatial-stage-eval.yaml`，因为训练配方关闭 visual hook。新 profile 的 176 B state、12 次实际 reads（上限 54）与 768 B instance 必须按实际包核对，不能套用历史两次读取声明。

待执行：三个实际 source 完整 D0、真正 summary-control 与 matched D1；Linux/NCCL 共享资源释放、动态语义 unused group、stop/resume 仍须目标实机检查。不要以 Windows fixture 代替平台或未见资产质量证据。构建 viewer 时先退出加载 Falcor DLL 的项目测试，防止链接目标被占用。

## 架构基线 Windows 检查

2026-09-05 本机为完整 Windows：RTX 4090、neural-shading、锁定 Windows Falcor Python 与 Release viewer 存在。本次检查的命令输出和具体结果保存在 `.trellis/tasks/archive/2026-09/09-05-architecture-reset-training-workflow/scratch/`，最终执行结果见该归档任务的 `research/validation.md`。Linux/NCCL 不在本机验证范围内。

```powershell
conda run -n neural-shading python -m pytest tests/unit -q
conda run -n neural-shading python -m ncls.runtime --device 0 -- -m pytest tests/gpu -q
.\scripts\build_viewer.ps1 -Configuration Release
git diff --check
git -C external/Falcor status --short
```

GPU 集合应按实际改动选择。方法目录重组重点覆盖 Nvidia training/latent/proposal、Metal budgeted model/sampler/runtime/residency、公共 package/reference 与 viewer path-surface；原生 reference 数学未修改时无需重复完整研究性能矩阵。

## 真实短流程

```powershell
conda run -n neural-shading python -m ncls train 0 --config configs/training/runs/nvidia-layer-stack-visual-smoke.yaml --stop-at-step 0
conda run -n neural-shading python -m ncls export outputs/<config>/<run>/checkpoints/latest.pt
conda run -n neural-shading python -m ncls train 0 --config configs/training/runs/nvidia-layer-stack-visual-smoke.yaml --resume outputs/<config>/<run>/checkpoints/latest.pt
conda run -n neural-shading python -m ncls validate outputs/<config>/<run>/checkpoints/latest.pt --batches 1 --device 0
conda run -n neural-shading python -m ncls eval outputs/<config>/<run>/checkpoints/latest.pt
```

检查新 run 隔离、初始化导出、optimizer/phase/query 恢复、数值日志、TensorBoard 标量和图像。只改 YAML 的 reference_spp 为 128 和另一正整数（本任务使用 33），核对实际 capture 的 target/actual、ready slot、PNG、float32 finite EXR。128 和 33 是测试取样，不是生产协议限制。小分辨率 headless 允许 width≥2、height≥1。

从较早 checkpoint 回退后，JSONL 不保留后续旧 step，TensorBoard purge 后的可见事件与恢复轨迹对应。公共 engine 测试还检查 Linux 空实现不改变 RNG、reference dispatch 或模型结果，仍产生数值 validation。

## Linux 实机待验证

需要原生 Linux、两张可用 NVIDIA GPU、neural-shading 与锁定 Falcor Linux/MDL backend。先执行部署/probe；使用可用卡号替换 0,1：

```bash
conda activate neural-shading
python -m ncls reference probe --device 0
python -m ncls train 0 --config configs/training/runs/nvidia-layer-stack-visual-smoke.yaml
python -m ncls train 0,1 --config configs/training/runs/nvidia-layer-stack-visual-smoke.yaml --stop-at-step 1
python -m ncls train 0,1 --config configs/training/runs/nvidia-layer-stack-visual-smoke.yaml --resume outputs/<config>/<run>/checkpoints/latest.pt
```

期望同一命令自动启动一个作业，物理卡映射正确；rank 0 单点写出 checkpoint/metrics/TensorBoard，数值 validation 有汇总结果。图像 cadence 虽为 1，run 中仍没有 eval 图像目录、renderer 进程或跨机队列。各 rank 有独立 RNG/query cursor，stop/resume 及 teardown 正常。

需要底层故障注入时，`tests/integration/test_distributed_training.py` 可在内部两 rank worker 中执行；它是开发测试，不是新的用户训练入口。NCCL 梯度通信、Falcor/Vulkan 互操作与实机卡号映射不能由 Windows unit 或 mock 证明。

## 仓库边界

构建后 `external/Falcor` 必须干净。outputs 是新训练成果；artifacts 中旧视觉证据保持原位置。本任务不迁移/删除旧成果，不进行正式模型训练或质量评选。
