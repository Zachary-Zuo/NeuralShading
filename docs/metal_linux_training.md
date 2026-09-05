# Metal 的 Linux 训练

当前 `metal` 方法位于 `src/ncls/learning/methods/metal/`，使用 MDL 原生参数和纹理及对应 reference 在线产生 GT。已有 hybrid/direct 和空间细节实验配置可作为后续实验起点，当前历史权重质量不作为新模型目标，后续按新架构重新训练。

在 `neural-shading` 环境中运行，设备列表直接填写目标机可用物理卡号：

```bash
python -m ncls train 0,1 --config configs/training/runs/metal-budgeted-hybrid-pilot.yaml
python -m ncls train 0,1 --config configs/training/runs/metal-budgeted-direct-pilot.yaml
```

这些是实际 pilot 配置，启动前自行确定实验预算；架构 smoke 使用 `nvidia-layer-stack-smoke.yaml`，不以完整 Metal pilot 代替入口检查。匹配对照按同样资源和数据预算分别运行，具体 batch、phase、calibration 和 validation 参数由 YAML 决定，不锁死 GPU 5–9 或某个历史 step。

```bash
python -m ncls train 0,1 --config <run.yaml> --stop-at-step 0
python -m ncls train 0,1 --config <run.yaml> --resume outputs/<config>/<run>/checkpoints/latest.pt
tensorboard --logdir outputs
```

Linux 保留数值 validation、DDP 汇总和 TensorBoard 标量。图像 eval 保留公共接口，当前绑定空实现；不会创建 eval 目录或跨机任务。新权重可在 Windows 使用统一 `eval`/`export`，需要对应源资产。

历史实验结论见 `research/experiment_log.md`。旧 viewer PNG/EXR 保持在原 `artifacts/` 位置，供新模型设计前分析细节；不迁移旧权重，也不保留历史训练 handoff 或格式兼容。
