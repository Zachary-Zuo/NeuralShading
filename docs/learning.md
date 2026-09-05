# 训练、验证与导出

## 入口与输出

以下命令均在 `neural-shading` 环境中运行。非交互调用可统一加 `conda run -n neural-shading`。

```bash
python -m ncls train 0 --config configs/training/runs/nvidia-layer-stack-smoke.yaml
python -m ncls train 0,1 --config configs/training/runs/nvidia-layer-stack-smoke.yaml
```

Windows/Linux 单卡进入同一 engine；Linux 多卡自动启动 torchrun/NCCL，Windows 当前只支持单卡。入口在导入 Torch/Falcor 前设置原生库与设备环境，每个 rank 的 Torch 使用 `cuda:0`，Falcor 使用对应物理卡。无需另写 shell 启动器或 `CUDA_VISIBLE_DEVICES`。

启动时打印一个 run 目录，DDP 启动前只分配一次：

```text
outputs/<config-stem>/<run-id>/
  config.yaml              # 入口配置副本
  resolved.yaml            # 展开后的实际参数
  run.json                 # 配置来源、提交、设备和执行状态
  checkpoints/latest.pt
  checkpoints/step-00005000.pt
  tensorboard/
  eval/step-00005000-<id>/  # replay、package、PNG、EXR 和 capture
  exports/
  logs/                    # metrics.jsonl、summary.json、review.json
```

子目录按实际使用创建。每次新启动创建独立 run；显式 `--resume` 才使用 checkpoint 所属目录。`artifacts/` 中的旧视觉证据原地保留，不作为新训练输入或默认 package。

## YAML

run 的 `compose` 选择 base/method/data/recipe；run 文件可覆盖 `training`、`execution`、`hooks`。fragment 直接写这些字段，不需要 schema/version、内部实现 key 或兼容标签。默认设置见 `configs/training/base/default.yaml`。

```yaml
compose:
  method: nvidia
  data: layer-stack-smoke
  recipe: nvidia-layer-stack-smoke
training:
  checkpoint_interval: 5000
  validation:
    interval: 1000
    batches: 8
hooks:
  visual_eval:
    enabled: true
    interval_steps: 5000
    reference_spp: 128
    neural_mode: deferred
    neural_spp: 0
    width: 640
    height: 360
```

`reference_spp` 只在 YAML 设置，改为 33 或 256 不需要改协议、worker 或测试常量。reference 与 neural 的 spp 独立；neural 若采用 `path-tracing`，将 `neural_spp` 设为正整数。renderer 最后一次 dispatch 使用剩余采样预算。

日志 cadence 位于各 phase 的 `log_interval`，checkpoint、数值 validation 和图像 eval 各用自己的 cadence。阶段切换可按 `checkpoint_boundary` 额外保存 checkpoint；图像不为自己保存 optimizer 快照。

`execution` 调节 host worker、队列、reference packing 和 residency 预算。GPU 主进程拥有 CUDA/Falcor 与资源 lease，worker 只处理 host 数据。队列深度或 packing 不改变逻辑样本的 RNG。

## 续训与观察

```bash
python -m ncls train 0 --config <run.yaml> --stop-at-step 100
python -m ncls train 0 --config <run.yaml> --resume outputs/<config>/<run>/checkpoints/latest.pt
tensorboard --logdir outputs
```

一份 checkpoint 保存模型、optimizer/precision 状态、phase/step、RNG 和在线 query cursor。续训检查实际训练定义与数据，不比较完整 YAML 或源码 hash。日志、TensorBoard、图像 spp、预取设置和相同卡数下的物理卡号可以改变；改变模型、训练 batch、phase 或卡数后请创建新 run。本次不实现不同拓扑的弹性状态迁移。

从较早 checkpoint 回退时，JSONL 截断到恢复 step，TensorBoard 使用相应 purge step 隐藏后续旧事件。完成态仍保存 optimizer 状态。

Windows 在训练线程的图像调用点使用当前模型编译 package、同步渲染，再把 comparison/difference 写入同一个 TensorBoard。材质选择与 compiler 使用隔离并恢复的 RNG；图像耗时单独记录。PT 与 deferred 对照中的阴影差异不能解释为纯模型误差。

Linux 数值 validation 正常执行，DDP 按窗口汇总各 rank 的数值；只有 rank 0 写日志。公共图像 hook 仍调用 `evaluate(model, context)`，Linux 实现直接返回 `None`，不准备模型快照、渲染资源、文件、队列或额外 collective。以后只需替换图像实现。

## 独立验证、图像与导出

```bash
python -m ncls validate <checkpoint.pt> --batches 8 --device 0
python -m ncls eval <checkpoint.pt> --device 0
python -m ncls eval <checkpoint.pt> --config <run.yaml> --device 0
python -m ncls export <checkpoint.pt> --material-index 0
```

`validate` 使用内嵌计划；`eval --config` 覆盖图像设置。Linux 图像 eval 当前为空实现。`export` 默认写入该 run 的 `exports/step-N/material-I/`，也可显式 `--output <新目录>`。

初始化状态和短训状态均可预览/导出，训练完成度与梯度覆盖是诊断信息。加载时只检查实际模型张量和资源，质量结论由实验评测给出。没有旧 checkpoint reader、转换工具、跨机图像 worker 或旧训练入口。
