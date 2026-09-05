# 统一训练与输出合同

## 1. 适用范围

修改训练入口、YAML、方法、在线数据、checkpoint、图像 eval、输出位置或部署时适用。用户于 2026-09-05 要求彻底重置：新训练重来，旧视觉证据原地保留，不迁移旧成果，不保留兼容层。

## 2. 签名

```text
python -m ncls train GPU_LIST --config YAML [--resume CHECKPOINT] [--stop-at-step N]
python -m ncls validate CHECKPOINT [--batches N] [--device N]
python -m ncls eval CHECKPOINT [--config YAML] [--device N]
python -m ncls export CHECKPOINT [--output DIRECTORY] [--material-index N]
RunPaths.create(config) / RunPaths.from_checkpoint(checkpoint)
TrainingPlanResolver.resolve(config, devices=...) -> ResolvedTrainingPlan
get_method(key) -> Method
TrainingEngine(method, session, config, visual_callback=..., ...).run(...)
save_checkpoint(path, checkpoint) / load_checkpoint(path)
VisualEvaluator.evaluate(model, VisualContext) -> VisualResult | None
```

## 3. 合同

- `launcher.py`/`runtime.py` 在导入 Torch/Falcor 前配置动态库和 GPU。单卡同一 engine，Linux 多卡自动 torchrun/NCCL。Windows 当前仅单卡。私有 `ddp_worker` 给每个 rank 设置物理卡，Torch 使用 `cuda:0`。
- GPU 只由命令参数指定；`NCLS_RUN_DIR` 由外层一次分配，`NCLS_DDP_GPU_LIST`、`NCLS_FALCOR_GPU_INDEX` 是内部派生值。
- 所有新训练产物位于 `outputs/<config-stem>/<run-id>/`，包含 config/resolved/run metadata、checkpoints、tensorboard、eval、exports、logs。新运行隔离，明确 resume 才续写旧 run。旧 artifacts 不迁移、不作为默认模型来源。
- YAML 直接组合 base/method/data/recipe 与覆盖项，用户不填写 schema/version/hash。检查拼写、重复字段和循环依赖；不把实验标签或历史数值锁进实现。
- `learning/methods/nvidia/`、`metal/` 并置真实模型、数据适配与编译；`Method` 直接提供 objective/lifecycle/compiler，没有六个转发 facet 或旧 definition。
- source 保持原生语义。唯一 `PipelineOnlineDataSession` 产生 GPU batch，控制有界队列、host 任务、reference dispatch 与 lease；不持久化 batch。
- checkpoint 保存当前模型/optimizer/precision、phase、RNG 与 query cursor；DDP rank 0 写完整状态，控制组传递 rank cursor 和 commit 结果。完成态保留 optimizer。
- 续训比较实际训练定义、source/resource 与逻辑 query partition；运行设置和源码来源不成为完整 hash 全等门禁。更改模型、训练 batch、phase 或卡数需要新 run。本次不做弹性拓扑迁移。
- 数值 validation 在两个平台保留，DDP 采用共同窗口汇总。checkpoint、日志、validation 和图像 cadence 独立。
- 图像 hook 两个平台调用同一 `evaluate`。Windows 在 cadence 同步编译当前模型并渲染，把返回图像写入本 run TensorBoard；默认 reference 128 spp，YAML 可改，最后 dispatch 截断至余量。Linux/禁用图像时直接返回 None，不准备快照、GPU、文件、队列或额外 collective。
- Windows 图像使用独立材质选择 RNG，compiler 调用后恢复全局 RNG。无需保存 optimizer 图像快照。没有跨机 spool/claim/worker/collector。
- TensorBoard 使用自身有界写队列。rank 0 发布事件，resume 以 checkpoint step+1 purge，JSONL 同时截断。
- eval/export 直接读同一当前 checkpoint；初始化状态也可以导出。完成度、实验类型、梯度覆盖是诊断。部署资源的 ABI、大小、实际 shader parity 保留。

## 4. 错误矩阵

| 条件 | 行为 |
|---|---|
| YAML 拼写错误、重复键、循环引用、spp 非合法采样数 | 配置边界报告具体字段 |
| 合法 spp 改成 33/128/256，或 neural spp 大于 reference | 直接执行 YAML |
| 缺少原生库、source 资源，tensor shape/dtype 不符 | 对应加载边界失败 |
| DDP rank/物理卡映射或 world size 不一致 | 进程装配失败，避免错误 collective/interop |
| 日志、图像 spp、预取、同卡数物理编号变化 | 可以加载并恢复 |
| 部分训练或初始化模型请求导出 | 正常编译，诊断记录实际训练 step |
| 图像 renderer 返回失败 slot | 报告 renderer 诊断，不发布成功图像事件 |
| Linux 图像调用 | None，无图像副作用；数值 validation 照常 |

## 5. 正常、基础与错误案例

- 正常：`train 2,5 --config run.yaml` 在 Linux 启动一个两卡作业，只有一处 run 输出。
- 基础：同一 YAML 单卡完成训练、validation、checkpoint；Windows cadence 渲染，Linux 相同调用点为空。
- 错误：要求改 spp 时同步改协议常量；为短训导出增加 formal/complete gate；将权重默认写到 artifacts。

## 6. 验证

unit 覆盖配置透传、入口与物理卡装配、run 隔离、checkpoint/optimizer 恢复、真实 tensor 结构错误、TensorBoard purge、Linux 空实现与公共 engine 的 cadence/validation。Windows 执行短训练/续训、真实图像与导出，检查两个 YAML spp 的实际 capture。Linux/NCCL 实机命令与未验证边界见 TESTING.md；Windows 不代替 Linux 证据。

## 7. 错误与正确

```text
错误：修改日志 → 完整 plan hash 改变 → 拒绝恢复。
正确：训练身份和运行设置分开；恢复实际 optimizer/cursor，记录新的运行设置。
错误：Linux 先复制模型、保存图像请求，再发现没有 renderer。
正确：公共调用点直接调用 NoVisualEvaluation.evaluate，返回 None。
```
