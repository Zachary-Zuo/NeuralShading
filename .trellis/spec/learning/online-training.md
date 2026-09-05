# Online 训练合同

## 1. 适用范围

修改当前方法、公共 engine、DDP、checkpoint 或观察时适用。入口/目录/图像和配置的完整合同见 [统一 pipeline](../project/unified-pipeline.md)。

## 2. 签名

```text
Method.create_trainable(context) -> nn.Module
Method.create_source_adapter(snapshots, device) -> MethodSourceAdapter
Method.training_objective(model, named_batches, phase) -> loss, metrics
Method.configure_phase / apply_phase_transition
Method.export_training_state / restore_training_state
TrainingEngine(method, session, config, checkpoint_callback, visual_callback).run(...)
```

## 3. 合同

- `Method` 直接绑定实现；公共 engine 不识别 Nvidia/Metal 名称。方法自身的 model/data/compiler 位于各自目录。
- `TrainingConfig` 一次解析；phase、route、优化器、精度、log/checkpoint/validation cadence 来自 YAML。方法只检查 objective 实际需要的 route/输入，不硬编码实验 seed、batch、阶段名称组合或验证批次数。
- phase 明确 active parameter 和 optimizer 状态策略；进入 DDP 前先配置 requires-grad，按 phase 构造真实 `DistributedDataParallel`。梯度由 bucket reducer 同步，不在 backward 后逐 parameter all_reduce。
- NCCL 处理 GPU gradient/scalar；Gloo 处理低频 descriptor、初始化全局样本、小型 rank RNG/query state、checkpoint commit 与 teardown。所有 rank 按相同顺序进入 collective。
- validation 在 Windows/Linux 均使用同一 engine，窗口保留逐 batch 数值，再做 packed DDP 汇总；rank 0 写结果。validation 与训练 RNG/数据用途分离。
- checkpoint 前 drain session，只有 rank 0 构造完整 CPU 模型/optimizer 副本并写文件，peer 接收写出状态。resume 恢复完整 optimizer、scaler、phase、RNG、query cursor；不因完成标签丢弃 optimizer。
- 实际结构与资源在加载边界检查。源码 hash、完整执行计划、日志频率、prefetch、图像 spp 是记录，不是恢复门禁；DDP 卡数决定 rank state 的可恢复性。
- 梯度 finite/nonzero/update coverage 用于诊断。loss/gradient 的非有限值仍是数值错误；零梯度或短训并不自动禁止保存/导出。
- checkpoint、validation、log 和 visual cadence 独立。Linux 共同图像接口为空实现，不准备 probe/快照/文件/GPU。
- Nvidia 保留 evaluator/sampler、encoder materialization 与 latent finetune 的实际模型语义。Metal 保留当前 budgeted asset/prepare/evaluator/proposal、train-only calibration 和 QAT；数值预算从 YAML 获取。
- Metal sampler 的 component index 与 distribution enum 独立；多个 GGX 合法。折回上半球后的 PDF 累加两个 preimage，sample 与独立 PDF 一致。资源 ABI 由当前 layout 和 Slang 对应，不保留 full Metal。
- calibration 使用训练 reference 数据；resume 恢复已有 buffer，不重新估计或消费 validation。

## 4. 错误矩阵

| 条件 | 行为 |
|---|---|
| active parameter 的跨 rank 结构不同 | DDP 构造前报告 |
| 必要 objective route 缺失 | 配置/方法边界报告 |
| loss 或梯度非有限 | 训练报告数值错误 |
| 模型 tensor shape/dtype 不同 | restore 拒绝，禁止广播掩盖错误 |
| 日志、预取设置、图像预算变化 | 保持逻辑样本和状态恢复 |
| rank 0 checkpoint 写入失败 | control group 传播失败 |
| sampler 折回后漏算镜像 PDF | 数学回归失败 |
| 初始化/短训状态请求预览 | 执行并记录 step，无 readiness gate |

## 5. 案例

- 正常：单卡或 DDP 使用相同 phase/objective，数值 validation 有记录。
- 基础：停止后续训结果与未中断训练一致；修改日志设置仍可恢复。
- 错误：把旧实验的 16,384 calibration、2,048 steps、固定 GPU 编号写成公共训练硬门。

## 6. 验证

`test_training_runner_phase_graph.py` 覆盖 optimizer/phase/cursor 恢复及公共图像空实现；`test_training_distributed.py` 覆盖 reducer/数值/control；`test_training_checkpoint.py` 覆盖单文件状态和真实结构错误。当前 Nvidia/Metal 模型与 package 的 GPU 回归维持 evaluate/sample/pdf 和资源绑定语义，实际运行范围见 TESTING.md。

## 7. 错误与正确

```text
错误：checkpoint.complete 且所有梯度 coverage 为真才允许普通 export。
正确：导出当前模型；训练诊断和研究质量结论单独记录。
错误：单卡一个训练循环，Linux 多卡另一个训练循环。
正确：共同 engine + phase-local DDP reducer。
```
