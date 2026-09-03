# Metal 神经训练与部署初始证据

## 调查边界

- 本文记录任务规划阶段的只读检查，不是最终根因报告，也不把 observed quality/time/memory 提升为 hard gate。
- 环境为完整 Windows：RTX 4090、`neural-shading` Conda 环境、Windows Falcor Release Python 构建均存在。可以验证 Windows checkpoint/package/Slang/viewer 链路；Linux 长训性能仍需目标机 trace。
- 检查对象为用户指出的 `artifacts/metal-linux-training/long/checkpoint.step00020000.pt`，其 SHA-256/sidecar 均为 `6694ef1ef8c13aa8ac8298a1af9d6aed9a359b04ae3888e9a64b8ffc11e4403a`。

## 已确认事实

### F1：20k checkpoint 的散射 evaluator 从未训练

- long config 的第一阶段是 20,000 step `codec-warmup`，第二阶段才是 `joint-appearance`；见 `configs/learning/metal-fused-full-linux-long.json:6974` 与 `:7091-7092`。
- `checkpoint.step00020000.pt` 的 resume cursor 为 `phase_index=1 / phase_name=joint-appearance / phase_step=0`。这表示下一步将进入 joint，而不是已经训练过 joint。
- 用 `scratch/inspect_checkpoints.py` 比较 5k 与 20k checkpoint：六个 codec group 的 tensor 已变化且 coverage 为 finite/nonzero/update；`typed_compiler`、`optimized_state_teacher`、`prepared_model`、`angular_bank`、`analytic_core`、`hybrid_evaluator`、`proposal_sampler` 全部逐位不变，coverage 三项均为 false，`last_audit_step=-1`。
- 因此 viewer 白模不能被当作“训练 20k 后 evaluator 仍学不会”的证据；该 checkpoint 只训练了纹理 codec，散射 evaluator 仍是初始化状态。

### F2：部署入口把不可用的中间 checkpoint 当成正常 Metal package

- `tools/viewer/prepare_metal_catalog.py:42-49` 把 20k checkpoint 与 `metal-step00020000` 设为默认入口。
- `validate_preview_checkpoint()` 只验证 method/component/tensor schema；`tools/viewer/prepare_metal_catalog.py:69-108` 没有检查 phase、runtime component gradient/update coverage 或 deployability。
- 通用 `compile_program()` 只要求 `model_state` 与 `model_context`，见 `src/ncls/learning/methods/metal_fused.py:1198-1208`，因此未训练 runtime component 也会被打包。
- catalog 将 resume cursor `joint-appearance@0` 写成 `checkpoint_phase=joint-appearance`；`tools/viewer/prepare_metal_catalog.py:383` 直接索引 `config.phases[checkpoint.phase_index]`。这既掩盖“本 phase 训练步数为 0”，又会在 complete checkpoint 的 `phase_index == len(phases)` 时越界。

### F3：checkpoint/runtime 实现漂移被 tensor shape 兼容策略放行

- 20k checkpoint 的 descriptor/implementation identity 分别为 `867083...` / `5fb8f2...`；当前代码为 `a3bd27...` / `b4e823...`。
- `tools/viewer/prepare_metal_catalog.py:78-108` 在 descriptor 漂移时只比较 component manifest 与 tensor name/dtype/rank/shape，返回 `state-schema-compatible-preview`；相同 tensor shape 并不能证明前向语义相同。
- 该策略已写入 `.trellis/spec/viewer/mdl-reference.md:41`，本任务若收紧它需要同步修正规范、测试和旧 preview 入口。

### F4：既有 viewer 验收没有检查“学到了目标材质”

- 已保存的 `09-03-metal-authored-preset-viewer/scratch/progressive-headless-smoke-display.png` 中，左侧 MDL reference 有明显金属/纹理外观，右侧 neural deferred 为近白色无信息材质。
- 对应 capture 把两个 slot 都标成 `ready`；slot 1 GPU 时间约 29.9 秒，证明 `ready/finite` 只代表执行成功，不代表语义或质量成立。
- package 的 `validation/parity.json` 状态为 `gpu-parity-required`，只保存当前 Python runtime 的 `expected_f`。它可验证 Python↔Slang 实现一致性，但不能证明该实现已学习 reference。

### F5：20k 后的性能问题包含确定性的 execution-group cache thrash

- evaluator route 从 `joint-appearance` 才开始，因此 20k 是预期的成本形态切换点，而非先验上的逐 step 泄漏证据。
- `OnlineTrainingProducer._select_group()` 每个 request 轮换到下一个 group，见 `src/ncls/learning/producer.py:211-214`。full cohort 有 178 个 execution group。
- reference backend 默认只保留 8 个 resident group，见 `src/ncls/references/backend.py:323` 与 `src/ncls/references/query.py:785`；达到容量后关闭 LRU group 并构建新 session，见 `src/ncls/references/query.py:842-868`。
- 复用距离 178 大于容量 8，顺序轮转在 steady state 形成近 100% miss/evict/recreate。每次 `_ReferenceExecutionGroupSession` 构造还会创建 `evaluate/sample/pdf` 三个 pass、material resources 和两个 query slot，见 `src/ncls/references/query.py:257-327`，即使训练 route 只使用 `evaluate`。
- 这违反了原任务“避免每 step 展开/重建”的性能意图，是当前最强的结构性性能根因候选。

### F6：现有 metrics 会隐藏冷 materialization 尖峰

- long config `log_interval=10`，runner 只把被记录那一个 step 的 `prepared.preparation_seconds` 写入 row；前九个 step 的分项被丢弃，见 `src/ncls/learning/training/runner.py:809-816` 与 `:908-924`。
- `steps_per_second` 从当前 run 起点累计计算，不能显示 phase-local 或近期吞吐，见 `src/ncls/learning/training/runner.py:914-924`。
- 现有 metrics 的 observed 结果：phase 0 中位约 1.55 秒/step；phase 1 的 10-step interval 中位约 3.60 秒/step、p90 约 5.27 秒/step，最慢两个 interval 分别约 45.06 与 24.76 秒/step。最慢 interval 对应 row 的单步 `batch_prepare_wall` 只有约 2 秒，直接证明中间九步成本被分项日志漏掉。
- 这些复制到 Windows 的记录尚不能解释用户现场“每 step 约 4 分钟”的准确口径；需要 Linux group ID、cache hit/miss、compile/resource materialization、rejection、route 和完整 step wall trace。

### F7：20k 之后已有局部优化信号，但不足以证明模型正确

- `scratch/summarize_metrics.py` 对 20,010–21,010 的 101 个 training rows 做 20-row 首尾窗口统计：总 loss、response robust、peak support、compiler distillation 的均值下降；linear energy 均值上升。
- 这说明 appearance 参数开始接收优化信号，但只有约 1k step，且 query group 持续变化。它不能证明真实 validation 泛化、输出尺度、checkpoint round-trip 或 viewer 语义正确。

### F8：静态完整 capability 与分阶段训练之间缺少 readiness 合同

- `MetalFusedMethodDefinition.descriptor` 从注册时就声明 `PREPARE/EVALUATE/SAMPLE/PDF` 全部 capability，见 `src/ncls/learning/methods/metal_fused.py:420-451`；当前 package/compiler 没有根据 checkpoint 训练状态缩减 capability。
- evaluator 所需的 `typed_compiler/prepared_model/angular_bank/analytic_core/hybrid_evaluator` 只在 `joint-appearance` 与 `qat-refine` 阶段训练，见同文件 `:261-335`；`proposal_sampler` 只在 `proposal-fit` 与 `qat-refine` 阶段训练，见 `:337-376`。
- 因此 20k checkpoint 不仅 evaluator 未训练，`sample/pdf` 也仍然由初始化 proposal 提供。当前“tensor 完整即可打包全部 capability”的合同无法区分结构存在、接受过训练、通过 audit、允许诊断预览和允许正式部署这五种状态。
- 本任务需要引入由 phase completion、parameter-group update coverage、audit 与实现 identity 共同决定的 deployability/readiness；否则后续任何中间 checkpoint 都可能再次被 viewer 的 `ready/finite` 假阳性掩盖。

### F9：NVIDIA reproduction 的阶段不包含无 appearance 监督的 codec-only 前缀

- `configs/learning/nvidia-rta2024-materialx-formal.json` 的 `bootstrap` 同时启用 `reference-evaluator`、`method-sampler`，并优化 `encoder/evaluator/sampler`；两个 loss 是 `evaluator_log1p_l1` 与 `sampler_forward_kl`。
- `src/ncls/learning/methods/nvidia.py:613-657` 表明每个训练 step 都计算 reference `target_f` 对 evaluator 的 log-space loss，并同时计算 sampler objective。
- 100k 后的 `materialize-assets` 把 source-parameter encoder 输出烘成 latent texture；`finetune` 改为优化 `asset/evaluator/sampler`，没有停止最终 appearance 目标。NVIDIA recipe 中 `mollification.steps=20000` 是训练早期逐步恢复窄方向峰的难度调度，不是 codec warmup。
- 因此 Metal 当前 20k codec-only 设计不能由 NVIDIA reproduction 的阶段化 lifecycle 支持。二者都使用 phase，但 NVIDIA phase 改变 latent 的产生方式，Metal phase 则让 runtime evaluator 在很长的前缀中完全不受目标约束。

## 待验证假设

- H1：Linux 现场 4 分钟尖峰主要来自 group shader/resource 冷构建和 LRU thrash；需目标机 profile 确认每个 group 的 compile、resource upload、dispatch 与 eviction 时间。
- H2：除未训练 checkpoint 外，旧权重由新 implementation 解释也可能改变输出；需在严格 identity 的同一实现中做 checkpoint eager→quantized Python→Slang round-trip。
- H3：equal-per-group 调度与 group size 不均衡可能改变 source/state 采样分布；需比较冻结 recipe 所声明的目标分布与实际 visitation histogram。
- H4：当前 finite/nonnegative/gradient tests 不足以发现输出塌缩、尺度错误或 reference-neural 偏离；需加入固定 query 的初始化→训练→部署分层检查。

## 复现命令

```powershell
conda run -n neural-shading python .\.trellis\tasks\09-03-metal-neural-root-cause-audit\scratch\summarize_metrics.py
$env:PYTHONPATH='src'
conda run -n neural-shading python .\.trellis\tasks\09-03-metal-neural-root-cause-audit\scratch\inspect_checkpoints.py
```
