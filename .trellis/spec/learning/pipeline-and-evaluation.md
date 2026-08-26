---
name: learning-pipeline-and-evaluation
description: pipeline 注册与训练配置、quality-v1 四层指标与 sanity 唯一否决、compare 的 matched 规则、checkpoint tail guard、experiment_log 登记格式、MethodBundle 导出边界
paths:
  - src/ncls/learning/**
  - src/ncls/bundle/**
  - configs/learning/**
  - configs/evaluation/**
  - docs/research/experiment_log.md
---

# pipeline、评测与导出规则

## 候选身份（`src/ncls/learning/pipelines/base.py`）

- `LearningPipelineDescriptor` 字段精确为：`data = {reader, partition, source_adapter}`、`model = {representation, architecture, latent}`、`fitting = {path ∈ gradient|direct-fit|hybrid, loss}`、`runtime = {compiler, exporter, deployment_candidate: bool}`，外加 `name / stage / supported_families / scope`。多一个或少一个字段都会 `ValueError`。
- 名称给人读（`lobe-residual-k2-v1`），descriptor 的 SHA-256 才是实现身份；`deployment_candidate` 是研究期分类，**不进 hash**，P1 v1 checkpoint 仍可评测。
- `register_pipeline(factory)` 在 `pipelines/__init__.py` 统一调用；重复名报错；没有 alias。
- partition 三选一并显式声明：`parametric-v1`（source split ∩ query role）、`target-visible-v1`（只按 query role，encoder / refinement 可读 source-test 的 train response）、`workflow-v1`（资产式族批量评测）。
- `parameter_costs()` 返回的 key 与现有 pipeline 一致（`test_deployment_budget.py` 的 `SOFT_BUDGET` 集合 + `B_shared`、`parameter_count`、`analytic_core_state_bytes`、`C_eval_excludes_analytic_core`），按部署实际 bytes 记账。

## 训练配置（`training/config.py`）

- 只接受完整 `training-config-v1`；`capacity` 可省略，P1 v1 历史配置保留其 S/M/L 字符串；默认 `seed=20260824`、`steps=25000`、`checkpoint_selection="median_then_p95"`（旧默认不进 hash）。新配置用 `"tail_guard"`：先剔除 validation p95 > 该 run 至今最小 p95 × 1.25 的 checkpoint，再取 median 最小。
- `dataset_selection` 只允许 `state_ids / asset_ids / family_ids`。
- 预算档位：快速档（≤ 30 min GPU，只做 smoke，不入注册表）、标准档（全量阶段数据、共同 seed、4,000 步后 validation patience 早停）、冲刺档（×5–10）。P1 主搜索只用一个 deterministic seed；只有差距接近或轨迹异常才追加 seed。
- `run_manifest.json` 记录解析后配置、Git 提交、reference 实现 hash、合同版本、seed、依赖版本、输入产物 ID；命令行只能覆盖配置里已声明的字段。
- `steps / batch_size / seed / validation_interval` 是方法或预算适配，不是任务质量门。复现方法若有权威训练定义，差异必须进入 method correspondence；没有用户授权时不得因为 observed quality 追加 seed 或增加预算。

### 长运行的进度与性能合同

- train step 和耗时明显的 validation / data collection 循环使用 `tqdm`。进度条以真实 batch、query 或 chunk 为单位，显示 `completed / total`、elapsed、rate、ETA；postfix 只放当前 phase、loss 等少量有用信息。
- 更新粒度以可见且不扰动训练为准：通常每个 batch 或合理 chunk 更新，不为逐 sample 展示强制 GPU 同步。阶段切换时结束旧进度条并为新阶段建立可解释的 total。
- smoke 负责发现正确性、显存和基本数据流问题。根据 smoke 或短时运行的实测速率估计长跑耗时；速率合理即可运行，不要求先建立覆盖所有 phase 的完整 timing 报告。
- 若吞吐显著低于方法与硬件规模应有的水平、ETA 不可接受或进度长时间不前进，先停止盲跑并 profile 实际慢段。只分解与嫌疑有关的 CPU data、transfer、forward、backward、optimizer 或 validation/I/O，再通过扩大有效 batch、合批、缓存、减少 host round-trip / kernel launch / 同步边界等方式优化主导成本。
- heartbeat、watcher、PID/start-time、liveness timeout 和进程树状态机不属于普通训练的默认合同。只有明确引入无人值守调度、多进程 wrapper 或断点恢复服务时，才为相应边界设计这些机制。
- 一个 formal run 完成 implementation/convergence/correctness 报告前，不因 observed quality 自动排队额外 seed、候选或 sampler run。质量较低但正确收敛时登记结果。

### 分阶段冻结与 sampler-only 梯度门

- sampler-only 阶段只允许目标 sampler head 的参数 `requires_grad=True`；shared prepare、latent、evaluator 与另一 sampler head 必须冻结，shared hidden 在进入 sampler head 前 detach。
- 在 optimizer step 之前执行结构门：目标 head 输出与 PDF 必须 `requires_grad=True`；backward 后目标参数梯度有限且至少一个非零，所有冻结参数保持 `grad is None`。
- 该门必须在先运行过冻结 evaluator/deployment 路径的 warm session 中覆盖。SlangPy callable 身份与 active-gradient mask 的具体约束见 `core/shared-slang-backend.md`。

## quality-v1（`evaluation/quality.py`，`configs/evaluation/quality-v1.json`）

- 候选只返回线性 RGB `f`；harness 在指标内乘一次 `|cos θi|`；candidate 不能替换 metric suite；报告记录 suite 的 SHA-256，`compare` 拒绝 suite hash 不一致的报告。
- 四层：① sanity（唯一会把报告标无效的层）；② 主指标：solid-angle weighted normalized L1 的 state median / p95，半球能量相对误差的 `(state, wo)` median / p95；③ scorecard：log-domain error、峰位角、峰高比、top-energy recall、source-aware reciprocity、分组 breakdown——解释长尾，不否决；④ diagnostics：绝对误差、reference SE 及比值——`model error / reference SE` 不是 kill gate。
- 当前参考线（可随证据修订）：state-median ≤ 0.05 且 p95 ≤ 0.15，能量 median ≤ 0.03；这是"值得进下一阶段"的参考，不是单指标 kill gate。
- `quality-v2` 与 v1 只差 `checkpoint_selection` 块；比较器接受两者。

## Baseline 复现状态与方法比较必须分离

### 1. Scope / Trigger

注册或复现 prior-art baseline、生成 `unified-method-selection` 证据、或把 quality 数值用于任务验收时触发。该合同防止把某个旧 run 的 observed metric 提升成跨材质硬门，并防止为了满足成本线先改小原方法后仍沿用 baseline 身份。

### 2. Signatures

正式比较入口：

```text
ncls learn select-unified-method \
  --inputs <unified-selection-inputs-v1.json> \
  --output <unified-method-selection-v1.json> \
  --source-git-commit <40-hex>
```

`configs/evaluation/unified-method-selection-v1.json` 只冻结 cell identity、bootstrap 和 relative Pareto 规则，不含 directional/energy 的绝对阈值。每个 input cell 提供：

```text
audit + checkpoint_label
implementation_correctness
evaluator_convergence
sampler_convergence
sampler_correctness
benchmark + compiled + parity
```

### 3. Contracts

- `implementation_correctness.passed`：method correspondence、独立 oracle 与声明的 adaptation 完整；不能由 quality 数值推导。
- `evaluator_convergence.passed` / `sampler_convergence.passed`：训练全程有限、validation 相对初始化改善、后期无可信发散、checkpoint 可恢复；只读 train/validation evidence，不读 test。
- `sampler_correctness.passed`：PDF/null/sample 数学门通过。
- `checkpoint_parity.passed`：SlangPy、Falcor 与 packed asset 是同一实现。
- cell `eligible` 只由上述四类证据合取；quality、time、memory 进入 `metrics/cost` 做 relative comparison，不改变复现状态。
- top-level 只要求四格共享 `data_id` 与 test protocol；每个 cell 分别保存 `slang_implementation_sha256` 与 `layout_sha256`。不同方法不得靠 padding 或共享假 hash 伪造相同私有 state/网络布局。
- 原规模 baseline 超过研究软成本线时写 `deployment_candidate=false` / 真实 runtime class，但仍可导出和在 viewer 显示。缩模必须使用独立 pipeline/config/checkpoint identity。

### 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| implementation / convergence evidence 缺失、hash 错或 identity 不匹配 | selection 输入失败，不生成 manifest |
| 数学 correctness 或 checkpoint parity `passed=false` | cell `eligible=false`；保留证据，不解释为低quality |
| quality 很差但四类复现证据通过 | cell 仍 eligible；报告 observed quality，不触发重训 |
| 四格 `data_id` 不同 | 拒绝 matched comparison |
| 四格 Slang/layout hash 不同 | 允许；在各 cell 内分别验证 compiled/parity identity |
| baseline 超软成本线 | 改成本分类，不改变 implementation/convergence status |
| test 被 convergence/checkpoint 选择读取 | convergence evidence 无效，必须重跑无泄漏流程 |

### 5. Good / Base / Bad Cases

- Good：原方法逐项对应、预先冻结的主 seed 稳定收敛；某些 LayerStack 结构 quality 较低。登记“复现成功 + 当前结构上的质量限制”；只有轨迹异常或用户明确要求时才补 seed。
- Base：baseline 与 candidate 都 eligible，分组 paired CI 在不同结构上给出不同 Pareto；保留多个条件性非支配结果。
- Bad：拿旧 run 的 median/p95 数值写进 `require_q1`，未过就修改网络/seed反复训练；或把缩小网络仍命名为原 baseline。

### 6. Tests Required

- protocol loader 断言不存在 `q1` / absolute quality threshold，baseline cell 指向原规模 pipeline。
- selection unit 用任意高的绝对 error 仍能在 implementation/convergence/correctness/parity 全通过时保持 eligible；relative paired evidence仍可工作。
- 任一 gate `passed=false` 时对应 cell ineligible；不得由更好 quality 覆盖。
- artifacts assembly 拒绝 data/checkpoint/pipeline/hash 篡改，允许 baseline 与 candidate 使用不同 Slang/layout identity。
- pipeline registry 不再暴露被淘汰的缩模 baseline 正式入口；若以后增加缩模，测试要求新 ID。

### 7. Wrong vs Correct

```python
# 错：用某次历史run的误差决定“是否复现”，并强制所有方法共享布局。
eligible = directional_p95 <= 0.10 and cell.layout_sha256 == baseline.layout_sha256

# 对：复现证据与质量比较分离，layout只在各自方法产物链内保持一致。
eligible = all((
    implementation.passed,
    evaluator_convergence.passed,
    sampler_convergence.passed,
    sampler_correctness.passed,
    checkpoint_parity.passed,
))
quality_comparison = paired_state_bootstrap(baseline.metrics, candidate.metrics)
```

## compare（`evaluation/comparison.py`）

- 只接受 `valid=True`、`evaluation_role=test`、hash 自洽的 quality 报告；baseline 与 candidate 必须同 `data_id`、完全相同的 test state，≥ 20 个 matched state，≥ 1,000 次 95% state-block paired bootstrap。
- `steps / seed / dataset_selection` 必须一致，否则直接拒绝；候选之间的差异只体现在 pipeline 与 model 字段。
- CI 跨零记"无显著差异"，不写"X 优于 Y"。30-state p95 的 CI 很宽，只作 selection 诊断；正式长尾结论用 ≥ 50 个 test state 并附 p90/p95、最差 state 清单、CI 与 leave-one-state-out。
- 有 core 的候选始终与 direct 候选配对报告；跨族汇总分层报告。

## experiment_log 登记（`docs/research/experiment_log.md`）

每个正式 run 一行：日期、run ID、候选+配置、数据版本（`data_id`）、预算档、seeds、方向 L1 med/p95、能量误差 med/p95、一句话结论（含是否部署候选、超了哪条线）、artifacts 路径。详细数值留 `artifacts/`；阶段结论写在表下方小节。

## 导出（`src/ncls/bundle/`）

- 从不可变 checkpoint 导出全新 bundle，计算全部内容哈希，记录源 run / checkpoint；bundle 只含推理必需内容与验证证据，不含 optimizer / TensorBoard。
- `runtime_class=realtime` 需要 descriptor `is_complete_realtime_backend`（`Prepare|Evaluate|Sample|Pdf|AnisotropicFrame`）且 `cost_claims` 满足硬线；否则 `diagnostic`。
- 通用 exporter（`bundle/compiled_set.py` 与共享 manifest/hash 工具）负责 manifest 组装、写文件、hash 与 parity probe；各方法只提供 descriptor、反射得到的 params/layout、`compiled_materials/` 与 `cost_claims`。不得为单个方法新增手写 bundle 序列化路径。

## 反例

- 训练循环里按 test 结果选 checkpoint 或超参。
- 为一个候选单独调 loss 权重后再与别的候选"同表比较"。
- 把 PyTorch `benchmark` 的 kernel 时间当最终 viewer 时间。
- 报告里用"上界"描述有限预算下的结果。
- formal 训练没有 `tqdm` 进度与 ETA，明显缓慢时仍不分析热点；或只用存活进程代替实际完成量。
- 前一方法还没完成 joint evaluator/sampler lifecycle，就排队多个 seed、缩模版本和其他 sampler。
