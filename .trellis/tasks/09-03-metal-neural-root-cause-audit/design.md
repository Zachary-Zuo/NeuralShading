# Metal 神经训练与部署根治设计

## 1. 设计目标与边界

本任务把当前故障视为同一条数据流上的三个相互放大问题，而不是互不相关的白模、慢训练和 viewer 兼容问题：

```text
源材质/typed state
    ↓ GPU online reference query
端到端 joint objective ──→ checkpoint readiness
    ↓                         ↓
Python eager           MethodBundle / Slang
    └──────── parity + learning evidence ───────→ viewer
```

- 训练目标必须从第 1 步约束最终 `prepare/evaluate/sample/pdf` 表示；不再用长时间 codec-only 重建代替 neural material 学习。
- Windows 与 Linux 只在 Falcor device API、启动器、工具链和硬件容量边界不同；上层 config、producer、runner、checkpoint、method、package 与验证语义相同。
- 性能优化必须保持 source/query/model/work 等价。不得通过缩模型、少算 reference、替换 sampler、降低 source coverage 或使用缓存 GT 制造加速。
- `TrainingCheckpoint@4`、`ScatteringPackage@2` 与 scattering ABI 保持不变；readiness 使用 checkpoint 已有的 phase graph、游标、identity 和 gradient coverage 推导，不增加旧格式 reader 或宽松迁移器。
- 120k 总预算保持不变。绝对 quality/time/memory 是 observed result；正确性、无稳态重建和资源有界是 hard contract。

## 2. 已确认根因与修复映射

| 根因 | 直接影响 | 设计修复 |
|---|---|---|
| 20k `codec-warmup` 没有 reference appearance 监督 | 20k evaluator 仍为初始化，viewer 白模 | 105k 端到端 joint + 15k QAT；所有 runtime 参数从第 1 步训练 |
| compiler/export 只检查 tensor 结构 | 未训练或实现漂移的权重被包装成完整材质 | 统一 readiness evaluator；formal fail closed，diagnostic capability 分级 |
| catalog 默认使用旧 20k 且 shape-only 放行 | 白模被标成 `ready` | 移除默认旧 checkpoint 和 `state-schema-compatible-preview` |
| 178 groups 逐 step 轮换、resident 仅 8 | 近 100% miss/evict/session 重建 | 确定性 group-local 调度、operation-lazy session 与有界 cache |
| session 总是创建 evaluate/sample/pdf 三个 pass | 训练只用 evaluate 却承担全部构建成本 | backend open 声明所需 operation，训练只 materialize evaluate |
| 每 10 步只记录最后一步分项、吞吐为 run-global | 冷构建尖峰和阶段退化不可见 | 每步低同步 wall/counter 累积，按 log window 汇总 phase-local/rolling 指标 |
| ready/finite/parity 不检查 reference 学习 | Python 与 Slang 可以一致地产生白模 | 固定 query 学习探针、collapse 诊断、holdout 与 viewer 双证据 |

调查期间新增的 schema、单位、frame、UV、颜色空间、fallback、梯度、资源增长或采样偏差问题沿用同一原则：先定位最早失效边界，再在共享所有者修复，并为每个确认根因增加回归。

## 3. 端到端 coarse-to-fine lifecycle

### 3.1 Phase graph

Metal 正式训练改为两个 phase，总预算仍为 120k：

1. `joint-coarse-to-fine`：105k step。每步包含 `asset-tile`、`reference-evaluator`、`method-sampler` 三条 typed route，训练 codec、asset adapter、typed compiler、optimized teacher、prepare、directional/evaluator 和 proposal 全部 parameter group。
2. `qat-refine`：15k step。继续 appearance、codec auxiliary 与 proposal 目标，同时用现有 functional-call 路径模拟最终 FP16/runtime grid 精度；training-only teacher 可以冻结，但不能改变部署状态的输入语义。

Windows quick smoke 与 full-cohort smoke 缩短预算，但保留相同模型形态、route、loss、parameter ownership、schedule 类型和 QAT 执行路径。Linux long 使用同一套由平台无关生成器产生的正式 recipe。

### 3.2 Joint objective

`joint-coarse-to-fine` 每一步都计算以下目标：

- 主目标：online reference `target_f` 对 `evaluate()` 的 response、energy、peak、reciprocity 与 analytic-core 约束；主目标权重从第 1 步非零。
- 表示辅助目标：现有 semantic/normal/structured/mip/grid-QAT codec loss。它是帮助空间表示稳定的正则，不是独立预训练目标；权重可按冻结 schedule 逐步降低，但 appearance gradient 始终能穿过 codec/compiler/prepare/evaluator。
- compiler/teacher 目标：保留 functional distillation 和 teacher response，但 teacher 不能替代 reference 主目标。
- proposal 目标：从第 1 步非零，按 config 中显式、由 global step 决定的 schedule 渐进增强。

所有 loss schedule 的 kind、端点、区间和作用域写入 `phase.recipes` 并由 `validate_training_config()` 严格验证。正式值只允许在一次有上限的 Windows pilot 后冻结；pilot 不扩大 120k 预算，不自动循环调参，结果写入 `research/`。uninterrupted 与 save/resume 必须在每个 global step 得到相同权重。

### 3.3 Proposal 梯度所有权

当前 proposal 单独成 phase 时，optimizer 隐式阻止 proposal loss 更新 shared evaluator；合并为 joint 后不能依赖这个偶然边界。模型提供显式 proposal-training view：

- evaluator `f`、learned frames、shared spatial/compiler conditioning 在作为 proposal target/conditioning 时 detach；
- proposal-specific heads 仍保留梯度；
- appearance loss 继续正常更新 shared representation；
- 自动测试分别断言 proposal loss 只更新 `proposal_sampler`，joint appearance loss 能更新 codec→compiler→prepare→evaluator 完整链路。

这与 NVIDIA reproduction 的 detached learned-evaluator target 原则一致，但保持 Metal 自己的 11-component proposal 和 runtime identity。

## 4. Checkpoint readiness 与诊断预览

### 4.1 单一 readiness 入口

新增共享 checkpoint deployment assessor，输入 `TrainingCheckpoint@4`、当前 `MethodDefinition`与部署模式，输出封闭的 readiness 结果：

- exact method key、descriptor、implementation、component manifest与tensor schema；
- 从training config解析`run_class`，并使用checkpoint的`phase_name/global_step`判定是否complete；
- 模式所需parameter groups的finite/nonzero-gradient/actual-update coverage；
- `formal`或`diagnostic-evaluator`状态与拒绝原因。source/config的外部配对仍由export/catalog各自在解析locator时验证，不在assessor中复制第二套来源。

公共默认策略是 formal 必须 `phase_name=complete` 且所有 required component coverage 完整。Metal diagnostic policy 显式登记：

- evaluator preview 需要 codec/asset/compiler/prepare/directional/analytic/evaluator 相关 group 已通过 audit；
- diagnostic始终是evaluator-only，不开放PT preview；即使proposal已有coverage也移除`sample/pdf`；
- 未完成QAT或非formal run class的包只能标记为diagnostic，不能成为正式比较或release package。

checkpoint 不新增冗余 readiness 字段，避免同一事实出现两个真相；package/catalog 中保存 assessor 的规范化结果和 checkpoint SHA-256。

### 4.2 Export 与 viewer 行为

- `ncls learn export` 只接受 formal-ready checkpoint。
- `prepare_metal_catalog.py` 默认同样只接受 formal；只有显式 `--diagnostic-preview` 才可请求已解锁的 evaluator-only profile。
- diagnostic package 的 manifest capability 按固定policy收窄到`prepare/evaluate/anisotropic-frame`，validation/provenance保存readiness结果；catalog/UI同时显示`diagnostic evaluator-only`、global step和checkpoint phase，不能复用formal ready文案。
- 删除 shape-only `state-schema-compatible-preview`。descriptor 或 implementation 漂移一律拒绝；旧 20k checkpoint 只保留为根因证据，不再作为默认 catalog 输入。
- complete checkpoint 的 phase 显示使用 `complete`，不再索引越界。

## 5. Reference 调度与性能

### 5.1 先观测再改热点

在改变调度前，增加低扰动累计计数和 wall-time：group ID、hit/miss/evict、group/session/pass/resource materialization、reference dispatch、rejection rounds、batch prepare、model forward/backward/optimizer、validation/checkpoint。普通 step 不增加 GPU synchronize；GPU event 仍只在 log/audit/profile cadence 使用。

runner 每个 step 记录轻量 wall time与 counter delta，在 log interval 汇总，而不是只保留最后一步：

- phase-local 与 rolling steps/work-units per second、ETA；
- window total/median/p90/max step wall 与 prepare wall；
- group visits、cache hit/miss/evict、materialization 次数和累计时间；
- rejection candidate/count/rounds；
- 当前/峰值 allocated 与 reserved GPU memory。

### 5.2 确定性 group-local schedule

用版本化 `group-block-balanced@1` 替代逐 request round-robin：

- evaluator 与 method-sampler 由同一 `(global_step, rank, validation flag)` 选择 execution group，避免 route cursor 漂移；
- 一个 group 在一个 visit 内连续服务若干 step，使 session compile/resource 成本被真实 query work 摊销；
- 每个调度 cycle 的 group quota 与 group record 数成比例，batch 内仍在 group records 中均匀取样，从而保持按 source/state 的目标分布；
- group order、64-step block、rank partition 和冻结的validation block offset写入`online_query` identity；调度游标或等价的确定函数可精确resume；
- block multiplier 只在有上限的 matched preflight 中选择，随后冻结到 config，不根据 OS、wall clock 或观测 loss 自适应变化。

回归同时检查完整 cycle 的 visitation histogram、uninterrupted/resume 序列一致、两个 route 同 group、DDP rank 分区与 source/query identity。

### 5.3 Operation-lazy 与有界资源

`ReferenceBackendCapability.open()` 增加平台无关的 requested operations；默认仍是完整 evaluate/sample/pdf，online trainer 明确只请求 `evaluate`。group session 只创建实际需要的 pass，static bindings/resources 与 slot 仍按 group/session 生命周期管理。

任何进一步的 compiled-program 或 resource cache 只有在 profile 证明其占主导时才实现，并必须：

- 以 program/group/backend identity 为 key；
- 有显式容量和 eviction；
- active lease 期间不可回收；
- Windows D3D12 与 Linux Vulkan 走同一 cache policy，差异只在 backend capability 内部；
- close 后资源可释放，长 smoke 中 allocated/reserved/host memory 不随 cycle 无界增长。

性能验收比较相同 source/query/model/batch/work units 的 before/after trace，按 wall-time 占比处理热点；不以减少工作量作为改进。

## 6. 学习正确性与部署闭环

验证分四层，任何一层失败都在该层停止，不用下游 finite 掩盖：

1. **数据层**：固定 source/state/UV/footprint/方向/seed，检查 reference target 的有限性、尺度、颜色、frame 与 provenance；训练/holdout stream 独立。
2. **优化层**：保存初始化 prediction、各 loss、每组 gradient norm/update delta；断言 appearance 梯度贯穿 codec→compiler→prepare→evaluator，proposal 梯度所有权正确，短 fixed-stream 实验能降低目标损失。
3. **checkpoint/package 层**：uninterrupted 与 resume 序列/权重一致；同一 exact checkpoint 的 eager FP32、部署量化 Python 与 Slang 在预先冻结的 dtype/oracle tolerance 内一致。
4. **viewer 层**：reference/neural 使用同一 authored material 与输入状态；capture 同时报告 readiness、reference-neural 误差、输出均值/方差/动态范围、constant/white collapse 诊断。`ready/finite` 只表示执行状态，不能代替学习结论。

早期 diagnostic checkpoint 的最低证据是：runtime group 已真实更新，固定 holdout 的 reference error 相对同 seed 初始化下降，输出不是常量/白色塌缩，并能经 exact checkpoint→package→viewer 保持同一语义。正式质量数值仍作为 observed result。

## 7. 平台无关配置与双平台验证

现有 `windows-smoke` / `linux-smoke` / `linux-long` 命名和“从 Windows config 派生 Linux config”的方向改为平台无关生成：

- 一个 canonical recipe 生成 stratified quick smoke、full-cohort smoke 和 120k long；文件/recipe identity 不包含 Windows/Linux。
- smoke/long semantic fingerprint 只允许预算、cadence 和明示硬件资源参数不同；model、route kind、loss、schedule 类型、source/query recipe 与 sampler 语义一致。
- PowerShell 与 shell launcher 只负责选择锁定 Falcor build/device API 和 Conda 环境，不改 config payload。
- 报告同时记录共同的 semantic fingerprint，以及有意不同的 backend/device/toolchain identity。

Windows 先执行 full-shape quick correctness、跨 cache 容量的 full-cohort scheduling/performance smoke、checkpoint/package/Slang/viewer 闭环；Linux 在同一 commit 上执行 full-cohort smoke、resume probe、profile 后才启动 120k。Linux 最终 checkpoint 再回到 Windows 做同一 package/viewer 验证。

## 8. 文件所有权与任务组织

- Lifecycle/objective/model：`src/ncls/learning/methods/metal_fused.py`、`src/ncls/learning/models/metal_fused.py` 及必要的 proposal 子模块。
- Config：`tools/learning/build_metal_training_configs.py`、`configs/learning/metal-fused-full-*.json`。
- Readiness/export：`src/ncls/learning/training/checkpoint.py` 或相邻 deployment 模块、`src/ncls/cli.py`、`tools/viewer/prepare_metal_catalog.py`。
- 调度/backend：`src/ncls/learning/producer.py`、`src/ncls/references/backend.py`、`src/ncls/references/query.py`。
- Metrics：`src/ncls/learning/training/runner.py`、review/handoff 工具。
- Tests/docs：对应 unit/GPU/integration tests，`.trellis/spec/` 与稳定中文文档。

这些修改共享 method/config/query/checkpoint identity，无法独立交付而不产生相互无效的 checkpoint，因此保留为一个任务内的可审查 work package，不拆成并行 child task。

## 9. 风险与 rollback

- 新 lifecycle、query schedule 或 implementation identity 变化后，旧 checkpoint 只可作为历史证据，不能迁移为新正式结果。
- 若 joint objective 出现梯度冲突，先用按 group 的 gradient ownership/norm 证据调整已批准的辅助 loss 权重；不得恢复长 codec-only 阶段。
- 若 group locality 造成采样偏差，以完整 cycle histogram 和 holdout 证据修正 quota/order；不得退回逐 step thrash。
- 若 cache 优化需要无界 residency、改变 reference 数学或跨平台上层分支，则回滚该优化并保留已完成的 instrumentation/lifecycle/readiness 修复。
- 若 Windows 或 Linux 发现 source/query/model/work 不一致，该平台结果作废并修复生成/launcher 边界，不以“平台差异”放行。
