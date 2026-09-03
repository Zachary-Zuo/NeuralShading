# Metal 神经训练与部署根治：实施计划

## 0. 开发前与基线冻结

- [x] 使用 `trellis-before-dev` 重读 project/core/data/learning/viewer specs，并重新报告开发机状态；确认 dirty worktree，只修改本任务关联文件。
- [x] 把旧 20k checkpoint、metrics、viewer 白模 capture 与当前 implementation identities 登记为 immutable root-cause evidence；不覆盖旧 artifact。
- [x] 在修改前用新增低扰动 instrumentation 或任务 `scratch/` 探针保存 matched Windows baseline：固定 full-shape model、source/query、batch/work units，覆盖超过 resident capacity 的 execution groups。
- [x] 冻结 correctness oracle/tolerance 来源、训练/holdout 固定 query、profile work definition 与 failure action；数值写入 `research/`，不事后放宽。

**Review 0**：能够分别复现“20k evaluator coverage 为空”“shape-only preview 放行”“group miss/evict/recreate”三条已确认根因；基线没有启动新的长训。

## 1. 端到端 Metal lifecycle

- [x] 将 phase graph 改为 `joint-coarse-to-fine` 105k + `qat-refine` 15k；两阶段都包含需要的 asset/evaluator/sampler route，首阶段启用全部 runtime parameter group。
- [x] 合并 codec、appearance、teacher/compiler 与 proposal objective；appearance 和 proposal 从第 1 步非零，codec reconstruction 仅为辅助项。
- [x] 在 `phase.recipes` 中定义并严格验证 loss-weight schedule；schedule 只依赖 global step/config，smoke 压缩预算但不改变拓扑或 loss 语义。
- [x] 为 proposal 构造显式 detached training view，确保 proposal loss 不更新 shared codec/compiler/prepare/evaluator，而 appearance loss仍贯穿最终 evaluator 路径。
- [x] 更新 `_COMPONENTS` active phases、required outputs、config builder 与 platform-neutral quick/full-cohort/120k configs；旧 config identity 不复用。
- [x] 做一次有上限的 Windows pilot，确认固定流下降、各 group 梯度所有权和显存可行后冻结 schedule 系数；失败只分类并修正实现/合同，不自动增加预算或循环调参。

**Review A**：任意新 checkpoint 的第一个 audit 已覆盖 evaluator 与 proposal；不再存在无 appearance 监督的长 phase，120k 总预算不变。

## 2. Readiness 与 fail-closed deployment

- [x] 实现共享 checkpoint readiness assessor，严格验证当前 descriptor/implementation/component/tensor/config/source identity，并从 `TrainingCheckpoint@4` 现有字段推导 completed phases、cursor、coverage 与 capability。
- [x] formal mode 只接受 complete checkpoint；Metal diagnostic mode 只允许显式请求并收窄到已通过 audit 的 evaluator 或 PT capability，未完成 QAT 必须标 diagnostic。
- [x] `ncls learn export` 接入 formal gate；`prepare_metal_catalog.py` 删除默认 20k 与 shape-only compatibility，增加显式 `--diagnostic-preview` 和 capability/readiness metadata。
- [x] 修复 complete checkpoint phase 显示越界；catalog/viewer 把 execution ready、training ready、formal/diagnostic 和 collapse/quality evidence 分开显示。
- [x] unit tests 覆盖 20k historical rejection、未训练 group、partial evaluator/PT、complete、implementation drift、tensor drift、source/config drift、phase boundary 与 resume cursor。

**Review B**：旧白模 checkpoint 无法进入正式 export；diagnostic 包不能宣称超出已训练范围的 capability，也不能显示为正式 ready。

## 3. 性能可观测性

- [x] 为 reference session/backend 增加 group hit/miss/evict、session/pass/resource materialization、dispatch 与累计 wall-time counters；无强制普通 step GPU sync。
- [x] producer 暴露每步/窗口 counter delta，并把 execution group、candidate/rejection 与工作量归入结构化 profile。
- [x] runner 累积每个 step 的总 wall/prepare wall，在 log cadence 输出 window median/p90/max、phase-local/rolling throughput/ETA、cache/materialization、rejection、forward/backward/optimizer 与 allocated/reserved memory。
- [x] 修复当前“10 步只记录最后一步 preparation”和 run-global rate 掩盖 phase 退化的问题；validation/checkpoint 作为独立 work unit/时间字段。
- [x] review/handoff 报告能从异常 step 追到 route/group/cache/compile/resource/query/model/validation/checkpoint，不依赖额外 watcher。

**Review C**：用合成慢中间 step 证明 window 聚合不会漏记；真实 full-cohort smoke 能准确显示每次 group transition 的成本来源。

## 4. Group 调度与 reference session 优化

- [x] 实现版本化 `group-block-balanced@1`：同一 step 的 evaluator/sampler 对齐同一 group，完整 cycle quota 按 records 平衡，validation/rank stream 独立且 deterministic。
- [x] 将 schedule recipe 纳入 query stream identity；checkpoint/resume 恢复或重算出逐 step 完全一致的 group/source/query 序列。
- [x] backend open 增加 requested operations，online trainer 只 materialize `evaluate`；默认 public reference session 仍支持完整 evaluate/sample/pdf。
- [x] 用 matched profile 选择并冻结 block multiplier；验证 resident group、active lease、close/evict 和 resource memory 上界。
- [x] 若剩余主热点仍是重复 compile/resource materialization，再实现 identity-keyed、有容量的 backend-internal cache；没有 profile 证据不增加额外 cache 层。
- [x] 检查 source/state visitation histogram、不同 group size 的权重、rejection bias、DDP rank partition 与完整 cycle coverage。

**Review D**：相同 work definition 下不再 steady-state 每 step 构建 session/pass/resource；所有加速都能由计数和 wall-time reduction 解释，没有少算 query。

## 5. 学习、checkpoint 与部署正确性

- [x] 固定 source/state/UV/footprint/direction/seed，验证 target provenance、单位/frame/颜色、有限性和动态范围；训练与 holdout stream 分离。
- [x] 记录初始化→短训后的 prediction/target、loss、每组 gradient norm/update delta；检查 NaN/Inf、零梯度、参数遗漏、输出常量/白色塌缩和错误 fallback。
- [x] 对 uninterrupted vs save/resume 比较 group/query 序列、loss schedule、optimizer/scheduler/RNG 与最终权重；严格 checkpoint round-trip。
- [x] 对同一 exact checkpoint 运行 eager FP32→部署量化 Python→Slang parity，并验证 `evaluate`、`sample/pdf` identity、support 和 bounded execution。
- [x] 生成代表性 authored/edited Metal 的 reference/neural viewer capture；同时写 readiness、reference error、输出统计和 collapse 结论，不再以 finite/ready 单独通过。

**Review E**：早期 diagnostic checkpoint 已有可辨识、非塌缩且相对初始化改善的 evaluator；formal checkpoint 只有完整 lifecycle 后才能生成。

## 6. Windows 共享路径验收

- [x] 在 RTX 4090 上运行 full-shape quick lifecycle smoke，覆盖 joint、QAT、所有 routes/loss/groups、gradient audit、checkpoint/resume/readiness。
- [x] 运行超过 resident capacity 的 full-cohort scheduling smoke，保存优化前后 matched profile，确认无逐 step materialization、资源有界且 rolling throughput 不随 step 系统性恶化。
- [x] 运行 package/Slang GPU parity 与 Windows viewer headless capture；必要时做一次交互检查，Falcor overlay 构建结束后保持上游 clean。
- [ ] 生成 Windows handoff：commit/config/query semantic fingerprint、backend/device identity、observed throughput/ETA/memory、known risks 和 Linux 精确命令。

**Review F**：Windows 证据来自未来 Linux 使用的同一上层实现和 canonical config，不存在 OS/device API 条件选择的训练、checkpoint 或 exporter 分支。

## 7. Linux full-cohort 与 120k 交接

- [ ] 在 Linux 目标机同一 commit 运行 environment probe、canonical config verification、full-cohort smoke 和 resume equivalence probe。
- [ ] 用 instrumentation 确认用户现场“约 4 分钟/step”的实际口径及修复后热点；若出现新的结构性 defect，停止 long run、回到对应 work package，不盲目继续。
- [ ] 通过 preflight 后启动单次冻结的 120k long run；保留 tqdm、metrics、checkpoint sidecar、phase-local/rolling throughput、cache/resource counters 与异常分类。
- [ ] 完成 formal review/export/package 后，把同一最终 checkpoint 带回 Windows 做 strict package/Slang/viewer 验证。
- [ ] 根因报告逐项记录 evidence→classification→fix→regression→observed quality/time/memory→remaining risk。

**Review G**：Linux 没有平台专用修复；120k 产物与 Windows 证据共享 semantic fingerprint，最终 checkpoint formal-ready 且 Windows 可部署。

## 8. 质量门与稳定文档

- [x] 运行受影响 unit tests、CPU/static contracts、Windows GPU/integration、config generation/check、compileall、`git diff --check`。
- [ ] 按环境能力执行 Falcor/Slang/viewer tests；Linux-only 命令与结果只在 Linux 实际运行后勾选，不由 Windows 代报。
- [x] 使用 `trellis-check` 做 spec compliance、跨层数据流、lint/type/test 和遗漏回归检查；修复后重新执行受影响门。
- [x] 使用 `trellis-update-spec` 更新 learning online-training、data reference-query、viewer MDL preview 与必要 core/backend 规范；同步中文稳定文档和命令。
- [x] 使用 `trellis-break-loop` 把“阶段游标冒充已训练能力”“shape-compatible 冒充语义兼容”“有限输出冒充学习成功”“LRU+round-robin 稳态 thrash”固化为防复发规则。

## 9. Rollback points

- Lifecycle rollback：允许回到最近一个端到端 joint recipe，不允许回到 20k codec-only；任何 recipe 变化产生新 identity 和新 run。
- Readiness rollback：若 diagnostic capability 分级实现有缺陷，先关闭 diagnostic，只保留 formal complete；不得恢复 shape-only 放行。
- Performance rollback：若 locality/cache 改变采样分布或资源无界，回滚该优化但保留 instrumentation；不得通过减少 reference work 掩盖。
- Platform rollback：若某修复只能靠 Windows-only/Linux-only upper branch 成立，停止并回到 shared owner 重新设计。
