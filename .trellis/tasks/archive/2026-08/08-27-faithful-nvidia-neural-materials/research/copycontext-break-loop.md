# Bug Analysis：MaterialX formal 长跑在 CopyContext 交接点退出

## 1. Root Cause Category

- **Category**：B（Cross-Layer Contract）+ D（Test Coverage Gap）。
- **Specific Cause**：live producer 建立了 CUDA/Falcor fence 顺序，却没有把一个训练 iteration 明确结束为一个 Falcor frame；tiled reference dispatch 因而跨 iteration 累积。输入 shared buffer 还长期持有 `to_torch()` 映射并原地覆写，没有采用 Falcor 示例使用的 `Buffer.from_torch()` 提交路径。

## 2. Bayesian diagnosis 与为何先前验证未发现

初始假设：缺少 `end_frame()` 45%，输入 shared mapping 用法不稳 30%，Falcor/驱动独立缺陷 25%。可靠证据是 formal 在约 step 650 的 `CopyContext::waitForFalcor()` 退出，而同一 numerical core 的 unit/GPU smoke 已通过；静态 MaterialX evaluator 每次求值会 `end_frame()`，live training 路径没有；Falcor 自带 CUDA 示例对输入使用 `from_torch()`。

修复后，262,144-query dispatch 连续 1,000 次通过，formal 几何的 65k+65k 完整 forward/backward/optimizer 连续 1,000 step 也通过。该证据把“组合修复正确”的置信度提高到 90% 以上；由于 frame rotation 与输入 transfer 同时修正，不能把二者的独立贡献伪装成已严格分离的结论。formal 必须以新 implementation identity 从 step 0 重跑，旧 run 只保留为 implementation-defect evidence。

先前 smoke 失败的原因不是容差或样本规模，而是只覆盖少量 dispatch，没有跨过长期 command/transient-resource 生命周期；数学 parity 对该边界没有判别力。

## 3. Prevention Mechanisms

| Priority | Mechanism | Specific Action | Status |
|---|---|---|---|
| P0 | Architecture | 全部 route lease 释放后一次 `device.end_frame()` | DONE |
| P0 | Runtime contract | 输入统一为 `Buffer.from_torch()` + `wait_for_cuda()`；输出 `wait_for_falcor()` 后映射消费 | DONE |
| P0 | Test coverage | 增加 frame-boundary unit regression 与 1,000-step 完整训练 soak | DONE |
| P1 | Documentation | 固化到 data spec 与 cross-layer thinking guide | DONE |
| P1 | Formal hygiene | 旧 metrics 独立保存，修复后新 identity 从 step 0 重跑 | DONE |

## 4. Systematic Expansion

- **Similar Issues**：以后所有 Falcor live source、Slang/Falcor shared output、跨 route in-flight lease 都需要同样的 frame/ownership 审查。
- **Design Improvement**：frame 边界由 live source 在 lease 集合归零时统一拥有，runner 不按 backend 分支。
- **Process Improvement**：formal preflight 除显存和单步吞吐外，必须包含足够长的完整训练 soak；不把单 kernel soak 当成 end-to-end 证据。

## 5. Knowledge Capture

- [x] 更新 `.trellis/spec/data/reference-and-corpus.md`。
- [x] 更新 `.trellis/spec/guides/cross-layer-thinking-guide.md`。
- [x] 增加 unit regression 和 task-scoped GPU soak。
- [x] 记录失败 artifact 的分类、处置和诊断证据。
- [ ] 随本任务完成统一归档并提交；项目没有 `src/templates/markdown/spec/`，因此无 project-local template copy 可同步。
