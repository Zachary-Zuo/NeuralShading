# Bug Analysis：统一材质接口下的单条 primary continuation 长尾

## 1. Root Cause Category

- **Category**：E - Implicit Assumption，同时暴露 D - Test Coverage Gap。
- **Specific Cause**：架构已经统一 `prepare/evaluate/sample/pdf`，MIS 公式也存在，但 renderer 隐含假设“一条 BSDF continuation 足以代表 primary 后的全部 path suffix”。4 个 light/BSDF direct samples 只能降低给定 suffix 条件下的方差，无法平均是否进入罕见高-throughput suffix 的上游事件。既有 static/API 测试验证了接口调用和 PDF multiplier，没有验证 path-prefix sample ownership。

## 2. Why Fixes Failed

1. **只怀疑 shading/geometric normal**：bounce 0/1 空间签名支持该先验，但 contribution AOV 显示 invalid geometry 份额接近零，继续修改 frame 会是 speculative surface fix。
2. **只把 environment MIS 从 4+1 改为 4+4 direct pool**：修复了 technique sample-count 不对称，却把 1 条独立 continuation 保留下来；secondary light/BSDF 两侧仍共同继承它的大 throughput，属于 incomplete scope。
3. **只匹配 environment CDF 与 bilinear lookup**：这是正确的 PDF/support 修复，但当前 HDR 上对目标亮点改善很小，不能解释 strategy AOV 的高相关性。
4. **试验自定义 rank-1 lattice sampler**：只小幅改变 observed 尾部，没有改变单条上游 path ownership；且引入新的 sampler 风险。正式实现删除该实验，改用 Falcor production generator。

## 3. Prevention Mechanisms

| Priority | Mechanism | Specific Action | Status |
|---|---|---|---|
| P0 | Architecture | primary BSDF pool 同时拥有 environment miss 与 geometry-hit suffix，不再另设单条 primary continuation | DONE |
| P0 | Test Coverage | static test 固定 4+4 technique PDF、primary path pool、secondary direct/continuation 分工 | DONE |
| P0 | GPU oracle | continuous/delta MIS 与 Falcor `UniformSampleGenerator` 在真实 Slang/D3D12 上验证 | DONE |
| P1 | Documentation | 更新 viewer convention 与 cross-layer estimator ownership checklist | DONE |
| P1 | Diagnosis | AOV 必须同时按 path prefix 与 strategy 拆分，并在用户问题区域而非整图选 residual | DONE |
| P2 | Separate research | ideal glass 的 delta/内部折射方差另建任务评估 specular splitting 或 bidirectional/caustic estimator | TODO，不属于本任务 |

## 4. Systematic Expansion

- **Similar Issues**：任何 layered/glossy/transmission source，即使 canonical sampler 完全正确，也可能因 renderer 对 path-prefix sample 数的选择形成共同上游长尾。
- **Design Improvement**：把 estimator ownership 从“每个 hit 有哪些函数”扩展为“哪组 samples 拥有路径空间中的 miss/hit 分支”；sample-count 常量、MIS math 和 path pool 必须由共享 transport owner 定义。
- **Process Improvement**：视觉 firefly 先做 bounce isolation，再做 prefix/strategy AOV；若两个 downstream strategy 在离群像素高度相关，应优先向共同上游追溯，不继续微调 downstream PDF 或 RNG。
- **Knowledge Gap**：统一 scattering ABI 只保证材质数学语义，不能自动保证 renderer 的 Monte Carlo 方差结构；这两个正确性层必须分开测试。

## 5. Knowledge Capture

- [x] 更新 `.trellis/spec/viewer/conventions.md`。
- [x] 更新 `.trellis/spec/guides/cross-layer-thinking-guide.md`。
- [x] 将任务级根因与视觉审计写入 `research/root-cause.md`、`visual-verification.md`。
- [x] 项目没有 `src/templates/markdown/spec/`，因此不存在需要同步的本地模板副本。
- [ ] ideal glass 高方差只有在用户授权扩大范围后再创建独立任务。
