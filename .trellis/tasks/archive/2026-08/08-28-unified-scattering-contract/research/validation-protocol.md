# Estimator 验证协议（implementation 前冻结）

## 1. 角色与来源

本协议只给数学/数值正确性设置 hard gate；capture 的 max/p99.99 与视觉 tail 是用户现场缺陷的诊断证据，不据历史观察值另造 hard threshold。

### 2026-08-28 协议修正

- trigger：OpenPBR car paint 的 coat roughness `0.02` 掠射扫描证明，Adobe native sample PDF 约为 `10^9` 且 native weight 约为 `10^-2–10^-1`；用已舍入 sample direction 调 independent PDF 会因 half-vector 切向量相消得到约 `10^-8` 的错误值，把重建 weight 放大到 `10^14`。
- invalidated evidence：此前 aluminum/glass 在非掠射随机方向通过的 sample→independent-PDF 点对点门，不能证明“用 independent query 重建 native tuple”在极窄掠射方向数值正确。
- scope impact：project-owned proposal 继续执行完整逐点恒等式；source-native API 改为分别验证 native sample tuple identity 与 independent evaluate↔pdf，同一数学 proposal 的极窄掠射 round-trip 漂移不再通过篡改 native tuple消除。
- rerun required：OpenPBR aluminum/car paint/glass 的 native identity、原冻结稳定方向域、追加 logarithmic grazing 域、Release car paint/glass capture 与 full GPU suite 全部重跑。

| gate | 类型 | source / scope | why hard | failure action |
|---|---|---|---|---|
| `evaluate.pdf == pdf(wi)` | 理论/数值正确性 | `INclsScatteringState`；所有连续 source/neural queries | 同一个 state 不能报告两种 proposal | implementation defect，停止对应 family 迁移 |
| `sample.pdf == pdf(sample.wi)` | 理论/数值正确性 | project-owned proposal；source-native 的冻结稳定方向域 | 实际采样分布与 MIS PDF 必须同式 | implementation defect，不允许 fallback |
| `weight == f·absCos/pdf` | 理论/数值正确性 | project-owned continuous sample；source-native 的冻结稳定方向域 | throughput 必须属于同一 estimator | implementation defect；检查 measure/RNG/PDF |
| native sample tuple == source oracle | 数值正确性 | source-native direction/event/PDF/weight；含 logarithmic grazing `wo` | adapter 不得拆开或重算权威 sample | implementation defect；恢复 native tuple |
| integral + null mass = 1 | 理论/数值正确性 | project-owned mixture sampler | 采样概率总质量必须守恒 | protocol 或 implementation defect |
| delta/absorb/null event matrix | 理论正确性 | MDL/OpenPBR/source proposal | 不同 measure 不能互相冒充 | implementation defect |

## 2. 冻结点对点容差

- 同一 project-owned analytic/target-code PDF 通过不同公共入口重算：`rtol=2e-6, atol=2e-7`。来源是现有 MDL formal/viewer 与 NVIDIA proposal float32 oracle 已使用并通过的容差。source-native sample tuple 与 source oracle 要求逐值 identity；independent query 之间继续用该容差。
- `sample.weight` 与重新 evaluate 经过 frame/color/measure 组合后的等式：`rtol=3e-5, atol=3e-6`。该门覆盖约几十次 float32 算术与一次 frame round-trip；不覆盖 stochastic evaluator 的独立 RNG 差异。
- unit-length direction：`rtol=2e-6, atol=2e-7`。
- 非负/finite/event/valid 是精确条件，不设宽松 epsilon。连续 valid sample 要求 `pdf.forward > 0`；delta 可为 0，null/absorb 必须返回 invalid/false 而不是有效连续 sample。

这些容差在正式结果前写定；若独立 oracle 证明不合理，先记录理论原因并修改协议 identity，不能看过失败值后直接调宽。

## 3. PDF 总质量与 sample 分布

- hemisphere numerical integral 使用固定 equal-area grid `512 × 1024 = 524,288` directions；被测 `pdf()` 在每格中心求值。
- sample valid/null probability 使用固定 seed `0x51ED270B` 与 `N=524,288` samples。
- 对允许 null mass 的 GGX mixture，比较 `integral(pdf)` 与 observed valid fraction；hard tolerance 为

  ```text
  6 * sqrt(max(p*(1-p), 0.25/N) / N) + 2/N_theta
  ```

  其中第一项是冻结的 6σ binomial uncertainty，第二项是 equal-area angular discretization 上限；不要求连续 PDF 单独积分为 1。
- 纯 cosine 或无 null 的分布要求 integral 落在同一 quadrature error 内，sample valid fraction 为 1。
- component selection 使用固定 component-frequency 检查，同样采用 6σ binomial interval；不使用看结果调整的 chi-square bin merge。

## 4. Query 与 tail 诊断预算

- 每个 family 的 contract probe 至少覆盖 normal/grazing `wo`、低/中/高 roughness、anisotropy rotation、reflection；支持 transmission/delta 的 family 额外覆盖两侧方向。
- MERL 选择 `chrome` 与 `specular-red-phenolic`；MaterialX 使用 synthetic low-roughness anisotropic fixture 和一个真实纹理资产；LayerStack 使用单界面 `alpha=0.01` 与多界面 coat/slab；OpenPBR 使用 coat/thin-film 与 transmission；MDL 使用 diffuse oracle、car paint 和 ceramic。
- 高动态范围 query tail 使用每个代表状态 `524,288` fixed-seed samples，报告 weight luminance 的 median/p99/p99.9/p99.99/max、PDF min quantile 与 invalid/null rate。数值是 report-only；project-owned 等式或 source-native identity 失败时按上表 hard gate 分类。
- viewer capture 固定 1024 spp，遵守 `.trellis/spec/viewer/capture-harness.md`。现场验收检查同一 replay 随 spp 是否持续新增空间孤立白点，并确认真实连续 flakes/highlight 未被消除。

## 5. 正式运行 identity

正式 GPU probe/capture 记录当前 Git commit、source snapshot/artifact hash、shader implementation hash、seed、query count、Falcor/Slang/MDL SDK identity。实现 bug 修复后的结果写新 artifact 路径，不覆盖旧 diagnostic。
