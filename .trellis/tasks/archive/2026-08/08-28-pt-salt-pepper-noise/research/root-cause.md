# 根因：第一条 continuation 的单样本 throughput 尾部污染后续所有 strategy

## 1. 归因方法

task-scoped diagnostic shader 不改变 `state.evaluate/sample/pdf`、RNG 分布或 sample tuple，只把第一条 continuation 之后的 radiance 分类到三个显示通道：

- R：reflection/transmission event 与 geometric hemisphere 不一致后产生的贡献；
- G：几何有效的 BSDF-sampled direction 直接命中 environment；
- B：secondary-hit environment NEE 与更深路径。

beauty 与 AOV 都使用同一 replay、480×540 单 slot、1024 spp。`scratch/analyze_contribution_aov.py` 在 beauty 上选取 top 512 个 3×3 local-median 正向残差像素，再读取同坐标的 AOV 通道并按残差加权。

## 2. 结果

| 材质 | invalid geometry | BSDF-hit environment | secondary NEE / deeper |
|---|---:|---:|---:|
| MDL ceramic | 0.000000% | 97.6461% | 2.3539% |
| MDL car paint | 约 0.000000001% | 97.8924% | 2.1076% |
| OpenPBR car paint | 0.0075% | 99.7204% | 0.2722% |

三份 AOV 与 raw EXR 均 finite。对应产物：

- `artifacts/diagnostics/pt-salt-pepper-noise/aov-ceramic-bounce1/`
- `artifacts/diagnostics/pt-salt-pepper-noise/aov-mdl-carpaint/`
- `artifacts/diagnostics/pt-salt-pepper-noise/aov-openpbr-carpaint/`

## 3. 第一次修复后的二次归因

第一版正式修复采用 `4 light direct + 4 independent BSDF direct + 1 continuation`。它消除了 BSDF-hit environment 的单样本 strategy，却仍能在 ceramic 底座看到少量孤立亮点。若在整张图选 top residual，会被球体上的合法连续高光污染；把分析区域固定为底座 `y=275..475` 后，新的 contribution AOV 显示 top 512 正向残差中 94.36% 来自第一条 continuation 命中几何后的 secondary direct。

再把 secondary direct 拆成 light/BSDF 两个 strategy 后，残差份额为：

| strategy | 底座 top 512 residual 份额 |
|---|---:|
| secondary BSDF | 81.34% |
| secondary light | 18.66% |

两个通道在这些坐标上的相关系数为 0.919。两侧同时变亮说明 secondary MIS 本身不是单侧失配；它们共同继承了那条罕见 primary continuation 的大 throughput。只增加 secondary direct samples 会平均条件方差，却无法平均“是否进入这条高-throughput suffix”的上游 Bernoulli 事件。

对应证据位于：

- `artifacts/diagnostics/pt-salt-pepper-noise/after-v2-contribution-aov/`
- `artifacts/diagnostics/pt-salt-pepper-noise/after-v2-secondary-strategy-aov/`

## 4. 最终结论与修复

H1（shading normal / geometric normal 路径域错误）不是本次 firefly 的主因，不实施 speculative normal adjustment。H2 成立，但更准确的根因不是“没有 MIS”：旧 PT 已有标准 environment NEE + power MIS。真正缺口是 primary BSDF strategy/continuation 只有一个样本；HDR environment、glossy lobe、复杂遮挡与底座互反射使它产生极长的正向 throughput 尾部，之后所有合法的 secondary contribution 都被一起放大。

最终实现把 4 个 primary BSDF strategy samples 直接迁成 4 条完整 path samples：

- miss environment 时按 `4·p_bsdf` 对 `4·p_light` 做 power MIS；
- hit geometry 时各自追踪完整 suffix，native weight 除以 4；
- secondary 及更深顶点保留 4+4 direct pool 和 1 条 continuation，避免指数 path tree；
- source/package 使用同一环境 math 与 Falcor `UniformSampleGenerator`，不重建 native tuple，不裁剪、不 denoise。

曾试验自定义 rank-1 lattice path sampler，只带来很小的 observed 改善，且会把 sampler 设计变成本项目自有风险；正式实现已删除该实验并使用锁定 Falcor 的 production generator。

## 5. 冻结视觉结果

以下统计只在 480×540 单 slot 的底座 `y=275..475` 上报告，阈值为 local-median score > 5 且正残差 > 0.1；它是解释视觉变化的 report-only 指标，不是新的验收 gate。

| 材质 | 指标 | before | after |
|---|---|---:|---:|
| MDL ceramic | 标记像素数 | 103 | 56 |
| MDL ceramic | score p99.9 | 6.74 | 3.36 |
| MDL ceramic | RSE | 0.02445 | 0.01749 |
| MDL car paint | 标记像素数 | 41 | 7 |
| MDL car paint | score p99.9 | 2.93 | 1.20 |
| MDL car paint | RSE | 0.02311 | 0.01892 |
| OpenPBR car paint | 标记像素数 | 40 | 26 |
| OpenPBR car paint | score p99.9 | 3.36 | 1.42 |
| OpenPBR car paint | RSE | 0.01966 | 0.01587 |

视觉上，after 中的剩余标记主要沿连续高光边界、tile 边、接触边和 car-paint flake 区域分布；不再呈现 before 中散落的稀疏白点。observed slot cost 从 ceramic 8.80 ms 增至 44.38 ms、MDL car paint 9.89 ms 增至 103.71 ms、OpenPBR car paint 6.48 ms 增至 47.05 ms；这是固定 4-path primary pool 的明确成本，不把一次 timing 写成硬门槛。

R7 视觉审计还发现 OpenPBR ideal glass 在 1024 spp 下仍有大面积颗粒。它来自 delta reflection/transmission 与内部多跳路径的高方差，空间签名和 ceramic/car-paint 的稀疏 first-continuation firefly 不同，也不是本次修改引入的回归。若要继续解决，需要单独评估 specular path splitting、bidirectional/caustic estimator 或更高效的路径采样，不能伪装成 environment MIS 参数修补。
