# 神经闭包代数方向可行性深评

> 对 `idea-neural-layered-materials-siggraph-research-2026-08-21.md` 的二次分析  
> 日期：2026-08-21  
> 当时核实的背景：NMA = EGSR 2026（Disney）；项目选择 Falcor 8.0（Slang/GFX，D3D12+Vulkan，支持 Python 与 PyTorch 互操作）；SM 6.9 于 2026-02 正式发布，Cooperative Vector 正被 DXLA 取代（2026-04 预览）；NLBRDF / NA 代码公开。项目实际锁定版本见 `AGENTS.md`，不依赖“最新版”这一表述。

> **2026-08-22 实测修订：** 本文关于“LTC-K3 大概率够用”的内容原本是实验前预判，现由实测结论取代。当前最好的基础是“精确计算顶层界面反射，再用 LTC 瓣拟合其余响应”。两个残差瓣的方向域 relative-L1 中位数/第 90 百分位为 6.73%/31.20%，三个残差瓣为 5.56%/25.24%。增加一个瓣确实有效，但导体基底和深层栈的长尾仍然过高，因此暂时不能把表示问题视为已经解决。当前应先改进残差表示，再训练结构化编译网络。完整数据见 `reports/oracle_ceiling_v0.md`。

## 0. 一句话结论

方向成立，空缺真实存在（NMA 只到 EGSR、固定三层各向同性；NLBRDF 逐方向解码服务 PT；NA 逐材质烘焙服务 PT；Belcour/Substrate 是解析近似）。但原文档把四个贡献（可组合代数、footprint 过滤、UE 系统、实测数据）全部压进一篇，按一人主力估算是 12–18 个月的量。建议把**结构化算子 + 固定 closure packet + Falcor 级 raster 系统**定为主线，UE 与自采数据降为可裁剪项；这样 8–10 个月可以出一篇 SIGGRAPH Asia / EGSR 质量的稿，再加 UE 证据冲 SIGGRAPH。

## 1. 可行性判断

### 1.1 成立的关键洞察（文档没有点透）

raster 管线里 **ω_o 在光照前就已按像素确定**（primary visibility）。所以"view-conditioned closures"在 deferred 主着色里不是妥协而是天然合法——每像素只需解码一次、对任意数量动态光复用。这是本方向相对 NLBRDF/NA 的根本优势，应写进论文 §1。代价是**次级光线**（SSR/Lumen 反射打到这类表面）需要另一个 ω_o，解法：packet 里附一个 view-averaged fallback closure，或反射时退化为 Substrate 参数混合；必须在 limitation 里讲明。

### 1.2 四个贡献的风险排序

| 贡献 | 技术风险 | 是否必需 | 建议 |
|---|---|---|---|
| 固定 K closure packet（K=2–3） | 中 | 必需 | 主线。关键实验是 iso-byte 对比 Substrate |
| 可组合/结合律代数 | 高 | 必需但要改造 | 见 §2.1，用结构保证结合律而非靠 loss |
| footprint-aware latent pyramid | 高（本身是一篇论文） | 可简化 | 见 §2.3 |
| UE 真实系统 | 工程极高 | 非必需 | Falcor 系统 + stock UE 测 Substrate 成本 |
| 自采 paired 实拍 | 中（设备/时间） | 非必需 | 用 MERL/RGL/OpenSVBRDF 替代，留 v2 |

### 1.3 文档里的两个硬伤

1. **OpenPBR 不能当 ground truth。** OpenPBR BSDF 参考实现是解析分层近似（albedo-scaling 式的 coat 组合），不是层间多次散射的物理模拟。用它做 teacher，学生学到的就是 OpenPBR 的近似，论文"逼近 path-traced layered appearance"的主张不成立。真正的 teacher 必须是随机游走：pbrt-v4 `LayeredBxDF`（Apache-2.0，可任意上下界面+中间介质）为主，PFMC（GPL，CPU，隔离）做交叉验证。OpenPBR 只用于 Stage A 单层 sanity 与"输入词汇"。
2. **"训练 2–4 层、测试 5–8 层"不该是头条主张。** 审稿人第一反应是"谁用 8 层"。真正有价值的泛化是：未见的（层类型 × 顺序 × 参数）组合零样本编译 + 改一层不重训。层数外推降为次要实验。

## 2. 可优化处

### 2.1 把“学习式叠加加倍”换成“学习基底上的散射算子”

文档把 Redheffer star-product 路线列为方案二并因"维度/正定性/尖峰"放弃。我建议反过来把它做成核心，因为这是和 NMA/NLBRDF 拉开距离的**结构性**差异，而不是"我们也能跑 raster"：

- 每层编码为学习到的方向基底（8–16 个基函数/半球）上的小矩阵 `[[R, T'],[T, R']]`；
- 层组合 = star product（含 `(I − R'_A R_B)^{-1}`）：**结合律由构造保证，能量守恒靠算子范数 ≤1 结构保证**，不需要 associativity loss 与 energy penalty 去"祈祷"；
- 尖峰问题用混合解决：近镜面能量走 Belcour 式解析通道（方差/粗糙度传播，闭式），低秩算子只承载平滑残差；
- 网络只做两件事：层参数 → 算子矩阵（encoder）；根算子 + ω_o + footprint → K closures（decoder）。

消融自然变成"学 compose 网络 vs 结构化 star product"，故事更硬，且 O(log N) 平衡树 / 子树缓存 / 编辑只重算到根路径全部免费。

### 2.2 和 Substrate 的对比必须使用相同字节预算

3 个 closure × (RGB F0 + rough xy + frame + tint) 在 fp16 下约 50–60 B/px，落在 Substrate 参数混合（~28 B）与全 slab（~108 B）之间。论文要证明的是"同 bytes 质量更好"或"同质量 bytes 更少"，两条 Pareto 都要画，否则"固定成本"会被质疑。

### 2.3 细节层次简化：不要单独学习过滤器

footprint → 层级统计量（法线协方差 → 各向异性粗糙度 LEAN/MIPNet 式；coverage；高度方差）喂给**同一个** encoder，pyramid 由"对 footprint 聚合后的层重新 compose"得到，再加 mip consistency loss 微调。保留"transport-aware filtering"的主张，但机制便宜一个数量级。dust/flake/glint 第一版只以统计 NDF（SpongeCake 式）进入，不做离散 glint。

### 2.4 其他

- 解码位置直接定在 pre-light compute pass（Falcor 里就是一个 RenderPass），不要花时间比三种位置。
- 互易性：view-conditioned 必然不严格互易，测量、报告、move on，不要过度投入 reciprocal variant。
- closure 基选引擎现成可积分的：Lambert、各向异性 GGX、clearcoat GGX、Charlie sheen；IBL 用 split-sum，但实验要**把 closure 表示误差和 split-sum 积分误差分开**（对 closure 也做一次 MC 积分对照）。
- 竞品监控：2026 年 NMA、Procedural Data Enhancement 已出，SIGGRAPH Asia 2026 录用结果将在 9 月前后公布，投稿前至少再扫一次。

### 2.5 闭包表达力：不是“解析与神经二选一”，而是选择哪一档可积分函数族（2026-08-21 补充）

原文档未分析"K 个解析瓣够不够"，这是整个方法的天花板问题。换问法：deferred 对表示的真实要求是**可积分性**——点光要逐光评估、IBL 要预过滤积分、面光要闭式积分。全神经 g（latent + 小网络逐 ω_i 评估）三项全破：成本重新与灯数耦合、IBL 退回 MC 或第二张"神经积分器"网、无 LTC。所以它只配做上界消融列。

函数族谱系：① 固定 GGX 参数族（NMA 所在档，原方案）→ **② LTC 风格瓣（候选）**：能拟合偏斜/非对称瓣、面光闭式、bytes 相近（4–5 参数/瓣），但标准 transformed-cosine LTC 并非 GGX 的严格函数超集 → ③ 固定基展开（SH / SG / 学习字典）：任意形状含瓣内颜色随 ω_i 变，点光逐基 ALU、IBL 每基一张 load-time 预过滤图；oracle 不达标时的升级路径 → ④ 全神经 g：仅上界。

实验前曾预判 LTC-K3 可能足够；2026-08-22 的 v0 实测否定了这个乐观结论。通用 LTC-K3 的方向域 relative-L1 中位数为 14.72%，而“精确顶层界面 + 三个 LTC 残差瓣”可降到 5.56%，说明先按物理路径拆分比单纯增加通用瓣更重要。不过它的第 90 百分位仍为 25.24%，导体基底和深层栈尤其困难。结论是：LTC 可以继续作为残差基元，但当前组合方式还不是最终表示。

表示上界实验已经完成第一轮。它把**表示误差**与**网络预测误差**分开后，得到的直接结论是：现在不应急着训练网络，而应先处理导体基底和深层多次传输的残差长尾。候选改进包括进一步解析分离主要基底路径、为困难材质使用不同残差函数族，以及只在必要时增加一个瓣。完成这一步后，再以相同数据划分训练结构化组合算子。

已写回主文档 §3.2 / §4.6 / §9.9 / §10 Claim 2 / §12.2 / §14 P1.0（修订表 R13），P0 清单新增 C6。

## 3. 实验搭建

### 3.1 Falcor 还是 UE：研究闭环放在 Falcor，UE 只作可裁剪的部署证据

Falcor 的优势对本题几乎是量身定做：

- 已内置 `PBRTCoatedConductor` / `PBRTCoatedDiffuse`（pbrt-v4 layered 移植）——GPU 侧随机游走 teacher 有现成起点，扩到任意层栈改动可控；
- Slang 统一训练核与渲染核（与你项目二 slangtorch/SlangPy 的做法同构）；Python 绑定 + PyTorch interop 让"Falcor 在训练回路里"成为可能（Stage D/F 的 image-space loss）；
- render graph 天然支持同一 GBuffer 走两条着色路径（见 3.2）；
- NA 基线就建在 Falcor 材质系统上，同框架复现。

劣势：2024-08 后无新 release、主线无 cooperative vector（RTXNS 用 Donut）、没有生产级 clustered 光照（自己写 1–2 周）。SIGGRAPH 近年大量神经着色论文（NA、NTC、STF）都只用 Falcor，审稿人接受。

UE：source-built + Substrate 扩展是整个计划里最贵的单项（熟悉 UE 渲染器的人 2–4 个月），且 Substrate 内部每版都在变。既然部署是"理论可行"定位，建议：**Substrate 成本基线用 stock UE 直接测**（同场景导出、同 GPU、同分辨率，记录 bytes/px 与 GPU 时间，零源码改动），我们的方法在 Falcor 里跑同一组数据；论文明说两者不在同一管线。若最后有余力，再用 NVIDIA RTX branch 或 Custom 节点做一个 UE microbenchmark 列。

### 3.2 一个项目内同时完成采集、训练、基线、PT-vs-raster 对比：可以

```
neural-closure/
├── teacher/      Slang 随机游走 layered BSDF（pbrt-v4 移植，Apache）；PFMC CPU 交叉验证（GPL，隔离目录，仅验证）
├── datagen/      Python + Falcor ComputePass：采样层栈/位置/ω_o/footprint，GPU 查 teacher，写 bin-averaged 角度 tile + variance + 计数
├── model/        PyTorch；closure 评估核用 slangtorch 与渲染端共享
├── viewer/       Falcor render graph：
│                   GBufferRaster ─┬─ ClosureDecode(compute) → DeferredLighting(解析灯+IBL+LTC)   ← ours
│                                  ├─ StochasticShade(teacher BSDF, 同灯同 IBL, 渐进收敛)          ← 材质级参考
│                                  └─ PathTracer(maxBounce=N)                                      ← 场景级语境
├── baselines/    Belcour 2018 移植、Principled/OpenPBR 参数拟合、NLBRDF（官方码，function-space）、NA（官方码）、NMA（复现）
└── ue/           可选：stock UE Substrate 场景 + 成本采集脚本
```

**PT vs raster 对比的正确做法**：两者共用同一 GBuffer、同一组灯和 IBL；参考路径只在材质内部做随机游走（相当于 PT 的 maxBounce=1 + NEE + env），**排除场景级 GI**，差异就只剩"材质表示误差 + IBL 积分近似"。再加一列完整 PT 作语境，说明本方法不承担的部分。三列同图即为 teaser 的骨架。

### 3.3 基线分三档（不必全进引擎）

- function-space（方向域误差、能量、互易）：Belcour、Principled fit、NLBRDF、NMA、ours；
- image-space 离线（同 IBL/灯下的 FLIP/LPIPS）：以上 + NA；
- real-time（GPU 时间、bytes/px）：Belcour、Principled、ours 在 Falcor；Substrate 全 slab / 参数混合在 stock UE。

## 4. 一次迭代要多久（数量级估算，未实测）

规格假设：RTX A6000（GA102）fp32 38.7 TFLOPS、bf16 tensor 155 TFLOPS、48 GB/768 GB/s；RTX 4090（AD102）fp32 82.6 TFLOPS、bf16 tensor 165 TFLOPS、24 GB/1008 GB/s。

### 4.1 瓶颈在参考数据生成器，不在网络

网络极小（encoder/compose/decoder 合计 ~10^5 参数，每样本 ~10^5 FLOP），closure 在采样方向上的解析评估也便宜。贵的是随机游走 teacher：fp32 + 分支发散 + 多次弹射。结论：**teacher 生成用 fp32 强的卡，训练用带宽高的卡**，且 teacher 结果必须缓存为角度 tile 复用，不能每 step 在线算。

### 4.2 两档配置

**Kill-test 配置**（Phase 1）：5k 层栈 × 32 texel × 16 ω_o × 128 ω_i bin × 64 walks ≈ 2×10^10 次游走。

| | 数据生成 | 训练（~10^5 step，batch 16k） | 评估 | 一次迭代 |
|---|---|---|---|---|
| 单卡 4090 | ~40–60 min | ~1 h | ~30 min | **~2–3 h** |
| 单卡 A6000 | ~1.5–2 h | ~1.2–1.5 h | ~30 min | ~3.5–4 h |
| 5×A6000（生成并行，训练单卡） | ~25 min | ~1.2–1.5 h | ~30 min | ~2.5 h |

**完整配置**（Phase 2，含空间 patch + footprint pyramid + image-space loss）：50k 层栈 × 64 texel × 8 ω_o × 256 bin × 64 walks ≈ 4×10^11 次游走。

| | 数据生成（缓存，偶尔重生成） | 训练（3–5×10^5 step，含可微 IBL 着色 loss） | 一次完整迭代 |
|---|---|---|---|
| 单卡 4090 | ~6–12 h | ~12–30 h | **~1–2 天** |
| 单卡 A6000 | ~12–24 h | ~15–40 h | ~1.5–3 天 |
| 5×A6000 DDP 单个大 run | ~3–5 h | ~5–12 h（小模型 DDP 有效加速约 3–3.5×） | ~0.5–1 天 |
| 5×A6000 跑 5 个消融 | ~3–5 h | ~15–40 h（5 个同时） | 单位时间产出最高 |

### 4.3 哪种更合适：两者分工，不是二选一

- **4090**：开发机、交互调试、Falcor 实时 benchmark（D3D12 只能在 Windows）、kill-test 的快速迭代。fp32 约等于 2 张 A6000，teacher 生成效率最高的单卡。
- **5×A6000**：Linux 集群走 Falcor Vulkan 后端做 headless 数据生成；并行跑消融/超参扫描（这类小模型 DDP 合并成一个大 run 的边际收益低，平行 5 个实验更划算）；48 GB 对"缓存 tile 常驻显存 + Falcor 在训练回路里"有实际帮助。
- 若只能选一张卡做主力：选 4090；5×A6000 在并行消融与生成上有 2.5–4× 的吞吐优势，但单实验 wall-clock 不比 4090 快多少。

## 5. 周期与难度

### 5.1 分阶段估算（一人主力 + 零星工程协助）

| 阶段 | 内容 | 原文档估计 | 我的估计 |
|---|---|---|---|
| P0 基础设施 | Slang 随机游走 teacher、datagen、Falcor 双路径 viewer、Belcour/Principled 基线 | 2 周（只算文献） | **6–8 周** |
| P1 kill test | 2–3 层 reflection-only，K=2/3，方向光+IBL，变深度初测 | 4 周 | 4 周 |
| P2 方法 | 结构化算子、spatial、footprint pyramid、function/image benchmark、NLBRDF/NA/NMA 对比 | 8–12 周 | **10–14 周** |
| P3 系统 | Falcor deferred + clustered + LTC + 性能；portable HLSL | 未单列 | 6–8 周 |
| P3' UE | source-built 插件 + Substrate 扩展 | 未单列 | +8–12 周（可裁） |
| P4 评估写作 | 实测数据集、时域视频、用户实验、多 GPU、写作 | 未单列 | 8–10 周 |
| **合计** | | 隐含 ~7 个月 | **不含 UE 8–10 个月；含 UE 10–13 个月** |

### 5.2 投稿窗口（按历年惯例的大致时间）

- SIGGRAPH 2027（约 2027-01 下旬）：距今 5 个月，完整版不可行；除非只投 kill-test 级别的 short/EGSR。
- EGSR 2027（约 2027-04）/ HPG 2027：无 UE 的精简版可达。
- **SIGGRAPH Asia 2027（约 2027-05）：建议主目标**，Falcor 系统 + 结构化算子 + LOD 简化版 + 公开实测集验证。
- SIGGRAPH 2028（约 2028-01）：加 UE 证据与自采数据的完整版。

### 5.3 难度：高（4/5），按难度排

1. **通用模型在未见层栈上匹配尖锐高光**——NA 正是因为这点选了逐材质训练；这是 kill test 要回答的核心问题，P1 止损条件里应放第一条。
2. GPU teacher 基础设施（任意层随机游走 + 方差控制 + 缓存格式）。
3. footprint-aware pyramid（即使简化也需 2–4 周调到不闪）。
4. UE Substrate 集成（纯工程，但吞时间）。
5. 实测/自采数据（受设备与许可约束）。

## 6. 建议调整后的主张

> 在限定原子层词汇内，把任意顺序与空间变化的层栈编译为固定 K≤3 个 raster closures；层组合由学习基底上的散射算子以 star product 完成（结合律与能量守恒由构造保证），未见的层类型/顺序/参数组合零样本成立、改一层不重训；在 deferred 管线中以接近 Substrate 参数混合的 bytes/px 与 GPU 成本取得接近随机游走参考的局部外观。

与原主张的差异：去掉"5–8 层"头条、去掉"UE 必需"、teacher 从 OpenPBR 改为随机游走、结合律从 loss 改为结构。

## 4.4 修订：4090 负责生成数据，A6000 负责训练（2026-08-21 补充）

用户提议 teacher 数据全部在 4090 预生成、A6000 只用缓存训练。采纳，并做如下约束：

- **缓存的半球响应 tile 足以计算任意光照下的 image-space loss**：像素 = ∫ L_i(ω) g(ω) dω，把 HDRI 投到同一套 ω_i bin，image loss = bins × HDRI 矩阵乘；Stage D/F 也不需要 teacher 或渲染器在训练回路里。
- **按数据用途与版本生成**：先做约 3.67 MB、adaptive 高 spp 的 `v0-oracle` 选表示；closure 定稿后才做约 4.59 GB 的噪声感知 `v0-train`。v1 完整约 92 GB 或 half-vector 128 bin 瘦身约 46 GB，每版带 teacher hash + 先验版本。
- **存储粒度**：每 bin 存 RGB A/B 两组独立半样本均值 + 计数（fp16，14 B/bin），固定 half/difference-angle 参数化；不存 loss/最终图。A/B 同时供 average-vs-average 监督与方差估计。
- **datagen 使用 Falcor 8.0 Python ComputePass 与同一 Slang kernel**；4090 走 D3D12，A6000 后续走 Falcor Vulkan，避免 P0 同时维护两代 Slang API。
- **保留在线 teacher 小通道仅用于验证**：对从未缓存的新层栈/新方向现算，最终泛化实验必须走此通道，避免只测"对固定 bin 集的记忆"。
- 训练数据 memmap fp16 shard，整 shard 拷入 48 GB 显存做 GPU 端采样，dataloader 瓶颈消失。
- 收益：单次训练迭代不变（A6000 15–40 h / 完整配置），省每轮前 6–12 h 生成，5 个消融并行不重算 teacher，Phase 2 吞吐约翻倍；4090 解放给 Falcor 开发与实时 benchmark。不能省：数据集版本切换仍需 4090 半天到一天；Stage F 若要 Falcor 真在回路仍需 4090。
