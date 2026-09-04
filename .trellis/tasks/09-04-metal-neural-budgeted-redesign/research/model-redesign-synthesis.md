# Metal NVIDIA-class 模型重构综合

## 1. 需要推翻的旧假设

旧 `metal_fused_full_v1` 按“先融合尽可能多的机制追求质量，再消融”的需求设计。当前证据说明以下推断不能继续使用：

1. **大结构学好后可以自然缩小**：full evaluator 的 185,088 MAC 和 2,816 B prepared state 与 NVIDIA faithful 的 9,664 MAC / 96 B 不在相邻容量区间；full 上的成功无法说明同拓扑缩到约二十分之一后仍成立。
2. **多分支等于多语义保真**：semantic/normal head 接受辅助监督，但 evaluator 实际消费的是跨 slot 平均后的 64D `structured`；监督与 runtime 数据流没有闭合。
3. **半分辨率 high grid 加大 decoder 就能恢复任意细节**：当前 high grid 先自适应平均到半分辨率，runtime 又在 patch 中取中心平均；没有显式 intra-cell 坐标重建，微小划痕丢失不是单纯训练步数问题。
4. **更多 learned frames、angular tables 和 lobes 会自动覆盖复杂 graph**：旧方法同时使用三 learned frames、四级 angular bank、六个 core lobe 和四个 residual lobe，但没有 matched 证据说明这些读取和状态对 Metal 质量有净贡献。
5. **一个联合 total loss 足以表示进度**：连续 PDF NLL 可为负，且旧 appearance 指标对 RGB 色度、亮峰和空间高频不敏感；total loss 下降不能排除偏色与细节消失。

## 2. Source 事实对模型的约束

vMaterials 2 Metal 的 692 个 opaque exports 不是 692 个互不相关的黑盒：

- 126/127 个 module 使用 GGX Smith，只有 `Aluminum_Anodized` 例外；
- 112 个 module 使用 weighted layer，68 个含 diffuse，80 个使用 tangent-space normal texture；
- 主体是 13 种 metal identity × 7 种 finish，特殊 recipe 再增加 paint、patina、rust、crack、damage 与结构纹理；
- 每个 export 有 9–31 个 typed 参数，UV/access 和 rounded-corner/frame 参数存在确定性执行语义，不应交给 MLP 猜测；
- texture replacement 的合法首版含义是替换已编译的 finish bundle，并保持 source-backed recipe compatibility，不是接受任意外部图片。

这支持“紧凑解析主结构 + 小型 neural correction + 统一压缩空间资产”，但不支持把 source 预先反演成统一 Principled/layer 参数后再称为 GT。解析 core 只是容量先验和 sampling proposal，最终输出仍直接拟合 reference 的线性 RGB `f`。

## 3. 候选比较

| 候选 | 主要形态 | 优点 | 本任务判断 |
|---|---|---|---|
| 缩小旧 full | 减少旧 U-Net、attention、angular bank、lobe 数 | 改动表面小 | 淘汰为主线：容量分配和信息路由问题仍在，且缩放结论无依据 |
| NVIDIA-style direct | 单一 z8、两个 frame、约 3×64 MLP 直接输出 RGB | 已有 shader/runtime 证据，天然落在目标预算 | 保留为 matched compact control；负责检验 analytic core 是否真的有净贡献 |
| semantic-hybrid | 统一空间 latent、response-ready semantic state、双解析 lobe、约 3×64 correction/gate MLP | 高频 frame/roughness 由直接路径承担，RGB core 显式，移动峰不全压给 MLP | 推荐主候选 |
| warped H/D plane | spatial/half/difference 显式 planes + 小 decoder | 可能直接存窄峰 | 首版不选：增加方向随机读取、过滤与 per-asset bytes，违反先从最低必要读取开始的原则 |
| 大 teacher / hypernetwork | 更宽 evaluator 或生成大量状态 | 可辅助归因容量瓶颈 | 只允许 ≤4× 主 profile neural MAC 的 diagnostic，不进入主结论 |

选择 semantic-hybrid 不是因为解析模型被当作目标表示，而是因为 Metal source 的主峰确有强 microfacet 结构，且现有 Hybrid Neural-Microfacet 与项目 LayerStack 结果都支持“analytic core 降低小网络追移动峰的负担”这一可证伪假设。任务仍用同预算 direct control 检查该假设；若 hybrid 没有质量净收益，选择更简单的 direct 形态。

## 4. 推荐表示：Budgeted Semantic Hybrid

### 4.1 三段状态

```text
native typed state
  → instance-time compact compiler
  → ProgramState

source texture bundle
  → offline asset compiler / bounded latent refinement
  → one Detail RGBA8 hierarchy + one Context RGBA8 hierarchy

ProgramState + two filtered latent reads + surface/frame + wo
  → small semantic decoder
  → PreparedState ≤ 192 B

PreparedState + wi
  → two analytic basis lobes + small correction/gate MLP
  → linear RGB f
```

关键变化是：runtime 不再逐 slot 解码。最多 9 个 source texture slot 只存在于离线 asset compile 和训练监督；部署 asset 把 compatible source roles 合成为固定两张 latent hierarchy。离散资源选择若无法安全合并，则编成 asset 内的 variant table，由 instance 选择一个 variant，但每次 `prepare()` 仍只读一对 plane。

### 4.2 空间资产

- `Detail`：每个 response mip 一张 full-resolution `RGBA8_SNORM` latent，承担 normal、scratch、roughness edge 和 mask 等高频信息。
- `Context`：每个 response mip 一张四分之一线性分辨率 `RGBA8_SNORM` latent，承担颜色、低频污染、recipe/context 与 correction code。
- 每个 response mip 独立对该 footprint 下的 reference/semantic target 拟合；fractional LOD 使用 NVIDIA-style adjacent-level stochastic choice，再在选中 level 内 bilinear filter，避免把语义不相同的 latent 直接 trilinear。
- decoder 输出必须被 evaluator 真实消费，至少包括 primary frame、roughness/anisotropy、RGB optical modulation、secondary mixture 和 8D local correction code；normal/roughness/mask 辅助监督不再通向旁路 head。
- asset plane 对同一 finish/asset 的全部训练参数状态共享，typed compiler condition 随 state 变化，防止把 authored default 烘焙成看似可编辑的 latent。

这不是预先断言 `RGBA8 + quarter context` 一定具有最佳 rate-distortion；它是满足两次固定读取、保住 full-resolution detail、并显著降低旧 per-slot 读取的首个可判定形态。`B_asset`、缓存和 streaming 结果保持 report-only，待 matched measurement 后决定后续压缩研究。

### 4.3 Typed compiler

旧四层 64D Transformer 改为 responsibility-aware set compiler：

```text
token_i = E_semantic + E_type + E_responsibility + E_discrete + W_value φ(value)
g = masked_mean(token_i) + E_graph + E_schema + E_recipe + E_metal + E_finish
ProgramState = MLP(g ⊕ canonical_optical)
```

- token width 16；最多 32 token 的 source 上限保留；无 attention block。
- UV/access、resource variant、rounded-corner/frame 等确定性责任字段不进入 learned guess，继续由 source adapter/renderer 执行。
- compiler 只在实例创建或 typed edit 时运行，不进入 per-hit 成本；ProgramState 只携带紧凑 lobe baseline、8D condition、asset variant 和 flags。
- optimized per-state state 只作为 target-visible control。pure compiler 的 G2/G2s 结果单独报告，不能由 optimized control 代替。

### 4.4 `prepare()` 与 prepared state

首版 dense 路径为约 `24→32→32→24`，约 2,560 MAC/shading point，再加两次 filtered latent read、frame 构造、LOD 和解析标量操作。输出布局目标为 128–160 B，上限 192 B：

| 字段 | 建议存储 |
|---|---:|
| filtered latent / local code | 8–16 FP16 |
| 两组 `(tangent, normal)` frame seeds | 12 FP16 |
| 两组 projected `wo` | 6 FP16 |
| 两个 compact analytic lobe state | 14–16 FP16 |
| correction condition / proposal weights | 8–12 FP16 |
| flags、variant、validity 与对齐 | 16–32 B |

bitangent 从 tangent/normal 重建，normalize、LOD、PDF normalization 与能量敏感累积使用 FP32。最终 layout 由生成器精确验证；表中只是设计分配，不作为放宽 192 B 上限的理由。

### 4.5 `evaluate()`（v1 初始形态，已由§10修订）

方向输入只保留无随机读取的表示：两 frame 下的 `wo/wi` 投影、稳定 half/difference 坐标、cosine/valid flags 和 8D local correction code。移除四级 angular bank。

主 MLP 采用 `28→64→64→64→6`，dense MAC 为：

```text
28×64 + 64×64 + 64×64 + 64×6 = 10,368 MAC/direction
```

六个输出分成 positive correction RGB 与 analytic gate RGB：

```text
f_hat = f_positive + gate * f_core
```

`f_core` 固定最多两个 basis lobe：primary anisotropic conductor/Beckmann-compatible specular，以及 optional dielectric-specular/diffuse-contamination secondary。recipe/type 由 ProgramState 决定，静态循环上限不变。解析标量、activation、normalization 和 transcendentals另行登记并以真实 GPU 时间约束，不能用 10,368 MAC 隐藏。

同预算 direct control 复用完全相同的 asset、compiler、prepared state、方向输入和 `28→64→64→64` body，只把输出改为 direct positive RGB，约 10,176 MAC。这样比较只回答 analytic hybrid head 的净贡献。

### 4.6 `sample()/pdf()`

首版不保留 11-component learned proposal。proposal 直接复用两个 analytic lobe 的 frame/roughness/weight，再加一个正权重 hemisphere fallback，共 3 components；`sample()` 与 `pdf()` 对同一 prepared state 和同一折叠映射做严格 parity。

neural correction 不必等于 proposal，但 PDF 必须等于实际采样分布，因此 estimator 仍无偏，差异只反映 variance。sampler quality 在 evaluator 稳定后报告；它不成为本任务扩大 evaluator 的理由。

## 5. 训练与观测重构

### 5.1 单材质先行

首个 probe 固定到已暴露问题的 `Tungsten Brushed / Medium Light Brushing` exact source。训练仍通过 GPU online reference 生成，不保存 response batch。训练与独立验证分别固定：

- coherent spatial tile、相邻 UV pair 和原生 footprint；
- uniform、near-reflection、grazing、cosine 四类方向 quota；
- RGB peak、brush/scratch spatial slice 和参数 state；
- eager → FP16/INT8 quantized Python → Slang/package 三层输出。

比较 old full（历史）、new direct、new hybrid；teacher 仅在 direct/hybrid 都失败且先完成 failure classification 后启用。

### 5.2 Loss

- `L_log_rgb`：逐通道 train-only scale 的 log-domain robust loss；
- `L_linear_rgb`：solid-angle weighted 线性 RGB；
- `L_chroma`：在有效亮度区比较去均值 log-RGB，直接暴露高光偏色；
- `L_peak_rgb`：按各通道 top-energy 支持集加权，不只使用 luminance；
- `L_spatial_gradient`：相邻 UV、同方向 query 的响应差分，检查 microdetail；
- `L_core`：只监督 analytic core 对 reference 的 response coverage，防止 gate/correction 完全绕开 core；
- `L_semantic_runtime`：监督 evaluator 实际消费的 normal/roughness/mask/optical fields；
- proposal objective 与 appearance objective 分离显示并保持梯度 ownership。

训练从 step 1 起包含最终 evaluator，不再有 codec-only 的不可部署阶段。若使用 curriculum，必须明确改变方向 mollification、LOD/空间频率或 peak quota，并在 identity 中冻结，不能只改变 proposal weight 却命名为 coarse-to-fine。

### 5.3 选择规则

observed quality/time/memory 不作为任务 hard gate。单材质 pilot 按以下预登记规则决定下一步：

1. hybrid 在峰值、色度或 spatial detail 上相对 direct 有稳定收益且成本未越 hard budget：保留 hybrid；
2. 两者质量统计不分或 direct 更好：选择更简单的 direct；
3. 两者都出现相同细节/颜色失败：先归类 asset/query/loss/quantization defect，不自动增宽 evaluator；
4. 仅 eager 成功而 quantized/Slang 失败：修量化或 parity，不能改训练结论；
5. 任何主候选 `evaluate > 20,000 MAC` 或 state `>192 B`：降为 diagnostic，不能进入主 profile。

## 6. Matched runtime 口径

同一 RTX 4090、同一 precision、同一 query/state buffer、同一 warm-up/sync 下测四条路径：optimized MDL reference、NVIDIA faithful、旧 full 和新候选。每条均报告 coherent 单材质与同 execution group 多 state 两种 workload：

- `prepare` only；
- prepared `evaluate` only；
- `prepare + 1 evaluate`；
- `prepare + N evaluate`，`N∈{4,8}` 的摊销；
- `sample/pdf` 解析成本另列；
- median、p90、吞吐以及 state write/read、weights、asset residency。

viewer 整帧 timing 只作为 lifecycle 补充，不替代 kernel matched result。测得的 `prepare`、读取、asset 与 latency 先登记 report-only，再在 formal 候选 freeze 前由用户确认是否升级为产品 hard target。

## 7. 任务组织判断

当前不拆成多个 Trellis child。原因是 observability、runtime harness、asset layout、evaluator 和 deployment 都共同决定同一个 method/profile identity；单独启动某一 child 会允许下游在上游身份尚未冻结时产生不可比较 checkpoint。执行计划仍设置独立 rollback point 和逐项验收，任一阶段可以停止并报告，不依赖子任务树隐式表达依赖。

## 8. 主要风险

- 两个 analytic lobe 对复杂 paint/patina recipe 可能覆盖不足；这由 direct control 与 recipe-stratified结果检验，不自动增加 lobe。
- 合并九个 source slots 可能损害 typed edit 的解耦；必须让同一 asset latent 跨参数 state 共享，并对 parameter sweeps 单独评测。
- full-resolution detail plane 可能仍有较高 asset bytes；首版先证明读取与高频路径，后续 bit-rate 只能作为独立 matched 研究。
- stochastic mip 会增加单 query 方差；需要 deterministic expectation probe 与 viewer temporal evidence，但不能改成未经验证的 latent trilinear。
- 旧 full checkpoint 与新 plugin identity 不兼容；旧实现只作为显式 historical benchmark/control，新的训练和 package 不提供 silent reader 或 converter。

## 9. 证据来源

- `research/initial-evidence.md`：20k 质量、loss 与旧成本审计。
- `docs/research/model_candidates.md`：M1/M2/M4/M6 机制与既有实验边界。
- `08-30-vmaterial-metal-neural-system/research/vmaterials2-metal-audit.md`：692 opaque、closure、texture 与 typed 参数事实。
- `08-29-neural-shading-appearance-literature/research/papers/zeltner-2024-real-time-neural-appearance-models.md`：z8、two-frame、small evaluator、filter 与 runtime 证据。
- `08-29-neural-shading-appearance-literature/research/papers/2026-hybrid-neural-microfacet-brdf.md`：analytic core + correction/gate 结构与边界。
- `08-30-metal-runtime-deployment/research/runtime-evidence.md`：旧 full 的静态与 viewer 成本。

## 10. step-512 修订：完整 semantic state 进入 evaluator

高吞吐v1 pair在共同step512显示：hybrid/direct总体appearance与peak均在学习，但双方spatial-gradient都停留在约`0.282–0.283`。实现审计进一步确认上述v1结构只把24维semantic state的前8维拼入28维方向输入；decoder后16维虽写入PreparedState，却没有进入neural response body。这使“全部response-ready语义都被最终求值器消费”的原候选前提失效。

当前v2保持其他matched轴不变，把完整24维semantic state拼入方向输入，形成`44→64→64→64→6`，evaluate dense MAC为11,392、PreparedState仍160 B、asset reads仍2。v1两种profile及其step512 artifacts作为历史诊断；v2使用新的profile/method/recipe身份fresh重跑。这个改动首先检验信息连通性，而不是用新loss或更宽网络掩盖通路缺失。

## 11. v2 空间诊断与 v3 短路径

v2共同step512证明完整semantic输入能改善direct的平均log/linear/chroma，却未改善one-texel spatial。在线paired分解显示target log梯度约0.285，而预测只有0.001–0.004；raw patch差异经过Detail与semantic decoder连续缩小约一个数量级和数倍。fixed-batch spatial-only高学习率实验仍无法充分拟合，说明此处不是简单增加训练step或loss权重的问题。

v3给Detail plane明确的高频frame责任：训练端role-aware encoder仍学习四通道含义，prepare把这四通道直接residual到semantic前四个frame分量，同时保留原semantic decoder修正。它不假定source texture的原始通道语义，不增加参数、state、read或evaluate MAC，只增加四次prepare加法。该修订是本轮最后一个自动结构修订；若合同正确但质量仍差，结果进入下一轮候选方向而不继续版本循环。
