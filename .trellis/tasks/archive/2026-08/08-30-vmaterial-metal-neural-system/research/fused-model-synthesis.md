# Metal-v1 质量优先融合模型综合

## 1. 当前结论

Metal-v1 不从 M1–M6 中选择一个“赢家”，而是把它们放回各自适合的生命周期：

- NTC-style hierarchical grids 承担可替换 finish asset 的空间高频与 mip；
- target encoder + autodecoder/QAT 承担固定 source texture bundle 的离线压缩；
- typed source compiler 承担未见参数状态的交互编辑；
- source-aware analytic lobe bank 承担移动高光和主要可解释光学结构；
- learned frame、stable half/difference、shared warped angular features 与小 MLP 承担方向残差；
- optimized target-visible state 只作为训练 teacher/control，不冒充即时编辑 compiler；
- matched sampler 使用 evaluator 已产生的 lobe/proposal state，在 evaluator 稳定后的里程碑实现。

目标是先实现一个联合训练、静态有界的最大质量形态，再以 matched 消融删除没有净贡献的分支。预训练/初始化阶段用于让完整模型可优化，不是“某组件先单独过门才允许加入”。

直接证据边界：

- [Random-Access Neural Compression of Material Textures](https://research.nvidia.com/publication/2023-08_random-access-neural-compression-material-textures)证明多张材质纹理及 mip chain 可以用 per-material 小网络和局部 latent grids 做随机访问压缩；它不处理方向散射或 typed 参数编辑。
- [Real-Time Neural Appearance Models](https://www.jannovak.info/publications/NAP/index.html)证明 hierarchical neural textures、learned shading frames、compact evaluator/sampler 与 shader 部署可以形成完整系统；其资产训练不能自动证明 Metal 全目录的 G2 参数编译。
- [A Hybrid Neural-Microfacet BRDF Model](https://arxiv.org/abs/2608.09604)支持 analytic core + multiplicative gate + positive correction 在同 network size 下的质量与编辑性价值；其单-lobe measured-BRDF 结论不能直接外推到 vMaterials 复杂 graph。
- MetaLayer、Neural Material Adapter 与项目 M6 研究支持“固定 native family 内参数→program/state”的 compiler 路线，但不支持把所有 source 先归约成统一 layer/Principled GT。完整 correspondence 见历史研究的 `current-nvidia-correspondence.md`。

## 2. 用户可见三部分与内部固定 recipe

用户可见组合保持三部分：

```text
MetalOpticalIdentity
× FinishNeuralTextureBundle
× NativeTypedParameterState
→ Metal neural material instance
```

内部还需要一个不可随意编辑的 `RecipeSignature`：它记录 native module/family、compiled graph/schema、bundle compatibility 和 capability。它不是第四个可组合外观轴，而是防止把 Steel paint/crack、Copper patina 等特殊 recipe 无依据地挂到其他 family。

### 2.1 `MetalOpticalIdentity`

保存 source 能够提供或从 source-backed training 中拟合的金属光学描述：反射颜色/Fresnel clue、主 roughness/anisotropy、GGX/Beckmann core 类型、metal identity code，以及少量 compiler-generated bounded correction。标准矩阵中的 metal identity 与 finish bundle 分开，以支持 source-backed 13×7 组合。

### 2.2 `FinishNeuralTextureBundle`

以 texture-set identity 为资产单位，保存：

- 每个 mip 的量化 high/low-frequency local grids；
- channel role、packed channel mapping、gamma/transfer function、normal convention、address/filter metadata；
- asset code、schema adapter 和可选的小型 asset-local low-rank modulation；
- source texture、module/graph/schema compatibility 与压缩 provenance。

bundle 可以携带数据与小型 adapter state，但不能携带一套完全不同的 method ABI 或 shader。换 bundle 不重训全局 evaluator、不重新编译 shader。

用户已确认允许固定 shape/precision 上限的 asset-local modulation；它计入 `B_asset`。该 modulation 由 shared texture encoder 生成或以 encoder 输出初始化后有界 refinement，不是每资产重新训练一套完整 decoder。

### 2.3 `NativeTypedParameterState`

保留 family-local schema、类型、范围、枚举域和存在性。连续、颜色、bool、enum、index 分别编码；不存在的参数不以零冒充。参数是材质实例级输入，编辑只重新运行轻量 compiler/state generation。

## 3. 完整运行时数据流

```text
离线 bundle compile
source texture set + mip targets
  → deterministic target encoder initialization
  → local grids + asset adapter direct optimization
  → joint appearance fine-tune + QAT
  → freeze quantized grids
  → decoder/adapter refinement
  → FinishNeuralTextureBundle

材质实例创建/编辑
native typed args + schema/graph tokens + MetalOpticalIdentity
  → TypedSourceCompiler
  → core params + texture modulation + evaluator FiLM/LoRA + sampler hints
  → bounded MaterialProgramState

prepare(surface, footprint, wo)
  → deterministic UV/projection/tiling + renderer-provided final frame
  → fixed-tap access to adjacent hierarchical grid levels
  → shared TextureDecoder + schema/asset adapter
  → spatial features + masks + lobe frames/roughness clues
  → view-conditioned prepared tokens + analytic lobe/proposal state

evaluate(prepared, wi)
  → raw direction + stable half/difference charts in learned frames
  → shared warped angular feature bank
  → source-aware analytic lobe bank
  → compiler/spatial-conditioned residual evaluator
  → linear RGB f
```

`prepare()` 把同一着色点、同一 `wo` 下可以复用的 texture decode、frame、core parameter 和 view condition 前移；`evaluate()` 不重新解码整套纹理，也不读取方向表。

## 4. 各融合组件

### 4.1 NTC-style spatial codec

采用每 mip 的 high/low-resolution grids。high-frequency grid 对当前 cell 的固定邻域直接 gather，避免先双线性平均掉细节；low-frequency grid 做硬件友好的双线性读取。相邻 mip 都在 `prepare()` 解码为 structured spatial state 后确定性插值，首个质量形态不采用单 level 随机选择。

encoder/decoder 采用分层语义共享：color/normal/scalar/packed channel 使用 role-specific stems，所有资产共享 multiscale spatial trunk，通过 role/schema/finish tokens 和 bundle set aggregator 保留语义，最后输出 grids 与 asset adapter。runtime decoder 使用 shared trunk + schema heads/adapter。完整数据量与 zero-shot 路径分析见 `texture-encoder-sharing.md`。

选择“decoded-state interpolation”而不是 latent interpolation的原因是：非线性 decoder 下 latent 混合不等价于 filtered response；两个 level 的解码成本可以在 `prepare()` 对多次 `wi` 摊销。后续成本消融再比较 stochastic one-level 与 latent interpolation。

纹理 decoder 采用双用途输出：

1. runtime structured feature head：输出 evaluator 使用的空间 feature、mask、frame/normal clue、roughness/mixture clue；
2. training-only semantic reconstruction heads：按 color、normal、roughness、mask/AO 与 packed-channel role 重建 source mip。

只重建 raw channels 会迫使 evaluator 再解释一次复杂 graph；只输出无语义 latent 又容易失去 texture replacement 的可审计性。共享 grid + 双 head 让 end-to-end appearance loss 与 source channel semantics 同时约束资产。

### 4.2 Typed source compiler

compiler 输入不是稠密 154D 向量，而是：

```text
[schema token, graph token, metal token, finish token]
+ set/ordered typed parameter tokens
+ source-computable canonical optical fields
```

公共参数 token encoder 共享；family/schema adapter 保留局部语义；graph token 保留 layer/mix/normal-routing 差异。compiler 不生成整套独立网络权重，而生成静态有界的：

- analytic core/lobe parameters 和 active masks；
- texture decoder 的少量 feature modulation；
- evaluator 每 block 的 FiLM 与低秩 LoRA 系数；
- sampler milestone 复用的 mixture/proposal hints。

这比只把一个 `z8` 拼到第一层有更深的条件化能力，也比 MetaLayer 式完整 hypernetwork state 更容易满足 shader state/bytes 合同。

### 4.3 Source-aware analytic lobe bank

不采用一个 universal GGX 把全部 vMaterials Metal 解释成同一 closure。core 是内部 prior/proposal：按 `RecipeSignature` 激活固定上限的 conductor GGX/Beckmann、coating/specular、diffuse/contamination 和 broad-scatter lobe。参数来自 native optical fields、decoded spatial clues 与 compiler correction。

core 训练保留独立 `L_core`/analytic loss，使主要 optical controls 仍有可解释作用；它不成为 source GT，也不要求 source graph 先归约成该 lobe bank。

### 4.4 Directional representation

方向输入并联三种信息：

- raw local Cartesian `wo/wi`、cosine 与 validity flags，保证退化/边界信息；
- stable half/difference chart，驻定随 view 移动的 specular peak；
- texture decoder 产生的少量 learned shading frames，表达 brushed/normal-driven mesoscale direction。

另加共享的 multiscale warped angular feature bank：高分辨率 half/slope plane 承担窄峰坐标，difference-direction 以低秩 vector/plane factor 补相关性。它作为 residual MLP 的 feature，不直接存每个材质完整 BRDF；因此固定读取、shared bytes 和资产替换都可控。

这吸收 M4 的“把容量放在 warp 后的显式场”机制，同时保留 MLP direct path，避免 raw-plane 历史失败或有限 rank 决定最终表达上限。

### 4.5 Hybrid residual evaluator

推荐输出形态：

```text
f_hat = f_core * exp(clamp(Δ_log, -b, b))
      + Σ_k f_positive_residual_lobe_k
      + softplus(f_free_tail)
```

- multiplicative branch 修正 core 的颜色/强度/方向形状；
- nonnegative residual lobes 补 core 未覆盖的额外峰，并直接为后续 sampler 提供 proposal；
- 小容量 free positive tail 保留复杂 graph、污染层和非单峰效应的表达出口；
- 禁止历史上已经证实会产生死区的 `clamp(core + signed residual, 0)`。

residual evaluator 使用 shared compact trunk，由 compiler state、spatial state 和 view-prepared tokens 逐 block FiLM/LoRA 调制。free tail 的容量与能量要登记并可消融，防止它完全吞掉 core；但首个质量形态不先删除它。

### 4.6 M3/M5/M6 的最终位置

- M5 target texture encoder：离线读取完整 source texture/mip tensor，确定性初始化 grids；runtime 删除。
- autodecoder/direct optimization：在 target encoder 初始化后优化 asset grids/adapter，提供 NTC-style rate-distortion control。
- target-response encoder/optimized code：训练期为 compiler 提供 canonicalized teacher/control；不能成为交互编辑路径。
- M6 pure compiler：产品中的 typed edit 路径，不读 reference response。
- bounded refinement：只用于 bundle-level固定资产压缩或 compiler-gap diagnostic，不允许每次滑动参数都做 reference query cook。
- M3 的 response-space top-2 字典已经在 matched bytes 下落后 PCA，不进入 evaluator；如果以后对 spatial grid block 做 VQ，它是独立 codec/profile，不沿用失败结论也不在首个质量形态中强塞 top-2。

## 5. 训练生命周期

### 5.1 Texture bootstrap

对 source-derived texture sets 建立角色感知 mip targets。target encoder 产生 grids 初始化，随后用 semantic reconstruction、normal/angular loss、mip consistency 与 QAT 联合优化。它建立可审计的 texture codec 起点，但不是单独进入完整系统的质量门。

### 5.2 Compiler/evaluator bootstrap

用 online MDL reference 在 source-train 的 graph、metal、finish 和参数状态上训练 analytic core、optimized-state teacher、typed compiler 与 evaluator。compiler 先做 state/functional distillation，再以 reference `f`、energy、peak 和 reciprocity loss 联合训练。

### 5.3 Full-stack joint quality training

完整候选同时启用 texture grids/decoder、compiler、analytic bank、warped angular bank 与 hybrid residual。loss 至少分账：

- transformed-domain robust response loss；
- linear-domain solid-angle weighted response/energy loss；
- peak/top-energy support loss；
- source-aware reciprocity；
- analytic core preservation loss；
- semantic texture channel/normal/mip reconstruction；
- quantization simulation 与 asset rate bookkeeping。

正式训练继续使用 GPU-resident online reference query，不保存 response batch。纹理 source/mip 是资产输入，不是持久化训练 response corpus。

### 5.4 Quantization freeze/refinement

量化 grids 和部署精度进入 forward path；固定离散 grids 后只 refinement shared decoder、asset adapter、compiler/evaluator，使最终结果对应实际 bundle。训练结束同时导出 pure compiler、bundle、runtime weights 与方法身份。

## 6. 泛化与评测分层

除 G1/G2 外，Metal-v1 需要把组合结论拆开：

- `G_asset`：未见 source texture-set/bundle；
- `G_metal`：未见 metal identity；
- `G_finish`：未见 primary finish asset；
- `G_pair`：metal 与 finish 各自见过，但二者配对未见；只在标准 13×7 矩阵成立；
- `G_param`：native family 内未见连续/离散参数状态；
- `G_recipe`：特殊 graph/module holdout，只衡量 compiler/runtime 可否处理注册 recipe，不宣称任意 overlay 组合。

评测同时记录 local direction/energy/peak、连续参数 sweep、连续 footprint/mip boundary、texture semantic reconstruction、bundle bytes、固定 reads、`C_prepare/C_eval` 与真实 Slang timing。observed quality/time/memory 用于 Pareto 与消融，不临时转成无来源 hard gate。

## 7. 当前互斥决策

| 互斥点 | 当前选择 | 未选方案的角色 |
|---|---|---|
| raw channel decoder vs direct appearance latent | shared grids + runtime structured head + training semantic heads | raw-only/direct-only分别作为消融 |
| per-asset decoder vs one global decoder | shared trunk + schema adapter +小型 asset-local modulation | 两端作为 matched memory/quality control |
| direct-only vs analytic-only | source-aware core + multiplicative/additive/free residual | direct/core-only均保留为消融 |
| raw direction vs fixed chart vs learned frame | 三者并联 + shared warped angular bank | 分支逐项消融 |
| stochastic one-level vs deterministic two-level | 首个质量形态用 two-level decoded-state interpolation | stochastic/latent interpolation用于后续成本消融 |
| per-state optimized code vs pure compiler | asset code可优化，参数 delta 必须 pure compiler | per-state optimized/target-encoded只作 control |
| full hypernetwork vs shallow concat | typed token compiler生成 hierarchical FiLM + bounded low-rank modulation | full generated weights为高容量 teacher，concat为低成本 control |
| response top-k dictionary | 不进入首个融合 evaluator | spatial-grid VQ未来另立 codec profile |

## 8. 运行成本可行性

用户已经确认 mip/footprint 使用 independently supervised per-mip grids，在 `prepare()` 解码相邻两个 levels 后对 structured state 做确定性插值；derived mip、latent interpolation 和 stochastic one-level 留作完整模型后的 matched 成本消融。

HLSL 审计表明，当前 MDL viewer 的 primary hit 成本可写为：

```text
current reference = 12 I_source + 4 E_source + 4 S_source + 20 Q_source
```

其中 `I` 是 generated `init`，`E/S/Q` 分别是 generated evaluate/sample/pdf。把 initialized state 提升到真正的 per-hit prepare，并直接复用 evaluate 已返回的 PDF 后，强 source control 约为：

```text
optimized source = I_source + 4 E_source + 4 S_source + 12 Q_source
```

因此 neural 方法相对当前 reference 获得的主要收益首先来自消除 11 次重复 `init` 和 8 次重复 PDF；这部分不构成相对 optimized source 的独立优势。相对强 source control，neural 更快在理论上仍然可能，但依赖以下联合条件：

- spatial latent、相邻 mip decode、structured state 与 view condition 只在 per-hit `prepare()` 执行一次；
- evaluator 使用 analytic core 驻定窄峰，使 direct neural tail 能缩到小型 correction，而不是每方向无条件执行大网络；
- sampler/pdf 复用 prepared lobe/proposal state，不为 sample 后的权重再次执行完整 evaluator；
- FP16/FP8 packed dot、cooperative vector/matrix 或等价硬件路径有效；以 `StructuredBuffer<float>` 标量循环执行 dense MLP 不视为目标部署性能；
- shared weights 连续布局并能由 cache 摊销，多材质 wave 不因 graph-specific code 产生大规模 instruction divergence；
- latent texture 的读取数、footprint 与工作集显著小于 conventional source textures。

现有 NVIDIA 原规模 reproduction 提供了量级参照：evaluator `20→64→64→64→3` 为 9,664 MAC/方向；sampler `11→32→32→32→9` 为 2,688 MAC/prepare，另有 96 MAC 的 frame extraction。它容易快过当前重复 init 的 MDL 路径，但不能据此断言会快过 optimized source；在 current viewer 的 4 evaluate + 4 sampled-direction evaluation 下，仅 evaluator 就约 77k MAC/primary hit。若这些 MAC 由普通标量 HLSL 执行，简单/中等 source graph 很可能更快。

本项目选择 hybrid evaluator 的一个成本理由正是降低这一 break-even：analytic lobe bank 承担 source 已知的窄峰与主要反射，小型 multiplicative/free-tail network 只补误差。质量优先完整形态允许暂时更大，但每个 branch 必须可被 active mask、蒸馏、低秩化、量化或消融形成部署 descendant；若完整结构只能依赖每方向大 MLP 才保持质量，则“比 optimized source 更快”不应被宣称。

预期分三层报告：

1. 相对当前 authoritative reference：neural 更快具有较强可行性，但其中包含 source integration 冗余收益；
2. 相对 prepare-hoisted/PDF-reuse optimized source：理论可行但需要实测，复杂 non-repeat、重 mixture 和多 graph workload 的机会大于简单 polished metal；
3. 相对 conventional compressed-texture control：同时比较质量、time、resident/delivery bytes，最终只在非支配区域宣称产品价值。

这里记录的只是理论可行性、当前调用公式和需要持续观测的成本来源。9,664 MAC、本轮 DXIL 数字以及三类对照都不反向构成 hard gate，也不在模型设计前冻结“单材质、分层 cohort 或 working set”哪一种成功聚合口径。等 full-quality、compact variant 与 source/conventional controls 都有实测 Pareto 后，再根据结果决定数值门槛和产品判断方式；当前分析不阻塞质量优先的完整模型设计。
