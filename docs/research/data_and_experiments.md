# 目标材质数据、监督审计与实验路线

## 1. 当前结论

项目已经拥有多种 reference package，但**目前真正能直接生成统一训练数据的仍是 LayerStack 随机游走路径**。`ncls.reference-dataset@2` 的 schema 和 reader 明确固定了 `ncls.layer-stack-ir@1`、`MaterialProgram`/canonical IR 文件与 LayerStack state；OpenPBR、MERL 和 MaterialX/Poly Haven 虽已接入 reference/viewer，并不因此自动拥有同合同的方向训练集。

因此当前最短路线不是继续下载更多材质，而是：

1. 审计现有 LayerStack `wo×wi` 监督是否覆盖尖峰、掠射角和长尾；
2. 用它确定 evaluator、方向编码和 target transform；
3. 在同一 decoder 下比较 autodecoder、读取完整目标 tensor 的 target encoder，以及 target encoder + latent refinement；
4. 再把这些 target-visible 压缩结果作为上界，与只读取原生材质输入的 source compiler 比较；
5. evaluator 成形后再设计 family-neutral spatial query contract，并引入测量与空间数据。

## 2. 三种数据不能混为一谈

| 数据层 | 内容 | 用途 |
|---|---|---|
| source material corpus | 原生参数、图、纹理、测量表和编辑状态 | compiler 输入与 provenance |
| reference response | reference 对 `(state,x,footprint,wo,wi)` 的线性响应、方差和计数 | evaluator/ compiler 监督与评测 |
| method artifact | latent、codebook、target encoder、source compiler、decoder、checkpoint、MethodBundle | 被比较的方法输出 |

MatSynth 的 PBR maps、MERL 的方向测量表和随机游走生成的 `response_cos` 分属不同语义。不能把“大量 PBR map”当成“大量复杂 BSDF GT”，也不能把某方法拟合出的 latent 再当成独立 reference。

## 3. 现有 reference portfolio 审计

| package | 原生 GT | 当前最适合回答 | 不能回答 |
|---|---|---|---|
| LayerStack random walk | 可编辑界面、均匀 slab 与局部 Monte Carlo transport | 极窄/多峰方向响应、参数状态、compiler 泛化；当前训练主线 | 空间纹理、真实测量分布、footprint/LOD |
| pbrt coated cross-check | pbrt 两界面 coated diffuse/conductor | 随机游走 reference 的独立正确性 | 任意 N 层训练语料；它不是新 source family |
| OpenPBR 1.1.1 | resolved inputs、原生图连接与 Adobe BSDF | 工业解析材质、编辑 round-trip、clean `eval/sample/pdf` | 复杂源材质的普遍物理 GT；未求值图不能伪装常量 |
| MERL 100 BRDF | 密集各向同性实测表及官方参数化 | 真实 BRDF 长尾、模型偏差、跨材质 held-out | 空间变化、各向异性、连续原生编辑参数 |
| MaterialX/Poly Haven 8×4K | 原始 `.mtlx`、纹理、物理尺寸与标准 surface | 空间 latent、跨通道相关、mip/filter 的小型闭环 | 大规模泛化；普通 PBR map 不覆盖复杂多 lobe 外观 |

这五个 package 的角色是互补的。LayerStack 是方法开发数据；pbrt 是 reference validation；OpenPBR/MERL 是方向函数外部检查；MaterialX/Poly Haven 是空间阶段 smoke。不能用一个总体平均指标把它们合并成“统一材质准确率”。

## 4. `ReferenceDataset@2` 的具体缺口

代码审计得到以下事实：

- schema 要求 `canonical_ir_id = ncls.layer-stack-ir@1`，不是 family-neutral；
- `material_states` 固定保存 `program_index` 和 `canonical_ir_index`；
- 每个 tile 只索引 `material_state_index + view_index`；
- response 保存固定 `light_count×RGB` 的 `response_cos = f*cos`、variance、两组 replica mean 和 sample count；
- 没有 `uv`、surface point、footprint covariance、mip/LOD、texture derivatives 或 spatial neighborhood；
- 当前生成器默认 `view_count=4`、`light_count=128`，文档中的较完整命令使用 16 views；`wi` 是固定等立体角 Fibonacci 半球，`wo` 最远到 82°；
- LayerStack prior 的 roughness 下界为 `0.025`，并覆盖各向异性和导体。

最后两项形成一个结构性风险：128 个近似等立体角 `wi` 对粗糙度 0.025 的极窄高光可能采不到峰值或峰形；固定少量 `wo` 也可能让网络只学会插值已有 view，而非完整查询域。这里是**根据采样密度与 prior 的审计结论**，不是已经测出的误差数字，必须由下一步 peak coverage probe 验证。

v2 仍然适合常量 LayerStack 的第一轮容量实验，因为多个 view tile 可按同一 `material_state` 组合。它不应继续被描述成已覆盖所有 source family 和 spatial material 的最终公共合同。

## 5. 下一版 family-neutral 查询合同

空间阶段前应新增版本，而不是向 v2 reserved bytes 偷塞字段。建议的逻辑结构如下：

```text
dataset manifest
  reference_id + reference_version + implementation hashes
  source_family_id + source corpus manifest/hash
  query_contract_id + response_measure + color/spectral model
  split policy + transform-statistics provenance

source state table
  source_asset_id
  native_state_id
  family-specific payload URI/schema/hash
  edit relation / parent state（可选）

query table
  source_state_index
  position_kind = constant | uv | surface-point
  uv / surface identifier
  footprint ellipse or covariance + derivative convention
  shading frame convention
  wo, wi
  requested wavelength/color channels

response table
  f and/or an explicitly named integration measure
  variance / standard error / sample count
  validity/event/capability flags
```

核心规则：

1. 不再强制 `LayerStackIR` 或任何统一 canonical material IR；
2. family-specific source payload 可以不同，但共同 query/output 语义必须可比较；
3. `f` 与 `f*cos` 要么同时存储，要么明确一个可稳定恢复另一个的规则；runtime evaluator 永远返回线性 `f`；
4. footprint 至少保留二维椭圆/协方差，不能只保存 integer mip；
5. Monte Carlo reference 必须保存不确定性，解析/测量 reference 也要记录插值与标定误差来源；
6. constant material 用显式 `position_kind=constant`，不要用假的 `uv=(0,0)` 冒充空间监督。

## 6. 方向与状态监督如何采样

### 6.1 训练 query 使用混合分布

单一等立体角网格对 diffuse 很公平，对 specular peak 不公平。训练 query 应混合：

1. 均匀立体角或 cosine-weighted `wi/wo`，覆盖整体能量；
2. half-vector / microfacet-aware proposal，覆盖镜面峰；
3. 掠射角 oversampling，覆盖 Fresnel 与 masking；
4. roughness、IOR、absorption、anisotropy 和 layer weight 边界状态；
5. 当前模型误差驱动的 adaptive queries，但 adaptive 样本不能污染固定 validation/test。

每个样本记录生成 proposal 或等价权重，loss/metric 才能区分“采样分布上的平均误差”和“立体角积分误差”。

### 6.2 validation 与 test 必须有固定、连续和对抗三部分

- 固定 probe：确定性方向集，便于版本回归；
- 连续随机 query：用未进入训练 grid 的方向检查插值；
- peak/adversarial probe：围绕 reference 最大值、镜面方向、掠射角和模型当前最大误差局部细化。

只在同一 128-bin grid 上 train/test，会把“记住采样表”误当成连续 evaluator。

### 6.3 split 的最小单位由 source 语义决定

| source family | 不得跨 split 的单位 |
|---|---|
| LayerStack | 同一结构 template 及其所有 local edited states |
| MaterialX/程序图 | 同一图拓扑、派生参数状态和同源纹理 crop |
| PBR/SVBRDF texture | 同一原始资产的所有 crop、rotation、mip 和渲染图 |
| measured BRDF/BTF | 同一物理样本的所有方向、空间点和处理版本 |

还应分开报告三种泛化：未见参数状态、未见资产/图拓扑、未见 source family。第三种通常需要 family adapter，不应仅靠把一个陌生 payload 塞进同一个 encoder 来宣称成功。

## 7. 动态范围与监督统计审计

训练前先落盘一份不含模型的 supervision audit，至少统计：

- 每 reference family/通道的 min、非零分位数、max、`max/median`、零值比例；
- 每 tile 的半球积分能量与 top 1%/5% 方向能量占比；
- `wo`、roughness、base type、层数/图拓扑分档的长尾；
- Monte Carlo standard error 相对信号大小；
- A/B replica 差异与模型候选目标误差的量级关系；
- 固定 128-bin 与局部高分辨率 peak probe 的能量/峰值差异。

target transform 的统计量只能由 train split 生成，并写入带 hash 的独立 artifact。首轮比较：

```text
T0  linear target baseline
T1  log1p(y / scale)
T2  log1p + channel mean/std
T3  energy + normalized directional shape
T4  analytic core residual + asinh(residual / scale)
```

模型逻辑输出仍为 `f`。可以在 `y=f*max(n·wi,0)` 上计算变换后 loss，但不能让 runtime 通过接近 horizon 的除法恢复 `f`。最安全的实现是 head 直接参数化非负 `f`，训练时由已知 `wi` 乘 cosine 后与 `response_cos` 比较。

## 8. 候选数据源与优先级

### 8.1 立即使用，不新增下载

1. **LayerStack**：完成方向监督审计、单材质容量、shared decoder 和 compiler。
2. **MERL**：选定少量 diffuse/glossy/metallic 长尾材质做外部分布测试，不参与第一轮模型选择时也可作为 stress probe。
3. **OpenPBR**：构造 clean、可编辑解析状态，检查 compiler 是否只记住 LayerStack prior。
4. **现有 8 个 MaterialX/Poly Haven**：只在 evaluator 方向结构稳定后验证 spatial smoke 和 mip contract。

### 8.2 下一批最有价值的数据

| 数据源 | 已知内容 | 最适合的研究问题 | 接入边界 |
|---|---|---|---|
| [RGL Material Database](https://rgl.epfl.ch/pages/lab/material-database) | isotropic/anisotropic、RGB 与 360–1000 nm 光谱 BRDF，附 compact eval/sample/pdf 实现 | 真实各向异性、光谱/RGB aliasing、测量 sampler | 先锁定许可和单材质文件 hash；不转成 GGX 后当 GT |
| [MatSynth](https://openaccess.thecvf.com/content/CVPR2024/html/Vecchio_MatSynth_A_Modern_PBR_Materials_Dataset_CVPR_2024_paper.html) | 4,069 个 4K、CC0 tileable PBR 材质，7 类 map，3,417,960 张增强渲染图 | 大规模 target-tensor encoder、跨通道压缩、asset split | maps 必须绑定固定 closure/颜色/尺度；它不是复杂 BSDF 测量 |
| [OpenSVBRDF](https://opensvbrdf.github.io/) | 1,000+ 个 1024²、9 类近似平面各向异性实测材质；GGX/local-frame maps、neural representation 与原始 HDR 照片分层提供 | 真实 spatial anisotropy、local frame、长尾与 acquisition gap | fitted maps 与 raw measurements 是不同 GT 层；下载可用性/许可逐项核对 |
| [Bonn UBOFAB19](https://cg.cs.uni-bonn.de/btf/bonn_svbrdf_database.html) | fabric calibrated measurements 与 anisotropic Ward/Fresnel SVBRDF fits | 织物各向异性、fit-vs-measurement 偏差 | research-use；Ward fit 不能冒充原始 TAC7 测量 |
| [UBO2014 BTF](https://cg.cs.uni-bonn.de/project/btfdbb) | 84 个物理样本，每个 151×151 light/view、512² HDR BTF 和 height | 完整 spatial `uv×wo×wi` 压力测试、parallax/BTF | 数据巨大且许可需确认；只选少量固定 sample，不先全量下载 |

### 8.3 可作为新 source family 的程序增强数据

[Toward Richer Material Generation via Procedural Data Enhancement](https://research.nvidia.com/labs/rtr/publication/yu2026toward/)把普通 PBR 材质扩展为 dust、clearcoat、多 lobe 和 layered scattering，可用于生成大规模复杂外观训练状态。若采用，它必须登记为新的“procedurally enhanced material” source family；reference 是增强后模型本身，不能说这些效果是原 MatSynth/Poly Haven 资产的原生 GT。

### 8.4 暂不作为第一批目标

- BSSRDF、hair/fiber、volume：查询域超出局部 surface BSDF；
- 大规模 Adobe Substance 资产：许可与可再分发边界复杂；
- 任意互联网材质图：缺少稳定 reference 和原生编辑 provenance；
- 场景级 GI/lightmap：适合 codec 类比，不是材质散射 GT。

## 9. 实验顺序

### E0：监督审计，不训练模型

输入：现有/重新生成的 LayerStack v2 数据 + 连续 reference probe。

输出：方向覆盖、动态范围、reference 噪声、峰值和能量报告。先决定是否需要提高/改变 `wi` 采样，再冻结训练集版本。E0 没完成前不比较网络。

### E1：单材质完整 evaluator 容量

每个候选拟合同一材质的全部训练 `wo×wi`，而不是逐 tile：

| 轴 | 最小候选 |
|---|---|
| 方向编码 | local Cartesian；Rusinkiewicz half/difference；learned frame/analytic warp |
| 网络 | 2–4 层小 MLP 的 width/activation/precision Pareto |
| target | T0–T4 变换 |
|表示 | direct neural；analytic core + residual |
| latent | 无 latent/单材质 code；dense material latent |

E1 只回答给定 runtime 预算是否有容量。若 optimized single-material 仍失败，不进入 compiler。

### E2：shared decoder + 压缩期 latent 获取

在多个材质/状态间共享同一个 decoder，并在相同 latent layout 下比较：

1. autodecoder：从随机 latent 开始逐资产直接优化；
2. per-asset target encoder：把该资产完整、规则排列的 reference response tensor 输入 `E(X)`；encoder 与 decoder 联合优化，压缩结束后丢弃 encoder；
3. corpus-shared target encoder：同一个 `E(X)` 跨训练资产摊销 latent inference；
4. target encoder + refinement：从 `E(X)` 初始化，再执行固定步数或固定时间的 latent 优化。

target encoder 与 decoder 联合训练，但导出资产只含 baked latent 和 decoder。无论 encoder 是 per-asset 还是 corpus-shared，它都看到完整 reference tensor，因此只回答压缩速度、优化稳定性和最终率失真，不回答从原生材质输入即时编译或未见编辑状态泛化。

这些路径都继续比较以下 latent 表示：

- dense latent；
- top-1 codebook；
- top-2 convex mixture；
- top-2 + residual；
- plane/CP/vector-matrix factorization oracle。

固定总 asset bytes 与 decoder query time，报告达到目标质量的 wall-clock/step、seed 方差、最终误差和 latent 空间噪声。方向表字典只能作为 oracle；最终方法必须连续 query `wi`。若 response 是不规则方向集合，必须明确 target encoder 所需的 canonical tensorization、mask 或 set encoder，不能偷偷给它比 autodecoder 更多监督。

### E3：source compiler 形态

在同一个 frozen/shared decoder 下比较：

1. autodecoder optimized latent：直接拟合上界；
2. target encoder latent：读取完整 reference tensor 的压缩基线；
3. source compiler latent：只从原生参数、图或资源前向生成；
4. source compiler initialization + 固定步数 refinement：发布 cook；
5. hypernetwork 从原生状态生成少量材质权重：MetaLayer 风格基线。

报告 source compiler 相对 optimized/target-encoded latent 的 gap、完整 reference tensor 的生成成本、纯前向 compile latency、编辑轨迹连续性、重复编译确定性和额外 asset bytes。target encoder 与 source compiler 的训练资产只能来自 train split；validation/test state 不参与训练期 latent statistics 或 codebook。target encoder 在 test 时可以读取该 test 资产的完整 reference tensor以测压缩能力，但这个结果必须标记为 target-visible，不得计作 compiler 泛化。

### E4：Slang 最小部署

只导出 E1–E3 的 Pareto 候选：

- Python/Slang 固定 query parity；
- `prepare` 与一次/多次 `evaluate` 分开计时；
- 普通 ALU、可用 cooperative execution 和 fp16/int8 分开；
- 单材质 coherent tile 与多材质 divergent tile 分开；
- asset latent、material-specific weights、shared weights 和 scratch bytes 全计入。

### E5：spatial latent 与 LOD

先用现有 8 个 MaterialX/Poly Haven 资产和新版 query contract 做 smoke，再决定 MatSynth/OpenSVBRDF 子集。比较：

- standard parameter mip / MIPNet；
- independent neural mip pyramid / NeuMIP；
- target-tensor encoder-generated pyramid；
- source compiler-generated pyramid；
- dictionary/codebook latent pyramid；
- footprint-conditioned `prepare`。

必须同时测近景高频、远景 alias/overblur、UV seam、时域相机缩放和 footprint 旋转。只有这里通过后，项目才可声称 random-access spatial neural material。

### E6：matched sampler 与 integration

冻结 evaluator 后再训练 sampler head；`sample` 和 `pdf` 必须来自同一 proposal。随后才做 PT variance、environment/area-light integration 和系统多灯 scaling。

## 10. 指标

### 函数与物理

- 半球加权 normalized L1：`sum Ω|y_hat-y| / sum Ω|y|`；
- log-domain error，但与线性指标并列；
- 每 tile/family 的 median、p90、p95，而不只总体 mean；
- top-energy region recall、peak value ratio、peak angular displacement；
- directional-hemispherical reflectance/energy error；
- reciprocity error、nonnegative/finite rate；
- 模型误差相对 reference standard error，识别 noise floor。

点对点 relative error 的 denominator 不能简单用每个 GT 值，否则大面积接近零区域会主导统计。应按 tile energy、稳定 epsilon 或对数域分别报告。

### 图像

- held-out directional lights、HDRI 和 roughness/view sweep；
- linear HDR MAE/PSNR 与 display-referred FLIP；
- reference-material-only image，避免 scene GI 混入；
- spatial 阶段的 zoom sequence temporal error、alias 与 overblur。

### 系统

- `B_asset`、`B_shared` 与摊销后的 scene bytes；
- `C_compile`、refinement steps/time；
- `C_prepare`、`C_eval`、多 query amortization；
- coherent/divergent、分辨率、材质数与 light query 数；
- 本地 fetch 数、codebook fetch 数和 quantized weight bandwidth。

## 11. 验收门槛如何产生

当前不凭空写一个统一的“误差 < x% 即成功”。E0/E1 pilot 先产生 reference noise floor、简单解析 baseline、Real-Time Neural Appearance/MetaLayer/Hybrid BRDF 等可复现基线和 runtime budget。之后把版本化门槛写入 `references/acceptance.json` 或独立 experiment acceptance manifest，再冻结 test。

晋级原则是：

1. E1 候选必须在完整连续 `wo×wi` 上进入可部署预算的 Pareto；
2. E2 shared decoder 相对单材质上界的退化可解释，且 target encoder 相对 autodecoder 的速度、稳定性和率失真收益分别可测；
3. E3 source compiler 必须在未见 state/asset 上接近 optimized/target-encoded latent，而不是只在 train asset 重建；跨 source family 只有在定义对应 family adapter 和共同查询合同后才单独验收；
4. E4 shader 实测确认有界成本；
5. E5 之前不宣称 spatial/LOD，E6 之前不宣称 matched sampling。

## 12. 最近的可执行任务

1. 增加一个只读 supervision-audit 工具，输出到 `artifacts/research/supervision-audit/<dataset-id>/`；
2. 用连续 reference query 对固定 128-bin 的峰值与积分覆盖做校验；
3. 定义新的 neural evaluator experiment manifest，锁定 direction encoding、target transform、latent scope 和 cost estimate；
4. 实现 E1 的最小候选：Cartesian vs half/difference、linear vs log1p-standardized、direct vs analytic residual；
5. E1 通过后再实现 shared dense latent 与 top-2 latent dictionary；
6. 依据结果决定 v2 数据扩采，避免在错误方向网格上盲目扩大 family 数。

所有单次审计、训练与报告进入被 Git 忽略的 `artifacts/`；本文只维护稳定结论、实验合同和后续决策。
