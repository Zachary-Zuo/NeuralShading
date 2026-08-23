# 目标材质数据、监督审计与实验路线

## 1. 当前结论

数据框架已经收敛为材质族无关的查询采集层。LayerStack、MERL、OpenPBR 和 MaterialX/Poly Haven 都能通过相同 provider 协议在 Falcor 中求值，并写入唯一的 `ncls.reference-dataset@3` HDF5；公共 collector 和 response-only learning reader 不含任何材质参数分支。

这解决的是“统一监督查询与持久化”，不是“把所有源材质变成同一参数表示”。各材质族仍保留自己的原生参数、图、纹理或测量表及权威 reference。source compiler 若要读取原生输入，仍需 family-specific encoder；它的输出才进入共享 evaluator/latent 合同。

当前应做的事情是：先审计状态和方向采样是否覆盖目标查询域，再生成正式 HDF5，然后按单材质容量、共享 decoder、source compiler 和 Slang 部署的顺序建模。不能因为接口统一，就假设当前均匀方向网格已经具有足够密度。

## 2. 三种数据层

| 数据层 | 内容 | 用途 |
|---|---|---|
| source material corpus | 原生参数、图、纹理、测量表、编辑关系和 manifest/hash | reference 任意查询、source compiler 输入与 provenance |
| ReferenceDataset | 已采样 state payload/identity、位置/footprint、`wo/wi`、proposal、响应与统计 | evaluator 监督、target-visible compression、固定评测与审计 |
| method artifact | latent、encoder/compiler、decoder、checkpoint、MethodBundle | 被比较和部署的方法输出 |

HDF5 能独立恢复第二层的完整监督快照，不能从有限 query 推回第一层的无限连续函数，也不应重复打包所有大型原始资源。MERL 表、MaterialX 纹理等仍由 source manifest 锁定；这不会让 response-only learning 依赖材质族。

## 3. 当前 reference portfolio

| package | 原生 GT | HDF5 provider/角色 |
|---|---|---|
| LayerStack random walk | 可编辑界面与均匀 slab | `layer-stack`；方向函数、连续状态与 compiler 主线 |
| pbrt coated cross-check | pbrt coated diffuse/conductor | 独立验证 LayerStack reference，不重复成为 source provider |
| OpenPBR 1.1.1 | 原生参数/连接与 Adobe BSDF | `openpbr`；83 个当前资产、反射/透射、reference PDF |
| MERL | 100 个实测各向同性 BRDF 表 | `merl`；真实方向长尾和跨资产 held-out |
| MaterialX/Poly Haven | 8 个原始图、4K 纹理和物理尺度 | `materialx`；UV、footprint、normal map 与 filtering smoke |

这些 family 可以存在同一 HDF5，但实验必须按 family 分层报告。把它们混成一个总体平均误差会掩盖测量、解析、Monte Carlo 和空间纹理的不同难度。

## 4. 统一合同与非统一源表示

固定公共样本是：

```text
state identity + opaque native payload
surface position / UV / footprint / frame
wo + wi + proposal PDF + solid-angle weight + seed
reference RGB response + uncertainty + validity/event/PDF
```

合同不要求 `LayerStackIR`、统一 closure 或统一参数向量。源参数编码分成两条路径：

- response/target-visible 模型只从公共 query/response 学习，不解码 source payload；
- source compiler 通过明确注册的 family adapter 读取 native payload 和必要资源，并映射到共享 latent/evaluator。

这样，新增已复现材质只增加 provider 和可选 source encoder，不会改写 collector、HDF5 或通用 batch reader。

## 5. 状态与方向采样

### 5.1 状态空间

`source_states()` 必须保存实际采样状态的完整描述和父编辑关系，不能只写一个连续编号。不同 family 的 split 最小单位分别是结构 family、物理样本、原生图/资产及其所有派生 edit/crop/mip。

状态密度由研究问题决定：LayerStack 重点覆盖 roughness、IOR、导体参数、吸收/散射、各向异性和层数边界；OpenPBR 重点覆盖各 lobe 开关、传输和编辑轨迹；MERL 使用资产级 held-out；MaterialX 使用资产级 split，并在每个资产内采 UV/footprint。

### 5.2 查询分布

当前默认 stratified `wo` + 均匀立体角 `wi` 只作为确定性基线。正式训练 query 应组合：

1. 均匀立体角或 cosine-weighted 样本，覆盖整体能量；
2. half-vector/microfacet-aware proposal，覆盖镜面峰；
3. 掠射角 oversampling，覆盖 Fresnel、masking 和透射临界区域；
4. reference/模型误差驱动的局部细化；
5. 对纹理材质的多尺度 UV、footprint 大小与旋转。

每个 query 都保存实际 proposal PDF 和积分权重。训练分布、均匀立体角指标和渲染分布指标因此可以分开计算，不会把采样偏好伪装成函数准确率。

### 5.3 固定评测域

validation/test 至少包含：

- 固定 deterministic probe，用于版本回归；
- 未进入训练 grid 的连续随机 query；
- peak、掠射角、临界透射与高频纹理等 adversarial probe；
- 与训练 proposal 不同的立体角加权积分 probe。

只在同一个离散方向表上 train/test 会把查表记忆误判为连续 evaluator。

## 6. 监督审计

训练前为每个 HDF5 生成只读 audit，至少包含：

- 每 family/通道 min、非零分位数、max、零值比例和动态范围；
- 每 query group 的积分能量、top 1%/5% 能量占比和峰值方向；
- 按 `wo`、状态边界、资产、UV/footprint 分档的长尾；
- Monte Carlo standard error、A/B replica 差异和 deterministic reference 标记；
- 当前固定方向集与局部高分辨率 probe 的峰值/能量差；
- train/validation/test 的 state、资产和参数覆盖，确认无 group 泄漏。

target transform 的统计量只能由 train split 生成并带 hash 保存。首轮比较 linear、`log1p`、标准化 log、energy+shape 和 analytic-core residual；运行时逻辑输出仍为 `f`，HDF5 的监督测度为 `f×|cos|`。

## 7. Learning 分层

### E0：监督审计

不训练模型。冻结 source-state distribution、训练 proposal、固定 validation/test probe 和 reference noise floor。若尖峰覆盖不足，先重生成 HDF5。

### E1：单材质完整 evaluator 容量

每个候选覆盖同一 state 的完整 `wo×wi`，比较局部 Cartesian 与 half/difference 编码、小型 MLP 宽深/激活、direct response 与 analytic residual、target transform 和 latent 预算。optimized 单材质仍失败时不进入 compiler。

### E2：共享 decoder + 材质 latent

在多个 state/asset 间共享 decoder，比较 autodecoder、target encoder、target encoder + 固定预算 refinement、dense latent 和 codebook/factorized latent。target encoder 能读取完整 HDF5 response，因此只代表压缩上界，不代表从原生材质自动编译。

### E3：source compiler

为每个 source family 使用明确 adapter，从原生参数、图或资源生成同一 shared decoder 的 latent。比较 pure feed-forward、compiler initialization + refinement 和 optimized latent 上界；报告未见参数状态、未见资产/图拓扑和跨 family 三种不同泛化。

### E4：Slang 最小部署

只导出 E1–E3 的 Pareto 候选，验证 Python/Slang 固定 query parity，并分别测 `prepare`、一次/多次 `evaluate`、shared weights、asset latent、scratch、coherent/divergent material tile 和 fp32/fp16 路径。

### E5：spatial latent 与 LOD

用 MaterialX HDF5 的 UV/footprint query 验证 latent texture fetch、mip/filter、UV seam、zoom temporal stability 和 footprint 旋转，再决定是否扩展 MatSynth/OpenSVBRDF。常量 LayerStack 数据不能支持 spatial/LOD 结论。

### E6：matched sampler 与 integration

冻结 evaluator 后才增加 sampler head；`sample` 和 `pdf` 必须对应同一 proposal。之后比较 PT variance、环境/面光积分和多灯系统成本。

## 8. 指标

函数指标至少包括半球/球面立体角加权 normalized L1、linear 与 log error、family 内 median/p90/p95、top-energy recall、峰值比例/角偏移、能量误差、互易性、finite rate、符合各 source/color-space 定义的数值范围，以及模型误差相对 reference standard error。ACEScg 转线性 sRGB 的 out-of-gamut 响应可以有负通道，不能统一截断后再称为 GT。

图像指标使用 held-out directional lights、HDRI 和 view/state/roughness sweep，报告 linear HDR 指标和 display-referred FLIP。空间阶段增加 zoom 时序误差、alias、overblur 和 seam。

系统指标分开报告 `B_asset`、`B_shared`、compile/refinement、`C_prepare`、`C_eval`、query amortization、材质分歧和显存/带宽。不能用网络 FLOPs 代替 Falcor 实测时间。

## 9. 最近可执行任务

### 2026-08-24 E0 pilot 审计结论

四个现有 pilot H5 已通过合同与内容哈希，但都没有通过 `configs/research/e0-supervision-gates-v1.json`。稳定发现如下：

- 四族的 `split_group_id`、source hash 与父子状态都没有跨 split 泄漏；
- 四族都在 train/validation/test 复用完全相同的 `wi` 表和 `wo` 集，因此不能把这些 pilot 当作连续方向泛化证据；
- 当前 proposal 只有均匀立体角。MERL 与 OpenPBR 的 top 1% 能量占比 p95 分别约为 `0.836` 与 `0.982`，但 peak 最近邻方向角 p95 仍约为 `8.73°` 与 `12.25°`，说明窄峰监督明显不足；
- LayerStack 的 replica normalized L1 p95 约为 `1.252`，相对 standard error p95 约为 `0.583`，未达到冻结 gate 的 `0.1`；正式监督必须增加自适应样本并针对高能量/窄峰查询诊断；
- OpenPBR pilot 虽含上下半球各一半方向，但没有临界透射 probe；MaterialX pilot 只有一个 footprint 尺度与一个轴向，没有 footprint 旋转、seam 或 zoom/LOD 证据。

因此这些文件只保留为 provider/合同 smoke，E1 不得直接消费。下一步先让 query plan 支持按 `wo` 的 peak-aware 方向与 split 独立 probe，再生成最小的 targeted H5；不能用扩大同一均匀表代替修正 proposal。完整运行结果位于 `artifacts/research/supervision-audit/<dataset-id>/`。

逐 `wo` 查询合同已经实现并通过 CPU 与 Falcor GPU 回归：`QueryPlan` 支持 `[view, light, 3]`，LayerStack shader 和四个 provider 都按 query group 消费真实方向。`ncls.e0-peak-grazing-mixture@1` 提供有显式归一化 PDF 的 uniform、三尺度 peak、grazing，以及完整球面的 transmission peak；uniform 基线也对不同 state split 使用确定性方位扰动。该结果只说明采集机制成立，尚不说明 E0 gate 已通过。同一 state 内 train/validation/test query 角色仍需在 E1 正式数据前显式建模，不能用 state split 方向扰动替代 held-out query。

真实 targeted probe 给出了两项稳定结果。MERL `black-obsidian` 使用 8 个 `wo`、每个 512 个 mixture query 后，peak 最近邻角 p95 从 pilot 的约 `8.73°` 降至 `1.15°`；OpenPBR `open_pbr_glass` 从约 `12.25°` 降至 `0.52°`，并实际包含约 `47.1%` 透射侧 query 与 `11.4%` 的五度内掠射 query。两个单资产 probe 的 proposal/profile 检查通过，gate 只因它们故意没有 validation/test state 而失败，不能把该预期失败解释为 proposal 失败。

三组 LayerStack 的相同 state/query 在四档自适应预算下测得以下 noise 曲线；`sample_count` 是合并两个 replica 后的总样本数上限：

| 最大总样本数 | relative SE p95 | replica normalized L1 p95 | gate |
|---:|---:|---:|---|
| `16,384` | `0.3445` | `0.3634` | 失败 |
| `131,072` | `0.1953` | `0.1649` | 失败 |
| `524,288` | `0.1152` | `0.0930` | 仅 relative SE 失败 |
| `1,048,576` | `0.0849` | `0.0624` | 通过 |

因此冻结 noise gate 可达，但百万样本是高方差状态的诊断上界，不是所有正式数据的默认预算。一次把单 batch 提高到 16,384 触发 Windows D3D12 TDR；保持 8,192 batch、增加迭代次数可稳定完成。audit v2 必须先按 state/split/积分能量列出最坏 query，再决定哪些状态需要高预算或更好的 reference proposal。

1. ~~扩展 query plan，使每个 `wo` 可以拥有独立的 `wi` proposal，并让 train/validation/test 使用互不重合但测度明确的方向 probe；~~ 已完成逐 `wo` proposal 与 state-split 独立方向，query role 合同仍待单独冻结；
2. 对 LayerStack 极低 roughness、MERL 高光材质和 OpenPBR transmission 做高分辨率 peak probe；
3. 把默认均匀 proposal 扩展成带显式 mixture component 的训练 proposal，并保持固定 validation/test proposal；
4. 针对 LayerStack reference noise 调整自适应采样后，只重生成最小必要 H5，并重新执行合同、hash、split 与 gate；
5. 用通过 E0 的 LayerStack v3 数据完成 E1 的方向编码、target transform 与 evaluator 容量比较；
6. 在同一公共 reader 上把 MERL/OpenPBR 加入 target-visible shared decoder 实验；
7. evaluator 成形后再进入 MaterialX spatial latent、sampler 和 integration。

单次数据、audit、训练和报告都位于 `data/reference-responses/` 或 `artifacts/`，不进入根 Git。本文只维护稳定结论、实验依赖和验收逻辑。
