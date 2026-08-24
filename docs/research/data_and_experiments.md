# 目标材质数据、监督审计与实验路线

## 1. 当前结论

数据框架已经收敛为材质族无关的查询采集层。LayerStack、MERL、OpenPBR 和 MaterialX/Poly Haven 都能通过相同 provider 协议在 Falcor 中求值，并写入唯一的 `ncls.reference-dataset@4` HDF5；公共 collector 和 response-only learning reader 不含任何材质参数分支。v4 分开保存 source state split 与 query role，支持在同一材质上严格隔离训练、validation、held-out test 和 adversarial probe。

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

target transform 的统计量只能由 source train × query train 生成并带 hash 保存。首轮比较 linear、`log1p`、标准化 log、energy+shape 和 analytic-core residual；运行时逻辑输出仍为 `f`，HDF5 的监督测度为 `f×|cos|`。

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

四个开始时已有的 v3 pilot H5 曾通过各自生成时的合同与内容哈希，但都没有通过当时冻结的 E0 v1 gate。它们的历史 audit 仍保留为 proposal 缺口证据；当前唯一 reader 已升级为 v4，因此这些文件不再是可消费数据，必须从锁定 source/reference/config 重生。稳定发现如下：

- 四族的 `split_group_id`、source hash 与父子状态都没有跨 split 泄漏；
- 四族都在 train/validation/test 复用完全相同的 `wi` 表和 `wo` 集，因此不能把这些 pilot 当作连续方向泛化证据；
- 当前 proposal 只有均匀立体角。MERL 与 OpenPBR 的 top 1% 能量占比 p95 分别约为 `0.836` 与 `0.982`，但 peak 最近邻方向角 p95 仍约为 `8.73°` 与 `12.25°`，说明窄峰监督明显不足；
- LayerStack 的 replica normalized L1 p95 约为 `1.252`，相对 standard error p95 约为 `0.583`，未达到冻结 gate 的 `0.1`；正式监督必须增加自适应样本并针对高能量/窄峰查询诊断；
- OpenPBR pilot 虽含上下半球各一半方向，但没有临界透射 probe；MaterialX pilot 只有一个 footprint 尺度与一个轴向，没有 footprint 旋转、seam 或 zoom/LOD 证据。

因此这些文件只保留为 provider/合同 smoke，E1 不得直接消费。下一步先让 query plan 支持按 `wo` 的 peak-aware 方向与 split 独立 probe，再生成最小的 targeted H5；不能用扩大同一均匀表代替修正 proposal。完整运行结果位于 `artifacts/research/supervision-audit/<dataset-id>/`。

逐 `wo` 查询合同已经实现并通过 CPU 与 Falcor GPU 回归：`QueryPlan` 支持 `[view, light, 3]`，LayerStack shader 和四个 provider 都按 query group 消费真实方向。当前 `ncls.e0-peak-grazing-mixture@2` 提供有显式归一化 PDF 的 uniform、以真实镜面方向为中心的三尺度球面 vMF peak、grazing，以及完整球面的 transmission peak；uniform 基线也对不同 partition 使用确定性方位扰动。随后 v4 加入显式 query role：train/adversarial 使用 mixture，validation/test 使用独立 fixed uniform probe。source split 与 query role 的独立性由 audit 和 `ncls.e0-supervision-entry@3` 分别检查，不能再用 state split 方位扰动替代 held-out query。

首个三资产 MERL v4 smoke 暴露了 source test × adversarial 与 source train × train 的三个精确 `wo` 碰撞；没有放宽 gate，而是改为由 `(source split, query role)` 联合生成非碰撞确定性方位。修复后全部 source/query partition 的 `wo/wi` overlap 为 0，adversarial raw-response peak 最近邻角 p95 为 `0.532°`，五度内 `wi` 掠射比例为 `0.119`。E0 gate v3 因而冻结 peak spacing p95 ≤ `2°`、掠射比例 ≥ `0.08`，并为 OpenPBR 冻结 adversarial 透射比例 ≥ `0.25`。

真实 targeted probe 给出了两项稳定结果。MERL `black-obsidian` 使用 8 个 `wo`、每个 512 个 mixture query 后，peak 最近邻角 p95 从 pilot 的约 `8.73°` 降至 `1.15°`；OpenPBR `open_pbr_glass` 从约 `12.25°` 降至 `0.52°`，并实际包含约 `47.1%` 透射侧 query 与 `11.4%` 的五度内掠射 query。两个单资产 probe 的 proposal/profile 检查通过，gate 只因它们故意没有 validation/test state 而失败，不能把该预期失败解释为 proposal 失败。

随后三资产 OpenPBR v4 smoke 把 `open_pbr_glass`、`open_pbr_soapbubble` 与 `open_pbr_aluminum_brushed` 放入独立 source/query partition。合同和内容哈希通过，全部方向 overlap 为 0；adversarial peak spacing p95 为 `1.410°`、掠射比例为 `0.115`、透射比例为 `0.475`，通过 `ncls.e0-supervision-entry@3`。这只确认 E0 查询覆盖，不代表 transmission evaluator 已通过容量或视觉 gate。

LayerStack 不再等待随机 prior 偶然采到边界状态。provider-local 的 `ncls.e0-layer-stack-boundary@1` 固定六个 coverage case：极窄 dielectric、极窄各向异性 conductor、旋转各向异性 dielectric、色吸收 slab、符合当前 v0 同消光实现约束的色散射 slab，以及多界面移动峰。它要求 6 个 family、每个 1 个状态，并把 profile/case ID 写进原生 payload 与 provider metadata。该集合是 E0 probe，不替代后续连续研究 prior；改动案例必须升级 profile ID。

首个 boundary v4 probe 使用合并总样本 8,192。全局 relative SE p95 为 `0.0188`、replica normalized L1 p95 为 `0.0411`，但多界面 case 的最坏 query-group relative SE p95 为 `0.1375`。因此不能用全局 p95 宣告 noise 通过；`ncls.supervision-audit@4` 与 `ncls.e0-supervision-entry@4` 增加最坏 query-group 的 `0.1` 阈值，并要求 LayerStack E0 数据显式使用六状态 boundary profile。该 probe 还显示 128-direction 下极窄各向异性 conductor 的加权能量估计不稳定，必须另做更高方向数 proposal 收敛检查后再冻结能量 gate。

固定把所有状态翻倍到合并 16,384 样本后，最坏 query relative SE 仍为 `0.1355`。改用 query-group adaptive 后，五个低方差 case 都在合并 16,384 样本停止；多界面 case 的六个 query 按需使用 40,960–262,144 个合并样本，最终全局 relative SE p95 `0.0117`、最坏 query relative SE `0.0534`、最坏 replica normalized L1 `0.0179`，通过 v4 gate。该配置中 8,192-sample 单 dispatch 触发一次 Windows TDR；降到 4,096 batch、保持总预算不变后稳定完成，因此安全 batch 结论必须同时记录 directions、query-group 数与 stack 深度。

极窄各向异性 conductor 的 16-seed proposal 重复实验进一步证伪了旧 `ncls.e0-peak-grazing-mixture@1`：在 128/1,024/8,192 directions 下，四个固定 `wo` 的最大能量 estimator 变异系数分别为 `1.446/0.715/0.241`。根因是分离的 `z/方位角` peak 在近法线退化为环状覆盖，也不能跟随旋转各向异性窄轴。`@2` 改用以真实镜面方向为中心、折叠到目标半球的三尺度球面 vMF，保持解析归一化 PDF；相同实验降为 `1.381/0.578/0.109`，8,192-direction 最大 RGB 总能量从 `3.518` 降到 `2.721`。`@1` 已停止作为当前采集入口，历史 H5 只由其原生成提交复现。

MaterialX 的第一轮三资产 spatial probe 进一步说明 peak gate 不能只看全体 query 的离散最大值。denim 的 top-1% 积分能量占比约 `0.065`，宽响应中的最大采样点没有窄峰位置语义；metal_plate 的集中响应却因 normal map 相对几何镜面移动最多约 `9°`，128-direction 几何镜面 proposal 的集中 query peak spacing p95 为 `5.352°`。把方向数提高到 1,024/2,048/4,096 后，该值仍分别为约 `2.234/2.043/2.171°`，并不单调稳定。因此没有继续扩大 H5，而是让 `QueryPlan` 支持按 `(surface, wo)` 的方向表，并发布 `ncls.materialx-local-normal-peak@1`：它用与 GT 相同的 normal texture filtering 求出逐 UV/footprint shading normal，再生成有解析 PDF 的局部峰 proposal。

`ncls.supervision-audit@6` 仍报告全体 peak spacing，但正式 2° gate 只作用于 top-1% 积分能量占比至少 `0.1` 的 query，并要求至少 4 个这类 adversarial query，防止“没有集中峰”自动通过。三资产 MaterialX local-normal 1,024-direction H5 有 19 个集中 query，p95 `1.926°`；同时实测 5 个 footprint 尺度、4 个旋转、U/V seam 两轴配对、掠射比例 `0.1218`，且 split/hash/finite/确定性 reference 检查均通过。LayerStack、MERL、OpenPBR 在相同 `ncls.e0-supervision-entry@6` 下的集中 peak p95 分别为 `0.063/0.232/0.066°`，OpenPBR 透射比例 `0.480`，四族当前 E0 H5 全部通过。

这标志 E0 入口 gate 已成形，不表示后续表示或 viewer 已通过。MaterialX 固定 spatial probe 只验证真实 filtering supervision 与 coverage；spatial latent、LOD、zoom temporal stability、alias/overblur 和 seam 视觉一致性仍属于 E5。E1 从通过 v6 的 LayerStack boundary 数据开始做单材质完整 `wo×wi` evaluator 容量实验，test 继续与 validation/model selection 分离。

E1 当前已有两个独立的一状态容量合同。极窄各向异性 conductor 数据在四个 query role 上各有 `64/16/16/16` 个独立 `wo`、每组 256 个 mixture `wi`；监督 gate 通过。七个同预算 direct small-MLP 变体覆盖 Cartesian、Fourier、half/difference、multiscale half-slope，以及 linear、q90-log1p、train-only standardized-log1p、energy+shape，train 和 held-out test normalized L1 median 都接近 1。在 `alpha_x=0.002`、`B_asset≤256 KiB`、`C_prepare/C_eval≤65k MAC` 的限定范围内，direct small-MLP 因高光能量和 top-energy recall 丢失而淘汰；这个结论不外推到宽峰或 residual。单界面 analytic core 在 noise floor 内通过，但 residual 近零，因此只作为解析 control。

第二个 `ncls.e1-layer-stack-multi-interface@1` 数据固定三界面、两 slab 的移动峰状态；adaptive reference 实际使用 16,384–36,864 个合并样本，relative SE p95 `0.0277`、最坏 query-group `0.0510`，通过独立监督 gate。core-only test normalized L1 median 为 `0.866`；64-wide neural residual 降到 `0.217`，同预算 direct dense 为 `0.290`，证明 residual 有非零贡献。加入 energy+shape、扩大到 64,603 参数、使用 cosine 后，SiLU 为 `0.0690`；同参数 GELU 为 `0.0462`，并通过 `ncls.e1-single-material-evaluator-acceptance@1` 的全部 31 项检查。GELU 候选的 test p95 `0.0576`、energy p95 `0.0204`、peak angle p95 `0.278°`、top-energy recall p5 `0.9463`、model/reference-SE p95 `2.835`；adversarial median/p95 `0.0462/0.0562`。静态成本 `B_asset=258,412 bytes`、`B_shared=0`、`C_prepare=13,716`、`C_eval=50,220` MAC。

这项通过只建立“固定多界面材质的 optimized-latent 容量候选”。它还不是 shared decoder、source compiler、Slang parity、GPU timing 或 viewer 证据；`prepare2/evaluate3` 同参数对照的 test median `0.0916`，因此保留 `prepare1/evaluate4`。完整逐 run 对比位于 `artifacts/research/learning-goal/e1/comparisons/multi-interface-residual-capacity.json`。

plane/tensor factorization 的 E1 smoke 已通过同一 lifecycle 完成。`ncls.pairwise-direction-plane-factorization@1` 把 `(wo.x,wo.y,wi.x,wi.y)` 组成六个成对 2D plane；32² direct 的 train/test median 为 `0.271/0.779`，改为 analytic residual 后为 `0.134/0.571`。降到 16² 后 test 改善为 `0.377`，但 train 退到 `0.328`，证明 v1 在高分辨率对未见 query 过拟合、低分辨率欠拟合。当前淘汰 raw-direction pairwise-plane v1，不继续扩大网格；这个结论不否定带 microfacet/half-difference warp 的新 factorization，也不涉及 E5 的 UV/spatial plane。

至此 E1 对实际候选需要的方向编码、target transform、direct/energy-shape/analytic residual、宽深/激活、prepare/evaluate 划分和 plane v1 都已有可复现结论，并保留一个通过冻结数值/静态成本 gate 的 optimized-latent 候选。下一步进入 E2 shared decoder + material latent；不能把 E1 的全部 asset-specific 网络 bytes 误写成最终 `B_shared/B_asset` Pareto。

E2 的共享表示监督入口已经通过冻结 gate。`ncls.e2-layer-stack-shared-decoder@1` 固定 12 个 family × 2 个同拓扑局部状态，以 family 为 split group 得到 20/2/2 个 train/validation/test state；每个 state 有 `16/4/4/4` 个独立 query role 和每组 128 个 `wi`。`ncls.e2-layer-stack-independent-peak-grazing-mixture@3` 对单界面 sheen 使用原生 roughness 条件峰中心，对其余 LayerStack 状态使用围绕几何镜面方向的 75 点窄 vMF patch，以覆盖随机游走 response 相对镜面方向移动约 4–11° 的多界面峰；它仍保留 34 个 uniform 与 19 个 grazing query，并持久化连续 mixture PDF 和积分权重。正式 v5 dataset `bbcd51e7451e7e2b8df705abc4eeb1382684c37bce1152284d9c4900dc5fd515` 的四 role peak spacing p95 为 `1.771/1.771/1.776/1.808°`，relative SE / replica L1 p95 为 `0.0398/0.0317`，最坏 query 为 `0.0813/0.0887`，split/source/direction leak 为 0，通过 `ncls.e2-shared-decoder-supervision-entry@4` 全部 61 项检查。只有 `family-0002`、`family-0003` 使用有 provenance 的 262,144 samples/replica 定向上限；没有扩大其他 family。

这个通过只允许公共 learning lifecycle 读取 v5，不是 shared decoder 已经成立。E2 必须按顺序报告 optimized/autodecoder latent 上界、target response encoder、encoder initialization + bounded refinement，以及 dictionary/factorized latent；target-visible 方法能读取 reference response，仍不得解释为 E3 source compiler。历史 v1–v4 H5 保留为掠射、response measure、moving peak 与 noise cap 的失败证据，不形成第二套 reader 或 runner。

三组 LayerStack 的相同 state/query 在四档自适应预算下测得以下 noise 曲线；`sample_count` 是合并两个 replica 后的总样本数上限：

| 最大总样本数 | relative SE p95 | replica normalized L1 p95 | gate |
|---:|---:|---:|---|
| `16,384` | `0.3445` | `0.3634` | 失败 |
| `131,072` | `0.1953` | `0.1649` | 失败 |
| `524,288` | `0.1152` | `0.0930` | 仅 relative SE 失败 |
| `1,048,576` | `0.0849` | `0.0624` | 通过 |

因此冻结 noise gate 可达，但百万样本是高方差状态的诊断上界，不是所有正式数据的默认预算。一次把单 batch 提高到 16,384 触发 Windows D3D12 TDR；保持 8,192 batch、增加迭代次数可稳定完成。audit v2 必须先按 state/split/积分能量列出最坏 query，再决定哪些状态需要高预算或更好的 reference proposal。

1. ~~扩展 query plan，使每个 `wo`/surface 可以拥有独立的 `wi` proposal，并让 train/validation/test 使用互不重合但测度明确的方向 probe；~~ 已完成逐 `wo`、逐 surface proposal、ReferenceDataset v4 query role 与三种版本化 partition policy；
2. ~~对 LayerStack 极低 roughness、MERL 高光、OpenPBR transmission 和 MaterialX normal-map 移动峰做高分辨率 probe；~~ 已完成并进入 v6 gate；
3. ~~把默认均匀 proposal 扩展成带显式 mixture component 的训练 proposal，并保持固定 validation/test proposal；~~ 已完成球面 vMF `@2` 与 MaterialX local-normal adapter；
4. ~~针对 LayerStack reference noise 调整自适应采样后，只重生成最小必要 H5，并重新执行合同、hash、split 与 gate；~~ 六状态 boundary adaptive H5 已通过 v6；
5. ~~完成 E1 的方向编码、target transform、容量和 factorization 比较；~~ 已保留一个通过数值/静态成本 gate 的多界面 analytic residual 候选，并淘汰极窄 direct MLP 与 raw-direction pairwise-plane v1 的限定范围；
6. 在同一公共 reader 上进入 E2 shared decoder，依次比较 optimized dense latent、target encoder、encoder initialization + bounded refinement、dictionary 和 factorized latent；
7. evaluator 成形后再进入 MaterialX spatial latent、sampler 和 integration。

单次数据、audit、训练和报告都位于 `data/reference-responses/` 或 `artifacts/`，不进入根 Git。本文只维护稳定结论、实验依赖和验收逻辑。
