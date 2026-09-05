# 空间编码、采样与 loss：证据和诊断

## 1. 当前结论

**应把“source/采样/部署读取对齐”放在重做 encoder 之前。** 当前确有 encoder，但它消费的是被手工压缩的 patch summary；另外，旧训练的亚 texel 位置、footprint target 和部署 bilinear 路径存在可定位的差异。现有实验不能把这些差异与 encoder 容量不足分开。

因此，对用户的回答是：encoder 可以学习保留任务需要的邻域信息，但前提是输入包含这些信息，GT 对应同一位置和过滤尺度，训练也约束真正部署的 latent 读取。当前三个前提均有待补齐的证据。继续把失败单独归因于“局部语义带宽不足”过于确定；“只需调 spatial loss”同样没有依据。

用户随后明确原始纹理经 encoder 生成 latent 是必修结构。这一要求已写入 R8：采样/目标对齐仍先于有意义的训练比较，但实现真正的空间 encoder 不再以读出实验先证明收益为条件。结构修复的必要性与其能够解释多少旧质量退化，是两个分别验证的问题。

本文保留最初研究时的证据：只读代码、历史文档、图像和论文，没有运行模型或生成新实验结果。代码定位固定到 Git `cc4d76bf4df089b725ad91b2a2673ca177edff86`，可用 `git show <commit>:<path>` 阅读；未另说明的路径/行号均指该快照。架构现已在 `ea2d743` 提交，当前 HEAD `e3f1c21` 的接口复核、补充发现与实施落点另记于 [提交后代码核对](post-architecture-code-plan.md)。原始源码锚点不改写成新行号；其他模型问题见 [模型设计审计](model-design-audit.md)，确定缺陷、语义缺口与模型近似仍分开处理。

## 2. 先前研究已经回答了什么

| 证据 | 原始定位 | 可保留结论 | 不能推出的结论 |
|---|---|---|---|
| 用户引用的四点总结 | Codex `01a07002-caaa-7c23-9b31-c6095ab05a97`，turn 7 | 当时展示的是 Tungsten 单材质 step2048 hybrid | 不是所有材质或完整 neural correction 的成功证明 |
| hybrid/direct matched pair | [single-material-selection.md](../../09-04-metal-neural-budgeted-redesign/research/single-material-selection.md:9) | 相同历史预算下 hybrid 的 appearance/peak/chroma 更好 | 无法单独分解 analytic core、gate 和 positive 分支的因果贡献 |
| v4 role separation、v5 center、v6 signed derivatives | [characteristic-probes.md](../../09-04-metal-neural-budgeted-redesign/research/characteristic-probes.md:59) | 这些具体变体没有同时改善划痕青铜和开裂钢的空间误差 | 没有检验完整二维 patch encoder，也没有隔离共同的采样/target 差异 |
| 128-step fixed-batch spatial-only | 同文 `:72`、`:104` | 现有模型、初始化、优化器和固定 batch 下拟合困难；梯度非零 | 不是自由 latent 的 optimized-code control；不能证明所有 encoder 或优化方案都失败 |
| mixed 512-step | 同文 `:106`；[final-conclusions.md](../../09-04-metal-neural-budgeted-redesign/research/final-conclusions.md:57) | 平均响应与空间响应的差距扩展到有限 mixed groups | 不是完整 692-source 泛化 |
| 前期 texture encoder 设计 | [texture-encoder-sharing.md](../../archive/2026-08/08-30-vmaterial-metal-neural-system/research/texture-encoder-sharing.md:29) | 曾明确提出 role stems、共享多尺度 trunk、跨图关联、encoder-only/refinement/direct control 分工 | 设计存在不等于当前 budgeted 版本已实现它 |
| 更早的 full model 审计 | [initial-evidence.md](../../09-04-metal-neural-budgeted-redesign/research/initial-evidence.md:21) | 辅助 semantic head 未必被最终 evaluator 消费；不能只看 auxiliary loss | 旧 full 的问题不能直接套到当前 v3，须逐符号重查 |

历史 pair 的 appearance 为 hybrid `1.14500`、direct `1.83045`；spatial 为 `0.28299/0.28120`。这些是旧报告的观测值。本轮不重新估计 CI，也不把它们设为新任务验收门槛。

旧资产审计还发现 692 exports 只对应 52 个唯一 texture sets。未见资产的 split 应按 texture-set identity 做；不同金属共享同一组纹理时，按 export 拆分会夸大空间泛化。该数量是旧 registry 的记录，未来正式实验重新核对。[texture-encoder-sharing.md](../../archive/2026-08/08-30-vmaterial-metal-neural-system/research/texture-encoder-sharing.md:3)

## 3. 当前 encoder 实际看到什么

主路径：

```text
原生纹理资源及 typed 参数
  → source adapter 的坐标变换和 mip 选择
  → 每 slot 两个 mip 的 8×8×4 patch
  → 手工 summary + role embedding
  → 小型 per-slot MLP + slot 聚合
  → Detail4 / Context4
  → prepare 的 semantic decoder 与 frame/core 参数
  → directional MLP + analytic lobes
```

`src/ncls/learning/models/metal_budgeted_asset.py:188` 的 `_encode_source_patches()` 是关键：

- v3 把每个被选 mip 的 patch 变成中心 2×2 均值 `c`、全 patch 均值 `m`、`c-m`，再拼接 8D role embedding。
- 神经层接收 20D 输入；其中 `c-m` 由 `c,m` 决定，不增加独立信息。除中心统计外，邻域的二维排列没有进入神经层。
- 两个 encoder 都是 `20→16→4` MLP，再以同一组 slot softmax 权重聚合到 Detail4、Context4。一个 slot 含多种 packed 语义时，role class 并不等价于逐原生通道语义。
- v4 改变角色聚合；v5 改为请求中心 texel；v6 改用中心和 x/y 中心差分。它们仍未让网络读取完整二维 patch。

保持中心统计与全局均值不变、重新排列其余 texel，会得到同一个 v3 输入。这证明该 summary 不保留一般二维结构；**不证明被丢弃的每一种排列都会改变当前 GT**。是否需要完整邻域，必须由正确对齐的 response 任务判断。

`src/ncls/learning/models/metal_budgeted_evaluator.py:124` 的 prepare 消费 Detail、Context、program condition 和 `wo`，v3 将 Detail4 加到前四个 frame semantic 分量；`:276` 起的方向表示消费完整 semantic state。因此“旧 v1 只读前 8 个 semantic 分量”的缺陷已不是 v3 的当前解释。

### 3.1 用户明确后的输入合同

原始纹理指经正确文件/通道/颜色/normal 解码后仍保留二维像素场的原生资源。中心值、均值、导数可作为补充特征，但不能成为网络唯一能看到的纹理内容。按语义分组的输入 encoder、共享空间 trunk、跨图融合与联合 latent 已在前期 [texture-encoder-sharing](../../archive/2026-08/08-30-vmaterial-metal-neural-system/research/texture-encoder-sharing.md:29) 中提出；本轮据用户要求将其落实为主路径。

整体是 encoder–decoder，后半段的目标是原生 reference 的散射 `f`。纯纹理 codec 的 decoder 重建 texel，本项目还需要原生参数/图条件与 `wo/wi`，不能因输入是纹理就丢掉这些语义。相关区别也见 [problem_definition.md §5](../../../../docs/research/problem_definition.md:139)。纹理重建 head 可作辅助监督，不能代替最终 response 学习。

较强 encoder 让任务相关的邻域信息有机会进入 latent；固定八通道仍是有损压缩，不能保证保留任意原图的一切排列。下一轮评测的是新结构的可用信号、质量和成本，而不是重新投票决定是否满足原始输入要求。

### 3.2 encoder-only 还需去掉资产专属记忆依赖

当前 [asset.py:172](../../../../src/ncls/learning/methods/metal/asset.py:172) 注册 `variant_scale_bias = nn.Embedding(asset_variant_count,16)`，`:327` 按 `resource_variant` 读取并调制量化后的 Detail/Context。raw grid 虽由 encoder 产生，最终 asset-conditioned 状态还依赖逐资产学到的参数，所以单凭 cook 的 `encoder-only` 标签不足以证明新纹理只需前向编码。

满足 R8 的首个方案取消这条独立 affine 表，保留基于原生材质参数的 compiler 条件；后续若确需资产 modulation，只能由 encoder 生成并计入成本。资源索引仍可定位原始/编译纹理。该项是用户明确 encoder-only 合同后的设计修正，未证明它是旧空间失败的主因，也未声称表已经在代码中移除。

### 3.3 同语义固定数值尺度：本轮读取审计

用户进一步明确 encoder 应接收统一语义范围的原始值，不允许每张 texture 按自己的分布重新拉伸。R9 已将此写成输入合同。统一尺度不要求每张图覆盖相同 min/max；不同图的实际亮度、roughness 或 height 幅值差异正是应当保留的信息。

本次读取工作树中的 `learning/methods/metal/native_assets.py`：`_read_block():391` 和 mip patch 的 decode 分支以 `np.iinfo(dtype).max` 做固定 UNORM 解码，并按 slot 声明使用标准 sRGB transfer；`_canonicalize_decoded_channels():80` 仅适配通道布局。`references/programs/mdl.py:90` 的 reference upload 也按声明处理格式和 gamma。在检查的纹理读取路径中未发现按图像内容统计的 min-max、均值方差或直方图值域变换；不能把这一点写成已发现的旧根因。

`_normalize_gpu_roles():633` 及 tile 路径含逐 texel normal 向量单位化，并重新打包到 `[0,1]`。这是方向处理而非逐图统计归一化，但它与源图幅值含义、mip/bilinear 前后顺序是否一致仍需 D0a 独立检查。允许固定解码不等于已经批准所有 normal 预处理。

`learning/methods/metal/data.py:44` 的 `_normalized_components` 使用参数声明的范围/default，作用于 typed material 参数，不作用于原始 texture pixels；它的名字不构成逐图归一化证据。训练 objective 的 reference percentile scale 也属于 loss 数值处理，不能混作 encoder 的纹理输入变换。

具体固定映射、禁止项及范围碰撞 witness 见 [design §2.1](../design.md)。这里是源码阅读结果，没有运行模型或用真实纹理测量该 witness。

## 4. 更优先的采样与过滤问题

### 4.1 亚 texel 坐标没有进入训练中的 latent 重建

`src/ncls/learning/mdl_metal_assets.py:591` 的 GPU sampler 使用 `floor(u·W), floor(v·H)` 取 patch，没有亚 texel 插值权重；CPU 对照在 `:417` 使用相同几何。`_encode_source_patches()` 不接收 texel 内位置。

在 source/state、方向、整数 mip 及所有 slot 的采样索引不变时，同一格内不同 UV 会得到相同 encoder 输入和同一预测。然而 MDL native lookup 使用 smootherstep remap 后的连续纹理采样，目标可能随亚 texel 位置变化：`shaders/ncls/reference_backends/mdl_runtime.slangh:431`、`:487`。

这是一种可构造的输入碰撞：多个不同 target 对应同一模型输入。更强的 spatial loss 无法补回缺失的条件；应先比较 texel 中心与亚 texel 网格的相同查询。不能仅把中心 2×2 平均改成单 texel 就认为问题解决了。

### 4.2 旧 footprint recipe 的输入与 target 尺度不一致

证据链：

1. `configs/training/recipes/metal-budgeted-hybrid-pilot.yaml:23` 固定 `evaluation_samples: 1`、`footprint_samples: 1`，同时使用 0/1/4 texel 三档 footprint；v4–v6 recipe 保留了同样设置。
2. `src/ncls/learning/source_adapters.py:763` 将 footprint 变成输入 source mip。单位 texture scale 时，4 texel 档选 mip2，0/1 档均为 mip0。
3. `shaders/ncls/reference_query/reference_query.cs.slang:55` 在 `sampleCount==1` 时返回零 footprint offset，因此只求中心点。
4. MDL lookup 走显式 `SampleLevel(...,0)`（`mdl_runtime.slangh:139`）；`mdl.slang:32` 构造的 target state 没有导数字段。不能依靠传入 `uvDx/uvDy` 自动得到空间平均。

因此，在该 MDL/recipe 路径中，粗 source mip 输入仍拟合中心的 LOD0 response。以单位 scale、同分辨率纹理的 mip2 为例，移动一个 finest texel 经常仍在同一个 coarse cell，模型输入不变，而目标细节可以变化。这个矛盾同时存在于旧对照与变体，**其负结果不能单独裁决 encoder 容量**。

公共 reference dispatcher 已支持多 footprint 点求完整线性 `f` 的平均。应先用固定查询比较 point target 与显式 footprint average，再决定训练合同；不能直接把“多采几个 `evaluation_samples`”等同于空间积分。MDL evaluate 在 `mdl.slang:101` 不使用传入的 sample generator，单点确定性重复与随机游走 reference 的重复采样含义不同。

### 4.3 训练 QAT 不等于部署读取已经匹配

训练 `_read_planes()` 在缺少显式 plane tensors 时直接编码 patch；启用 QAT 时，对单个 encoder 输出执行 SNORM8 STE。部署则在 texel 中心 cook 两套 mip hierarchy，再读取四个量化邻居做 bilinear：

- cook：`src/ncls/learning/metal_budgeted_asset_cook.py:144`；
- Python runtime：`src/ncls/learning/metal_budgeted_runtime.py:180`、`:212`；
- 训练分支及 QAT：`src/ncls/learning/models/metal_budgeted_asset.py:290`、`:313`。

训练近似 `Q(E(P_cell))`，部署是 `Σ w_j Q(E(P_j))`，一般不同；量化应发生在 texel 值上，之后才插值。若把已插值 latent 再量化，也会引入额外差异。

Context 在 v3 部署中只有 Detail 的 1/4 线性分辨率，而训练每条 query 直接算 Context。v6 增加 Context 分辨率能改变部署近似，却不能独立验证训练阶段是否缺少该分辨率带宽。旧 v6 pilot 未部署到完整资产，所以其 `1.882×` 是预估 bytes，不是已测运行结果。

另外，训练 fractional mip 用 UV hash 决定，部署用 `filter_random`；paired UV 可能选择不同 mip。旧单位 scale 的 0/1/4 档通常没有 fractional mip，故这一点是后续范围风险，不能当作已有主实验的确定根因。

### 4.4 需固定 witness 核实的坐标疑点

原生 payload 在 asset sampler 与 reference upload 两侧都规范为 top-left 行序：`mdl_metal_assets.py:391`、`src/ncls/references/programs/mdl.py:112`、`src/ncls/references/query.py:189`。MDL lookup 随后显式执行 `v→1-v`，而 adapter 的 `access_uv` 直接进入按行号取 patch 的路径。

这构成 V 坐标方向不一致的高优先级疑点。具体 graph 仍可能有自身变换，当前没有对每张真实 slot 做 witness，因此不能宣称所有材料已经证实上下翻转。下一步用非对称纹理、已知坐标和真实 material slot 对照有效采样值；同时检查 gamma、packed mapping、normal 切线基与 authored transform。不能用最终 `f` 的大误差反向选择“看起来更好”的翻转规则。

## 5. 采样与 loss 已有何种监督

不是完全没有邻域监督：

- adapter 生成 x/y 轴配额相等的一 native texel pair：`source_adapters.py:885`、`:937`；
- producer 对两点复用同一 `wo/wi` 和 reference seeds，并联合筛选 valid 行：`src/ncls/learning/producer.py:852`、`:864`；
- `src/ncls/learning/appearance_metrics.py:144` 计算 `|Δlog(1+f_pred/s) − Δlog(1+f_ref/s)|`；
- appearance objective 已给该项 `0.50` 权重：`src/ncls/learning/methods/metal_budgeted.py:574`。

需要澄清三个含义：

1. 名称为 spatial gradient，但公式没有除以 `Δuv`，实际是固定距离的 log-response 差分误差。跨分辨率、变换或距离比较时须同时报告距离，不直接当导数大小。
2. `semantic_runtime` 是 ungated analytic response 对 reference 的误差（同文件 `:568`），没有监督局部 color/normal/roughness/mask 的重建。名字不能代替职责证据。
3. 主 run 在 `[0.371,0.619]` 周围 `1/64 × 1/64` UV tile 采样；对于 4096²、单位 scale，约为 64² texel，面积占完整 UV 域 `1/4096`。这适合作为局部诊断，不能支持整个纹理或未见区域的恢复结论。

仍应检查每个 loss 对共享参数的梯度范数、夹角、量化占用与激活饱和，以及 x/y、mip、方向类别和 mask 边界的误差。非零梯度只证明有数值通路，不证明信号方向有用。旧 proposal 目标在 `metal_budgeted.py:751` 将非 proposal 参数 detach，不能直接把 sampler loss 数值变化解释为 encoder 梯度竞争。

## 6. core/gate/residual 的结论需收紧

`metal_budgeted_evaluator.py:442` 的 hybrid 为：

`f = softplus(r) + sigmoid(g) · a`。

`:468` 的 `positive_rgb` 是 `mean(positive²)`。旧记录的 `2.4e-6` 对应 RMS 约 `1.55e-3`，不是幅度 `2.4e-6`，更不是相对贡献率。analytic trace 又在分 lobe 后平方，与最终 gated/summed response 的能量不是同一口径。

此外，gate 本来就可以衰减 analytic core。因此问题不是当前模型“完全不能做负修正”，而是三者可互相补偿，难以解释各自职责；positive-only 分支不能独立拟合 `reference − core` 中的负部分。最后一层 positive bias 初始化为 `-5`（`:389`），是优化行为的候选影响因素，尚无消融证明它导致退化。

下一轮先报告相同查询下 `a`、`g·a`、`p`、`g·a+p` 的误差及移除分支的变化，按亮/暗区分层，再检验固定 core 的有界 signed correction。是否有用由 held-out 误差增益判定，不要求 residual 幅度一定变大。

## 7. 相关论文如何约束方案

- **Neural Graphics Texture Compression Supporting Random Access（ECCV 2024）**：论文以原始多通道 texture set 为输入，使用卷积 encoder 与随机访问的全连接 decoder；其 grid constructor 读取最高分辨率纹理，再生成供不同 mip 查询的表示。它直接支持“原始纹理内容经过 learned encoder”的结构参考，但 decoder 重建的是纹理值，不能将其质量结果当成本项目 neural scattering 的证明。[ECVA 论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05476.pdf)
- **Real-Time Neural Appearance Models（2024）**：§5.1 使用的是点式 MLP encoder，输入原生材质属性，粗 mip 以 LEAN 辅助预过滤；decoder 收敛后烘焙 latent，再通过实际纹理读取微调。由此不能推出必须上 CNN；应该首先补齐有语义的输入与真实读取训练。已复用本地作者论文文本 `.trellis/tasks/08-25-03-neural-baseline-and-candidate/scratch/nvidia-neural-materials-author-paper.txt:454`、`:467` 及既有 correspondence。[作者论文](https://research.nvidia.com/labs/rtr/neural_appearance_models/assets/nvidia_neural_materials_author_paper.pdf)、[作者项目页](https://research.nvidia.com/labs/rtr/neural_appearance_models/)
- **NeuMIP（2021）**：把位置、方向与过滤尺度共同作为材质查询，优化 neural mip hierarchy。它支持“尺度是 response 任务的一部分”的设计，不证明本项目八通道配置一定充分，也不作为未见资产 encoder 泛化证据。[作者项目页](https://cseweb.ucsd.edu/~viscomp/projects/NeuMIP/)
- **Filtering After Shading with Stochastic Texture Filtering（2024）**：非线性 shading 前过滤参数与 shading 后过滤 response 一般不同。这里借鉴其语义区分来核查 GT；本轮不因此增加随机读取数或更换 viewer estimator。[作者论文页](https://research.nvidia.com/publication/2024-05_filtering-after-shading-stochastic-texture-filtering)

本轮通过上述作者/出版方站点核对论文信息；ECCV 论文已在线读取，NVIDIA 大 PDF 的在线正文抓取受工具大小限制，其 §5.1 具体内容来自先前已保存的作者文本。没有将论文重新下载或复制到工作区。不同论文的 encoder 输入与优化生命周期并不相同；R8 是用户明确的本任务结构要求，不是从所有 neural texture 方法都使用相同 encoder 推导出的普遍规律。

## 8. 研究优先级

1. **固定查询的坐标与尺度 witness**：判断是否在同一位置、同一 footprint 上学习同一个函数。
2. **完整训练/部署读取对照**：texel 中心、亚 texel、mip、wrap 边界，区分 source 查值、量化和 latent 插值。
3. **落实原始纹理 encoder 并隔离表示/优化问题**：实现保留二维结构的主路径；旧 summary 与自由 latent 仅作为对照，在同一个正确读取路径上比较。
4. **loss 与 correction 职责**：前面路径一致后，再做局部监督/梯度冲突和 signed correction 的独立消融。
5. **跨区域/资产与成本**：局部结论成立后再覆盖 held-out tiles/texture sets，并实测成本；不自动进入全量长训。

具体可执行分支见 [design.md](../design.md)，图像观察见 [viewer-evidence.md](viewer-evidence.md)。
