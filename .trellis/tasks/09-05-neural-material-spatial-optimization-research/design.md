# 技术规划：先对齐学习问题，再比较表示

## 1. 目标与边界

本设计交付原始纹理 encoder 的必修结构、诊断顺序、已知缺陷修复与分支决策，本轮不实施。依据见 [证据分析](research/evidence-and-diagnosis.md)、[viewer 观察](research/viewer-evidence.md) 和 [模型设计审计](research/model-design-audit.md)。用户已经明确结构目标；尚不确定的是各项改动的质量收益和其他失败原因。

公共行为仍是 source-native reference 产生线性 `f`，编译器输出运行成本静态有界的 neural material program，`prepare` 获取并复用局部状态，`evaluate(wo,wi)` 做方向查询。诊断可以读取 native source 属性，但不能把新的“八通道语义”强加给所有原生材质族。

架构已在 `ea2d743` 提交，本次规划基线为 `e3f1c21`；当前接口和代码锚点见 [提交后代码核对](research/post-architecture-code-plan.md)。本轮按用户要求将文档细化为代码计划，仍保持 `planning`，没有实施。后续直接接入当前 `Method`、通用 online session、训练 engine、checkpoint 和 package compiler。旧 checkpoint、包、runner、CLI 和 identity 仅用于解释旧证据，不要求新系统兼容或恢复它们。

## 2. 需要对齐的数学对象

设 `T` 为原生资源与参数，`A` 为声明的纹理坐标变换，`u` 为 surface UV，`F(T,u,wo,wi)` 为权威 reference 的线性散射。

点查询目标为 `y0 = F(T,u,wo,wi)`；过滤目标为：

`yρ = ∫ Kρ(ξ) F(T,u+ξ,wo,wi) dξ`。

不能直接用 `F(filter(T),u,wo,wi)` 代替后者。normal、roughness 与 mask 的相关性通过非线性 scattering 影响平均值，辅助统计只负责帮助编码，GT 仍由原生 reference 求值。

部署读取应明确写成：

`X_s = S_g(s)(Decode_d(s)(T_tex,s))`

`H = E_spatial({X_s}, role/schema, declared_coordinate_maps)`

`Z_l,j = Q([G_l(H)]_j)`

`z_l(u) = Σ[j∈4 neighbors] w_j(u) Z_l,j`

`fθ = D(prepare(z_l(u), program, wo), wi)`。

`Decode_d(s)` 只按资源声明的存储/原生编码解释像素；`S_g(s)` 是同一语义定义与单位下共享的固定映射，首个实现默认 identity，不依赖第 s 张图的内容统计。`E_spatial` 包含分组 encoder 与空间融合，`G_l` 从学到的特征生成第 l 级 latent。高分辨率原始信息应进入可学习空间层；不能只给网络已经求均值的 coarse patch，再称其拥有原始邻域。原生 mip/局部统计可作为补充输入，其过滤定义须明确。

QAT 作用于每个 texel 的 `Z`；bilinear 在量化之后。Detail 和 Context 各自使用实际尺寸、texel 中心及 address mode。训练可以按带足够 halo 的 tile 计算查询所依赖的 latent 邻居，覆盖所选网络的完整感受野，无需把整张训练纹理常驻 autograd，也不保存训练 batch。weight/program 量化与 deferred prepared state 的 FP16 pack/unpack 分开加入对照，不能把一种量化通过当作所有存储路径均已覆盖。

fractional LOD 先在诊断中固定上下级选择，分别核对两个输出；若使用随机混合，再复用同一选择变量比较 pair，并单独估计其期望与方差。把随机差分噪声混入空间目标会妨碍定位。

### 2.1 原始纹理与语义分组

“原始”指材质原生资源中的 texel 数据，包括二维排列、必要通道、原生分辨率及坐标关系。解压文件、依声明做 sRGB/linear 转换、解开 packed channels、解释 normal 编码属于必要的输入适配。height 图仍保留原始 height；若派生 normal/导数，作为附加信息，不能取代原图。没有权威语义的 packed channel 不能自行解释为 roughness 或层参数。

**同一语义采用统一数值尺度，禁止逐 texture 的内容自适应变换。** 解码规则可以因文件格式、transfer function 或原生编码声明而不同，结果必须恢复各自真实的语义数值，不能把每张图单独变成“看起来一样的值域”。不同实际内容可以占用不同范围，例如两张 roughness 图分别只使用 `[0.1,0.3]` 和 `[0.7,0.9]`，它们应保留这种差异。

| 操作 | 输入合同 |
|---|---|
| UNORM8 除以固定的 255、UNORM16 除以固定的 65535 | 格式解码，分母由声明的格式决定，不能替换成该图实际最大像素值 |
| 按声明将 sRGB 解码为 linear；按声明将 normal 分量从 `[0,1]` 解码到 `[-1,1]` | 对相同编码使用相同规则；不自动对 scalar/data channel 做颜色转换 |
| 同语义、同单位的固定值域映射 | 可以明确声明后全组一致使用；首个实现不额外引入这种变换，也不按资产重估参数 |
| 每图/每通道/每 tile/每 batch 的 min-max、去均值除方差、直方图均衡、按分位数调 exposure/gamma/contrast | 不允许作为原始纹理 encoder 的输入标准化，会隐藏绝对幅值或使同一 texel 依赖其他区域/批次 |
| 将有效 HDR color、signed height 等统一裁剪到 `[0,1]` | 不允许；合法范围由原生语义决定，不能为迁就输入尺度丢掉信息 |

例如 `T_A=[0.1,0.2,0.3]` 与 `T_B=[0.7,0.8,0.9]` 逐图 min-max 后都变成 `[0,0.5,1]`。没有绝对尺度条件时，后半段无法知道是哪一种原始 roughness。即便额外保存 min/max 在数学上可能恢复信息，也属于另一种显式条件编码方案，不作为本任务 R9 的默认例外。

normal 的逐 texel 单位向量归一化仅在原生定义要求方向单位化时成立，它不等于按整图统计拉伸值域。若向量长度承载信息，或先归一化会改变插值/mip 语义，应保留原始值，按 native reference 的顺序处理。原生材质图中 authored scale/bias、height 单位和 normal strength 仍是 GT/参数条件的一部分；不能将其悄悄移到输入“标准化”中，更不能以统一尺度为由删掉可编辑参数。

R9 的 witness 在第一个学习层之前检查：改变图外其他 texel、tile 划分或 batch 组成，不应改变相同 texel 的固定解码值；不同绝对范围的图不能被标准化成同一输入；同语义的不同合法文件编码应在格式精度内得到一致值。该不变量不要求 encoder 输出与邻域无关，学习网络本来可以利用邻域。首个 encoder 也不能把同样的逐图/patch 归一化移到唯一输入分支中，换个位置后重新抹掉幅值。

第一个可学习空间层直接接收该二维数据，可为整图或等价的完整局部 patch。pooling、降采样和 bottleneck 可以存在于学习网络内部，但不以人工 summary 充当唯一原始输入。固定格式解码不等于手工信息瓶颈。

首个结构采用前期研究的分层共享：color、tangent normal、height/bump、scalar/data、packed 共五类输入 stem，normal 与 height 使用不同前端。组内跨纹理资产共享权重，随后接共享空间主干和跨 slot 的可学习融合。多张纹理按相同 surface 位置及各自原生 transform 对齐，保留 role/schema、有效通道和缺图 mask；不是将各图独立压成标量后平均。组间的专用前端不要求为每个 finish 或资产建立互不共享的完整网络。

首个实现以保留二维结构的局部卷积和多尺度特征作为具体方案；运行时仍只读取联合编码后的两个 RGBA plane，不按语义组增加 texture reads。具体层宽、感受野、数据关联和 cook 方式固定在 §8；这些选择影响质量，不改变原始输入的要求。

### 2.2 encoder + decoder 的职责及生命周期

```text
color 原始纹理 ─────── E_color ───┐
normal 原始纹理 ───── E_normal ───┤
height 原始纹理 ───── E_height ───┤
scalar 原始纹理 ────── E_scalar ──┼→ 共享空间主干/跨图融合
packed 原始纹理 ────── E_packed ──┘       → 量化 latent grids
                                                  ↓ 实际 mip/bilinear 读取
原生参数/图条件 ─────── source compiler ───────→ prepare(z, program, wo)
                                                  ↓
                                             evaluate(wi) → 线性 f
```

整体是以原生纹理为输入、以 scattering response 为主要监督的 encoder–decoder。runtime decoder 包括 `prepare` 与方向 evaluator；hybrid 时保留 analytic core、gate、correction 的现有职责供后续独立归因。原始纹理本身不能替代源材质的非纹理参数与图语义，这些条件仍必须进入 compiler/decoder。原生参数编辑的影响分清由运行时条件处理还是触发相应资产重编码，不能隐藏成新资产重训。

E/D 通过在线 reference 的 response loss 联合训练，梯度经过真实 latent 读取与量化近似回到 encoder。原生 texture/semantic 重建可设训练期辅助 head，但最终 decoder 不必重建整张纹理，也不必须先还原一套解析参数再求 `f`。未注册语义不能仅靠训练期 head 名称获得 GT 地位。

编译新同语义资产时，固定已训练 E/D，以原始纹理和声明元数据前向生成 `Z`；运行时只需要烘焙 latent 与 decoder，encoder 留在资产编译/编辑路径。存在 encoder 不等于已证明跨资产泛化，质量仍用按 texture-set 隔离的未见资产评测。

当前 `variant_scale_bias` 是按 asset ID 独立学习的表。首个满足 R8 的实现取消这种训练资产专属调制，以 identity 处理，保留来自原生参数 compiler 的条件；如果未来确需资产专属 modulation，须由 encoder 生成并独立登记预算。资源 ID 可继续用于查找纹理，但不能决定任意 learned latent/affine 值。用于归因的 summary control 同样去掉这条独立表，避免同时改变 encoder 和资产记忆通路。

encoder-only 是主路径；bounded refinement 另行报告，不能成为“新纹理直接编码”的隐藏步骤。自由 latent 只作为 optimized-code control，不能代替应当训练的 encoder，也不能将其结果登记成 encoder-only。

### 2.3 本次规划修正的范围

| 字段 | 内容 |
|---|---|
| trigger | 用户明确要求 texture latent 必须由原始纹理经 encoder 产生、不同语义组保留输入编码，并进一步要求同语义共享数值尺度，禁止逐图自适应值域 |
| invalidated evidence / decision | 旧实验数字和采样/部署缺陷不变；取消先前“只有完整 patch 读出更好才实现空间 encoder”的决策条件 |
| scope impact | R8 固定 encoder 结构，R9 固定原始输入的数值语义；D0 对齐采样、解码和监督，D1 测量新结构的质量及优化问题。当前仍只交付规划；架构提交后的具体代码落点和 C5 状态成本变更见 §8 |
| rerun required | 新 encoder/资产调制路径必须 fresh 训练；对齐后的 summary 仅作旧表示 control。旧 v4–v6 结果不能代替新结构实验 |

结构满足要求不保证质量必然提升。达到冻结 cap 后无收益仍按 empirical outcome 记录并停止该配置，不自动追加网络或预算；不能因此把 summary-only 重新登记为已满足 R8。

## 3. D0：无训练的 source/target/read witness

首轮只使用 Tungsten、划痕青铜、开裂涂漆钢各一个冻结原生 state。Tungsten 保留平滑/刷痕对照；另外两项复用旧失败案例。尚不扩到 692-source；黄铜、阳极氧化铝保留为后续确认集。

| probe | 固定项与唯一变化 | 记录什么 | 能区分什么 |
|---|---|---|---|
| D0a 坐标与资源 | 同一 slot 和已知 native UV；非对称纹理、已知绝对值范围与真实 source slot；记录坐标和解码值 | source payload 行序、graph transform、sampler 坐标、有效通道、固定值域、gamma、normal frame | V 翻转、逐图归一化、packed/normal 语义偏差与模型问题 |
| D0b 亚 texel | 同一材质、整数 mip、`wo/wi`；扫描一个 texel 内位置并包含四邻域边界 | native lookup、完整 patch、summary、latent、prepare、最终 `f` 是否变化 | 不可辨识的输入碰撞与 learned encoder 压缩 |
| D0c footprint | 同一 UV/方向，0/1/4 texel footprint；point query 与完整 response 空间平均分开 | reference center、reference averaged `f`、input mip、pair target 差分 | 当前 coarse input 是否在学习 point target |
| D0d 训练部署 | 同一权重和输入，逐级启用 texel quantization、bilinear、真实 Context 尺寸、mip 选择 | 每级 latent/prepare/`f` 差异和位置分布 | 量化损失、读法差异与网络拟合误差 |
| D0e target 稳定性 | 固定 source/query，复用及改变 reference seed；独立改变 footprint 样本数 | 重复性、空间积分收敛趋势、paired covariance | 确定性 MDL、随机 reference 噪声及不足的空间积分 |

这些 probe 不训练模型。非对称资源只可作为 reference 能表达的诊断 source，使用与原生图一致的变换和 sampler；不凭“翻转后误差更小”替代语义检查。

D0 的预期不要求误差变小。只要求相同数学路径的语义一致，浮点容差在观察正式质量前按 dtype 与独立 oracle 冻结。出现正确性差异即停止受影响的模型比较，记录影响范围，修复后 fresh 生成新查询/训练；不继续用旧负结果裁决新表示。

### 最小分支决定

- 若坐标/资源不对齐，先修 source adaptation 再接入已经确定的原始纹理 encoder；不以此取消结构修复。
- 若 point target 与 coarse input 不一致，先拆出 point-only 诊断，再恢复明确的 filtered response 合同；不能把换 GT 仅作为“调 loss”。
- 若只在 quantized/bilinear/cooked 路径退化，先让训练消费该路径；不能仅报告 FP16 weight MAE 后称部署训练一致。
- 全部对齐后才进入 D1，保留 D0 witness 为新实验的输入边界检查。

### 模型错误的修复依赖

模型审计发现的错误按受影响阶段处理。各项可独立验证，现阶段仍合在一个研究任务中交接；实施时即使拆分任务，也必须保留下列依赖，不能只靠任务树表达顺序。

| 项目 | 具体修复/决策 | 验收性质与来源 | 阶段依赖 |
|---|---|---|---|
| C1 control 初值 | 使用保持 initial 的参数化，明确可达域；第 0 步不得再次压缩 decoded state | 语义正确性：refinement 对照的起点等价与确定性字段不可优化 | 在使用 optimized program-state control 前完成；不与自由 asset latent control 混称 |
| C2 softplus | 用稳定的浮点实现，并做负尾部至正尾部的独立 oracle 比较 | 数值实现正确性：float32 误差分析；容差在部署质量结果前冻结 | 在 Slang/package 数值结论前完成；不阻塞仅 Python 的表示诊断 |
| C3 validity | 内部 evaluate 返回并传播 validity，真实零与无效零分开 | 语义正确性：公共非有限量规则及 Python/Slang 的行为一致 | 在部署质量、sample weight 结论前完成 |
| C4 frame 连续性 | 在允许域内采用连续的局部基，明确 rotation angle 与原生 frame 的关系 | 数学不变量：正交/手性、局部连续性和各向同性不变性 | 依赖 D0a；在新的 frame/方向表示训练比较前完成 |
| C5 reverse PDF | 使用 view-independent proposal 参数和独立 proposal frame，细节及额外存储见 §8.6 | 语义正确性：交换方向后独立 prepare 的真实 proposal 密度；导出 capability 一致 | 在 sampler 正式合同验证前关闭；不作为 forward evaluator D1/D2 的质量前置 |
| C6 未见资产编译 | 移除训练 source 名单准入与 asset-ID 学习依赖，保留原生 schema/资源兼容检查 | R8 需求交付与角色隔离；不能依靠额外训练生成新资产 | 在 encoder-only 生命周期验收前完成 |

每个 witness 都有独立的预期行为，不以“loss 变小”或两份相同错误实现相互 parity 作为正确性依据。C2/C3 的 shader 运行归入阶段收尾的部署轨道。若数学/参数化发生改变，新的研究对照使用 fresh identity；不将旧 checkpoint 的语义迁移设为本任务目标。

Beckmann 的 `G`、secondary 类型与 view-conditioned core 的物理含义先按审计 §7 明确为模型近似。若决定改成特定物理 core，则独立登记公式与成本后再比较，不与空间 encoder 变更同时实施。

## 4. D1：把表示问题与优化问题分开

在同一正确的 source/query/读取合同下，落实 §2 的原始纹理 encoder，再固定一小组方向，沿 `原始二维输入 → learned features → z → prepared → f` 观察信号。旧 `P → summary` 作为对照保留在诊断中。单个 batch 只驻留 GPU，退出诊断即释放；只保存配置、seed、摘要和图，不保存 batch。

### D1a：信号能否在各阶段被读出

比较 raw native 值/完整二维 patch、新 encoder 的空间特征、旧 summary、量化 latent、prepared state 对相同局部 response 差分的可预测性。每条路径必须带上正确的亚 texel 重建关系、role/schema 和原生变换；不能再次把缺少位置的 patch 当作充分输入。读出结果用于解释压缩与优化，不决定是否实现原始纹理 encoder。

先使用同容量的简短读出器或固定方向直接回归，不同时扩宽方向网络。报告 target/predicted 差分幅度、符号一致率、相关性、误差及 residual 频谱；低幅度和错误方向分开看。零 target 区域单独统计，不用除以近零幅度制造增益。

读出成功是“该阶段保留了可利用信号”的正证据。读出失败仍可能由优化器/读出容量造成，不自动证明信息论上不可恢复。

### D1b：同 runtime decoder 的三种资产获取路径

| 路径 | 允许优化的内容 | 研究角色 |
|---|---|---|
| encoder-only | shared encoder 与 decoder 联合训练；新 asset 只编码 | 未来同 schema 未见资产的 encoder 能力 |
| encoder + bounded refinement | 以 encoder latent 初始化；冻结共享 decoder，仅微调有界局部 latent | 编码器到逐资产可达质量的差距 |
| optimized-code control | 同形状、同读取/量化的自由 latent；保持已选 decoder 固定；必要时另列同预算 joint control | 局部代码获取与 decoder 表达的诊断，不是 zero-shot，也不是上界 |

decoder 固定时，自由 latent 也受该 decoder 已学表示影响。若三者都失败，不能立即归为八通道带宽不够；应结合 D1a、简单可知解的 scattering source 和优化轨迹继续分类。

### 必修 encoder 的验证与预算

按 §2 实现原始纹理空间 encoder 是固定要求，保持下游 prepare/evaluator 和读取次数，避免同时改 head 掩盖来源。以具有相同旧 summary、不同二维图案的纹理检查进入学习层的数据仍可区分；检查完整图与带 halo 的 tile 编码在对应区域一致，跨 slot 对齐正确，response loss 的有效梯度能够到达各输入编码分支。有限 latent 本来允许多对一压缩，不要求任意两张不同纹理的每个 latent 都不同。

在 encoder-only 检查中冻结共享权重：新资源 ID 只改变资源定位；同一原始内容、相同语义/参数应得到一致的编码行为，不能读取训练资产专属的自由参数。未知资产的 response 质量另列为观测值，不从这一结构检查推断 zero-shot 已成功。

不先扩大 runtime MLP。encoder 在资产编译/训练期运行，较强空间编码主要增加 cook 与训练成本，不必增加单次 `evaluate` 成本；这些成本仍须报告，且 patch/cook 工作量有界。

八个通道不是充分性的保证。RGB3 + normal2 + anisotropic roughness2 + mask1 已占满八维，尚未显式容纳 tangent rotation、多重 normal、coat color 等；这些只说明预算竞争，不能把源材质改成这八个参数。

首先比较保持原 Detail/Context 尺寸与 bytes 的候选。若将两个 RGBA plane 都提升到全分辨率，则是额外 asset-memory 轴，旧比例约 `2/(1+1/16)=1.882`，不能再称完全同成本。每个方案记录真实 mip bytes、prepare/evaluate MAC、state bytes 和读取数。

## 5. D2：在正确输入上检验 loss 和 correction

### 5.1 loss 的作用

保留 appearance 的线性、log、chroma、peak 与空间差分观测。每项按 mip、x/y、方向模式及纹理边界分层，并记录作用于 encoder/prepare 的梯度范数与夹角、SNORM8 占用、clamp/tanh/softplus 饱和。

局部辅助监督必须作用于 evaluator 真正消费的表示。native reference 能给出有权威含义的局部属性时使用 color、normal/frame、roughness、mask 的适当监督；否则以固定方向的 response 签名作训练期辅助目标。不能把推测的“层参数”或 packed channel 解释当 GT。

只有表示保留信号但 appearance 无法有效利用时，才冻结一个 loss 消融：相同 model/query/optimizer/budget，对比有无该辅助项。参数级梯度统计用于解释，不强迫所有 loss 梯度平行。

### 5.2 core、gate 和 residual 的独立归因

先在同一查询上评价 `a`、`g·a`、`p`、`g·a+p`，做分支移除与固定值干预；分别记录线性/log 误差、相对完整输出的差异和暗区/亮区贡献。不同分支的平方 trace 不直接相除作贡献率。

signed correction 的第一项诊断冻结 core/gate，设 `b=stopgrad(g·a)`，以 `y-b` 为 correction 目标。可研究：

`r = R · tanh(hθ)`，`f = max(0, b+r)`。

`R` 是训练前基于独立 calibration 选定的逐通道尺度，可依赖停止梯度的 `b`；其数值、单位和最大范围进入候选配置。报告 target 超出 correction 范围及最终非负投影的比例，不裁剪原始 GT，不按 test 结果扩大 `R`。

这个形式提供有符号、有界修正和非负输出，但不自动保证能量守恒/reciprocity，非负投影还可能使梯度失活。它是待检验候选。先证明 correction 在 held-out 查询有增益，再研究放开 core/gate 的联合训练，避免一开始三者互相补偿。若 residual 不提升质量，即使幅度变大也不算成功。

## 6. 评测、预算和停止条件

- 先固定上述三个 diagnostic source state、区域与方向配额。0/1/4 footprint、x/y pair、texel 中心/亚 texel分层报告；另留未见 tile 和方向，不用同一个被调参的 probe 充当 final test。
- 共享 decoder 的跨资产实验按 texture-set identity 分组，同源资源不跨 split 泄漏；参数泛化、方向泛化、未见区域、未见 texture set、工作流 W 分开表述。
- 历史 anchor 为两次 asset read、160 B packed state、11,392 dense evaluate MAC。160 B 不代表展开 state、MLP 临时数组或实际寄存器占用。它们用于对照与成本登记；不因新架构 ABI 改动而要求保留旧封装。静态有界是工程合同，实测时间另列。
- 单 source 诊断的 paired CI 以冻结 UV tile/query block 为独立单元，方向/通道是块内样本；多 source 结论以 source state 为外层，并对共享 asset 保持分组。单 seed CI 不覆盖训练 seed 方差，不能把 256 batch rows 当作 256 种材质。
- 新执行前冻结 seed、累计 reference query 数、optimizer step cap、工作量口径和选择规则。比较 encoder/损失时保持这些轴 matched；修正过滤协议时 fresh 重跑受影响对照，不按旧 step 数直接拼表。
- D0 source/query/read 正确性失败：停止受影响的 D1/D2；C1–C5 按上表阻塞对应阶段。实现修复以新记录重新验证，无效/非有限/资源失败分别归类。
- D1/D2 达预登记 cap 后无收益：记录 empirical outcome 并停止该候选。不得自动加 seed、加步、扩大网络或生成连续版本。
- 提升质量不是本轮研究文档完成条件。数值阈值只在有来源的正确性判据或经确认的正式研究协议中冻结，不根据旧观测编新 hard gate。

## 7. 后续部署和暂缓项

新的 evaluator/asset 路径形成后，在阶段收尾执行一次当前 package/Slang parity、同场景 viewer 和实际 query 成本测量；不要求每项机制 probe 都部署。matched sampler 保持与 evaluator 的公共接口关系，但 PT 方差、多灯 scaling、完整 UE 工作流不作为这轮空间诊断的前置。

本轮规划已无阻塞的用户范围问题。当前架构的具体代码落点见 §8，首轮 diagnostic 配置、预算、实施顺序与真实 CLI 见 [implement.md](implement.md)。本轮只完成规划，不执行这些命令；实测资源与质量仍待实施后的证据。

## 8. 当前架构上的具体实现合同

实施中用户明确要求按 UV 分组。以下 §8 原有单组形状以 §9 修订为准；C1–C6、原始数值、共享资源、独立 reference 和诊断预算保持原职责。

### 8.1 共享 tile 与 batch row 分开

新增 `learning/conditioning_resources.py`，只负责通用的不可变 GPU 资源集合、row 到资源的关联和 lease 生命周期，不认识 Metal 或 CNN。每项包含 CPU 上已知的资源 key、metadata 与 GPU tensor；原始纹理使用现有 `NativeAssetTileRequest/NativeAssetTile` 的 role/domain/halo 合同。

- `source_adapters.py` 的 `sample_tensors` 返回显式 `AdaptedConditioning(tensors, provenance, resources, bindings)`，替代二元组。Nvidia 的三个 adapter 返回空资源集合，只有返回封装变化。
- `TrainingConditioning.tensors` 继续只放第 0 维为 B 的 query tensor。`bindings` 是具名 int64 `[B]` 索引，每个值指向共享资源集合中的一个 tile bundle；bundle 包含该 source 的最多 9 个 slot tile、各自尺寸/原点/角色及坐标映射。未知 slot 不能靠零值伪装成已知语义。
- `TrainingConditioning.select_rows/concatenate` 一起操作 row tensor 和 binding；concat 合并 CPU key，并通过 GPU index remap 修正 binding。资源 tensor 不随 B 拼接。select 可保留资源全集，避免为删除少量未用 tile 触发 GPU→host 同步。
- producer 的候选拼接、主/paired 联合 rejection、accepted batch 截取都用这两个操作。资源在被保留的最终 batch 中持有引用；每个 lease 在最后一个 owner 结束时释放。异常、空选中、重试、提前停止、phase drain 和 session close 均须覆盖。
- Metal adapter 复用 CPU source cohort 和已预定的 tile origin；同一个逻辑请求的 rejection 重试留在该 tile 计划中。原始 tile 仍通过现有 host pipeline/residency 获取，不能调用第二个私有 GPU cache 或每步 `.cpu().tolist()` 决定资产集合。
- checkpoint 只保存可复现的 query/tile schedule、RNG 与模型/optimizer 状态；不保存 conditioning、原始 tile、GT batch 或未完成的 autograd graph。恢复通过共享 online session 重新获取数据。

`EvaluatorBatch`/`MethodSamplerBatch` 通过 `conditioning.resources/bindings` 访问共享输入，`.tensors` 仍只返回 row 数据。`Method` 的 descriptor 同时声明本方法需要的资源 binding，generic engine 校验存在性和 device；不能通过方法名称补数据。现有 `asset-tile` route 保持原职责，不把它当作与 evaluator 独立抽样却能自动对上的第二条 GT 流。

### 8.2 输入、网络形状与多尺度 hierarchy

新增 `methods/metal/spatial_encoder.py`，由 `asset.py` 持有并供 `asset_cook.py` 调用。预定接口为 `encode_tiles(tile_bundle, requested_levels)`，输出带 level shape、global origin 和 grid phase 的 `EncodedAssetTile`；`read_encoded_tiles(encoded, read_plan, qat)` 完成查询读取。

| 层 | 固定形态 | 信息与职责 |
|---|---|---|
| 语义输入 | 每 slot 的原生 mip0，最多四个值通道，加四个有效通道位；R9 固定解码 | 缺失通道只填占位值并提供 mask；正常原始值不裁剪、不按 tile 标准化 |
| 五类 stem | 各自 `Conv3×3(8→16) → SiLU → Conv3×3(16→16) → SiLU` | 同组共享权重；先在各 slot 原生网格学习，感受野为 5×5 |
| 对齐与融合 | 对齐 learned stem feature；每 slot 拼 16D feature、8D 共享语义 embedding、1D presence；9×25=225 通道，经 `Conv1×1(225→32)` | role 按声明映射，不再仅凭通道数/关键词猜组；固定 slot 顺序由 schema 给出，缺图 mask 防止缺失与合法零混淆 |
| 共享空间 trunk | `SiLU → Conv3×3(32→32) → SiLU → Conv3×3(32→32) → SiLU` | 同分辨率、同坐标时 `H0` 感受野 9×9；学习跨图相关性 |
| 共享 mip block | `Conv2×2(32→32,stride=2) → SiLU → Conv3×3(32→32) → SiLU`，跨 level 共享参数 | 从 learned `H_l` 生成 `H_(l+1)`，不对 raw texture 预求 coarse 均值后重新编码 |
| latent heads | 独立 `Conv1×1(32→4)`，随后 tanh 与 SNORM8 STE | Detail mip l 取 `H_l`；Context mip l 取 `H_(min(l+2,L))`；L 为最后一个 1×1 level |

所有 conv 的边界读取遵循对应 domain 的 repeat/clamp；不使用 zero padding 代替真实纹理边界，不用 BatchNorm/InstanceNorm 或输入幅值被消去的唯一路径。R9 的固定 mapping、原始 height 保留、normal 长度/顺序检查仍按 §2.1 执行。packed stem 只接明确的 packed 布局；已声明的各语义通道保留身份，未声明语义不能自行命名为 roughness。

联合 latent 网格以当前资产所有 slot 的最大宽/高定义。不同原生尺寸或声明的相对坐标变换，在 **stem 后**以 native mapping 对齐 feature；其所需原始 texel 必须完整进入 stem。当前公用 scale/rotate/translate 仍由 prepare 变换 query。若编辑改变 slot 间的相对 mapping 或资源，重新 cook；原生 scalar/颜色/strength 等条件仍进入 compiler。支持边界由当前 source adapter 的声明决定，不凭图像看起来对齐来推导 transform。

canonical mip 尺寸沿用 `max(1, floor(size/2))`；奇数尾部按这一定义处理，单维为 1 时按 address mode 扩出 stride-2 所需样本。卷积 stride 的 global phase 由全图原点固定，不能每个 tile 从局部零点重新开始。

同分辨率、无相对变换时，以上网络的 `H_l` 原始感受野边长为 `r_l = 9 + 5(2^l−1)`。实际 halo 由各 slot 的坐标逆映射、stem receptive field、所需 latent 四邻居及 stride phase 推导；不能所有 mip 固定拿一个 8×8 patch。首轮 LOD≤2 时，Context 最深到 H4，`r4=84`；预设 128×128 的 query core、对齐到 16 的原点及保守 64 texel halo，仅适用于等尺寸/同坐标情况。其它情况由 read-plan 推导扩大，超预算即报告资源问题。

训练仅为本 step 的被查询 tile 建图，主/paired query 共用同一份编码。可以在本 step 使用 activation checkpointing；不能跨 optimizer step 缓存 detached learned feature。cook 使用同一网络按层流式生成 hierarchy，必要的中间特征以受限 host staging 暂存，GPU 保持 tile 工作集；只写最终两张含完整 mip 的 SNORM DDS。cook 中间值不是训练 batch，结束释放，不进入 checkpoint。

### 8.3 读法、过滤条件与 reference target

新增 `methods/metal/asset_read.py` 定义纯 tensor 的 read-plan，供训练与 cooked Python 共用；shader 独立实现同一数学合同。

1. 由声明的二维仿射 A 变换 UV 以及 `uv_dx/uv_dy`。令 `J=[diag(W,H)·A·uv_dx, diag(W,H)·A·uv_dy]`，W/H 为 Detail base 尺寸。
2. `rho=max(length(J[:,0]),length(J[:,1]))`，`lambda=clamp(log2(max(rho,1)),0,L)`。零 derivatives 始终得到 LOD0；非等比 scale、旋转及非方图不用单个最大 scale 近似。
3. 显式 query `filter_random∈[0,1)` 选择 `floor(lambda)+[r<frac(lambda)]`；主/paired 共享 r。Detail/Context 分别 clamp 到自己的 mip 数。新增 `ScatteringQuery.filter_random` 缺省为现有 0.5；producer 显式传新 Metal 的值。
4. 每张 latent plane 按 `p=u·(W_l,H_l)−0.5` 取四邻居；repeat 使用 remainder，clamp 按实际地址规则。四个 texel 分别量化到 SNORM8 后再插值，保留插值权重对 query 的变化；不会用 `Q(E(patch(u)))` 代替此步骤。
5. 保留 prepare 的输入宽 24：program8 + latent8 + query8。新的 query8 为 `wo3 + vec(J)/(1+||J||F) + frac(lambda)`。这个全组固定、可逆的有限 J 映射保留 point/filtered 区别及 footprint 方向；不依赖纹理内容统计，也不把不同绝对 raw roughness 拉到同一尺度。
6. 新 profile 取消 asset-ID scale/bias；原生 program 的 spatial 条件保留。训练与 cook 的权重、program FP16、每 texel SNORM8、prepared FP16 pack/unpack 分别有开关 witness；新正式比较的 loss 路径消费最终部署等价的 fake-quantized 状态。

point-only D0 使用零 derivatives 与一个 footprint sample。filtered recipe 使用已有 dispatcher 的完整 response 空间平均，初始诊断配置取 16 点，D0 另以 64 点观察积分差异；`evaluation_samples=1`，避免把确定性 MDL 的重复方向求值当空间积分。16 是 diagnostic 工作量选择，不能宣称已得到所有 source 的收敛 GT。

在 fractional LOD 上，随机选 mip 后再做非线性 evaluate 的期望不必等于目标空间积分。D0 分别验证读法相同和 target 积分，D1 报告估计器的偏差/方差；两次读取的近似能力不写成数学等价。新对照统一使用上述 query features，不能把 encoder 改善与更换 footprint 条件混为一个因素。

### 8.4 Method、checkpoint 与新资产编译

- `model.py` 的新 context 使用 `metal_spatial_hybrid_v1`，删除 `asset_variant_count`；原始 encoder 是默认资产路径。对照使用明确的 `metal_spatial_summary_control_v1`，其量化、read-plan、下游网络、asset-ID 处理与主路径相同。它是旧表示的修正对照，不能叫原 v3 的重现。
- `method.py` 更新 required tensors/resources、组件、参数组、loss、phase 配方和状态 schema。`asset_encoder` 包含分组 stem/trunk/mip/head；没有空的 `asset_variant` 组。proposal 使用独立参数组；只有 descriptor 中启用的 phase 才参与 optimizer。
- objective 先编码共享 tile，再为主/paired query 分别 gather；QAT 经过 FP16 program、权重、prepared state 和 SNORM texel 的真实读法。filter random、tile schedule 和模型配置随当前 checkpoint 恢复；没有每次 resume 重新估计的输入统计。
- `compile_program` 只打包 shader 消费的 prepare、evaluator、proposal 权重；typed compiler 和原始 texture encoder 在离线 compile/cook 中运行，保留在训练 checkpoint。同步更新 runtime 参数清单与真实 bytes，不能把未调用的 compiler 网络当 runtime 工作量。
- C6：`_deployment/compile_asset` 用传入的 source snapshot 及其资源声明生成原生 program 和 latent；不要求 source 出现在训练列表，也不把旧 52 个资产的枚举当作模型容量。校验已支持的 graph/schema/语义/资源，拒绝真实不兼容。新增资源元数据不新增 `nn.Embedding` 行、不创建 optimizer，不做隐藏 refinement。
- 新 tensor/ABI 使用新 identity，`metal_budgeted_layout_v2.json` 作为当前 layout 源，生成对应 Slang 常量。旧 checkpoint 不按新形状尝试恢复；不添加历史格式转换器。现有引用新 Metal 默认入口的配置/测试更新到新合同，历史 v3–v6 recipe 保留历史身份，不能静默成为同名的新实验。
- 包仍是 `ScatteringPackage@2`，program/asset/instance、sampler usage 与普通初始化导出规则不变；不增加 formal/quality readiness 门。

### 8.5 C1–C4 的确定修复

**C1**：保留 initial decoded state 为 buffer，以零初始化 delta 优化 condition 和 spatial 字段，`value=initial+R_delta·tanh(delta)`；R_delta 在 control 配置中固定。lobe 有界字段按其合法域投影，合法 initial 不被再次压缩；angle 保留 initial 的数值约定，不为 canonical wrap 改变第 0 步。proposal 在原始 prior 上做归一化的乘性 delta。检查所有初始字段及输出在浮点容差内一致，确定性 access/frame/resource 不参与优化；超合法域的输入显式失败，不用 atanh/clamp 偷换 initial。

**C2**：Slang softplus 使用 `max(x,0)+log1p(exp(-abs(x)))` 的稳定计算。为避免依赖未知的 `log1p` intrinsic，首个实现固定令 `y=exp(-abs(x))`、`t=y/(2+y)`，用 8 项奇次级数 `2·Σ[k=0..7] t^(2k+1)/(2k+1)` 计算 `log1p(y)`。`0≤t≤1/3`，截断误差至多 `2t^17/(17(1−t²))<1.1e−9`；舍入和 exp intrinsic 误差由独立 float64 oracle/GPU witness 再界定。保留可表示的负尾响应；underflow/FTZ 区域另报，不能用大 atol 掩盖正常数的小值。

**C3**：内部 evaluator 返回 `{f, valid}`，外层 evaluate/sample 保留失败标志；真正有限的零反射可有效，半球外、退化 half-vector、非有限内部结果保持无效。受控注入仅放测试 kernel，普通 renderer 得到零占位与 invalid，不能把 NaN 清成“有效零”。

**C4**：对单位 `n` 的正 Z 有效域令 `a=1/(1+n.z)`，`t=(1−n.x²a,−n.x n.y a,−n.x)`，`b=(−n.x n.y a,1−n.y²a,−n.y)`。该基在有效域内连续，`t×b=n`；n=(0,0,1) 对应原生局部 X/Y。D0a 先确认 authored rotation 的轴约定，再按相同旋转公式处理。local slope 和 half-vector 各自验证输入有效域，无效输入先标记，不能把此公式泛化为无接缝的全球球面参数化。

### 8.6 C5、状态布局与成本

新增 proposal parameter head `16→16→13`，输入为 program condition8 与读取后的 latent8，完全不含 wo 或 view-conditioned semantic。13 个输出分别为两 frame 的 slope4、angle2、roughness4、mixture logits3；使用当前有界 modulation、program prior 和 2% uniform fallback。实际反射轴按输入方向和 proposal frame 构造，所以这只是 **参数独立于视角**，不意味着最终 `q(wi|wo)` 不依赖 wo。

evaluator 继续使用自身 view-conditioned frames；sampler/pdf 使用新的 proposal frames。将其 compact slope4/cos2/sin2 作为 8 half 存入 prepared state。`pdf.reverse` 交换方向后使用同一 view-independent 参数，必须等于独立 `prepare(wi)` 所得 proposal 的 forward density；latent 的位置/footprint/filter random 在该 witness 中保持不变。`sample` 仍只调用一次 evaluator，`pdf` 不重读纹理、不运行 semantic decoder。

| 成本项 | 首个新 profile 的结构推算 | 对照口径 |
|---|---|---|
| Detail/Context | 两个 RGBA8 SNORM plane，Context base 每轴为 Detail 的 1/4 | 与原 v3 的分辨率/完整 mip bytes 对齐；非方/尾 mip 按实际数组计数 |
| prepare dense MAC | `2560 + 16×16 + 16×13 = 3024` | 历史真实值为 2560+72=2632；proposal 正确性修改增加 392 |
| evaluate dense MAC | 11,392 | 网络仍为 44→64→64→64→6，解析/非线性 ALU 另列 |
| packed prepared state | 80 half + 4 uint32 = 176 B，44 words | 比旧 160 B 增加 16 B；不称完全同 state 成本 |
| runtime asset reads | prepare 两次；evaluate/pdf 零次 | 保留当前 profile 的硬件可部署边界 |

新 layout 的顺序为 semantic24、query8、evaluator compact frame8、lobes16、proposal12、access4、proposal compact frame8、flags4。half 字段占 160 B，flags 从 byte160 开始，总计176 B。prepare 有三层 semantic dense 与两层 proposal dense，对应 `maximum_prepare_steps=5`；不能继续沿用旧声明 4。同步 Python pack/unpack、Slang struct/pack、生成器、descriptor、program capability 和真实 static accounting。仍在既有候选的 192 B / 20k evaluate MAC 界内；实际寄存器、临时数组与耗时不从这些推算中推出。

### 8.7 correction 和辅助诊断的实施边界

首轮 D1 保持现有 hybrid core 与 positive 分支，只修输入/读取和 C1–C6。D2 分别暴露 `analytic_f`、gated core、correction、final，并报告 MAE/log error、分支 RMS、符号及干预差值；不再把 `mean(p²)` 写成 `p` 幅度。

需要 signed 对照时，将现有方向 MLP 的最后六维拆成两个等价的 `64→3` head。固定 E、prepare、方向 hidden trunk、core 和 gate head，只更新 correction head；gate 输出因此确实不变。此处 positive control 用 `R·tanh(max(h,0))`，signed 用 `R·tanh(h)`，两者 head 均初始化 weight=0、bias=0.01，初值输出相同且有非零梯度；该 bounded positive control 有独立 identity，不冒充旧 softplus 分支重现。R 取 train-only calibration 中 `|y−b|` 每通道 p99 的两倍并冻结，零通道按数值 epsilon 处理；范围外 target 和最终 clamp 比例单独报告，不改 GT。

这项最初只回答“固定已学特征上的 signed readout 是否更有用”。放开共享 hidden 层会改变 gate，另加 residual trunk 会增加成本，均不自动纳入首轮。D1b 的 bounded refinement/free-latent control 和辅助 response-signature head 保留为按结果选择的诊断分支；不要求它们成为新资产编码或普通导出的必经步骤。

## 9. 实施中的 UV 分组修订

**trigger**：用户明确澄清「应该只融合相同uv的纹理，uv不同的纹理分开处理」。源码同时证明三个 diagnostic source 含 nonrepeat 多位置读取、青铜 slot 独立坐标和非空间 lookup。**invalidated evidence**：旧 sampler/cook 将这些资源放在共同 surface UV；旧空间误差不能只归因于 encoder/loss。**scope impact**：取消单个共同 latent 网格与全资产两次读取假设；原始像素与语义 stem、后半段 decoder、原生参数编辑仍保留。**rerun required**：新模型必须 fresh；旧 matched 结果不作为修正后数值。

- UV 组的 identity 包括坐标来源、相对 mapping、address/filter 与 tiling 表达式；相同默认参数值不代表不同可编辑表达式永久兼容。
- 每个 UV 组分别调用共享语义 stem、融合/trunk/mip/head，输出本组的两张 latent plane。原生分辨率不同但 mapping 相同，可以在 stem 之后对齐；不同 mapping 的组不在 cook 时被强行重投影为一张图。
- prepare 按每组自己的坐标读取，再把所有组的 latent、presence 和 footprint 条件交给共同 decoder。原生 nonrepeat 的三个位置共享同一组编码的纹理，lookup 坐标和权重保持原生定义；不能把 `infinite_tiling` 解释为 `frac(uv)`。
- BSDF 三维 lookup 保留原生 slice 结构，颜色 lookup 保留自身表域，通过原始资源 encoder 提供离线条件。它们不决定 surface UV 网格大小，也不因夹取地址与表面 wrap 不同而禁止 cook。
- 保持原生 UV 与坐标编辑，不采用有限 UV 区域预烘焙方案。不同组单独寻址；改变只影响坐标的参数可更新对应 program binding，改变组兼容关系或资源时按显式编译生命周期处理。
- 读取上限由已支持的最多 9 个原生 slot 给出：最多 9 个 UV 组，每组两个 latent plane，原生三位置 tiling 最多 6 reads/组，因此保守上限为 54 reads/prepare。实际 program 编译时只保留存在的组/lookup，并登记准确次数。该值是形态静态上限，不能宣称达到实时性能；没有增加诊断 optimizer steps、seed 或 source 数。
- decoder 保留 32→32→24 后半结构；每组输入 latent8、Jacobian4、fracLOD1、presence1，最多 9×14；加 program8 与 wo3，固定输入 137。proposal 只看 program8 与各组 latent72，采用 80→16→13，不依赖 view。prepare dense MAC 为 137×32+32×32+32×24+80×16+16×13=7,664；evaluate 仍为 11,392。176 B prepared state 保留 query8 与独立 proposal frame，不存全部 decoder 输入。
- 该修订按用户指示修复输入合同；超出旧研究软成本线如实记录。若实际资源/吞吐不可行，按预先规定分类停止，不偷偷合并不同 UV 或减小原始感受野。
