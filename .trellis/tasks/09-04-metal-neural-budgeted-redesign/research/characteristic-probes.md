# Metal budgeted 特征材质专项 probe

## 它是什么

本协议落实用户授权的“主交付完成后，先对有特点的材质专项研究，再决定是否扩到普适性检查”。它只使用已经入选的`metal_budgeted_hybrid_v3`，不新增模型、loss、seed或部署预算，也不从Tungsten checkpoint恢复。

## 为什么选择这四项

四项均来自锁定的`metal-opaque-v1.json`，各取一个默认原生参数state；选择依据是机制互补，不是根据训练结果筛选：

| probe | 原生材质 | 要隔离的机制 |
|---|---|---|
| base | `Brass_Sheet_Yellow_Splotchy_Streaks` | 无涂层的有色金属基线，同时含sheet纹理和污渍 |
| high-frequency | `Bronze_Scratched_Yellow_Heavy_Worn_Scratches` | 深划痕、磨损及强空间高频 |
| chromatic-distribution | `Aluminum_Anodized_Pink` | 有色阳极氧化层、7个texture slot及Beckmann例外 |
| composite | `Steel_Painted_Light_Blue_Cracked_Matte` | 漆层、裂纹、粗糙金属基底的复合响应 |

四项分别使用versioned data fragment和run config；source locator、registry、seed、在线方向/footprint/paired-UV recipe与Tungsten主实验相同口径。

## 冻结执行

- 用户在专项probe尚未执行前明确要求继续检查更大`batch_size`，因此原先固定的每rank 512只保留为候选基线，不再是专项probe最终几何。这不改变已经运行的Tungsten matched pair。
- 在主pair和package交付完成后，先对同一Tungsten/hybrid结构执行每rank`1024/2048`的fresh 96-step吞吐probe，和主run最初96 step的512基线比较；三者都不经过step128 validation。候选必须五卡完成、无OOM/DDP错误、finite且低于8 GiB/rank residency预算，再按steady training row的median global work units/s选最高者。该选择只决定执行几何，不是质量结论。
- GPU固定物理GPU 5–9、DDP world size 5；入选batch在任何特征材质启动前一次性写入全部四个run，之后不再改变。四项必须使用同一batch，并同时报告per-rank/global batch与累计work units；它们不能按step与batch512的Tungsten主实验合并比较。
- 每项fresh初始化，单seed `2026090401`，在step 256停止；执行step 128和256两次冻结validation，不延长、不换seed。
- 每项累计work units按`256×(5×入选per-rank batch)×2 routes`从resolved plan计算。该数值是执行账本，不是质量门。
- 每项记录appearance、RGB、chroma、peak、spatial gradient、semantic-runtime、analytic/gate/positive分工、proposal和所有parameter group梯度/更新覆盖。
- 只有四项都通过实现正确性与有限性检查，才汇总共同结构趋势；256-step observed quality不称为泛化结论，也不与2048-step Tungsten绝对值直接排序。

## 大 batch 选择结果

实现 `1d5f813` 下的五卡 fresh 96-step probe 已完成；统计排除首个16-step warm-up row，使用step 32–96的五个steady training row。per-rank batch `512/1024/2048` 的median global work units/s分别为`21,135/42,364/87,677`，median step wall分别为`0.2351/0.2389/0.1995 s`，Torch peak分别为`747.65/965.80/1,399.98 MiB/rank`。三者均无OOM、DDP错误或非有限值，均低于8 GiB门槛。

按预登记规则选择per-rank batch `2048`、global batch `10,240`，四个特征材质run统一写入`batch_size_multiplier: 4`。每项256 optimizer step对应`256×10,240×2=5,242,880`个evaluator/sampler route work units；该几何选择只表示执行效率，不是质量排序。

## 四项 observed result

四项均以五卡完成step256并写出`complete=true` checkpoint/review，没有OOM、DDP错误、非有限metric或required parameter group失效。steady median global work units/s为黄铜`37.4k`、划痕青铜`37.0k`、阳极氧化铝`30.3k`、开裂涂漆钢`30.7k`；Torch peak分别约`1.74/1.74/1.82/3.30 GiB/rank`。较重材质的时间主要消耗在GPU-resident source reference生产与固定256-batch validation，不是模型显存。

同一固定validation recipe下，step256减step128的256条同序row paired bootstrap结果如下；区间为20,000次bootstrap的95% CI，负值表示改善：

| 材质 | appearance Δ | peak Δ | spatial Δ | 解释 |
|---|---:|---:|---:|---|
| 黄铜板 | `-0.13639 [-0.13699,-0.13578]` | `-0.12940 [-0.13046,-0.12832]` | `-0.001943 [-0.002053,-0.001835]` | 平滑金属的均值、色度与peak快速改善；spatial只有小幅下降 |
| 划痕青铜 | `-0.01865 [-0.02235,-0.01506]` | `+0.01408 [+0.00762,+0.02062]` | `-0.001116 [-0.002202,-0.000013]` | 总分几乎平台且peak显著退化；强纹理细节未被恢复 |
| 阳极氧化铝 | `-0.12996 [-0.13261,-0.12745]` | `-0.09630 [-0.10021,-0.09234]` | `-0.000148 [-0.000227,-0.000067]` | 平滑材质均值响应明显改善，原始spatial量级很小 |
| 开裂涂漆钢 | `-0.09274 [-0.09552,-0.08990]` | `-0.03340 [-0.03937,-0.02746]` | `-0.000166 [-0.000961,+0.000626]` | 均值/色度/peak改善，但spatial CI跨零，裂纹响应没有可确认进展 |

内部路径也呈一致模式：step256的positive RGB trace在黄铜/铝仅约`7.0e-6/5.8e-6`，主要依赖analytic gate；开裂钢虽升到`3.83e-3`，spatial仍无显著改善；划痕青铜positive trace仅`1.66e-4`且peak退化。因此本轮可以把Tungsten观察扩展为两条有范围的结论：当前hybrid适合快速拟合平滑金属的平均响应；对划痕、裂纹等高频纹理，Detail/Context经过slot聚合和4通道瓶颈后仍不能保留足够局部信号，增加step或仅调spatial loss不是优先方向。

下一小实验只诊断结构敏感度：在已训练的划痕青铜与开裂钢checkpoint上分别放大asset encoder的high-pass输入和semantic decoder的Detail输入，不更新权重，测量raw patch→Detail→semantic→final response的信号传递。它是机制probe，不作为新候选或质量对照；若单纯增益不能实质降低spatial error，下一正式候选应采用role-separated slot聚合和显式局部高频通道，而不是继续调gain。

## high-pass/Detail增益机制probe

在step256 checkpoint上固定四个online validation batch，不更新权重，分别执行baseline、asset high-pass输入×8、Detail→semantic权重×8和两者同时×8。划痕青铜baseline的target/predicted log-gradient绝对值为`0.50373/0.00354`，spatial error为`0.50414`；high-pass×8把预测幅度升到`0.02246`，但error反而升到`0.50735`，组合×8时为`0.02487/0.50810`。开裂钢baseline为`0.25800/0.00122/0.25804`；high-pass×8为`0.00585/0.25866`，组合×8为`0.00604/0.25874`。单独Detail→semantic×8在两项上几乎不增加final gradient。

因此瓶颈不是单纯的数值增益。现有共享slot softmax先让所有语义角色竞争同一权重，再把每个slot的四维输出混入同一个Detail RGBA；放大后虽然有更多局部变化，方向却未与target对齐。后续pilot必须改变信息路由，使Detail四通道各自对应`color/normal/scalar/packed`角色；Context仍保留共享低频聚合。这个变化不增加纹理读取、PreparedState或evaluator MAC，也不改变source/query/loss。

## role-separated Detail matched pilot

实现`a5e4de7`新增`metal_budgeted_hybrid_role_detail_v4`：它只把Detail改为四个角色各自在同角色slot内softmax，并分别写入一个RGBA通道；v3 shared聚合control路径保持原样。两项材质都在物理GPU 5–9上fresh运行v3/v4到step256，per-rank batch为`2048`、global batch为`10,240`，每组累计`5,242,880`个route work units。四组均完整写出checkpoint/review，无OOM、DDP错误、非有限metric或显存回归；bronze约3分25秒/组、steel约4分7秒/组，峰值分别约`1.66/3.26 GiB/rank`。

共同step256的256条同序validation row做20,000次paired bootstrap，candidate-minus-control结果如下；负值表示v4改善：

| 材质 | appearance Δ | peak Δ | spatial Δ | 判断 |
|---|---:|---:|---:|---|
| 划痕青铜 | `+0.000237 [-0.000090,+0.000557]` | `+0.004411 [+0.003617,+0.005190]` | `-0.000140 [-0.000179,-0.000101]` | aggregate无可确认变化，spatial改善极小且peak显著退化 |
| 开裂涂漆钢 | `+0.000742 [+0.000568,+0.000918]` | `-0.001371 [-0.001781,-0.000955]` | `+0.0000546 [+0.0000410,+0.0000684]` | aggregate与spatial均显著退化，只改善chroma/peak |

角色分离能改变误差分配，却没有形成跨材质净收益；因此v4停在diagnostic candidate，不替换可交付v3，也不继续增加step。结合增益probe，下一结构问题已经缩小为：四通道Detail本身能否从局部patch提取与paired target梯度方向一致的充分统计量。后续小诊断应先测raw patch feature到paired reference差分的可预测性，再决定是重做离线局部编码器、显式保留导数/方向通道，还是需要增加有界asset带宽；不再优先调整softmax、loss权重或主干宽度。

## fixed-batch容量与中心采样诊断

fresh v3 checkpoint各加载一个固定online batch，只优化现有非sampler参数128步。划痕青铜的target/predicted log-gradient为`0.50342/0.00309`，spatial error从`0.50398`变为`0.50463`，预测梯度只到`0.00806`；开裂钢为`0.25905/0.00112`，error仅从`0.25912`降到`0.25857`，预测梯度为`0.00219`。asset encoder、typed compiler、semantic prepare和directional evaluator梯度均finite/nonzero，因此不是断梯度；当前局部表示连单batch都难以承载目标变化。

检查patch几何发现，偶数尺寸8×8 patch以索引4对应请求texel，但v3 Detail的`center`使用索引3/4的2×2均值。四个固定batch中，若只观察真实中心texel，相邻UV的raw变化在青铜从`0.01198`增至`0.01948`，在钢从`0.00445`增至`0.00631`，分别多出约63%和42%。这说明高频信号在learned encoder之前已被额外平滑，足以支持一个不增加运行时成本的center-texel matched pilot；它仍需实验确认，不能由输入幅度直接推断最终质量。

## center-texel Detail matched pilot

实现`55640e0`只把Detail encoder的局部中心从索引3/4的2×2均值换成请求texel索引4；v3 control、其他网络和运行时预算不变。两材质各fresh运行v3/v5到step256，四组均完成且复现control轨迹。candidate-minus-control paired bootstrap结果如下：

| 材质 | appearance Δ | peak Δ | spatial Δ | 判断 |
|---|---:|---:|---:|---|
| 划痕青铜 | `-0.000756 [-0.001004,-0.000515]` | `+0.004659 [+0.004059,+0.005223]` | `+0.000136 [+0.000093,+0.000180]` | 平均log/linear改善，但peak与spatial显著退化 |
| 开裂涂漆钢 | `+0.000172 [+0.000048,+0.000295]` | `+0.001209 [+0.000928,+0.001491]` | `+0.0000620 [+0.0000489,+0.0000751]` | aggregate、peak与spatial均显著退化 |

v5停止在diagnostic candidate，不延长、不替换v3。v4和v5共同说明：消除角色竞争或增加单点变化幅度都只能重新分配误差，不能恢复正确空间方向。当前encoder还把8×8 patch压成center/high-pass/context三个无方向均值；而青铜活跃变化以normal/packed为主，钢同时有color/normal/packed。后续优先项因此是两个高分辨率RGBA plane共同承载带signed x/y局部导数的8维特征，并显式报告相对当前“一张全分辨率Detail + 一张1/4线性分辨率Context”的asset bytes增量；它不是本轮已经验证的结论。

## dual-local signed-derivative matched pilot

实现`560e2ca`新增`metal_budgeted_hybrid_dual_local_v6`：两张RGBA plane均由请求texel、signed x/y中心差分与role embedding形成20维per-slot输入，第二张plane也使用全分辨率。它仍固定两次RGBA8读取、160 B PreparedState和11,392 evaluate MAC；如果部署，两张全分辨率mip相对v3的一张全分辨率加一张1/4线性分辨率预计需要约`1.882×`asset bytes。

首次运行在step128 validation后、checkpoint写出前发现profile与公共checkpoint tensor schema不一致。DDP reducer、跨rank异常传播和清理均正确；失败属于profile注册实现缺陷，不是NCCL问题，旧artifact已标记无效且没有用于质量解释。修复`ba024f0`将输入保持为20维并在config resolve时前置验证所选profile的全部checkpoint tensor shape，随后四组均以新identity fresh运行到step256。

共同step256的candidate-minus-control paired bootstrap如下；负值表示v6改善：

| 材质 | appearance Δ | peak Δ | spatial Δ | 判断 |
|---|---:|---:|---:|---|
| 划痕青铜 | `-0.003751 [-0.004244,-0.003264]` | `-0.008728 [-0.009912,-0.007545]` | `+0.0000339 [-0.0000088,+0.0000763]` | 平均响应与peak改善，但spatial无可确认变化 |
| 开裂涂漆钢 | `-0.000485 [-0.000679,-0.000291]` | `+0.003180 [+0.002738,+0.003640]` | `+0.000130 [+0.000116,+0.000145]` | 平均响应改善，但peak与spatial显著退化 |

v6的额外局部导数与asset带宽只能改善部分平均响应，没有形成跨材质空间收益。固定batch的128步spatial-only容量复核进一步显示：青铜target gradient为`0.503423`，预测只从`0.005933`到`0.007018`，error从`0.503352`到`0.503186`；钢target为`0.259045`，预测从`0.001920`到`0.004989`，error从`0.259046`到`0.257854`。这排除了“v6只需更多step或更高spatial loss权重”的优先解释。v6停止、不创建v7、不替换v3；下一轮必须改变source语义到runtime latent的映射，而不是继续围绕同一个patch-summary encoder调幅度或导数。

## mixed cohort 普适性检查

青铜与钢的fresh对照、机制probe和fixed-batch容量复核均指向同一局部语义带宽瓶颈，因此满足预登记的普适性检查触发条件。仓库已有冻结的`mdl-metal-budgeted-full`数据fragment：692个registry source、每export四个typed state、按`metal-core/finish-microstructure/aging-contamination/coating-composite`四类责任分组调度。本轮直接复用它和v3 hybrid recipe，单seed、per-rank batch`2048`、global batch`10,240`，最多512 step；run identity为`metal-budgeted-hybrid-mixed-cohort-probe`。它只检查专项结论能否扩到更广source，不是formal 692-source质量结论，也不与单材质绝对metric直接排序。

首次有效DDP启动暴露了两个公共实现缺陷：多source下五个rank各自估计appearance calibration，导致初始化descriptor分叉；MDL content cache只有atomic publish而没有per-key跨进程互斥，冷cache时五rank重复编译并遗留3,317个loser partial目录。修复后16,384个全局calibration样本按`3277/3277/3277/3277/3276`分片、按rank合并，再由所有rank初始化同一identity；cache在锁内复查并清理loser/异常partial。`mixed-cohort-v2`随后完成512 step，但它又揭示`group-block-balanced@1`每个256-batch validation window只反复使用每rank一个group、不同milestone再换group，因而其DDP/吞吐证据有效，质量轨迹失效。

commit`400a984`增加版本化`group-block-balanced@2`：training仍按64 optimizer step成块，validation改用window-local batch index，每64 batch换组。这样DDP5的一个256-batch window稳定覆盖20个group，所有milestone重放同一group cohort；旧`@1`语义保留供已有checkpoint复现。最终fresh artifact为`artifacts/metal-budgeted-probes/mixed-cohort-v3/`，step512 checkpoint SHA-256为`61a2b4aae12fa3a1bdd222f51139b6f5092ec66568ef496fbde01b672692b656`，无DDP错误、非有限值或cache partial。

同一20-group分布上，step128→512的appearance从`1.77925`降到`1.69251`，log/linear/chroma分别变化`-0.07866/-0.19901/-0.00130`；但peak退化`+0.13626`，spatial只改善`-0.00217`。四个单GPU exact-query group block的matched复评给出log/chroma/peak改善`-0.04966/-0.02299/-0.21679`，linear和spatial却退化`+0.02608/+0.000104`；四组平均target log gradient为`0.14743`，预测从`0.00313`变为`0.00300`，始终只有目标约2%。20-group与4-group的部分指标方向不同，说明有限512-step训练没有形成跨group一致收益；两种口径却共同确认空间响应仍近似常数。结论是局部语义瓶颈可扩到mixed source观察，不能扩成692-source泛化质量声明。

## 预先解释规则

- high-frequency与composite的spatial仍接近target自身梯度尺度、而其他appearance下降：支持“Detail高频通路是跨材质瓶颈”，下一轮优先显式role-separated Detail/Context，而不是增加主干宽度或仅加spatial loss权重。
- anodized的chroma/peak显著落后base，同时analytic占主导：支持增加closure/distribution-aware analytic basis或让neural residual承担有色涂层校正；不据此把MDL改写为层GT。
- 四项行为差异明显：保留机制特定结论，下一轮做小型混合cohort，不宣称一个局部修复普适。
- 四项出现一致的finite/梯度/身份失败：先分类实现或protocol缺陷，不解释模型质量。

四项probe与增益诊断固定写入`artifacts/metal-budgeted-probes/characteristic-v1/`；role-separated、center-texel与dual-local matched pilot分别写入`role-detail-v1/`、`center-detail-v1/`与`dual-local-v2/`；mixed cohort的最终有效证据写入`mixed-cohort-v3/`，v1/v2分别只保留失败和协议诊断。stdout、metrics、checkpoint、review与dmon不进入根仓库。本文件只追加实际结果和跨材质解释，不反写选择规则。
