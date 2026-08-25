# 实验框架：数据采集、拟合协议与检验流程

本文是当前研究的权威流程框架。它回答四个问题：

1. 什么样的材质用什么样的采样密度，数据要多大才够；
2. 「泛化」对每个材质族分别指什么，怎么考核；
3. 候选方法怎么「训练」（梯度训练、直接拟合或两者混合），预算怎么统一；
4. 怎么测试、怎么比较、怎么记录，使不同方法能在同一个稳定框架下反复迭代。

模型候选本身的设计（结构、公式、容量档位、动机）见 [`model_candidates.md`](model_candidates.md)。HDF5 字段与采集命令的稳定合同见 [`../data.md`](../data.md) 与 [`../contracts/`](../contracts/)。

## 0. 框架定位

- **基准优先**：先冻结「数据 + 评测协议」的 v1 基准，方法迭代只在 Python 评测框架内进行。MethodBundle/Slang/viewer 是独立的部署轨道，每个研究阶段收尾时对当期最优候选执行一次，不阻塞日常实验。`evaluate()` 返回线性 `f`、`prepare/evaluate` 划分、单 query 随机访问这些运行时合同保留为设计期静态约束，每个候选注册时检查。
- **容量分档**：每个候选按 S/M/L 三档容量测「容量–质量曲线」。`C_eval`、`B_shared`、`B_asset` 和实测时间完整记录，用于 Pareto；部署压缩是后置的独立阶段。
- **足量语料**：正式语料按第 2 节密度表生成，覆盖未见方向、未见参数状态和未见结构 family；长尾分位数只从满足 state 数量要求的冻结 test 集计算。

术语规范：禁止用「上界/下界/ceiling/floor」描述有限实验的容量结论。统一句式为「配置 C 在数据 D、预算 B 下达到 X」。「运行成本静态有界」这一工程含义（固定循环数、固定访存数）不受影响。

## 1. 材质难度分级

采样密度由响应的**方向频率**决定，而方向频率主要由最窄 lobe 的角宽度决定。定义五个可叠加的难度标记：

| 标记 | 判据 | 典型例子 |
|---|---|---|
| **W** 宽响应 | 最窄峰半宽 ≥ 10°（roughness ≳ 0.2，或纯 diffuse/sheen） | sheen slab、粗糙 dielectric、MERL 漫反射类 |
| **G** 光泽 | 峰半宽 2°–10°（roughness 约 0.05–0.2） | 中等粗糙 conductor、brushed metal |
| **S** 窄峰 | 峰半宽 < 2°（roughness < 0.05、clearcoat、镜面导体） | 极窄各向异性 conductor |
| **T** 透射 | 存在透射事件；需要完整球面与临界角覆盖 | glass、soapbubble、薄介质 |
| **M** 移动/多峰 | 峰位显著偏离几何镜面方向，或存在多个同时峰 | 多界面 LayerStack（实测偏移 4–11°）、薄膜 |

分级来源：

- **参数式族**（LayerStack、OpenPBR 常量参数资产）：从源参数直接推出。规则示例：`min(alpha) < 0.05 → S`；对外部查询域存在透射事件 → `T`；界面数 ≥ 2 → `M`。当前 LayerStack v1 以不透明 base 收尾，因此只使用 W/G/S/M；T 配置供具有外部透射的 source state 使用。
- **测量/资产式族**（MERL、MaterialX）：源参数不可用或不可信，用低成本 `difficulty-probe-v1` 实测：取 4 个代表性 `wo`，每个采 4,096 个均匀 `wi`，按 top-1% 积分能量占比与峰半宽估计分级。probe 成本远低于正式采集，分级结果随资产 manifest 版本化。

采样密度取所有命中标记的最大要求。

### 为什么必须 peak-aware，而不是加大均匀采样

半宽 σ（弧度）的峰在半球上占立体角约 `π σ²`，均匀采样命中概率约 `σ²/2`。要在峰内得到 ~20 个样本，纯均匀需要约 `40/σ²` 条：σ=2° 时约 1.6 万条尚可接受，σ=0.1° 时约 650 万条完全不可行。所以密度表的原则是：**均匀分量只负责能量背景与宽结构，窄峰覆盖全部来自以实际峰位为中心的多尺度 vMF proposal**。当前统一实现名为 `peak-aware-v1`；LayerStack 的 M state 先用 4,096 方向 reference probe 实测逐 `wo` 峰位，再生成三尺度 vMF，不再使用固定峰位 patch。

## 2. 采样密度表（v1 基准）

### 2.1 每个 state 的 train 查询

| 分级 | train `wo` 数 | 每 `wo` 的 `wi` 数 | proposal 构成 |
|---|---:|---:|---|
| W | 48 | 512 | 60% uniform/cosine、25% 宽 vMF、15% grazing |
| G | 64 | 1,024 | 40% uniform、40% 三尺度 vMF、20% grazing |
| S | 96 | 2,048 | 30% uniform、50% 三尺度 vMF（最窄档 κ 对齐实测峰宽）、20% grazing |
| +T | 追加 32 个临界区相关的 canonical `wo.z>0` | ×1.5 | 追加 25% 透射峰 + 10% 临界角带分量 |
| +M | 与 S 相同 | 与 S 相同 | vMF 中心改为 4,096 方向 reference probe 得到的实测峰位，每峰三尺度 |

`wo` 分布：按 `cos θo` 分层，外加固定掠射带（`θo ∈ [75°, 89°]` 至少占 20%）。每条 query 照旧落盘 mixture 的解析 PDF 与 `1/(N·p)` 积分权重，训练分布与均匀立体角指标因此可以分开计算。

### 2.2 validation / test / adversarial 查询

- validation：16 个独立 `wo`，每个 256 个 uniform `wi`（独立 seed，方向表不与任何其他 role 重合）。
- test：24 个独立 `wo`，每个 512 个 uniform `wi`；含透射的 state 用完整球面。
- adversarial：延续现行机制——peak、grazing、临界透射定向 probe，只做冻结后评测，不参与选优。
- **稠密切片**（诊断专用，不参与训练与排名）：每个 state 选 4 个代表性 `wo`（法向、中角、掠射、峰最劣），每个采 8,192–16,384 个均匀 `wi`。用途：峰形可视化、峰位/峰高指标、离散表记忆 vs 连续插值的检查。

### 2.3 reference 噪声预算（沿用已校准机制）

`adaptive-v1` 与 `peak-aware-v1` 固定以下合同。预算按用途分层，避免诊断量反过来支配 GT 采集成本：

- validation/test 主 response：目标 relative SE p95 0.04、最终 group 上限 0.10、基础 cap 262,144；有实测失败证据的 state 可在 CorpusPlan 中按 role 显式晋升样本 cap 与最终 group 门，其他 state 不受影响。达到对应 state-specific cap 仍不合格时 shard 失败，不写入语料。
- train 主 response：目标 0.06、最终 group 上限 0.25、cap 262,144；训练含大量 peak-aware 方向，噪声无偏且 SE 逐 query 保存，不把每个训练 group 强压到排名 GT 的上限。
- adversarial/dense 主 response：目标 0.08、报告参考线 0.50、cap 262,144；达到 cap 后无条件落盘 SE，不作为 shard 拒绝门。它们只做冻结后的结构诊断，不参与训练、checkpoint 选择或 test 主排名。
- reciprocal 只进入 source-aware scorecard，不进入训练或主排名：train 为目标 0.50、报告参考线 0.999、cap 4,096，只保留低成本诊断；validation/test/adversarial/dense 均为目标 0.20、报告参考线 0.999、cap 65,536。reciprocal 达到 cap 后无条件落盘 variance/sample count，不作为 shard 拒绝门；validation/test 仍提供阶段 scorecard，但报告必须同时读取 reciprocal SE，高噪声项不得支撑模型质量结论。
- 单 dispatch query 数 ≤ 4,096；方向更多时由 provider 自动切 tile，限制随生成配置记录。
- 峰覆盖审计从实际方向、PDF 和 response 重算。集中 query（top-1% 积分能量占比 ≥ 0.1）的 peak spacing p95 必须 ≤ 2°；不满足的 state 从 8,192 晋升到 16,384 方向并独立成 shard。
- 原始与 reciprocal response 都保存样本数和方差；replica 差异与 SE 保留为诊断证据。

### 2.4 语料规模与生成策略

| 族 | 状态/资产规模 | 估算 |
|---|---|---|
| LayerStack | 24–32 个结构 family（层数 1–4 × 界面类型 × 有无介质 × 各向异性），每个 family 用 LHS 采 8–12 个连续参数状态，共 **250–350 states** | 原始与 reciprocal response 合计约 **6–9 GB**，以首轮实测 manifest 为准 |
| MERL | 全部 100 张测量表，按分级采样（多数为 G/W 级） | 约 0.6–1 GB |
| OpenPBR | 83 个资产；常量参数资产按参数式处理，纹理绑定资产按资产式处理 | 约 1–1.5 GB |
| MaterialX | 8 个资产，空间查询按现行 `footprint-scale-rotation-seam` 合同；密度升级推迟到 spatial 阶段 | 维持现有规模 |

每条 direction 未压缩约 110–125 B（`wi`、双 replica moments、PDF、权重、flags 与 reciprocal moments）；HDF5 gzip 后以实际 manifest/file size 为准。生成策略：

- 按结构 family 分文件生成，支持文件粒度断点续采；每个文件独立通过合同/hash/审计后合入语料清单（manifest 列出全部文件与 hash，语料版本 = 清单版本）。
- 先生成 P1 需要的代表性子集（约 30 个 state，见第 6 节），P2 语料在 P1 训练期间继续生成。
- 生成端报告总样本支出与 wall-clock，写入语料 manifest。

**关于迁移前数据**：当前 reader 不读取、转换或持久化迁移前 HDF5；需要核查历史结果时使用对应 Git 提交。任何写入实验注册表的正式比较必须使用 v1 corpus manifest。

### 2.5 State 语料的 split（LayerStack）

- 按结构 family 分层：约 70% train、10% validation、20% test。
- 每个进入 train 的 family 保留 ≥ 1 个 test 状态 → 考核**参数内插**（G2）。
- 另留 3–4 个完整结构 family 整体只进 test → 考核**结构外推**（G2s），与 G2 分开统计。
- test 状态总数 ≥ 50。理由：报告 state 分布的 p95 至少需要几十个样本；迁移前以极少 test state 报告的 p95 没有统计意义，这类结论一律作废。
- 同 family 的父子编辑状态不得跨 split（沿用现行 `split_group_id` 机制）。

## 3. 泛化合同：按源表示类型分别定义

「泛化」不是一个笼统层级，而是每个材质族根据其**源表示类型**声明的考核轴。

### 3.1 参数式族（parametric）

材质由少量结构 + 连续参数完全决定：LayerStack、OpenPBR 常量参数资产。

| 轴 | 含义 | 考核方式 |
|---|---|---|
| **G1** 方向泛化 | 同一 state，训练中未见的 `(wo, wi)` | held-out query role + 稠密切片上的插值检查 |
| **G2** 参数泛化 | 同结构 family，未见的连续参数状态 | family 内 held-out test 状态；分「target-visible 路径」（encoder/refinement 可读该状态的 train query）与「pure feed-forward 路径」（source compiler 不读任何 response）分开报告 |
| **G2s** 结构泛化 | 未见的层数/拓扑（仅 LayerStack） | 整族 held-out 的 test family；单列统计，不并入 G2 |

source compiler（[`model_candidates.md`](model_candidates.md) 的 M6）只对这一类族有意义，G2/G2s 是它的主考核。

### 3.2 资产式 / 测量式族（asset-bound）

材质本质绑定具体纹理或测量表：MERL、MaterialX、OpenPBR 纹理资产。**每个资产就是一个压缩对象**——一个资产压缩为一份 latent（decoder 可共享），这是材质压缩问题，不是参数外推问题。

| 轴 | 含义 | 考核方式 |
|---|---|---|
| **G1** 方向泛化 | 同一资产，未见 `(wo, wi)` | 同上 |
| **W** 工作流稳健性 | 同一套 pipeline、同一组超参，能否处理该族**全部**资产 | 批量拟合全族资产，报告质量分布（median/p95/最差资产清单）与拟合成本分布；不允许逐资产调参 |
| （后续）**G-spatial** | 同一资产未见的 UV/footprint 采样点 | spatial 阶段定义（MaterialX） |

对这一类族**不设**「未见资产 zero-shot 编译」考核——没有源参数可供外推，这样的考核没有意义。跨资产共享 decoder 的率失真收益是一个可选研究问题，其结论表述为压缩摊销，不表述为泛化。

### 3.3 跨族

「同一套框架能处理不同的材质族」本身就是项目要交付的泛化，它的证据是：各族都能通过同一采集合同、同一评测入口、同一 evaluator 家族完成 W 指标——即第 3.2 节的工作流稳健性跨族汇总。零样本跨族预测不是 v1 考核项。

## 4. 拟合协议

### 4.1 三类拟合路径，同一评测入口

| 路径 | 内容 | 例子 |
|---|---|---|
| **A 梯度训练** | 反向传播联合优化 decoder、latent、encoder 或 compiler | shared evaluator + autodecoder latent |
| **B 直接拟合** | 无梯度或封闭解算法：聚类、最小二乘、VQ | K-means++ 字典 + top-2 凸混合权重的闭式解 |
| **C 混合** | 直接拟合或 encoder 给初始化，有界梯度精调 | K-means 初始化 codebook 后联合精调；encoder 输出 + bounded refinement |

三类路径产出统一的 checkpoint/manifest，进同一个 `ncls learn evaluate` 入口。B 类没有 step 概念，报告 wall-clock 与确定性配置；比较时与 A/C 在相同数据、相同 latent bytes 下同表列出。

### 4.2 预算档位

| 档位 | 用途 | 预算 | seed |
|---|---|---|---|
| 快速档 | 实现正确性 smoke，不进注册表比较 | ≤ 30 min GPU、缩减数据允许 | 1 |
| **标准档** | 正式主搜索 | 全量阶段数据；最大步数按数据规模折算（目标约 30–50 个 query-group epoch），达到最小步数后允许用冻结的 validation patience 早停 | 先用 1 个共同 deterministic seed；差距接近或轨迹不稳时自适应追加 |
| 冲刺档 | 里程碑候选 | 标准档 ×5–10 预算 | 只对晋级候选按证据决定追加 seed |

随意缩减 step 或数据的运行只属于快速档；版本化的阶段子语料（例如 P1 selection）属于该阶段的标准数据，不算临时缩减。checkpoint 选择：validation 上的 solid-angle normalized L1 state-median（决胜看 p95）；test 只在配置冻结后读取一次。paired state bootstrap 描述冻结 state 集上的外观差异，不替代 optimization seed 方差；只有差异接近或训练轨迹异常时才追加 seed，避免把所有候选的成本无条件乘倍。

### 4.3 监督量与输出（沿用既定结论）

- HDF5 监督量是 `y = f·|cos θi|`；evaluator 运行时语义输出线性 `f`（或 `Δf`），cosine 只在 loss/metric 内乘一次。禁止运行时除以接近零的 cosine。
- target transform 只是训练内部参数化：非负量用 train-only scale 的 `log1p` + 通道标准化；带符号残差用 `asinh`。统计量只由 source-train × query-train 拟合并带 hash 保存；per-state 统计若运行时需要则计入 `B_asset`。
- loss 组合与各候选的输出参数化细节见 [`model_candidates.md`](model_candidates.md) 第 1 节。

## 5. 指标体系

四层，只有第一层是 pass/fail。

### 5.1 硬性 sanity（pass/fail，失败即结果无效）

split/hash/泄漏检查、输出 finite、范围符合族颜色合同、checkpoint 可恢复、fitted 统计只来自 train。这些检查统一由 `quality-v1` sanity 执行。

### 5.2 主指标（用于排名，只有两个）

1. **方向域**：solid-angle weighted normalized L1。query 层面聚合到 state，报告 test state 分布的 median 与 p95。
2. **能量域**：半球（透射为球面）积分能量相对误差，`(state, wo)` 层面的 median 与 p95。

当前达标参考线（v1，可随证据修订）：state-median ≤ 0.05 且 state-p95 ≤ 0.15，能量 median ≤ 0.03。这是「候选值得进入下一阶段」的参考，不是单指标 kill gate。

### 5.3 结构/长尾 scorecard（解释失败位置，不单独否决）

log-domain error、峰位角（到 95% 峰高支持集的最近角距）、峰高比、top-energy recall、source-aware reciprocity deviation、按分级/族/state 的误差分布、稠密切片上的峰形对比图。

### 5.4 诊断量（只解释，不考核）

`model error / reference SE` 只作诊断，不设固定 kill gate：该比值在 SE→0 的 query 上发散，且不能单独代表感知质量。绝对误差与 SE 同时列出。

### 5.5 里程碑指标（每阶段一次，部署轨道）

viewer 固定 light/view/state sweep 的 HDR 误差与 display-referred FLIP、sweep 时序连续性（峰位跳变/闪烁）、Python/Slang parity、真实 GPU 时间与显存。

### 5.6 成本记录（每 run 必录，研究期不淘汰）

`B_asset`、`B_shared`、`C_prepare`、`C_eval`（MAC 与实测）、拟合/编译 wall-clock。用于最终画质量–成本 Pareto，不用于研究期 kill。

## 6. 阶段路线

每个阶段回答一个问题，阶段内实验并行、不设瀑布。

| 阶段 | 回答的问题 | 数据 | 主要候选 |
|---|---|---|---|
| **P0 基准建设** | — | 实现密度表 → provider 配置；生成语料；评测 harness 按第 5 节调整；建实验注册表 | — |
| **P1 表达力（G1）** | 哪个表示族、在什么容量档位，能把最难分级（S/T/M）做到达标线？给出容量–质量曲线 | 代表性子集：每个分级组合选 4–6 个 state，约 30 个 | M1 S/M/L、M2、M3 oracle、T 诊断 |
| **P2 共享与状态泛化（G2）** | 共享 decoder + 每态 latent 的容量；三种 latent 获取路径（autodecoder / M5 encoder / M3 字典）的质量差距 | LayerStack 全语料 | M1（P1 胜出配置）、M3、M5 |
| **P3 编译（G2/G2s）** | pure feed-forward compiler 与 target-visible 路径的差距；refinement cook 能收窄多少 | 同 P2 | M6 + M5 对照 |
| **P4 资产式工作流（W）** | 同一 pipeline 不调参能否吃下 MERL 100 表 + OpenPBR 全资产？质量分布如何 | MERL/OpenPBR 语料 | P2 胜出配置 |
| **P5 及以后** | spatial latent/LOD、matched sampler、integration | MaterialX 扩容 | P2/P3 胜出 evaluator |
| **D 部署轨道** | 每阶段收尾：当期最优 S 档候选 → MethodBundle → Slang parity → 成本实测 → viewer capture | — | — |

P1 与 P2 的语料同批生成；P1 的结论决定 P2 的 decoder 起点。P4 可与 P3 并行。

## 7. 实验记录与比较规则

- **实验注册表** `docs/research/experiment_log.md`：每个正式 run 一行——日期、候选+档位、数据版本、预算档、seed 数、两个主指标、一句话结论、artifacts 路径。这是回答「现在做到哪了」的唯一入口；详细数值留在 `artifacts/`。
- **可比性**：只有数据版本、预算档、split 全部相同的 run 才能同表比较。
- **结论强度**：任何「X 优于 Y」写入注册表前，用 test state 上的 paired difference + bootstrap 置信区间（≥1,000 次重采样）确认区间不跨零；跨零则记为「无显著差异」。
- **对照要求**：每个结论必须有 matched 对照（同数据、同预算、只差声明的机制组）。允许一次实验同时改变一组相关机制（例如「chart + FiLM」作为一个 bundle），但 bundle 内要保留能识别主要贡献的消融。
- **test 治理（简化）**：test 日常不读；阶段结论读取。若某次 test 结果驱动了后续设计，下一阶段对该考核轴更换新 test split 版本。不再维护更复杂的 sealed/development 双层机制。
- **回归**：已解决的失败（峰位、能量、范围、parity）固化为固定 probe，新 run 自动跑；回归项退化需要在注册表中显式说明权衡。

## 8. 与部署合同的静态约束

研究期每个候选注册时检查（纸面检查，不需实现 shader）：

1. `evaluate()` 语义输出线性 `f`，不含几何余弦；
2. 单次 `(state, wo, wi)` 查询的读取量固定：有限个 latent/codeword/权重块，与分辨率和历史查询无关；
3. `prepare()` 结果可被同一着色点的多个 `wi` 复用；material-static 的调制参数要么计入 `C_prepare`，要么烘焙后计入 `B_asset`——二选一，注册时声明；
4. 不把完整方向表藏进 `prepare()`。

满足静态约束的候选才有资格进入 D 轨道；D 轨道的实测数据反过来只影响部署阶段的压缩设计，不回头否决研究期结论。
