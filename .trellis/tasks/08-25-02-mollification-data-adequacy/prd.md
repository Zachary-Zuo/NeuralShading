# 02 Directional Mollification 数据充分性

## 它是什么

本任务在 neural 方法训练开始前，独立判定冻结的 `layer-stack-p1-v1` v5 corpus 是否足以重建 NVIDIA directional mollification。Directional mollification 是训练前 `M=20,000` 次迭代内，将 query 的 `ωo` 在一个从 10° 余弦衰减到 0° 的 cone 内平均 256 次，以便让极窄峰先宽后窄进入网络；它不是 learned frame、sampler KL 或解析 `sample/pdf` 的数据要求。

若 v5 的离散 `(wo, wi, response)` 支持和数值误差全部通过冻结 gate，`03` 继续使用原 corpus。若任一 gate 失败，本任务必须先完成一个引用原 v5、只补充早期 curriculum target 的版本化 mollification corpus；不能把邻近点启发式留给 `03` 临时解释。

## Scope And Dependencies

- 前置任务 `.trellis/tasks/archive/2026-08/08-25-01-reusable-scattering-math` 已完成、提交并归档；其公共方向、solid-angle measure 和 LayerStack reference 语义是本任务的输入。
- 基础数据固定为 `artifacts/corpus/layer-stack-p1-v1.json` 指向的 30-state v5 corpus；不覆盖、转换或删除现有 HDF5。
- 本任务拥有 protocol/schema、matched audit、必要的 supplement 生成与 reader；`03` 只消费本任务输出的数据入口，不重新选择 corpus 或阈值。
- 新 HDF5 只进入 `data/reference-responses/`，运行报告只进入 `artifacts/`，稳定 schema/config/docs/tests 进入根仓库。

## Frozen Adequacy Protocol

查询前必须由 `freeze` 阶段生成带 hash 的 anchor lock；`run` 阶段拒绝缺失、被改写或与基础 corpus identity 不一致的 lock。

### 代表状态

| 角色 | state ID |
|---|---|
| 纯 diffuse 控制 | `1b6dc2ae36e7fb076e1942242317b9af5b8bc723d71e17e5bccf33402403c49f` |
| 单层窄导体峰 | `3d925881a55b0bbee135592d539b71dec694be51852793d74071d68d161ce5b5` |
| 旧尾部：三层 conductor | `bd6de2e9d0cf5b32e6259d90af99b718983783cbfff1a0a9c3025a54ff3672db` |
| 旧尾部：四层 diffuse A | `6fff05aa8f142b26631bd354c21c871e900a75bbd0f33e8426d3e05d73b9a3e9` |
| 旧尾部：四层 sheen | `4ebd9258461716ed23d523a0b221d015feb190270471e2a55b63c3a219fcb1e7` |
| 旧尾部：四层 diffuse B | `1796065779d0932fe7ded3cc2c40b84a8a19190dd7e68732f06bc518ae7fe54a` |

每个 state 使用现有 dense slice 的四个精确 `wo`，覆盖近法向、中角、斜角和 `wo.z≈0.138` 的 grazing。每个 `(state, wo)` 从该 dense group 确定性锁定四个 `wi`：最大 response 峰、峰外 2°–5° shoulder、`wi.z∈[0.02,0.15]` 的 grazing-light 最大值，以及 `wi.z∈[0.4,0.8]` 的 response 中位背景；空集合按配置中的确定性回退规则选择，最终向量写入 anchor lock。

### Cone 与 reference

- 论文 schedule：`r(t)=5°·(1+cos(πt))`，`t∈[0,1]`；audit 正半径层为 `t={0,0.25,0.5,0.75}`，即 `r={10°,8.5355339059°,5°,1.4644660941°}`，`t=1` 的 0° 由基础 v5 target 表示。
- 每个 cone 用 256 个 deterministic scrambled-Hammersley solid-angle-uniform `ωo`；cone 越过下半球时，从同一无限确定性序列接受 `ωo.z>0` 的前 256 个，因此分布是 upper-hemisphere 与 spherical-cap 交集上的均匀分布。
- 每个 jitter direction 的 LayerStack reference 使用两个独立 replica，各 512 paths，最大路径深度 64；报告从 replica cone mean 得到 target SE。fresh query 固定 `wi`，只 mollify `ωo`。
- response measure 始终是线性 RGB `f·|cos(wi)|`；不得对 `f`、reciprocal 或 reference PDF 做隐式替换。

### 旧 corpus 重建与通过门

- 重建只使用该 state 的 v5 train pairs；以 `[wo, wi]` 的 chord embedding 做固定 `k=32` 最近邻 Shepard-2 插值，再对与 fresh query 完全相同的 256 个 `ωo` 平均。
- support 门：至少 95% jitter query 同时存在 `angular(wo)≤2°` 且 `angular(wi)≤1°` 的旧样本；不满足时该 target 直接不充分，仍保存数值结果用于诊断。
- noise 门：fresh cone target 的 relative SE p95 `≤0.04`。
- 数值门：所有 state-target 的 normalized RGB L1 median `≤0.025`、p95 `≤0.05`、worst `≤0.10`。归一分母使用 `mean(abs(reference RGB))` 与配置 floor 的最大值。
- repeat 门：同一 lock 与 seed 重跑的语义 report hash 完全相同；浮点明细差异不得超过 `1e-7`。
- 只有 support、noise、数值和 repeat 全部通过才输出 `reuse-v5`；其余情况唯一决定为 `use-mollification-supplement-v1`。

## Failure Path: Versioned Supplement

若 audit 未通过，生成 `layer-stack-p1-mollification-v1`：

- identity 同时绑定 base `corpus_id`、protocol/anchor-lock hash、reference implementation、state selection 和全部 shard hash；它是 supplement，不复制 v5 的 validation/test/dense 数据。
- 覆盖 P1 selection 的全部 30 个 state。每 state 从 train role 确定性选择 8 个 view（球面 farthest-point，并强制包含 grazing）和每 view 64 个 `wi`（16 peak、16 log-response 分层、32 proposal-index 分层）。
- 保存 `t={0,0.25,0.5,0.75}` 四个正半径 curriculum level；每 target 仍平均 256 个相同分布的 jitter `ωo`。起始每 jitter/replica 64 paths，首个 plan 的 cap 为 512；按 cone-mean relative SE 将整个 target group 倍增。train target p95 `≤0.06`、group hard limit `≤0.25`，达到 frozen cap 仍失败则不发布 manifest，而是走下述 versioned budget refinement。
- 新 `mollified-reference-shard` 保存 anchor/group、精确 curriculum `t/radius`、jitter 规则与 count、双 replica mean/variance/sample count和来源 v5 pair；`mollified-reference-corpus` manifest 保存完整 provenance 与 semantic identity。
- learning reader 以训练 progress 选择最近的已存正半径 level；`t≥0.875` 切回 base v5 的 0° target。这个有限离线 curriculum 是项目对论文在线 reference 采样的显式适配，不能表述为连续 schedule 的无误差等价物。

首轮 `reference-se-v1` 在冻结的 512 paths/jitter/replica cap 下由 sheen state `4ebd925…` 的 view 2 / level 0 实测失败（p95 `0.152108`、worst `0.365159`），不发布 manifest。版本化 rollback 使用独立 budget plan `layer-stack-p1-mollification-reference-se-v2`：保持 threshold、256 jitter、双 replica、anchor 与 level 不变，按 `ceil-power-of-two((0.152108/0.06)^2×512)` 将 cap 提升到 4096；该 plan 与新的 collection lock 必须在 v2 fresh query 前冻结。若 v2 仍失败，继续用相同规则生成下一版本，不得修改旧 plan 或失败证据。

`reference-se-v2` 在 4096 cap 下继续由同一 sheen state 的 view 4 / level 0 实测 p95 `0.0689937`（worst `0.113843`），仍不发布。`reference-se-v3` 按同一规则计算 `ceil((0.0689937/0.06)^2×4096)=5416`，向上取二次幂将 cap 固定为 8192；其余合同不变。

`reference-se-v3` 的 p95 已过门，但 tail diffuse state `179606…` 的 view 2 / level 0 实测 worst `0.386128` 仍高于 hard limit `0.25`。v4 对实际失败 gate 使用相同平方反比原则：`ceil((0.386128/0.25)^2×8192)=19543`，向上取二次幂把 cap 固定为 32768；p95 与 hard threshold仍不变。

`reference-se-v4` 在 32768 cap 时同一 target 的 p95 降到 `0.0307`，worst 为 `0.319926`，仍只差 hard gate。v5 继续按 `ceil((0.319926/0.25)^2×32768)=53663`，向上取二次幂把 cap 固定为 65536。

batched accumulator 落地后的 `reference-se-v6` 在 65536 cap 下由 `179606…` 的 view 4 / level 0 实测 p95 `0.0844163`、worst `0.151923`。v7 保持 accumulator 不变，按 `ceil((0.0844163/0.06)^2×65536)=129727`，向上取二次幂把 cap 固定为 131072。

逐 state 证据晋升从 v7 起成为正式合同：已经通过自身 `0.06/0.25` 门的 shard 不因 global plan 更新而重算；collection lock 逐项冻结其 URI、dataset/file hash、metrics 与 source budget plan/lock。v7 复用 v6 的 27 个通过 shard，只采 `179606…/bd6de2…/6fff05…`。其中后两者在 v7 分别以 p95/max `0.0588622/0.241481` 与 `0.0594174/0.232417` 通过；`179606…` 在 view 6 / level 0 以 p95 `0.107237`、max `0.19514` 达 cap 失败。

v8 按同一规则计算 `ceil((0.107237/0.06)^2×131072)=418694`，向上取二次幂把 cap 固定为 524288；其 multi-source collection lock 精确复用 v6 的 27 个 shard与 v7 的 2 个 shard，只晋升 `179606…`。该 state 最终以 shard p95/max `0.0588824/0.234875` 通过，30-state composite corpus 合法发布，阈值、256 jitter、双 replica、anchor 与四个 level 始终未变。

## Acceptance Criteria

- [x] protocol、阈值、代表 state、精确 anchor lock、cone/jitter/reference budget 在任何 fresh matched result 产生前冻结并可验 hash。
- [x] matched audit 可重复，覆盖 diffuse、窄导体峰、grazing 和四个旧尾部 state，且给出唯一二选一决定。
- [x] `reuse-v5` 路径包含确定性重建、support/noise/数值/repeat 全部通过证据；不通过时不得选择该路径。
- [x] supplement 路径具有版本化 schema/manifest、30-state 完整生成、hash/measure/noise/provenance 验证和唯一 corpus identity。
- [x] `03` 获得一个明确、可读取、可追溯的数据入口和 curriculum reader，不存在局部静默重解释。
- [ ] unit/GPU/reference/corpus gate、Quality Check、scoped local commits 与任务归档全部完成后，父任务才进入 `03`。

## Out Of Scope

- neural 模型、loss 优化、sampler family、MethodBundle 和 viewer。
- 因 learned frames、sampler KL、解析 `sample/pdf` 或 reciprocal scorecard 产生的数据升级。
- 改写或删除合法 v5 shard；扩展到 transmission/delta 或其他源材质族。
