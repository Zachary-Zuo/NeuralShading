# 模型候选设计

本文给出当前全部模型候选的完整设计：每个候选说明它是什么、为什么可能有效、数学结构、容量档位、怎么拟合、已知风险，以及什么实验结果会让它保留、调整或放弃。实验流程、数据密度与指标见 [`experiment_framework.md`](experiment_framework.md)；文献细节见 [`prior_art.md`](prior_art.md)。

候选一览：

| 编号 | 名称 | 一句话定位 | 适用族 | 拟合路径 |
|---|---|---|---|---|
| M1 | conditioned shared evaluator | 主线：canonical 方向前端 + FiLM 调制的共享 MLP | 全部 | A 梯度 |
| M2 | analytic core + neural residual | 解析 core 承担主结构，网络只补残差 | 有可信 core 的族 | A |
| M3 | sparse dictionary / top-k mixture | 共享 codebook + 每态少量系数，直接拟合优先 | 全部 | B/C |
| M4 | warped tensor field | half/difference warp 后的低秩显式表格 | 全部 | A |
| M5 | target response encoder | 压缩期从响应数据推断 latent 的 encoder | 全部（压缩期工具） | A/C |
| M6 | typed source compiler | 从原生参数前向生成 latent | 仅参数式族 | A |
| T | 诊断 teacher | 无 latent 瓶颈的大容量单态拟合，归因工具 | — | A |

M1 是所有共享/编译实验的宿主：M3 替换它的 latent 资产表示，M5/M6 替换它的 latent 获取方式，M2 替换它的输出参数化。M4 是唯一的独立表示族，用来检验「显式表格 vs MLP」这一根本分歧。

## 1. 公共约定

### 1.1 符号与运行时分解

```text
m            : 一个 source material state（或资产）
z_m ∈ R^D    : 可部署 material code
wo, wi ∈ S²  : 局部 frame 中的出射/入射方向
f(m,wo,wi)   : evaluate() 的语义输出，线性 RGB，不含几何余弦
y = f·|cosθi|: HDF5 监督量（response_cos）

c      = Condition(z_m)          # 每材质一次，可烘焙
p      = Prepare(c, wo)          # 每 (材质, 着色点, wo) 一次，供多个 wi 复用
f_hat  = Evaluate(p, wi; c)      # 每 wi 一次
```

训练 batch 形状为 `wo:[G,3]`、`wi:[G,N,3]`、`y:[G,N,3]`，但模型不得依赖固定 `N`：单次随机 `wi` 查询必须成立，`prepare` 不得隐藏完整方向表。

### 1.2 方向 chart 前端（M1/M2/M4/M5 共用）

镜面峰在 raw `wi` 坐标里随 `wo`、法线和粗糙度高速移动；chart 的作用是把「移动的结构」变成近似驻定的规范坐标，让网络/表格学的是驻定形状而不是追峰。

**反射 chart（Rusinkiewicz half/difference）**：

```text
h  = normalize(wo + wi)                    # ‖wo+wi‖ 过小时置 invalid
(θh, φh) = 局部 frame 中 h 的球坐标
d  = R_h(wi)                               # 把 wi 旋到以 h 为极轴的 frame
(θd, φd) = d 的球坐标
slope     = (h.x/h.z, h.y/h.z)             # half-vector slope
log_slope = sign(slope)·log1p(|slope|)     # 掠射安全
```

窄峰主要沿 `θh`/slope 轴变化，GGX 类 lobe 在 slope 坐标下接近固定形状——这是 [NBRDF 2021]、[Neural Appearance 2024] 与 [Adaptive Parameterization 2018] 共同验证过的先验。

**透射 chart（generalized half vector，Walter 2007）**：

```text
h_t = -normalize(η_o·wo + η_i·wi)          # η 来自族的原生参数
```

只有原生 source 确实提供 η 的族（LayerStack、OpenPBR）使用；MERL 等无 η 族不伪造参数，透射侧保留 raw features + invalid flag，或使用 learned frame 变体（作为单独候选比较，不作为不透明 fallback）。

**合同要求**：chart 返回显式 validity/mode flags；half-vector 退化、临界角、反射/透射边界不得产生 NaN；归一化 epsilon 版本化；raw features（`wo`、`wi`、双 cosine、flags）始终与 chart 特征并联保留，防止 chart 在退化区丢信息。chart 进入部署轨道前需通过 Python/Slang parity（near-grazing、临界透射、方向交换、随机 probe）。

### 1.3 输出参数化与 target transform

- **direct head**：预测 `f_hat`。非负色域用版本化非负映射（如 softplus/exp 族）；合法负通道（如 ACEScg→sRGB 的 out-of-gamut）用 unconstrained head，按族颜色合同评测，禁止统一 clamp。
- **residual head**（M2）：预测带符号 `Δf`，`f_hat = f_core + Δf`。
- transform 只存在于 loss/metric 内部：

```text
非负监督: t = log1p(y/s)，逆为 y = s·expm1(t)
带符号残差: t = asinh(r/s)，逆为 r = s·sinh(t)
s 及通道 mean/std 只由 source-train × query-train 拟合，带 hash 版本化
```

per-state 统计（E2 已证明跨态共用 scale 会把 1e-8 与 1e-1 的残差压进同一尺度）若运行时需要则计入 `B_asset`（LayerStack 现值 36 B/state）。

### 1.4 loss 库

总 loss 是有明确职责的项的加权和，各候选按需启用，权重在读 test 前由 train/validation 对照冻结：

| 项 | 职责 |
|---|---|
| `L_transform` | 变换域 Huber/Charbonnier，保住暗部与长尾梯度 |
| `L_linear` | 按 solid-angle/proposal 权重的线性域误差，防止 transform 域改善但线性能量恶化 |
| `L_energy` | evaluator 自身积分出的半球能量误差（不用旁路 energy head 替身） |
| `L_peak` | train query 的 top-energy 支持集加权项，防窄峰被低值区淹没 |
| `L_recip` | source-aware reciprocity（沿用 E2 修正：对已知非互易源扣除固有偏差，不强加错误零约束） |

### 1.5 容量档位（S/M/L）

所有神经候选按三档报告，构成容量–质量曲线；数值是设计基准，允许按证据微调，实际参数量/MAC 逐 run 记录：

| 档 | 定位 | width | blocks (prepare/evaluate) | latent D | 量级 |
|---|---|---:|---|---:|---|
| S | 部署候选形态 | 128 | 2 / 3 | 32 | ~0.2M 参数，`C_eval` ~1e5 MAC |
| M | 结构完整的研究主力 | 256 | 3 / 6 | 64 | ~1.3M 参数，`C_eval` ~1e6 MAC |
| L | 容量诊断 / distillation teacher | 512 | 3 / 8 | 128 | ~6M 参数 |

block = pre-norm residual block（linear→GELU→linear + skip；LayerNorm 只作用于 hidden，不跨 query/材质统计）。旧 65k MAC 预算在此体系内约等于 S 档以下，不再作为淘汰条件。

---

## 2. M1：conditioned shared evaluator（主线）

### 2.1 动机

共享 decoder + 每态低维 code 是全部压缩收益的来源：跨材质的散射结构相似性进共享权重，材质差异进小 latent。E2 已经证明最小版本（concat 条件、width 108）能把 24 个 state 压到 test median 0.053，但长尾（p95 0.169）在该架构族内扩容无改善——瓶颈在**条件化方式和结构容量**，不在 latent 维数。M1 的设计目标就是解决这两点。

### 2.2 结构

```text
c = Condition(z_m):
    对每个 residual block l 生成 FiLM 参数 (γ_l, β_l) = A_l·z_m + b_l
    另生成全局 color/energy context 向量 g(z_m)

p = Prepare(c, wo):
    enc_wo = [raw wo 特征, 掠射特征]
    p = PrepareTrunk(enc_wo ⊕ g)，trunk 各 block 受 FiLM(γ,β) 调制

f_hat = Evaluate(p, wi; c):
    q = [raw 特征, reflection chart, transmission chart, validity/mode flags]
    h = EvaluateTrunk(q ⊕ p)，各 block 受 FiLM 调制
    f_hat = 按 mode 组合各 head 输出（见 2.3）
```

**为什么 FiLM 而不是 concat**：concat 只在第一层线性注入 latent，等价于对第一层 bias 的低维扰动，后续层只能被动传播；FiLM（`h ← γ_l(z)⊙h + β_l(z)`）让每个 state 逐层重标定每个通道，表达力接近给每个 state 一组对角调制权重，而参数开销只有 `O(L·width·D)`。这是 E2 长尾停滞后的第一个结构假设，[Neural Appearance 2024] 的 conditioning 也属于这一类。

**可选的低秩权重调制**（FiLM 不足时的下一级）：

```text
h_{l+1} = act( γ_l(z) ⊙ ((W_l + U_l·diag(a_l(z))·V_l)·h_l) + β_l(z) )
```

`U_l/V_l` 共享，`a_l(z)` 逐 state。`Condition(z)` 的生命周期二选一并声明：逐着色点执行计入 `C_prepare`，或编译期烘焙计入 `B_asset`。

### 2.3 mode heads 与 lobe experts（配置轴，不是遥远的未来阶段）

- **mode heads**（M 档默认开启）：shared trunk 输出 hidden，reflection head、transmission head、diffuse/low-frequency head 各自小 MLP，按 chart 的 mode/validity mask 组合。动机：反射与透射的规范坐标不同，单 head 在临界边界发生模式竞争（E2 的 sheen 能量错误属于此类）。
- **lobe experts**（可选配置，K=2/4）：`prepare` 从 `p` 生成 K 个 lobe token（6D 旋转表示的 frame + 类型 logits + 带宽/幅值 hint），每个 token 驱动共享 expert 在各自规范坐标下输出非负 contribution，另加一个受容量限制的 signed correction head。启用条件：稠密切片显示多个同时峰被单 head 平均。启用时必须报告逐分支消融，防止 correction head 学走全部函数、lobe 退化为装饰。

### 2.4 拟合与训练

路径 A。latent 获取三种方式（P2 的比较对象）：autodecoder（`z_m` 注册为自由参数直接优化）、M5 encoder 初始化 ± bounded refinement、M3 字典。训练分布沿用 coverage warm-up → peak curriculum → train-only hard-query refinement 三段；loss 用 1.4 全库。

### 2.5 风险与判定

- 风险：FiLM 仍不足以覆盖跨态长尾（→ 升低秩调制）；lobe 路由在 view sweep 上抖动（→ 时序连续性指标把关）；latent 维数在语料扩大后成为新瓶颈（→ D 是档位轴，随曲线报告）。
- 判定实验：P1 用 M1-S/M/L 对最难分级 state 画容量–质量曲线。若 M 档在 G1 上达标 → P2 起点；若 L 档仍不达标而 T 诊断（无 latent 瓶颈）达标 → 瓶颈在条件化/latent，升级调制方式；若 T 也不达标 → 瓶颈在方向表示或监督，转 M4/数据侧。

## 3. M2：analytic core + neural residual

### 3.1 动机

E1 唯一通过达标线的候选（多界面 state，test median 0.046，65k MAC 小模型）就是这个形态。解析 core 精确承担最难的移动峰主结构，网络只补 core 之外的多次散射/层间耦合，网络负担的方向频率大幅降低。[Hybrid Neural-Microfacet 2026] 在同内存下优于纯 neural 表示，证明这不是权宜之计而是正式竞争路线。

### 3.2 结构

```text
f_hat = f_core(m, wo, wi) + Δf(z_m, wo, wi)
```

- `f_core`：族 adapter 提供。LayerStack 用 direct-top 界面散射（已实现）；无层语义的族可用拟合的 GGX proxy core（参数由直接拟合得到，计入 `B_asset`）或不用 core。
- `Δf`：宿主为 M1 的任意配置，输出 signed，transform 用 per-state `asinh`。
- core-only 指标每 run 报告，量化网络的净贡献（E1 实测：core-only 0.866 → +residual 0.046）。

### 3.3 风险与判定

- 风险：core 掩盖网络容量问题，跨族结论被 core 依赖污染（→ 始终与 M1 direct 配对报告）；无 core 的族退化为 M1，形成不对称比较（→ 跨族汇总时分层报告）。
- 判定：P1/P2 中始终作为 M1 的 paired variant 跑。若 direct M1-M 达标且差距 < 显著区间 → core 变为可选优化；若 residual 显著更好 → core 进入部署形态，需在 D 轨道验证 core 的 Slang 成本。

## 4. M3：sparse dictionary / top-k mixture

### 4.1 动机

来自用户已验证的工程经验（时序 lightmap 的 K-means++ top-2 轨迹字典优于 tri-plane 优化；完整算法记录见 [`problem_definition.md`](problem_definition.md) §4）。机制上：大部分信息进入少量共享原型，每个查询位置只保存 2 个 ID + 1 个权重，凸混合是最便宜的 decoder，且天然满足随机访问。[Dictionary Fields 2023] 是同构的学术印证。启发式初始化（聚类）可能显著快于纯梯度自发现——E2 已证明随机初始化的 hard top-k 失败，但这恰恰没有测试过聚类初始化。

### 4.2 两个层级

**(a) response-space oracle（直接拟合，P1 的廉价判定实验）**——只回答「字典结构是否存在」，不是部署形态：

```text
1. 每个 state 在冻结的 canonical probe 网格上求变换后响应，展平为 x_m ∈ R^P
   （两种向量化单元都测：per-state 全响应；per-(state, wo) 方向轨迹）
2. 对全部 train 单元 K-means++（K = 64 / 256 / 1024）得 codebook
3. 每单元取最近 5 个候选原型；固定最近为 c1，枚举 c2 ∈ 第 2–5 近：
      w* = clamp( <x−c1, c2−c1> / ‖c2−c1‖², 0, 1 )     # 闭式最小二乘
   取重建误差最小的 (c1, c2, w)
4. 报告 rate-distortion（2 ID + 1 权重 vs 误差），与同 bytes 的 PCA/低秩基线对比
```

**(b) latent dictionary（部署形态）**——替换 M1 的 `z_m` 资产表示：

```text
z_m = Σ_{j∈top-k} w_j · c_j        # codebook C 共享计入 B_shared
资产保存: k 个 uint16 ID + 权重；k ∈ {1, 2, 4}
初始化: 对 P2 已拟合的 optimized z 做 train-only K-means → 联合精调（路径 C）
变体: top-2 凸混合 / top-k soft / codeword + 连续 residual / residual VQ
```

### 4.3 风险与判定

- 风险：凸混合偏向条件均值，压峰值（→ 峰指标把关；residual 变体兜底）；top-k 线段表达界（→ k 与 residual 是档位轴）；codebook 随语料增长的规模律未知（→ K 是档位轴）。
- 判定：oracle 的 rate-distortion 明显优于同 bytes PCA → 字典假设成立，进入 (b)；(b) 在同 bytes 下逼近 dense `z` 且拟合更快 → 成为资产表示默认；否则保留为低 bytes 端点或放弃。

## 5. M4：warped tensor field

### 5.1 动机

与 M1 的根本分歧：高频规范场到底该由 MLP 隐式表示，还是由显式表格直接存储。E1 的 raw 方向 plane 分解失败，但失败原因是坐标——raw 坐标下峰跨 plane 移动，表格存的是移动目标。warp 到 half/difference 坐标后峰驻定，低秩表格才有机会。[TensoRF 2022] 的 vector-matrix 分解与 [Dupuy 2018] 的自适应 warp 是直接依据。

### 5.2 结构

三个 branch（reflection / transmission / boundary+low-freq），各自在 warp 后坐标上做多尺度 vector-matrix 分解：

```text
q_r = (u_h, v_h, u_d, v_d)     # reflection：half-slope 为高分辨率轴
F_l(q) = Σ_r [ H²_{l,r}(u_h,v_h)·Dᵘ_{l,r}(u_d)·Dᵛ_{l,r}(v_d)
             + D²_{l,r}(u_d,v_d)·Hᵘ_{l,r}(u_h)·Hᵛ_{l,r}(v_h) ]
```

第一项让高分辨率 half plane 直接承载窄峰，第二项补 difference 方向的二维相关。各尺度特征 concat 后交给小 decoder 输出 `f` 或 `Δf`。2D plane bilinear fetch、1D factor linear fetch；chart seam 用周期坐标或成对 seam 特征；单 query 固定读取数，随机访问成立。

起点：单尺度 64² half plane × rank 8，先验证误差随 half-plane 分辨率因果下降，再扩 128²/多尺度/更高 rank。每次只动 half 侧的一个规模旋钮。

### 5.3 风险与判定

- 风险：资产 bytes 随分辨率平方增长（→ 它天然是高质量 Pareto 端点/teacher，不必是部署形态）；多峰材质 rank 爆炸（这本身是重要结论：说明峰间相关性不能被该结构解释）。
- 判定：P1 中与 M1-M 同 state 比较。若 M4 在窄峰上显著更准 → 作为 distillation teacher 或高质量端点保留；若 warp 后仍需极高 rank → 放弃，同时这一结果反过来支持 M1 的隐式路线。

## 6. M5：target response encoder

### 6.1 定位（与 M6 的边界）

M5 的输入是**被压缩目标的响应数据本身**（train query 的 `(方向, 响应)` 集合），输出 latent。它是压缩期工具：资产烘焙后 encoder 丢弃，运行时只剩 latent + decoder。它能读 reference 响应，所以永远不等于 source compiler——M6 的输入是原生材质参数，不读任何响应。两者不可混淆（[NTC 2023] 是 autodecoder，[NGTC 2024] 是 target encoder，[Neural Appearance 2024] 是 source-parameter encoder：三种输入合同支持三种不同结论）。

用途：① 资产式族（MERL/MaterialX）的正式压缩路径——确定性、可摊销的 latent inference；② P2 中与 autodecoder 的收敛速度/质量对照。

### 6.2 结构

E2 的 DeepSets（逐点 MLP + mean/max pooling）已验证生命周期且 encoder-only 略优于 matched autodecoder（0.086 vs 0.092 @3k；8k 时 0.060），但 pooling 压掉峰间相对位置与同 `wo` 的完整 shape。升级为两级结构：

```text
token = [chart 坐标, validity, 变换后响应, reference SE, solid-angle 权重, train-only peak flag]
第一级: 同一 wo 的 query group 内局部 attention/pooling → group 表示
第二级: 跨 group 的 induced attention（16/32 个 inducing token，Set Transformer）
输出: 与当前 M1 配置一致的 z_m
```

计算随点数近似线性；分块累积规则进 fitted-state hash。训练：固定 P2 稳定的 decoder，以 query-space 重建为主 loss（latent L2 只作弱对齐，因为多个 latent 可表示同一函数）。之后可对 `z_0 = E(X_train)` 做版本化 bounded refinement（路径 C）；encoder-only 与 refinement 后结果分开报告，压缩时间计入成本。

### 6.3 判定

P2 三路径对照：encoder ± refinement 逼近 autodecoder 质量且更快/更稳 → 成为资产式族默认压缩路径；只在初始化上有优势 → 降级为 autodecoder 的初始化器；无优势 → 保留 DeepSets 结论，放弃 attention 升级。

## 7. M6：typed source compiler（仅参数式族）

### 7.1 定位

输入原生材质定义 `(M, θ)` 的 typed token/graph，输出 `z_m`，不读取任何 reference 响应。它是「编辑后即时前向编译」这一目标 claim 的唯一载体，主考核是 G2（未见参数状态）与 G2s（未见结构），见 [`experiment_framework.md`](experiment_framework.md) §3.1。资产式族没有可外推的源参数，不进入 M6 的范围。

### 7.2 输入与结构

family adapter 输出版本化 typed token：node type、原生连续/离散参数、资源引用、有序位置/深度、typed edges、合同版本。LayerStack 保留 interface/medium 顺序与邻接。

两级实现：

1. **compiler control**：order-aware token 投影 + 小型 sequence encoder（先 Transformer encoder 而非 GRU——E3 的 GRU smoke 过拟合是该最小实现的结论），从 P2 起与 shared decoder 并存，随时量化 pure feed-forward gap；
2. **quality compiler**：只有 control 的误差随层数/拓扑/长程相互作用系统增长时，才升级 order-aware graph Transformer。[MetaLayer 2023]（从层参数生成完整网络权重）作为 baseline 变体在同 bytes 下对照。

**不预测 target transform 常量**（E3 教训）：per-state normalization 由 train-corpus 族级统计、显式 energy/color head 或源可计算物理量替代；compiler 考核的是材质函数 code，不是外推一个依赖响应统计的坐标系。

### 7.3 训练与三种结果角色

```text
1. control: 保存 P2 的 optimized code + functional metrics
2. canonicalization: 共同初始化 + code 正则 + decoder 固定，压掉 latent 排列/尺度不定性
3. code distillation: 只作初始化
4. functional distillation: 固定 decoder，在 source-train × query-train 上以响应/能量/峰 loss 为主目标
5. joint fine-tune: 仅在 validation 明确改善时有限解冻 decoder，保留 control
```

结果按三种 manifest role 报告，不得互相冒充：

- `pure feed-forward`：`z = C(M,θ)`，不读响应 → 即时编辑结论的依据；
- `compiler init + bounded refinement`：用新状态的 query-train 响应做固定预算 cook → 实用离线 cook 结论；
- `optimized / target-encoded control`：同 decoder 的质量参照，量化 compiler gap。

### 7.4 判定

P3：pure feed-forward 在 G2 上与 control 的 gap 落入显著区间内 → 编译主线成立；gap 大但 refinement cook 能收窄到达标 → 产品形态调整为「编辑后秒级 cook」；两者都不行 → 检查 canonicalization 与 corpus 规模，必要时收缩 claim 到 refinement-only。

## 8. T：诊断 teacher（工具，不进排名）

M1-L 的 per-state 变体：去掉 latent 瓶颈，直接拟合单个 state 的完整 `wo×wi`（可加 J=1/2 learned shading frame：6D 旋转表示 + Gram-Schmidt，material-static，禁止逐 query 旋转）。用途只有归因：

- M1 失败而 T 成功 → 瓶颈在 latent/条件化；
- T 也失败 → 瓶颈在方向表示或监督覆盖，先修 chart/数据再谈架构；
- T 的 direct vs residual 差距 → core 的真实贡献量。

T 参数量可能超过有效 train query 数，必须用稠密切片排除离散表记忆后才能引用其结论。

## 9. 已有实验证据边界

以下结论全部限定在「几 MB 数据 + 65k MAC 小模型」范围内，作为设计输入而不是任何候选的最终判决：

| 证据 | 结论 | 限定 |
|---|---|---|
| E1 极窄 conductor | direct 小 MLP 丢峰淘汰；7 种方向编码/transform 组合 normalized L1 ≈ 1 | 只否定该预算的 direct 小网络 |
| E1 多界面 | M2 形态（energy/shape + GELU + multiscale half-slope）test median 0.046 达标 | optimized-latent 单材质；core 可能承担了最难部分 |
| E1 plane v1 | raw 方向 pairwise plane 高分辨率过拟合、低分辨率欠拟合，淘汰 | 只否定 raw 坐标 + 该分解；不涉及 M4 的 warp 版本 |
| E2 shared concat 小 MLP | 24 state 压到 0.053/0.169（median/p95）；width/latent 局部扩容无改善 | 只说明 concat 条件化 + 该容量族到头；FiLM/调制/lobe 未测 |
| E2 hard top-k | 随机初始化纯梯度字典失败 | 未测聚类初始化（M3 的核心假设仍开放） |
| E2 rank-4 factor | 52 B/state 接近 matched dense | 低 bytes 端点可行的初步信号 |
| E2 DeepSets encoder | encoder-only ≈ 或优于 autodecoder；refinement 增益小 | 该 pooling 结构 + 该 decoder |
| E2 SE-ratio gate 系列 | SE-floor loss、CVaR25 均使独立 test 变差 | 该指标不可作为优化目标（已废除为诊断量） |
| E3 GRU compiler smoke | 最小 compiler + 联合训练 + 预测 transform 常量 → validation 过拟合 | 实现级结论；typed 输入与 lifecycle 有效 |
| E0 全部校准事实 | peak-aware vMF、adaptive SE、TDR batch 上限、峰覆盖 gate | 直接进入 v1 采集参数（见 framework §2.3） |

## 10. 文献 → 候选映射

| 候选 | 直接依据 | 关键边界 |
|---|---|---|
| M1 | Neural Appearance 2024（conditioning、learned frame、log loss）；NBRDF 2021（half/difference） | 两者均单资产/单族训练，共享与编译要本项目自证 |
| M2 | Hybrid Neural-Microfacet 2026；E1 多界面结果 | 单 GGX core 未必覆盖多峰层栈 |
| M3 | 用户 lightmap 字典经验；Dictionary Fields 2023 | 固定帧轨迹可全存，连续方向不能；凸混合均值偏置 |
| M4 | TensoRF 2022（VM 分解）；Dupuy 2018（warp）；NDGI 2026（非对称轴预算） | 通用 field 工作未测移动峰下的 rank 行为 |
| M5 | NGTC 2024（target encoder 不破坏随机访问）；Set Transformer 2019 | per-asset encoder ≠ 跨资产摊销 |
| M6 | MetaLayer 2023；Neural Appearance 2024 的 source-parameter encoder | 前者逐材质生成权重、后者逐材质训练 encoder，均不直接证明跨状态泛化 |
