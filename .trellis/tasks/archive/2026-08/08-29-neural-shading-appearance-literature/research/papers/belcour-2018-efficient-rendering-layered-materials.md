---
paper_id: "belcour-2018-efficient-rendering-layered-materials"
title: "Efficient Rendering of Layered Materials using an Atomic Decomposition with Statistical Operators"
authors: "Laurent Belcour"
year: "2018"
venue: "ACM Transactions on Graphics 37(4), Proceedings of SIGGRAPH 2018, Article 73"
doi: "10.1145/3197517.3201289"
report_status: "evidence-reviewed"
main_source: "https://hal.science/hal-01785457v3"
supplemental_status: "available"
official_code_status: "audited"
official_code_commit: "not-applicable; official HAL archive locked by hash"
author_worker: "/root"
reviewer: "/root/belcour2018_review"
last_verified: "2026-08-29"
---

# Efficient Rendering of Layered Materials using an Atomic Decomposition with Statistical Operators

## 1. 研究对象与报告边界

Belcour 研究的是平行层状材质的快速近似：不显式追踪所有界面间路径，而把反射、折射、体吸收和体散射写成作用于方向分布统计量的 atomic operators（原子算子），再用 adding-doubling 合并任意数量的层。运行时把每组统计量重新实例化为一个 GGX lobe，最终得到 GGX mixture，可求值也可按 mixture 采样。[P Abstract, §§1,3–6]

它是 `local-material transport` 的解析/统计 baseline，不是 neural network，也不是 source-native GT。其正式 domain 是宏观平面平行、微观 isotropic GGX、入口和出口可视为同一着色点的 geometric-optics layered BSDF。参与介质原型只处理 optically thin、强前向 HG 的 single scattering；空间扩散、BSSRDF 位移、非平行界面和一般 diffuse interface 不属于已实现方法。[P §§1,3,4.4,7.3–8]

本报告覆盖正式 TOG 版本、8 页 supplemental、作者项目页与 talk、官方 HAL code/data archive，并重点回答它作为 Neural Layered BRDFs、MetaLayer 和当前 NeuralShading 层栈研究的 load-bearing baseline 到底近似了什么、成功依赖什么、已知哪里失败。报告不会把 Belcour 的统计摘要称为层栈 reference，也不会把后续 neural residual、compiler 或 sampler 设计反向归因给 2018 论文。

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | [official HAL v3 record](https://hal.science/hal-01785457v3)，项目页的正式 paper 入口；归档的官方 HAL `belcour2018.pdf` | 2026-08-29 | SHA-256 `62262BABBB613FE66F65F021CB121BD6DF89908269017A65C9C1B159B75045ED`；16 个 PDF 物理页，其中第 1 页为 HAL cover、正式论文 15 页 | TOG 37(4) Article 73 正式方法、结果、限制与 Appendices A–E |
| Supplemental `S` | 作者[项目页](https://belcour.github.io/blog/research/publication/2018/05/05/brdf-realtime-layered.html)指向 HAL v2 `suppl.pdf` | 2026-08-29 | SHA-256 `3B628A8155BFC32BA17802A593BE02A01F71715024F11FC2A264D81410FF704A`；8 页 | atomic operators sweep、TIR ablation、Forward/Symmetric/WW07/LayerLab 对照及 Roadster 参数 |
| Official code/config/data `C` | 项目页指向 HAL v3 `suppl.zip` | 2026-08-29 | ZIP SHA-256 `895CF107DAB6D06680CB80F302CEAA9A3D7C0F1B73E7A67BE8596DF978AEF327`；MD5 `D446AB4042DAF81B0C903D877A0F6362`；无 git commit | Mitsuba plugins、Fig.12/13 XML、`FGD.bin`、`TIR.bin`、WebGL operator validation；静态审计，不等同于 production real-time engine source |
| Author page/talk `A` | [official project page](https://belcour.github.io/blog/research/publication/2018/05/05/brdf-realtime-layered.html)、[official slides/talk transcript](https://belcour.github.io/blog/slides/2018-brdf-realtime-layered/slides.html) | 2026-08-29 | stable URLs | 书目信息、公开附件边界、方法动机、Unity/GTX 980 实时演示语境 |
| NeuralShading evidence `N` | `docs/contracts/scattering_backend.md`、`docs/research/experiment_framework.md`、本任务的 NLB/MetaLayer/Guo 报告与 NVIDIA correspondence | 2026-08-29 | repo-local | 只用于 §§13–15 的项目分析，不回填为论文事实 |

2026-08-29 直接访问 HAL 文件端点受到 Anubis 拒绝；本地 `P/S/C` 是 Internet Archive 对同一官方 HAL 文件 URL 的公开快照传输，不是第三方改写版。正式 identity 由作者项目页、HAL record、DOI 和文件内 TOG metadata 交叉锁定。未使用账号、SSH、私有令牌或凭据登录。

Code archive 没有 git history、release tag、compiler manifest 或明确 license 文件；因此只能以 ZIP hash 固定内容，不能声称对应某个 commit，也不能从“可下载”推断任意再分发授权。作者项目页还提供 video 和 slides；本报告使用 slides 的方法/硬件语境，没有把视频画面当成数值证据。

## 3. 原论文的问题、假设与贡献边界

### 3.1 问题与假设

精确 layered BSDF 包含无限多条在界面间往返的路径。完整 stochastic evaluation 对实时渲染太贵，而高维 per-material tabulation 会阻碍纹理化参数和编辑。作者选择的压缩对象不是材质参数，而是每组路径形成的方向分布：在正交投影到切平面的圆盘中，只追踪能量 `e`、二维均值 `μ` 和一个各向同性标量方差 `σ`。[P §§1,3, Fig.3]

正式假设包括：

1. 各层在宏观上平面、平行，入口/出口横向位移可忽略；
2. 界面是 geometric-optics isotropic GGX microfacet，允许 dielectric 或 conductor；
3. 低到中等 roughness 下，相关方向分布可由 GGX lobe 的前三类统计量近似；
4. adding-doubling 合并的 lobe 具有相同或足够接近的 mean，因而方差可作能量加权合并；
5. 介质原型是均匀、不发光、optically thin，phase 为强前向 HG，single scattering 占主导；
6. 实时 engine 已有 GGX area/environment light preintegration，可消费少量 GGX lobes。[P §§3–6]

### 3.2 作者声明的贡献

- 为 reflection、refraction、absorption、scattering 建立作用于 `(e,μ,σ)` 的 atomic operators；
- 把 classical adding-doubling 从能量扩展到方向方差和相应的 transmission scaling；
- 将每个被合并的 path group 映射为 GGX lobe，形成可编辑、无需 per-material BSDF bake 的 layered BSDF；
- 提供适合 forward/path/real-time 的 Forward model，以及为 BDPT/MLT 构造的 ad-hoc Symmetric model；
- 对 lobe mixture 给出 energy-proportional selection、visible-normal sampling 与 balance-heuristic mixture PDF；
- 在 Mitsuba 和 commercial real-time engine 中展示 textured layers、multiple inter-interface scattering、能量保持和有限的 participating-media 支持。[P Abstract, §§1,6–7]

“arbitrary number of textured layers”是 offline analytic construction 的层数能力，不表示所有类型的 interface/volume 都任意，也不表示 realtime variant 无固定上限。正式实时实现明确限制为 3 层、2 个 outgoing lobes，且第 2 个 interface 固定为 participating medium。[P §7.2]

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Source/material input | 有序的 rough interfaces 与可选 homogeneous slabs；每界面 `η+iκ, α`，介质 `h,σ_a,σ_s,g` | 平面平行；isotropic GGX；介质 HG | [P §§1,3–5; C `mitsuba/plugins/layered.hpp:62-248`] |
| Runtime query | 给定 incident/view direction，构造 layered BSDF 对 outgoing/light direction 的 response；Mitsuba plugin 提供 `eval/sample/pdf` | reflection；`layered_dielectric` 另含 transmission | [P §6; C plugin entry points] |
| Direction coordinates | 方向 `ω` 正交投影到切平面，`[u,v]=[ω_x,ω_y]`；均值在投影圆盘中 | `u²+v²≤1` | [P §3, Fig.3] |
| Statistical state | energy `e`、二维 mean `μ`、scalar isotropic variance `σ`；论文故意用 `σ` 而非 `σ²` 记方差 | 每个 forward/backward reflection/transmission field 一组 | [P §3, Table 1, footnote 1] |
| Output quantity | 多个 GGX BRDF/BSDF lobes 的能量加权和；每 lobe 参数由 `e_k, μ_k, σ_k` 映射 | offline 可保留每层一个 lobe；realtime 2 lobes | [P Eqs.39–40, §6] |
| Validity restrictions | low/moderate roughness、平行界面、radially symmetric statistics；介质 single scattering/forward HG | 不是一般 BSSRDF、diffuse layer 或 arbitrary NDF | [P §§7.3–8] |

论文中 `ω_i/ω_o` 与本项目 ABI 的 view/light 命名不能只靠字母对应；项目接入时必须按实际渲染器 convention 校验。Mitsuba 0.5 的 `eval` ABI 还包含其 cosine-weighted convention，不能把 release plugin 返回值未经 adapter 就标成项目要求的 bare linear `f`。[C `layered_forward.cpp:287-341`; N scattering ABI]

## 5. Representation、atomic operators 与完整数据流

### 5.1 总数据流

```text
layer parameters + incident direction
  → 在投影方向域初始化 (e=1, μ=project(ωi), σ=0)
  → 为每个 interface/slab 建立 R12/T12/R21/T21 的 energy、mean、variance operator
  → adding-doubling 合并当前 virtual slab 与下一层，并解析求和内部往返路径
  → 每加入一层，记录“到该层并返回”的一个 reflection lobe
  → (e_k, μ_k, σ_k) → incident/fake direction + α_k=f⁻¹(σ_k) + energy coefficient
  → 求和 GGX lobes得到 evaluator；按能量选择 lobe并以 mixture PDF做 sample/pdf
```

该表示没有 learned latent。持久化容量主要是源层参数、公共 `FGD∞`/TIR lookup tables 和代码；每个 query 的 view-conditioned state 是 adding-doubling 产生的少量 lobe coefficient/roughness/direction。[P §§3–6]

### 5.2 方向统计与 roughness↔variance 映射

作者在投影圆盘上用 directional distribution 的零阶、一阶、二阶统计表示 energy、mean、isotropic variance。对一个 GGX response，能量由 integrated Fresnel directional albedo `FGD(ω_i,α,η+iκ)`给出。因为 GGX convolution 后不仍是精确 GGX，作者用等效 roughness 近似其 variance，并拟合线性化映射：

\[
f(\alpha) \simeq \frac{\alpha^{1.1}}{1-\alpha^{1.1}},
\qquad
f^{-1}(\sigma)=\left(\frac{\sigma}{1+\sigma}\right)^{1/1.1}.
\]

这样连续两次 rough reflection 的方差可近似相加：`σ_o=σ_i+f(α)`。Fig.4 与 Appendix A 表明线性关系最可靠的是小 roughness；Appendix E/Fig.21 的多层 Monte Carlo 核对在 roughly `α∈[0,0.5]` 较近，高 roughness 明显偏离。[P Eqs.5–6, Figs.4,20–21, Appendices A,E]

正文 Eq.6 和 WebGL scalar validation 都使用指数 `1.1`；但 official Mitsuba archive 的 `layered.hpp:38-52` 把 `USE_BEST_FIT` 注释掉，默认实际执行 `f(a)=a/(1-a)` 及其逆，而不是上式。archive 没有 build manifest，无法排除作者正式渲染曾由外部 compiler define 打开该宏，因此本报告把 Eq.6 作为正式 paper identity，把 release C++ 默认值登记为 `paper-code-gap`，不据此反推 Table 2/Fig.22 使用了哪个分支。[P Eq.6; C `layered.hpp:38-52`, `webgl/shaders/library/covariance.shader:9-38`]

### 5.3 四类 atomic operators

| Operator | Energy | Mean | Variance | 关键近似/边界 | locator |
|---|---|---|---|---|---|
| Rough reflection | `e_R=e_i·FGD∞` | `μ_R=-μ_i` | `σ_R=σ_i+f(α)` | `FGD∞`含 microfacet multiple scattering 的 directional albedo；shape仍用单 GGX等效 | [P Eqs.7–9, §4.1] |
| Rough refraction | `e_T=e_i(1-FGD∞)` | `μ_T=-η_12 μ_i` | `σ_T=σ_i/η_12+f(sα)` | 用 reflection fake transmission；`s=1/2[1+η_12〈ω_i,n〉/〈ω_t,n〉]`；高 roughness/掠射 deformation不能由radial lobe精确表达 | [P Eqs.10–13, §4.2, Appendix B] |
| Volume absorption | `e_T=e_i exp[-σ_t h/〈ω_o,n〉]` | `μ_T=-μ_i` | Table 1 为 `σ_T=σ_i` | attenuation在mean direction求；Eq.17 却印成 `σ^R=σ_i`，上下文是 transmitted variance，属于正文符号不一致 | [P Table 1, Eqs.14–17, §4.3] |
| Volume scattering | `e_T=e_i[σ_s h/〈ω_o,n〉]exp[-σ_t h/〈ω_o,n〉]` | `μ_T=-μ_i` | Table 1 为 `σ_T=σ_i+σ_g`，`σ_g=[(1-g)/g]^0.8/(1+g)` | Eq.24 同样印成 `σ^R=σ_i+σ_g`；optically thin、single scattering、strong forward HG；`g<≈0.7` 的 backscatter会形成第二个 mode | [P Table 1, Eqs.18–24, §4.4, Fig.8] |

`FGD∞`是 Heitz 等 stochastic microfacet multiple-scattering directional albedo，不是普通 single-scatter Fresnel。offline table 参数为 elevation、roughness、complex IOR 的 4D grid。反射/折射 operator 仍把输出 shape 约为 GGX，所以 energy correction 不等于完整 angular distribution exactness。[P §§4.1–4.2]

折射 operator 还有不能静默合并的 `P↔C↔A` 差异。正文定义 `η_12=η_1/η_2`，Eq.10 为 `s=1/2[1+η_12〈ω_i,n〉/〈ω_t,n〉]`，Eq.13 为 `σ_T=σ_i/η_12+f(sα)`。official talk slide 则逐字显示 `σ_t=η_12 σ_i+s`，同一 slide 没有重新声明 ratio convention，且把 roughness contribution 简写成 `s`，所以只能作为不可执行的作者说明冲突，正式方法以正文为准。release Mitsuba Forward/Symmetric 又令 `_cti=_ctt=1`，并用 `0.5|1-η_12 c_i/c_t|` 型 roughness scale；代码注释明确这是因为正文的 angular scale 在 grazing 会 overblur。`layered_dielectric` 保留实际 `cti/ctt`，但仍使用这个绝对差形式。以上是三个可定位 identity，不能选一个冒充全部实现。[P §4.2, Eqs.10–13; A talk “Statistical Analysis: Framework”; C `layered_forward.cpp:163-170`, `layered_symmetric.cpp:163-170`, `layered_dielectric.cpp:162-166`]

### 5.4 Adding-doubling、TIR 与 lobe construction

对上下两个 slabs，论文先对反射/透射 energy 使用 classical adding equations。例如上行总反射为

\[
r_{13}=r_{12}+\frac{t_{12}r_{23}t_{21}}{1-r_{23}r_{21}},
\]

其分母解析求和所有内部往返。方差 operator 是 affine 的；当被合并分布 mean 相同/近似相同时，方差可按 energy 加权，内部多次反射的方差形成 arithmetico-geometric series，利用 `Σ k r^k=r/(1-r)^2` 得到闭式。算法从空 virtual layer 开始，自上而下加入界面，持续更新双向 `R/T` 的 energy、variance 和折射 Jacobian。[P Eqs.25–31,36–38, §5]

对 homogeneous participating medium，正文还给出 classical doubling 路径：先取极薄 slab（typical `h=10^-8`），把同一层与自身反复 doubling，直到达到目标 depth；每次仍用 Eqs.28–31 更新 `R/T`。这与“按界面逐层 adding”是同一 algebra 的另一用法，不能只凭 release parser 的单次 `depth` branch 推断完整正文算法。[P §5.1]

普通 adding-doubling 会漏掉下行分布被上层界面 total internal reflection 后返回的能量。作者把 `(1-F)D` 的方向积分从 transport integral 解耦为 3D table，并按正文定义更新：

\[
r_{21}\leftarrow r_{21}+(1-\mathrm{TIR})t_{21},\qquad
t_{21}\leftarrow \mathrm{TIR}\,t_{21}.
\]

这里变量名 `TIR` 实际是该预计算积分保留下来的 transmission fraction；不能仅按名称把两项互换。官方 code `layered_forward.cpp:190-199` 与 Eqs.34–35 一致。[P Eqs.32–35, §5.1; C]

Forward model 对每个加入的界面记录一个 `(e_k,μ_k,σ_k)`，将其映射成 `α_k=f⁻¹(σ_k)` 的 GGX lobe，并按 Eq.39 求和。论文说可以把方差/能量相近的 lobes merge，但正式实现保留全部 lobes。实时 variant 才显式裁为 2–3 lobes。[P §6.1, Eq.39]

Symmetric model 为 BDPT/MLT 需要的 reciprocity 构造一个 ad-hoc symmetrization：假设 incoming/outgoing directions 共享 microfacet/difference-vector `θ_d`，在 half-vector normal 上注入 adding-doubling roughness，并以 `cos θ_d`评估 FGD。作者明确它不保证与 Forward 的统计完全相同，但实验中往往更接近 stochastic reference。[P §6.2; S §6]

### 5.5 Evaluator、sample 与 mixture PDF

每个 lobe 使用 GGX visible-normal sampling。仅按能量随机选择一个 lobe、然后把其余 overlapping lobes 从 proposal/evaluation 权重中省掉会产生 fireflies。令 `e_all=Σ_i e_i`，`p_i` 是第 `i` 个 vNDF sampling density，则实际 mixture proposal 为

\[
q(\omega_o)=\sum_i \frac{e_i}{e_{all}}p_i(\omega_o).
\]

正文 Eq.40 写的不是单独的 PDF，而是用 balance heuristic 后的完整 sample contribution：

\[
W(\omega_o)=\frac{e_{all}}{\sum_i e_i p_i(\omega_o)}
             \sum_i e_i\rho_i(\omega_i,\omega_o,\alpha_i)
          =\frac{\sum_i e_i\rho_i}{q(\omega_o)}.
\]

官方 `layered_forward.cpp:344-470` 和 `layered_symmetric.cpp:348-473` 静态体现了按平均 RGB energy 选 lobe、汇总所有 lobe sampling densities 为 `q`，最后返回 cosine-weighted evaluator/`q` 的路径。`layered_dielectric` 的 reflection/transmission 两个 outgoing supports 分居半球，sampling实现不同，README也把它限定为 dielectric microfacet interfaces；该路径不能由 opaque mixture 的 Eq.40 correspondence 自动证明。[P Eq.40, §6.1; C README]

## 6. 数据、GT/reference 与 query/sampling recipe

本论文无训练数据。验证来源是 parameter sweeps、作者场景与一个 stochastic layered-structure implementation。

| 项 | 具体配置 | locator |
|---|---|---|
| GT/reference | Appendix D：在 layer structure 中逐 interface 随机追踪路径；official `layered_ref` README 明确它只实现 sampling procedure，只能搭配 `direct` integrator 且 `emitterSamples=0` | [P Appendix D; C `mitsuba/README.txt`; C `layered_ref.cpp`] |
| Reflection/refraction sweeps | smooth/rough coating；`η=1.05,1.1,1.2,1.5,2,3`，base/interface roughness `0,.01,.02,.1,.2,.4,.6` 等网格 | [S §§1–2] |
| Scattering sweep | depth `d=1`、`σ_a=0`，vary `σ_s` 与 HG `g=.5,.6,.7,.8,.9,.99`；reference也只计 single scattering | [S §4] |
| Optical-depth sweep | mirror base，`σ_a=0,σ_s=1,g=.9`，`h=.02,.04,.1,.2,.3,.4`，environment lighting | [S §4] |
| Fig.13 public archive example | 2层：dielectric `η=1.49,α=.1` + conductor `η=1,κ=[1,.1,.1],α=.001`；ours `16` BSDF + `16` emitter samples，reference `32` BSDF + `0` emitter samples；camera `4` spp、512²。README称其复现 Fig.13，但该公开预算不等于正文所述 1024 spp | [C `frosted_metal_ours/ref.xml`; P Table 2] |
| Fig.12 public archive example | 6 interfaces：5个 smooth dielectric 交替 `η=1.33/1.0`，pure mirror base `η=0,κ=1`；constant radiance `.5`；direct integrator、256 BSDF samples、camera 4 spp、512²。正文“1024 spp”如何映射到两级 sampler 未报告 | [C `white_furnace_ours/ref.xml`; P Table 2] |
| Main render protocol | Fig.13/12 各 1024 spp、direct；Fig.14 512 spp、path；硬件 16-core i7、32 GB RAM、GTX 980 | [P §7, Table 2] |
| Train/val/test split | 不适用；正式 scene/sweep 不构成学习 split | [P/S full] |
| Filtering/LOD | 未报告；这不是 spatial neural material/texture LOD 方法 | [P full] |

official reference 的 `eval` 注释明确“不考虑 media scattering”，而 `sample` 中的 media multiple-scattering macro 默认关闭，并在反向散射时直接返回零以保持 forward single-scattering prototype。因而它不是一个可任意替换进 integrator 的通用权威 `evaluate/sample/pdf` 三元组；作者 README 的 direct-integrator 限制是复现的必要配置，不是可忽略的示例默认值。[C `layered_ref.cpp:238-240,411-513`; C README]

## 7. Loss、optimizer 与训练 lifecycle

本论文没有 neural fitting、loss、optimizer、learning-rate schedule、batch、steps、seed selection 或 checkpoint。把 lookup table 的 numerical precomputation 或 roughness fit 称为“training”会混淆方法身份。

| 项 | 正式配置 | locator |
|---|---|---|
| Roughness transform fit | 由一次/两次 GGX bounce 的数值方差拟合 `α↔σ`；完整采样数、optimizer、seed 未报告 | [P Appendix A, Fig.20] |
| `FGD∞` precompute | 4D `64^4` grid；`t,α∈[0,1]`、`η,κ∈[0,4]`；float32 data | [P §7; C binary header/`layered.hpp:275-400`] |
| TIR precompute | 3D `64^3` grid；`t,α∈[0,1]`、IOR ratio `[0,4]`；float32 data | [P §7; C binary header/`layered.hpp:403-515`] |
| Precompute algorithm/time | WebGL `compute_FGD/TIR` demo/source可得；正式 sample count、seed、convergence criterion、生成硬件与耗时未报告 | [C WebGL; P §7] |

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime call path | query direction → adding-doubling over layers → lobe list → GGX mixture eval/sample/pdf | [P §§5–6; C plugins] |
| Network params/MAC | 不适用；无 network | [P full] |
| Shared storage | offline `FGD.bin=67,108,912 B`（header + `64^4` float，约64 MiB）；`TIR.bin=1,048,612 B`（header + `64^3` float，约1 MiB） | [P §7; C binary audit] |
| Per-material storage | layer parameters/textures；无 parameter-dependent BSDF bake；确切 bytes取决于源 texture | [P §§1,7.1] |
| Texture fetches | 4D linear FGD interpolation最多16邻点，3D TIR最多8邻点；每加入层会使用 operator/table；正式GPU fetch/cache计数未报告 | [C `layered.hpp:317-386,440-510`; I static derivation] |
| Precision | binary tables float32；Mitsuba build和commercial shader precision未报告 | [C binary/layout; P unreported] |
| Offline hardware | 16-core i7、32 GB RAM、NVIDIA GTX 980；CPU/GPU分工和型号细节未完整报告 | [P §7] |
| Offline time | ours：Fig.13 `46s`（无table interpolation `24s`），Fig.12 `1.8m`（`35s`），Fig.14 `1.84m`（`1.64m`）；WW07 `17s/20s/1.60m` | [P Table 2] |
| Real-time configuration | commercial forward engine；3 layers、2 outgoing lobes，第2 interface固定 participating medium；1920×1080 fullscreen | [P §7.2; A talk] |
| Real-time time | layered shader/full frame `1.9–2.1 ms`；engine standard shader `1.7–2.0 ms`；talk把演示机标为 GTX 980 | [P §7.2; A talk] |
| Included/excluded | full-frame timing，未分离 material evaluation、lighting preintegration、raster/driver；table precompute和authoring不在frame time | [P §7.2; I] |

offline “arbitrary number of layers”意味着 loop 成本随层数增长，且 mixture保留每层 lobe；它在给定 scene/material 后有限，但不是跨任意 layer count 的固定 shader budget。实时 3-layer/2-lobe variant 才具有直接静态上界。论文没有报告 per-query latency、instruction count、register/state bytes、cache hit 或 wave coherence，不能用 full-frame 1.9–2.1 ms 推导本项目单次 `evaluate()` 成本。

## 9. 实验 protocol、baseline、指标与结果

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| Five two-layer appearances | symmetric model vs stochastic reference；环境球图；每例参数列在图下 | stochastic reference | image RMSE `Δ`，定义/色域/aggregation未进一步报告 | Metal Foil `.010`；Metallic Paint `.015`；Frosted Metal `.016`；Rough on Rough `.015`；Gold Coated `.025` | [P Fig.22, Appendix E] |
| Inter-interface color saturation | coated gold sphere；比较作者完整model、只保留`R+TRT`的restricted reference与含所有内部往返的stochastic reference | 两种stochastic reference path sets | visual only | 完整model接近包含`TR^+T`的reference饱和度；只计一次内反射的reference较不饱和 | [P Fig.11, §7.1] |
| Energy conservation | 5层 dielectric交替IOR + mirror base；official config实际 `nb_layers=6`（含base interface） | stochastic reference、WW07 | visual white furnace；无numeric residual | ours保持亮度，WW07因漏失 inter-layer multiple scattering变暗 | [P Fig.12; C XML] |
| Frosted Metal | rough dielectric over smooth conductor | stochastic reference、WW07 | visual；Table 2 time | ours传播上层roughness到base，WW07未能重现 | [P Fig.13, Table 2; C XML] |
| Participating medium | gold dragon + forward medium，`g=.7,σ_a=0,σ_s=1`，vary optical depth | stochastic single-scattering reference | visual | rougher/darker趋势接近；只验证single scattering | [P Fig.14; S §4] |
| Textured parameters | Plates/Robot，2–3 layers，多参数 texture mapped | visual only | editability/appearance | on-the-fly处理 textured layer combinations，无 per-material BSDF bake | [P Figs.15–16, §7.1] |
| LayerLab cost examples | 5 appearance files；LayerLab使用Beckmann，不能严格match GGX reference | LayerLab/Jakob 2014 | precompute storage/time | `2.4GiB/22m`, `1.7GiB/21m`, `40MB/10s`, `6.8MB/2s`, `6.7MB/4s`；正文另称 Metallic Paint generation 峰值最多 `15.5GB` RAM | [P §7.1; S §6] |
| Roadster authoring showcase | 3个layered paints；公开参数包括 copper `η=[.27+3.6i,.98+2.37i,1.33+2.3i]` + dielectric `η=2,α=.1`，以及 red conductor `[1+i,1+.1i,1+.1i],α=.1` + smooth dielectric `η=1.5` | visual only；无stochastic side-by-side | editability/appearance | 展示 car paint、coated copper/chrome；第三套chrome的完整数值参数未报告 | [S §7] |
| Real-time | fullscreen 1080p commercial engine，same adding code但Schlick/split-sum与preintegrated lighting | offline Forward、stochastic reference、standard shader | full-frame ms、visual | appearance接近；layered `1.9–2.1ms` vs standard `1.7–2.0ms` | [P Fig.17, §7.2] |
| Roughness failure | dielectric `η=2` over conductor `η=.01+i`，both `α=.3,.6,.9`；隔离inter-layer approximation，关闭microfacet geometry multiple scattering | stochastic reference | visual | roughness越高偏差越大，`α=.9`显著 | [P Fig.19, §7.3] |

这些结果不能合并为统一 Pareto 排名：LayerLab 使用不同 NDF 和 precompute产品；WW07、Belcour、stochastic reference解决的成本域不同；real-time 与 offline 的 Fresnel、lighting integration 和硬件路径也不同。尤其 NLB 后续 correction 的 equal-time Belcour对照应以 NLB correction 为准，不能拿本论文 CPU Table 2 或 1080p engine frame time替代。[N downstream NLB report]

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释 `[I]` | locator |
|---|---|---|---|---|---|
| `author-negative` | mixture按能量选一个 lobe，只用单proposal而不汇总其他lobe PDF | overlapping lobes产生fireflies | proposal overlap未在权重中体现 | matched sampler必须评估完整mixture PDF；这不是可省略的渲染技巧 | [P §6.1] |
| `ablation-inferior` | 不使用 TIR factor | TRT等短路径被高估；但construction仍energy conserving | upper-interface TIR energy未反馈到下行transport | energy conservation test不足以证明angular/path decomposition正确 | [S §5] |
| `author-negative` | Forward model在掠射与reference比较 | Fresnel被高估 | 使用average Fresnel | forward view-conditioned统计与symmetrization不能混作同一模型身份 | [S §6] |
| `known-limitation` | 多层 high roughness (`α=.3,.6,.9`) | roughness上升后显著偏离reference | GGX shape和roughness→variance线性化只对低/中roughness可靠 | 若用作teacher/control，必须按roughness stratify，不能用全域平均掩盖失败区 | [P Fig.19, §7.3] |
| `known-limitation` | HG×GGX，`σ_s=.01` sweep | approximation不能复制heavy tails | 单一等效GGX/variance不能保留乘积的尾部 | 少量moment不是高动态范围target的充分统计量 | [S §4] |
| `baseline-inferior` | WW07 | 漏掉上层roughness传播与inter-layer multiple scattering；能量/色饱和有偏 | 层间效果过度解耦 | 是specific baseline failure，不证明Belcour在所有domain exact | [P Figs.12–13; S §6] |
| `baseline-cost` | LayerLab tabulation | clear-coat类case可达2.4GiB/22min；texture轴导致存储爆炸 | per-material high-dimensional precompute | 说明shared operator/table的编辑优势，不等于单query更快 | [P §7.1; S §6] |

第一方材料没有报告先尝试 neural compression、不同 optimizer 或 residual network 后失败；不得从 2018 最终解析设计虚构这些历史。作者在 Discussion 中提到未来可拟合4D/3D regular tables，但这只是 future direction，不是已做失败实验。[P §§8–9]

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Layer schema | plane-parallel isotropic GGX；surface或media operator | sweeps覆盖coating、scattering、TIR | `nb_layers`；每层`eta/kappa/alpha`或`depth/sigmas/sigmaa/g`；`depth>0`会传播上一IOR并把`alpha`替为HG variance | code与paper operator身份对应；release parser用depth判断media，不能同时表达同层surface+volume复合 |
| Volume operator separation | absorption保持variance；scattering才增加`σ_g` [Table 1] | absorption/scattering分别验证 | `parseLayers`对任意`depth>0`都把`alpha`覆写为`gToVariance(g)`；三种plugin的media branch随后无条件令`s_t12=s_t21=alpha`，即使`σ_s=0` | `paper-code-gap`；release Mitsuba没有保持pure-absorption variance的独立branch，且公共XML未覆盖media，未动态量化影响 |
| FGD/TIR | `64^4≈64MB`、`64^3≈1MB`，linear interpolation | TIR ablation | binary headers确为`(64,64,64,64)`和`(64,64,64)`；范围与float layout已核 | 对应；文件大小含header，分别67,108,912与1,048,612 B |
| Roughness/HG | Eq.6用`α^1.1/(1-α^1.1)`；Eq.21 HG variance | WebGL scalar helper使用1.1；HG sweep含`.5–.99` | Mitsuba `USE_BEST_FIT`默认关闭，实际为`α/(1-α)`；`gToVariance` assert `g>0` | roughness map是`paper-code-gap`，正式build flag未知；code不支持`g≤0`，与paper focus forward HG一致，但不是一般HG implementation |
| Refraction variance | Eqs.10–13给出正式`η_12`、`s`和`f(sα)` | fake-refraction grid | Forward/Symmetric把角度固定为normal后用absolute-difference scale；dielectric保留角度但仍用该scale | `paper-code-gap`；代码注释把改动归因为grazing overblur，不能把release默认当正文逐式实现 |
| Author refraction slide | Eq.13：`σ_T=σ_i/η_12+f(sα)` | 未新增 | talk逐字为`σ_t=η_12σ_i+s` | `author-paper-gap`；talk未在该slide重述ratio convention且是示意，formal identity以P为准 |
| Reference | Appendix D stochastic trace | 所有GT称stochastic reference | README规定`layered_ref`只作sampling、direct integrator、zero light samples；`eval`注释不支持media | 不能把release reference当作通用eval/sample/pdf oracle |
| Bounce cap | formal随机reference未给统一cap | 未报告 | plugin default `max_bounces=20`；Fig.12/13 XML都设`100` | formal result cap必须从scene config锁定；default不等于paper配置 |
| FGD/TIR enable |正式offline结果使用tables | TIR on/off ablation | Forward/Symmetric/Dielectric default均`false`；ours XML显式为`true`；ref XML虽携带同名key，但`LayeredReference`不读取它们 | analytic plugin复现必须显式enable；不能把reference XML中的无效key当作reference使用table |
| TIR application | Eqs.34–35把预计算积分作为保留transmission fraction | TIR on/off ablation | Forward/Symmetric与P一致；`layered_dielectric.cpp:195-198`却执行`Ri0+=TIR·Ti0, Ti0*=1-TIR` | code archive内部冲突；不能用reflection plugin correspondence替代transparent plugin验证 |
| Lobe sampling | energy selection + vNDF + full mixture balance PDF | 未新增 | Forward/Symmetric eval/pdf/sample静态对应；dielectric plugin sample返回selected-lobe weight path，不与opaque代码完全相同 | opaque路径对应较清楚；transmission需独立sample/pdf验证，paper未给raw normalization test |
| Symmetry | ad-hoc Symmetric，用于BDPT/MLT | usually closer；Forward掠射Fresnel高估 | separate `layered_symmetric.cpp` | 两个model是不同evaluation identity，不能只换query order宣称reciprocity |
| Real-time | 3层/2lobes、commercial engine、split-sum Fresnel/preintegrated lighting | Roadster参数 | real-time engine shader未在archive；只有Mitsuba+WebGL operator code | 论文实时结果不能由公开archive端到端复现 |
| Eq.17/Eq.24 | Table 1与段落语义均为transmitted variance | 未勘误 | release media branch把HG variance无条件加入所有`depth>0`层 | 两式都写`σ^R`而非上下文`σ^T`；此外C没有忠实分开pure absorption/scattering，两个问题分别保留 |
| Public result XML | Table 2称Fig.12/13为1024 spp | 未新增 | Fig.13 XML为camera 4、ours 16 BSDF/16 emitter、ref 32 BSDF/0 emitter；Fig.12为camera 4、256 BSDF | README“reproduces”不等于formal immutable config；Fig.13尤其存在明确budget gap |
| Mitsuba ABI/build | paper只描述Mitsuba implementation | 未新增 | scene version `0.5.0`；`eval`实现返回cosine-weighted quantity，且大量`T x[nb_layers]`依赖VLA compiler extension | 接本项目需bare-`f`/measure adapter；archive无exact Mitsuba revision、SCons patch或compiler flags，未动态验证 |

official archive包含5个 Mitsuba plugin/header源文件、两个结果场景对、FGD/TIR data和WebGL validation，但没有 exact Mitsuba revision、SCons patch、third-party dependencies、production engine shader、build output或raw reference images。README要求把文件手工移动到 Mitsuba `src/bsdfs` 并修改SCons；本 author pass 只做静态审计，没有 build/run，不能宣称动态复现成功。

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

- **roughness**：统计/GGX approximation对smooth surfaces可靠，高roughness显著偏离；Fig.19直接展示 `.3→.6→.9`退化。[P §7.3]
- **color shift/TIR**：强色散式金属Fresnel在掠射改变color，方法不能精确保留multiple-scattering tint；Frosted Metal漏掉绿色tint。[P §7.3, Fig.22]
- **media multiple scattering**：prototype只有single scattering；要表现dust retroreflection需增加retro-reflective direction/state。[P §7.3]
- **interface family**：只能统一使用一种NDF；不能在同一stack混GGX/Beckmann/GTR/Student-t，也不能准确表达Lambertian layer。[P §7.3]
- **non-parallel interfaces**：production中给coat/base不同normal可运行，但对inter-layer multiple scattering物理上不正确。[P §7.3]
- **grazing transmission**：Fresnel extinction令transmitted lobe skewed/anisotropic，radially symmetric statistics不能表达。[P §7.3]
- **anisotropy**：正式实现只追踪isotropic scalar variance；作者估计扩展需从4个scalar state增至8个。[P §8]
- **spatial diffusion/BSSRDF**：不追踪横向位置/空间moment；仍是同点BSDF approximation。[P §§2,8]
- **complex appearance correlation**：可替换FGD以加入iridescence或glint，但离散glint与上层Fresnel的correlation会丢失。[P §8]

### 12.2 未报告/材料不可得

- roughness fit、FGD/TIR precompute 的sample count、seed、误差、运行时间和生成precision；
- Fig.22 RMSE的color space、tone mapping前后、pixel mask、聚合公式和reference noise；
- 每个正式scene的完整immutable assets、raw HDR outputs、重复runs和confidence intervals；
- stochastic reference的正式path length distribution、termination/RR语义与standard error；
- offline精确 CPU型号、thread schedule、Mitsuba commit/compiler flags；
- commercial engine名称/版本、完整shader、GPU counter、precision、frame内容和重复timing协议；
- per-query eval/sample/pdf time、register/state bytes、table cache behavior和lobe-count scaling；
- release archive license、git history、production shader/config/checkpoint（无checkpoint概念）；
- official archive formal build flags（尤其`USE_BEST_FIT`）、exact Mitsuba revision和VLA-capable compiler；
- talk折射方差公式、paper Eqs.10–13与release C++ scale之间没有第一方勘误/correspondence说明；
- release media branch对`σ_s=0`仍加入HG variance的动态影响与正式结果覆盖范围；
- README所称Fig.13复现XML与正文1024 spp之间的正式sampling映射；
- transmission plugin的sample↔pdf normalization、reciprocity residual和white-furnace numeric residual；
- Eq.17/Eq.24上标不一致的正式erratum；第一方入口未发现勘误，但不能证明不存在未公开说明。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

Belcour 2018把无限path expansion压成三部分容量：

1. **源层参数**保留可编辑的`η/κ/α/h/σ_a/σ_s/g`及textures；
2. **公共物理近似**放在roughness transform、adding-doubling algebra、`FGD∞` 4D表和TIR 3D表；
3. **per-query view-conditioned state**是每层/每group的energy、mean、variance与最终GGX lobes。

它不学习任意appearance residual，因此成本不会随训练集扩大；相应地，表达上限被GGX mixture和少量moments的充分性假设锁死。对当前项目，Belcour更适合作为 `optimized-code control`、bounded analytic proposal、compiler prior 或 auxiliary diagnostic，不是 source GT，也不是“high-capacity teacher”。[I]

### 13.2 成功所依赖的假设

1. native source本来就是plane-parallel GGX/slab family，或能在不改GT语义的前提下选择该restricted subset；
2. 大部分重要path groups在投影域近似unimodal/radial，前三类moments足以；
3. low/moderate roughness占主要工作区，heavy tails、skewness、multimodality不是质量主导；
4. FGD/TIR shared tables覆盖需要的IOR/roughness范围并有足够插值精度；
5. lobe energy是可用的mixture selection proxy，且所有lobes的PDF都可廉价评估；
6. runtime lighting pipeline能消费GGX lobes/preintegrated lights，或把lobe list作为proposal而非最终response；
7. 实时产品接受固定3-layer/2-lobe裁剪与single-scattering medium限制。[P/C; I]

### 13.3 可迁移机制与不能迁移的部分

可迁移机制：

- 把 `prepare(P,wo)`解释为一次view-conditioned operator propagation：读取源状态/latent后缓存少量lobe energies、roughnesses和fake directions，供同一着色点多次`evaluate(wi)`复用；
- 用Belcour 2/3-lobe mixture作为matched sampler baseline，始终计算完整mixture PDF，测试learned sampler是否在matched evaluator上真正降低方差；
- 给neural evaluator增加不承担最终表达的辅助moment heads或analytic low-frequency core，让网络容量集中在Belcour失败的tail、skew、multimodal residual；
- 将TIR、energy conservation、layer-order effect、high-roughness和HG-heavy-tail分别做stratified diagnostics，而不是只看aggregate RGB loss；
- 以shared FGD/TIR approximator替换大table是可证伪的deployment研究，但要保持原table control和插值domain；
- 把Belcour和Guo/current random-walk reference分层：前者是fast approximate control/proposal，后者才可在其source-family domain承担GT。

不能迁移的部分：

- 不得要求所有source material先反演为Belcour layer parameters；这会违反native source GT边界；
- `e/μ/σ`不是一般BRDF/BTF/BSSRDF的充分统计量，不能作为统一IR的强制输出；
- realtime 3-layer/2-lobe裁剪不能自动推广到当前G2/G2s的任意层参数状态；
- 2018 GTX980 full-frame time不能当现代Slang单query预算或NVIDIA MLP速度结论；
- Symmetric model的ad-hoc构造不能替代严格的sample/eval/pdf reciprocity与measure测试；
- single-scattering HG approximation不能作为当前uniform slab random-walk reference的替代GT。

### 13.4 与本项目 runtime contract 的关系

本项目要求 `prepare(context,compiledMaterial)->State`，`evaluate()`输出不含cosine的linear `f`，连续`sample()`返回与同一solid-angle `pdf()`匹配的 `f|n·wi|/pdf`，并要求读取、state和control flow静态有界。[N `docs/contracts/scattering_backend.md:3-5`]

- **offline full Belcour**：每层循环/每层lobe，给定最大层数后可有界，但原文“arbitrary number”本身不是固定budget；Mitsuba ABI还需bare-`f` adapter。分类为 `optimized-code control / analytic proposal`。
- **paper realtime variant**：3 layers、2 outgoing lobes、固定table reads，具有部署可能；但public archive缺production shader，需重新实现并做formula/table parity。分类为 `deployable analytic control`。
- **hybrid residual candidate**：允许用Belcour state/core，但residual MLP、table approximator、sampler都必须各自注册静态shape和reads；Belcour不是目标representation的强制closure词汇。

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

Belcour 2018不是 NVIDIA Real-Time Neural Appearance Models 的架构或训练规范，因此对当前NVIDIA correspondence的直接fidelity分类是 `not-applicable`。它不能用来修改`20→64→64→64→3` evaluator、`11→32→32→32→9` sampler、mollification或300k lifecycle的faithful/underspecified判定。[N archived correspondence]

| 主题 | Belcour 2018 | 当前 NVIDIA functional/formal identity | 分类与影响 |
|---|---|---|---|
| Representation | per-query GGX lobe mixture + shared tables | encoder latent + fixed small MLP evaluator | `not-applicable`；可做matched analytic control或hybrid core，不是复现替代 |
| Training | 无训练 | formal neural training/loss/schedule | `not-applicable`；Belcour failure strata可用于validation slicing，不改training事实 |
| Sampler | energy-weighted vNDF mixture + full mixture PDF | learned bounded sampler | `not-applicable`；适合作为sampler control，比较variance/time而非强制proposal identity |
| Source domain | native plane-parallel GGX/slab parameters | 当前LayerStack source adaptation及未来native families | `interface-adaptation`只在新candidate中成立；不能提升为公共source IR |
| Runtime | fixed realtime 3-layer/2-lobe variant或O(layer) offline | fixed `prepare/evaluate/sample/pdf` state | `candidate-control`；需bare-f/measure adapter和static cap |
| GT | approximate analytic model | GPU-resident online source reference | `not-applicable`；Belcour不得替换random-walk/native reference |

对当前复现最直接的价值是建立三个独立controls：相同GT下的Belcour evaluator control、相同evaluator下的Belcour proposal control、相同runtime budget下的Belcour-core+residual control。任何收益必须在当前冻结source/query recipe中以matched对照与bootstrap CI验证，不能把NLB correction里的单场景 equal-time结果当成本项目预期门槛。[N experiment framework; downstream NLB]

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| H1：Belcour analytic core + bounded neural residual在low/moderate roughness层栈上优于同MAC direct MLP | Belcour low/moderate roughness较准且high roughness/tails有系统残差 [P Figs.19,21–22; S §4] | residual比从零学习全部energy/peak更省容量 | direct MLP vs analytic-core+residual；同parameter/MAC/read cap、同online GT和training work | source/query/split、optimizer/loss、seeds、precision、state bytes、sampler | G1/G2/G2s分层error、peak/tail error、single-query time、bytes、bootstrap CI | evaluator candidate | 同budget下residual无质量—时间—内存Pareto收益，或只在Belcour easy subset改善且hard strata更差 |
| H2：把Belcour lobe state放入`prepare(P,wo)`可降低multi-light repeated-evaluate成本而不降质 | state由view direction和material决定，paper realtime只需2 lobes [P §§6–7] | 同一shading point多次`wi` query能复用coeff/roughness/direction | per-evaluate重算 vs prepare缓存；相同公式与precision | layer cap、light/query count仅作为报告轴、table/cache layout、backend、output measure | prepare time、evaluate time、state bytes、parity ULP/error | deployable analytic control | 缓存state成本抵消复用收益，或跨wi复用不成立/产生parity偏差 |
| H3：Belcour 2/3-lobe proposal在当前layered evaluator上是learned sampler的强control | paper正式用full mixture proposal与balance contribution修复single-lobe fireflies [P Eq.40] | 当前target主要modes与R/TRT/TT类lobes对齐 | cosine、Belcour mixture、learned sampler；固定同一evaluator/SPP/scenes与sample/pdf checks | evaluator、integrator、MIS heuristic、seeds、runtime precision、proposal support | variance/time、tail weights、PDF normalization、support failures、bootstrap CI | matched sampler/control | Belcour proposal相对cosine无variance/time收益，或learned sampler不能显著超过它且成本更高 |
| H4：用小型bounded approximator替换64MiB FGD表能改善memory Pareto且保持operator parity | paper称tables regular、可能高效近似 [P §9] | FGD函数在formal domain足够平滑且峰/边界可小模型表达 | exact 4D linear table vs compact MLP/polynomial；TIR表保持不变 | input domain/grid、precision、fetch/MAC budget、sampling weights、compiler/backend | max/percentile table error、white furnace、Fig.22 states、latency、bytes | shared runtime component | edge/grazing/conductor domain error破坏energy/tint，或saved bytes换来不可接受latency且无Pareto收益 |
| H5：moment-consistency辅助目标能降低compact evaluator seed variance | Belcour前三类moments对低/中roughness是有效summary；Taming表明compact网络optimization variance重要 [P; N Taming report] | 对GT response的稳定quadrature moments提供比逐query RGB更低方差的global约束 | 原loss vs 加energy/mean/variance auxiliary loss；同network、queries、steps和seed set | loss weights预先freeze、quadrature recipe、training budget、optimizer、data streams | seed success rate、G1/G2 error CI、moment residual、time | training-only auxiliary | seed variance/最终Pareto无改善，或moment bias把high-roughness/heavy-tail states拉向Belcour失败模式 |
| H6：high-roughness、grazing-skew、HG-heavy-tail三类slice能预测analytic-core candidate的失败 | 作者分别给出明确failure cases [P §7.3; S §4] | 当前source recipe覆盖同类状态且指标对tails敏感 | aggregate-only诊断 vs预冻结stratified slices；不改model | state locator、query distribution、reference SE、metrics、seeds | per-slice error/CI、aggregate masking ratio、failure localization | evaluation protocol | slices与实际worst cases无关联，或reference/query语义不同导致slice不可比 |

这些假设都是候选排序与protocol研究，不是本任务新增hard gate。H1/H5需要在提出具体candidate前再次通过 `method-constraints.md`；H2–H4必须注册固定shape、reads、precision和backend cost。H6只改变报告分层，不改变formal test data或事后选择规则。

## 16. 证据索引

### `P` Main paper

- Abstract、§§1–2：问题、per-material precompute边界、贡献与related work。
- §3、Fig.3、footnote 1：正交投影方向域、energy/mean/scalar variance与GGX mapping。
- §4、Table 1、Eqs.2–24、Figs.4–8：FGD、roughness↔variance、reflection/refraction/absorption/scattering operators；正式公式已按rendered pages视觉核对。
- §5、Eqs.25–38、Fig.9：adding-doubling energy/variance、TIR decoupling和共同mean假设。
- §6、Eqs.39–40、Fig.10：Forward/Symmetric model、lobe mixture、vNDF sampling、balance PDF与realtime裁剪。
- §7、Figs.11–19、Table 2：offline/real-time protocol、timing、texturing、energy与明确failure cases。
- §§8–9：anisotropy/spatial diffusion/complex appearance future work与table approximation建议。
- Appendices A–E、Figs.20–22：roughness fit、refraction scaling、variance algebra、stochastic reference、multi-layer validation和RMSE。

### `S` Supplemental

- §§1–3：smooth/rough coating与fake-refraction parameter grids。
- §4：single-scattering sweeps、`σ_s=.01` heavy-tail failure与optical-depth sweep。
- §5：TIR on/off unit test；no-TIR仍energy conserving但高估short paths。
- §6：Forward/Symmetric/WW07/LayerLab visual与LayerLab storage/precompute costs。
- §7：Roadster的copper/red conductor/coating参数。

### `C` Official code/config/data

- archive SHA-256 `895CF107...78AEF327`，MD5 `D446AB...F6362`；无git commit/license manifest。
- `mitsuba/plugins/layered.hpp:55-248`：HG variance、layer parser与media-over-surface schema。
- `layered.hpp:38-52`：Mitsuba默认roughness map与正文Eq.6的宏控差异；`:275-515`：FGD/TIR binary layout、range和linear interpolation。
- `layered_forward.cpp:103-267`：adding-doubling energy/variance/TIR；`:287-470` evaluator/mixture PDF/sample。
- `layered_symmetric.cpp:104-250,291-473`：Symmetric coefficients与eval/pdf/sample。
- `layered_dielectric.cpp:162-198,290-465`：reflection/transmission plugin、折射scale、与另两plugin相反的TIR application及sample路径；README声明只支持dielectric microfacet interfaces。
- `layered_ref.cpp:238-240,379-573`：reference限制、max-bounce path sampling、media single-scattering branch与approximate PDF。
- `frosted_metal_ours/ref.xml`、`white_furnace_ours/ref.xml`：README所称Fig.13/12 public example configs；不冒充正文formal immutable config。
- `FGD.bin`：`64^4`、range `[0,1]×[0,1]×[0,4]×[0,4]`；`TIR.bin`：`64^3`、range `[0,1]×[0,1]×[0,4]`。
- `webgl/`：atomic operators和FGD/TIR interactive validation；未当作formal raw metric dataset。

### `A` Author material

- official project page：作者、TOG/SIGGRAPH 2018 identity及paper/supp/code/video/slides入口。
- official talk：GTX980 realtime demo、3-layer realtime stack、statistical state直觉、high-roughness failure，以及与P Eq.13不一致的schematic refraction variance slide。

### `N` NeuralShading evidence

- `docs/contracts/scattering_backend.md:3-5`：`prepare/evaluate/sample/pdf`、bare linear `f`、solid-angle PDF、matched tuple与static runtime contract。
- `docs/research/experiment_framework.md`：冻结source/query recipe、matched control、bootstrap CI与结果登记边界。
- `research/papers/fan-2022-neural-layered-brdfs.md`：timing correction撤回faster-than-Belcour，保留corrected equal-time单场景语境。
- `research/papers/2023-metalayer.md`：Belcour-style R/TRT/TT proposal是外接analytic sampler，不是MetaNet输出。
- `research/papers/guo-2018-position-free-layered-bsdfs.md`：position-free stochastic reference与Belcour fast approximate baseline的身份边界。
- archived NVIDIA correspondence：正式evaluator/sampler shape、mollification、300k lifecycle与当前适配分类。

### `I` Derived/transfer notes

- 4D/3D linear interpolation分别最多16/8邻点、offline O(layer count)、realtime 3-layer/2-lobe static class、Belcour作为control/proposal而非GT，以及H1–H6均为本报告分析，不是作者原结论。
- archive只做static audit；没有build/run Mitsuba或commercial shader，不声称dynamic reproduction。

### 建议继续追踪的 load-bearing related

- **Jakob et al. 2014 / LayerLab**：若综合需要量化“预计算高维layer representation为何失去可编辑性”，以`key-baseline`提升；当前只能使用Belcour第一方对照中的不同NDF/成本语境，不能直接质量排名。
- **Weidlich and Wilkie 2007**：若需要追踪production clear-coat解析近似的layer-decoupling失败，以`failure-explanation`提升；当前已有明确author-negative，但尚未独立锁定WW07正文/实现。
- **Heitz et al. microfacet multiple-scattering FGD工作**：若实施compact FGD approximator或需要严格区分`FGD`与`FGD∞`，以`direct-inheritance`提升。

## Evidence review

```text
author_worker: /root
reviewer: /root/belcour2018_review
reviewed_at: 2026-08-29
sources_rechecked:
  - main PDF SHA-256 62262BABBB613FE66F65F021CB121BD6DF89908269017A65C9C1B159B75045ED, all 15 formal pages independently rendered and visually rechecked
  - supplemental PDF SHA-256 3B628A8155BFC32BA17802A593BE02A01F71715024F11FC2A264D81410FF704A, all 8 pages independently rendered and visually rechecked
  - official HAL code/data archive SHA-256 895CF107DAB6D06680CB80F302CEAA9A3D7C0F1B73E7A67BE8596DF978AEF327, all five Mitsuba source/header files, four XMLs, both READMEs and binary headers independently audited
  - official author project and full talk transcript/slides rechecked for source identity, refraction formula and realtime context
findings_closed:
  - corrected Eq.40 from a mislabeled standalone mixture PDF to the full balance-heuristic sample contribution and separated proposal q
  - added the Eq.24 R/T superscript gap alongside Eq.17
  - recorded paper Eq.6 versus Mitsuba default roughness-map divergence and unknown formal build flag
  - recorded paper, talk and release-code refraction variance/scale identities without reconciling them by guess
  - recorded Forward/Symmetric versus layered_dielectric TIR application conflict
  - recorded that the release media branch unconditionally adds HG variance even for pure absorption
  - downgraded public Fig.12/13 XML from exact formal config to README-linked examples and exposed the Fig.13 sampling-budget gap
  - preserved Mitsuba cosine-weighted ABI, direct-integrator-only reference boundary, VLA/build and no-commit/license limits
remaining_evidence_gaps:
  - no git commit/build manifest/license for official archive
  - no production realtime shader or complete engine benchmark protocol
  - no raw outputs/seeds/reference variance/RMSE definition
  - no dynamic Mitsuba reproduction in author or review pass
  - no formal correspondence for roughness macro, refraction slide/code scale, dielectric TIR branch, public XML sampling gap, or Eq.17/Eq.24 superscripts
review_status: passed-with-explicit-gaps
```

### 完成检查

- [x] main paper 已完整阅读，关键公式/图/表/图注/脚注已视觉核对；
- [x] supplemental/appendix/勘误的可用性已检查；
- [x] official code/config/data 的可用性与版本边界已检查；
- [x] representation、runtime 和主要结果均有 locator；
- [x] 失败尝试与较差消融正确分类；
- [x] paper/code gap 和“未报告”保留；
- [x] `I` 分析晚于事实层，没有改写作者结论；
- [x] NVIDIA 影响引用真实 `N` 证据；
- [x] 假设包含 matched control、部署类别和证伪条件；
- [x] 独立 evidence review 已完成。
