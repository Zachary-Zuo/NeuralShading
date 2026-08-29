# Metal reference HLSL 与 per-pixel 计算量

## 1. 结论先行

当前 viewer 的 `1 spp` 不是每个材质像素调用一次 BRDF。固定 environment-only 配置中，一个可见的 primary material hit 会发出 4 个 environment NEE 查询和 4 个 BSDF path samples；按当前 backend 的实际合同展开后，是：

- 4 次公开 `evaluate()`；
- 4 次公开 `pdf()`；
- 4 次公开 `sample()`。

公开 `evaluate()` 内部又执行 `init + generated evaluate + 2 × generated pdf`；公开 `pdf()` 执行 `init + 2 × generated pdf`；公开 `sample()` 执行 `init + generated sample + generated reverse pdf`。因此一个完全可见的 primary hit 会重复执行 target-code `init` 12 次，而不是一次。

对两个典型材质，默认 authored 参数下的纹理工作量为：

| 材质 | 单次公开 evaluate | 单次公开 pdf | 单次公开 sample | primary hit、1 spp 合计 |
|---|---:|---:|---:|---:|
| Aluminum Scratched | 约 10 次 `SampleLevel` | 约 10 次 | 约 10 次 | 约 120 次双线性 texture samples |
| Copper Patinated | 6 次 | 5 次 | 6 次 | 68 次双线性 texture samples |

一次硬件双线性 `SampleLevel` 概念上读取 4 个 texels；忽略 cache、事务合并和格式压缩时，上表对应约 480 与 272 个 texel taps。它们只是 material texture footprint，不含 environment lookup、visibility rays、BVH traversal、G-buffer/vertex reads 和路径续传。

这已经给出可用预期：当前 Metal reference 不是“几张图加一次 GGX”，而是每个方向查询都重复空间初始化，再执行 diffuse/metal mixture、microfacet evaluate/sample 和正反向 PDF。Aluminum Scratched 的 non-repeat hash/tiling 使其静态 shader 和执行开销明显高于 Copper Patinated。

## 2. 审计对象与方法

选择的是前一轮已经完成 runtime 构建和 viewer 计时的两个代表：

| 材质 | generated HLSL | 代表性 |
|---|---|---|
| Aluminum Scratched | `build/mdl-reference/cache/79f055.../generated.hlsl` | 3 个 source textures，但包含 non-repeat triangular patch lookup、双 bump/normal 合成与 rough metal layer |
| Copper Antique Brushed Patinated | `build/mdl-reference/cache/e5d297.../generated.hlsl` | 6 个 source textures，包含 patina diffuse 与 brushed copper glossy 的空间 mixture |

计数分三层：

1. **生成源码动态路径**：读取 authored default argument block，沿典型有效表面、有效纹理与混合 lobe 路径，数 `SampleLevel`、argument reads、特殊函数和主要表达式；
2. **优化后静态 DXIL**：用锁定 Falcor 的 Slang/DXC，以 `sm_6_6 -O3` 编译读取动态方向并写回结果的最小 core-equivalent entry，统计全部 CFG 分支中的静态指令站点；
3. **viewer 调用乘数**：按 `ReferencePathTracer.cs.slang` 的实际 4 NEE + 4 BSDF pool 与 bounce cap 2 展开。

DXIL wrapper 包含 target state normalize/init 与对应 generated evaluate/pdf/sample，但不包含 scene lighting、ray query、最终 response cosine 修正和完整合法性检查。因此它用于说明 material core 的静态规模，不是完整 viewer shader 的机器执行次数。原始诊断产物为：

- `artifacts/reference-cost-baseline/metal/hlsl-source-cost.json`
- `artifacts/reference-cost-baseline/metal/dxil-static-costEvaluate.json`
- `artifacts/reference-cost-baseline/metal/dxil-static-costPdf.json`
- `artifacts/reference-cost-baseline/metal/dxil-static-costSample.json`

## 3. Aluminum Scratched

### 3.1 `init()` 做了什么

authored default 为：

- `infinite_tiling = false`；
- `bump_factor = 0.187`；
- `scratches_bump_factor = 0.435`。

因此 default path 包含：

- texture 1 与 texture 2 各一次普通 lookup；
- texture 3 的 non-repeat helper 两次，每次选择一个三角 patch 分支并做 3 次 texture lookup；
- texture 3 的两个普通 bump lookup；
- 合计约 10 次动态 `SampleLevel`。

一次选中的 non-repeat helper 分支还包含 3 次 texture samples、约 54–60 个 bitwise/hash 运算站点、2 次 `floor`、2 次 `frac`、1 次 `sqrt`，以及约 76–82 个未按 vector width 展开的算术运算符站点。该 helper 在生成源码中有两个互斥三角分支，所以优化 DXIL 中每次 helper 有 6 个静态 `SampleLevel` 站点、运行时只选其中 3 个。

不展开 helper 与通用 texture runtime 时，`init()` 自身有 17 个 argument reads、79 个乘法、10 个除法、73 个加法、29 个减法符号站点，以及 6 `sqrt`、1 `pow`、1 `sin`、1 `cos`。这些是源码表达式计数：vector 运算只计一个站点，负常数也可能被词法计入减号，不能直接叫 FLOP。

### 3.2 单次公开 evaluate

生成的 directional 部分为 diffuse/rough-metal weighted layer：

- generated `evaluate`：97 `*`、9 `/`、66 `+`、11 `-`，5 `sqrt`、1 `pow`、2 次 normal adaptation；
- generated `pdf` 每次：59 `*`、6 `/`、42 `+`、10 `-`，3 `sqrt`、1 `pow`、2 次 normal adaptation；
- 公开 `evaluate` 会调用 generated `pdf` 两次。

把 default `init`、两次选中的 non-repeat helper、generated evaluate 与 2×pdf 相加，得到约 760–772 个未按 vector width 展开的算术运算符站点，外加 10 次动态 texture samples、19 个源码 `sqrt`、4 个 `pow`、4 `floor`、4 `frac`、1 组 `sin/cos` 和 7 次 normal adaptation。backend wrapper 还会执行约 11 次 direction/frame normalization。

优化后 core-equivalent DXIL 静态体为：

| 项目 | 数量 |
|---|---:|
| LLVM/DXIL instruction lines | 3,028 |
| `fmul / fadd / fsub / fdiv` | 963 / 612 / 171 / 100 |
| `Dot3 / Rsqrt / Sqrt` | 41 / 27 / 21 |
| `Log / Exp / Sin / Cos` | 2 / 2 / 1 / 1 |
| static `SampleLevel` sites | 28 |
| default path dynamic `SampleLevel` | 约 10 |
| branches (`br`) | 85 |

28 与 10 并不矛盾：28 是所有 CFG 分支的静态 texture instruction sites；`infinite_tiling=false` 且每个 triangular helper 只选择一边时，动态路径约执行 10 次。

## 4. Copper Patinated

### 4.1 `init()` 做了什么

authored default 的 `patina_metal_blend=0.5`、`patina_bump_amount=0.35`、`copper_bump_amount=1.0`，两套 normal/bump 都启用。`smudge_amount=0`，但 evaluate/sample 编译路径仍保留该纹理读取。典型 init 依次读取 normal、patina color、smudge、roughness/mask、copper color 和第二 normal，共 6 次动态 `SampleLevel`。

`init()` 源码有 12 个 argument reads、49 `*`、8 `/`、46 `+`、19 `-`，3 `sqrt` 与一组 `sin/cos`。它没有 Aluminum 的 non-repeat hash helper。

### 4.2 单次公开 evaluate

典型混合像素同时计算 patina diffuse 与 copper glossy：

- generated `evaluate`：95 `*`、10 `/`、77 `+`、17 `-`，5 `sqrt`、1 `pow`；
- generated `pdf` 每次：61 `*`、7 `/`、51 `+`、13 `-`，3 `sqrt`；
- 公开 `evaluate` 同样调用 generated `pdf` 两次。

合计为 585 个未按 vector width 展开的算术运算符站点，外加 6 次 texture samples、14 个源码 `sqrt`、1 个 `pow`、一组 `sin/cos` 和 4 次 normal adaptation；wrapper 同样有方向与 frame normalization。

优化后 core-equivalent DXIL 静态体为：

| 项目 | 数量 |
|---|---:|
| LLVM/DXIL instruction lines | 1,413 |
| `fmul / fadd / fsub / fdiv` | 562 / 278 / 67 / 53 |
| `Dot3 / Rsqrt / Sqrt` | 26 / 18 / 14 |
| `Log / Exp / Sin / Cos` | 1 / 1 / 1 / 1 |
| static/dynamic `SampleLevel` | 6 |
| branches (`br`) | 61 |

## 5. 三个公开 operation 的静态体

下表是同一 `sm_6_6 -O3` 诊断 wrapper 的静态体。`FP arithmetic sites` 只相加 DXIL 的 `fmul/fadd/fsub/fdiv`，没有把 `Dot3`、min/max、compare、integer/hash、texture、SFU 或 control flow 混入。

| 材质 | operation | DXIL instruction lines | FP arithmetic sites | static `SampleLevel` |
|---|---|---:|---:|---:|
| Aluminum Scratched | public-evaluate core | 3,028 | 1,846 | 28 |
| Aluminum Scratched | public-pdf core | 2,605 | 1,557 | 28 |
| Aluminum Scratched | public-sample core | 3,126 | 1,930 | 28 |
| Copper Patinated | public-evaluate core | 1,413 | 960 | 6 |
| Copper Patinated | public-pdf core | 993 | 679 | 5 |
| Copper Patinated | public-sample core | 1,561 | 1,070 | 6 |

这里的“public core”表示已按当前 public operation 展开 `init` 和内部 PDF，但省略少量公共合同检查。静态体统计所有分支，不能当成实际执行的 SASS 指令数；它适合判断 code/body 量级与材质之间的相对复杂度。

## 6. 一个材质像素到底乘多少次

### 6.1 primary hit，1 spp，environment-only

假设 4 个 NEE visibility rays 都可见且 4 个 BSDF samples 都有效：

| 材质 | public 调用 | 动态 material `SampleLevel` | 概念 texel taps | 累计静态 DXIL body lines | 累计静态 FP sites |
|---|---|---:|---:|---:|---:|
| Aluminum Scratched | 4 eval + 4 pdf + 4 sample | 约 120 | 约 480 | 35,036 | 21,332 |
| Copper Patinated | 4 eval + 4 pdf + 4 sample | 68 | 272 | 15,868 | 10,836 |

后两列是把各次调用的静态体规模相加得到的“body exposure”，不是动态 profiler counter；实际执行会因 lobe、hemisphere、mixture 与 visibility 分支减少。相反，表中也没有 scene/ray/lighting 运算，所以不能用它推导完整 frame time。

### 6.2 bounce cap 2 的完全存活上界场景

当前 primary 产生 4 条 path branches。若四条都连续命中两个 suffix surfaces，且所有 environment visibility 均通过，则整个像素最多形成：

- 36 次公开 evaluate；
- 36 次公开 pdf；
- 40 次公开 sample；
- 合计 112 次 material public operations。

对应 material texture work 约为：

| 材质 | 动态 `SampleLevel` | 概念 texel taps | 累计静态 DXIL body lines | 累计静态 FP sites |
|---|---:|---:|---:|---:|
| Aluminum Scratched | 约 1,120 | 约 4,480 | 327,828 | 199,708 |
| Copper Patinated | 636 | 2,544 | 149,056 | 101,804 |

这是路径完全存活、visibility 全通过时的 workload 上界构造，不是 shaderball 平均像素。背景像素没有 material call；NEE 在 visibility 失败时会在调用 material 前退出；path miss、absorb 和 invalid sample 也会显著降低平均数。要得到真实平均动态指令数，需要在 viewer 增加 operation/visibility/path-length counters，并用 Nsight/PIX 采集 wave、cache 和 SASS 指标。

## 7. 为什么实测 Aluminum 更慢

前一轮 RTX 4090、`320×360` 单 panel、1 spp 的 GPU slot median 为：

- Aluminum Scratched：1.111 ms；
- Copper Patinated：0.638 ms。

两者约 1.74× 的差异与本次结构审计一致：Aluminum 的公开 evaluate 静态体约为 Copper 的 2.14×，primary-hit texture samples 约为 1.76×，并额外含大量 non-repeat integer hash/control flow。frame time 不是纯 material time，所以不能要求比例完全一致，但方向和量级相互支持。

## 8. 对 neural 系统设计的直接约束

### 8.1 `prepare()` 必须真正复用

当前 `NclsMdlReferenceBackend.prepare()` 只保存 context；每次公开 evaluate/pdf/sample 都重新调用 `nclsPrepareMdlTargetState()` 和 generated `init()`。这使 primary hit 的空间纹理初始化从本可一次的 10/6 次，膨胀到约 120/68 次。

目标 neural 方法必须把 latent 获取、相邻 mip 过滤、structured texture decode、normal/frame 与 view-conditioned reusable encoding 放进真正的 per-hit `prepare()`，然后让 4+4 个方向查询复用。否则即便单次 MLP 很小，也会重复支付最贵的 texture/decoder 工作。

### 8.2 reference 需要 optimized-code control

当前 reference 仍是权威语义 oracle，但不能成为最终效率比较的唯一 source control。正式设计应同时测：

1. 当前 authoritative reference；
2. 把 initialized target state 提升到 backend `prepare()` 后复用的 optimized-code control；
3. viewer 直接复用 `evaluate()` 已返回 PDF、避免随后再次调用公开 `pdf()` 的 control；
4. conventional BC4/BC5/BC7 + mip/filtering deployment control；
5. neural method。

若只与当前重复 init、重复 PDF 的路径比较，会高估 neural 方法的相对加速。优化 control 不改变 MDL source 语义，只消除 backend/viewer 组织造成的冗余。

### 8.3 当前可用的预算直觉

在真正 profiler 数据补齐前，可以采用以下量级预期，而不把它写成 hard gate：

- 一个复杂 Metal reference 的公开 evaluate 是约 `1.4k–3.0k` 静态 DXIL instruction lines、约 `1.0k–1.85k` 静态基本 FP arithmetic sites 的量级；
- 一个完全可见 primary hit 在当前 4+4 estimator 下是 12 次 material public calls，而不是一次；
- 当前未复用 prepare 时，典型材质约为几十到一百余次 bilinear material texture samples；复杂 non-repeat 材质进入百次级；
- 因此 neural `prepare + N×evaluate` 应分别记账：`prepare` 的固定成本可以略高，但必须由同一 hit 的多次方向查询摊销，`evaluate` 必须足够小且固定读取。

这些数字是本轮 diagnostic observed result，不是产品完成门槛。正式候选仍按质量—时间—内存 Pareto 与 matched controls 判断。
