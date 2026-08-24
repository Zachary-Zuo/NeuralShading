# 实时神经材质编译：问题、表示与研究路线

## 一句话定义

NeuralShading 要把保持原生语义的复杂源材质编译成一种**随机访问、运行成本有界的神经材质程序**：小型 MLP 在运行时直接实现 `evaluate(wo, wi)`；`prepare()` 为同一着色点的多次方向查询准备可复用编码；需要材质驱动方向采样的路径再使用与 evaluator 匹配的 `sample()/pdf()`。

目标不是让 deferred renderer 复现完整场景 path tracing，而是让同一份编译后材质资产能够进入 deferred、hybrid ray tracing 和 path tracing 的材质求值位置。解析公式和固定 closure 继续作为基线，也可以成为 neural evaluator 的物理 core 或 sampling proposal，但不再是目标表示必须归约到的词汇。

## 项目试图解决什么

相机看到一个表面点时，renderer 已知位置、法线、切线、纹理坐标、纹理 footprint 和观察方向。材质需要回答：从方向 `wi` 到达的光，有多少沿方向 `wo` 离开。局部表面散射写成：

```text
f(wo, wi)
```

像素最终亮度还取决于所有方向上的入射辐亮度：

```text
Lo(wo) = ∫ Li(wi) * f(wo, wi) * abs(dot(N, wi)) dwi
```

这里有两个相互连接但必须分开验证的问题：

- **材质表示**：以有限存储和有限查询成本近似 `f(wo, wi)`；
- **光照积分**：由 renderer 得到 `Li(wi)`，处理灯光、阴影、环境可见性、间接光和互反射，再对它积分。

本项目的核心贡献域是材质表示及其可用的采样/积分接口。场景可见性和光传输仍由 renderer 负责。神经网络替换的是材质的运行时微程序，不是光线遍历、可见性或整个 path integrator。

工业问题的关键也不只是“更快算一个 BRDF”。Substrate 等实时系统需要在 GBuffer bytes、closure 数量、灯光循环和平台预算之间折中；复杂源材质通常要被人工简化为有限的运行时词汇。神经材质希望把这种关系改成：

```text
复杂且可编辑的源材质
        ↓ compile / cook
紧凑 latent 资产 + 共享小网络
        ↓ GPU shader 随机访问
有界成本的 evaluate / optional sample / optional integrate
```

源材质复杂度由编译阶段吸收，运行时成本主要由 latent 规模和网络结构决定。这使“共享 decoder 代码 + 材质专属数据”成为一种新的材质执行方式，而不要求每种复杂外观都扩展引擎的手写 closure 集合。

## 目标运行时结构

```text
原生材质
  │  offline optimization / cook / feed-forward compile
  ▼
NeuralMaterialAsset
  ├─ latent textures / material code / LOD metadata
  └─ source provenance 与编辑状态

MethodBundle / runtime
  └─ shared network weights、shader 与 capability

NeuralMaterialAsset + shared network weights
  │
  │  每个可见 pixel 或 ray hit 执行一次
  ▼
h = prepare(material latent, uv footprint, shading frame, wo)
  ├─ evaluate(h, wi)       → BSDF / 散射值 f(wo, wi)
  ├─ sample(h, ξ)          → 新方向 wi 与该样本信息
  ├─ pdf(h, wi)            → 该 proposal 在 wi 的方向概率密度
  └─ integrate(h, light)   → 可选：环境光/面积光的有界积分
```

材质专属数据与共享代码必须分开理解。`NeuralMaterialAsset` 是某个源材质编译后的 latent、material code、mip/LOD 和 provenance；网络权重通常由一个方法或材质族共享，属于 `MethodBundle`/runtime。若某种方法允许少量材质专属权重，也必须把它计入资产 bytes 并明确编辑和 cook 语义，不能隐藏在全局运行时中。

### 六个操作的权威定义

- **`compile_material(source) -> NeuralMaterialAsset`**：在离线优化、资产 cook、feed-forward compile 或“compiler + refinement”阶段，把源材质的原生参数、图和资源变成 view-independent 的 latent asset。它不接收当前相机 `wo`，也不执行逐灯光照。
- **`prepare(asset, surface, footprint, frame, wo) -> h`**：读取 material/spatial latent texture，按 UV footprint 选择并过滤 mip/LOD，建立或读取切线 shading frame，编码 `wo`，并运行可被多个方向查询复用的 shared trunk。它可以把 sampler proposal 参数缓存进 `h`，但不消费随机数选择下一方向。
- **`evaluate(h, wi) -> f(wo, wi)`**：目标方法的核心小型 MLP。它接受共享 state `h` 和当前入射方向 `wi`，直接返回不含几何余弦的 BSDF/散射值。renderer 在积分时恰好再乘一次 `abs(dot(Ns, wi))`。训练可以在 `f*cos` 测度下构造 loss，但不能改变运行时返回语义。
- **`sample(h, ξ) -> ScatteringSample`**：根据 `h` 得到或预测重要性 proposal，并用随机数 `ξ` 从中生成下一方向 `wi`。返回结果还需要携带对应 PDF、event 和路径 throughput 所需信息。proposal 可以由 `prepare` 预计算参数，也可以由单独的 `SamplerHead(h)` 生成；这两种实现都不把实际随机采样并入 `prepare`。
- **`pdf(h, wi) -> p`**：计算 `sample()` 所用同一 proposal 在方向 `wi` 上、相对于立体角的概率密度。Path tracing 用它构造 `f*cos/p` 权重，MIS 用它和光源采样 PDF 比较。能够输出方向但不能求实际密度的网络不满足 sampling capability。
- **`integrate(h, light) -> contribution`**：可选能力。它在声明的 light descriptor、查询数和误差范围内，近似由同一个 `evaluate` 定义的环境光或面积光积分。它不应暗中包含场景阴影、互反射或 GI；若包含，就必须作为独立 light-transport 方法命名和评测。

其中基础材质合同是 `compile_material + prepare + evaluate`。`sample/pdf` 是 path tracing 或 stochastic integration profile 成对增加的能力；`integrate` 是为 deferred 环境光/面积光降低查询成本的可选能力。

### 三个生命周期不能混为一谈

1. `compile_material()` 在资产导入、编辑或 cook 阶段把原生材质变成运行时资产；
2. `prepare()` 在着色点获取和过滤 latent，编码 shading frame、footprint 与 `wo`，形成可复用的 view-conditioned state `h`；
3. `evaluate()` 对每个 `wi` 运行主要的 neural scattering decoder。

`prepare()` 可以预先计算 sampler 的分布参数，但实际使用随机数产生方向的操作仍属于 `sample()`。这样 deferred 中多个灯光可以复用 `h`，path tracer 中同一 hit 的 BSDF evaluation、next-event estimation、sampling 和 MIS 也可以复用 `h`。实现可以把 prepare inline 到 lighting/ray-hit shader；逻辑阶段不要求一定落一张独立 GBuffer。

`h` 的生命周期是当前可见 pixel 或当前 ray hit，而不是新的作者材质格式。只要材质状态、UV/footprint、shading frame 或 `wo` 改变，就必须重新 prepare。多个 `wi` 查询可以复用同一个 `h`；不同观察方向不能错误共享 view-conditioned state。

## Neural evaluator 是目标表示的主体

目标求值器写成：

```text
z = material/spatial latent
h = Prepare(z, surface, footprint, wo)
f = EvaluateMLP(h, wi)
```

需要通过建模实验确定的不是“要不要把网络输出成若干固定 closure”，而是：

- latent 是每材质向量、空间 latent texture，还是二者组合；
- `wo/wi` 使用何种局部方向编码；
- MLP 的深度、宽度、激活、精度和共享方式；
- `prepare` 是否包含 view-conditioned shared trunk，`evaluate` head 每个方向还需多少计算；
- 输出直接表示 RGB BSDF、`f*cos` 的训练量、log-domain 响应，还是解析 core 上的 neural residual；
- 如何处理非负、动态范围、互易性、能量和各向异性；
- 空间变化材质如何支持 mip、footprint、tile 与 random access；
- 权重是所有材质共享、按材质族共享，还是允许少量材质专属参数。

解析 core 或 hybrid 设计可以用于稳定极窄主反射、约束能量或提供可采样 proposal，但最终散射能力由 `EvaluateMLP` 直接补全，不要求神经网络只预测传统 closure 参数。

## 编译和编辑有三种需要区分的形态

同一个 neural evaluator 可以配合不同资产生成方式，它们对应不同的研究与工业价值：

1. **逐资产优化 latent**：每个材质 cook 时优化自己的 latent；适合作为同 decoder 下的 `optimized-code control`，但编辑代价可能较高。
2. **target-tensor encoder**：把已生成的完整 reference response/多通道 texture tensor 输入 encoder 得到 latent，烘焙后丢弃 encoder；它可能提高压缩优化效率和确定性，但仍需先取得完整目标数据。
3. **feed-forward source compiler**：共享编译网络从原生参数、图或资源直接生成 latent，不读取完整 reference tensor；适合未见材质和即时参数编辑，是项目希望最终达到的形态。
4. **source compiler + optional refinement**：先即时生成 latent，发布 cook 时再短程优化；兼顾交互工作流和最终资产质量。

因此“单材质能被一个 MLP 拟合”只证明函数容量；“共享 decoder + 每材质 latent”证明统一运行时表示；“target encoder 更快地产生 latent”证明的是压缩算法；“未见材质由不读取完整 GT 的 source compiler 直接生成 latent”才证明通用编译和编辑工作流。实验必须把这四层结论分开。

## Deferred、path tracing 与 UE 的调用方式

二者可以消费同一 neural evaluator，但调用方式不同：

| renderer 路径 | 对神经材质的主要调用 |
|---|---|
| deferred 方向光、点光、聚光 | `prepare` 一次，对每个已知 `wi` 调用 `evaluate` |
| clustered/forward lighting | 对每个有效 pixel-light pair 调用 `evaluate` |
| path tracing / ray hit | 每个 hit 执行 `prepare`，用 `evaluate + sample + pdf` 更新路径 |
| 环境光、面积光 | 有界方向查询、stochastic sampling，或专用 `integrate_*` |

典型 deferred 调用如下：

```text
for each visible pixel:
    h = prepare(asset, surface, footprint, frame, wo)

    for each affecting analytic light:
        wi = direction_to_light(...)
        f  = evaluate(h, wi)
        Lo += visibility * Li * f * abs(dot(Ns, wi))

    Lo += optional integrate(h, environment_or_area_light)
```

这里 `prepare` 的成本每个 pixel 支付一次，evaluator 的成本按实际 pixel-light query 数支付。若 state 不落 GBuffer，也可以在 tile/cluster lighting shader 内 inline prepare；性能报告仍要把逻辑上的共享编码成本和逐方向 MLP 成本分开。

典型 path-tracing hit 调用如下：

```text
h = prepare(asset, hit_surface, ray_footprint, frame, wo)

// next-event estimation：光源先给出 wi
f       = evaluate(h, wi_light)
p_bsdf  = pdf(h, wi_light)             // sampler capability 存在时用于 MIS

// material importance sampling：材质选择下一方向
s       = sample(h, ξ)                  // s.wi、s.pdf、event/weight
beta   *= s.weight                       // 非 delta 时等价于 f*cos/pdf
trace_next_ray(s.wi)
```

Path tracer 没有必须先执行的全屏 prepare pass；每个任意 ray hit 都可以 inline 执行同一 prepare 逻辑。关键不是 pass 名称，而是 evaluator、sampler 和 PDF 使用同一份材质 latent、同一个 `h` 和一致的散射语义。

UE 的材质图在 base pass 求值，Substrate 把供光照使用的材质状态写入 GBuffer，再由 deferred/forward/environment lighting 消费。NeuralShading 对应的不是一次独立的通用 NNE 推理任务，而是 shader 内部的小网络求值：CompiledMaterial/latent 在 base pass 或 ray hit 可见，`prepare` 生成或重建 `h`，Substrate 式 lighting 位置调用 neural `evaluate`。实际 UE 接入需要新的 shading model、Substrate adapter 或等价的 engine integration；仅有模型文件并不会自动进入 UE lighting。

Substrate 的 Blendable/Adaptive GBuffer、每像素 closure/bytes 预算和材质简化机制提供工业对照。神经方法要比较的是同一真实工作负载下的质量—时间—内存 Pareto，而不是只比较网络参数量。官方背景见 [Substrate Materials Overview](https://dev.epicgames.com/documentation/unreal-engine/overview-of-substrate-materials-in-unreal-engine?lang=en-US) 和 [Programming with Substrate GBuffer Formats](https://dev.epicgames.com/documentation/en-us/unreal-engine/programming-with-substrate-gbuffer-formats)。

## 工业界关注的深层变化

这条路线可以从四个层面理解：

1. **新的 shader 执行介质**：利用 GPU 的矩阵/AI 计算能力执行固定形状的小网络，把一部分手写 closure 组合转成 learned computation。它不是在 renderer 外调用一个大型模型，而是把 neural decoder 放进 shader hot path。
2. **代码与材质数据分离**：共享 evaluator/sampler weights 类似一种神经 shader ISA，材质差异主要存在 latent texture 和 material code 中。复杂源图不必在运行时原样展开。
3. **作者复杂度与运行时成本解耦**：多层、程序图、测量外观或高分辨率资源在 compile/cook 阶段被吸收；运行时预算由 latent bytes、prepare trunk 和 evaluator 网络形状控制。
4. **同一资产服务多种 renderer**：deferred profile 使用 `prepare + evaluate` 和可选 integration；PT profile 复用同一 evaluator/latent，再增加 matched `sample/pdf`。这样 shading 与 sampling 不再是两份无关的材质近似。

这并不自动保证更快。逐 pixel、逐 light 的小 MLP 是否能有效使用矩阵硬件，取决于 tile 内材质一致性、batch/cooperative execution、网络宽度、分支、latent 带宽和灯光数量。复杂材质、较少有效灯光和高 closure 成本可能更有利；简单材质或大量独立灯光可能仍由解析路径占优。因此系统阶段需要质量—时间—内存和 workload scaling，而不能只用 FLOP 或模型参数量推断工业价值。

Neural Dynamic GI 对本项目最有价值的不是“它也使用网络”，而是它体现的 GPU-native neural codec 模式：

- 大量资产信息存入可 tile/stream 的 latent feature，而不是运行时展开原始复杂表示；
- 小 decoder 支持任意 pixel 的 random access，而不是依赖整帧顺序推理；
- 计算与显存/带宽互换，训练时就模拟压缩、量化和实际部署约束；
- mip、tile、virtual texture、缓存和常规 shader pipeline 仍然存在，神经网络只是其中的解码执行单元；
- 资产专属 latent 与共享 decoder 分开版本化，便于跨场景和跨引擎部署。

更准确地说，动态 lightmap 的核心是压缩查询域 `(t, u, v) → RGB`：空间邻近 texel 的时间轨迹相关，而空间维和时间维也不必具有相同频率。NDGI 用 `uv`、`ut`、`vt` 平面和低分辨率 `uvt` 体积显式分配这些相关性，再由小网络随机访问解码。项目经验还提出另一个可验证假设：把每个 texel 的 26 个 RGB 时刻视为 78 维向量，以 K-means++ 建立共享原型字典；每个 texel 只保存两个原型索引和一个凸混合权重，查询时间 `t` 时读取两组 RGB 并混合。它不是 NDGI 论文的方法，而是说明“高维查询不等于每一维都需同等高频容量”。完整编码过程已固化在[问题定义](research/problem_definition.md)中；该启发应作为 dictionary oracle 与 neural factorization 做同 byte/同时间比较，而不能只凭经验认定更优。

材质问题在这个模式上增加了新的难点：查询不仅依赖位置和 footprint，还依赖 `wo/wi`；evaluator 之外还要构造匹配的 sampling distribution；环境光和面积光需要对方向函数积分。因此 NDGI 是系统模板和工程先验，不是可直接套用的材质解法。

对这一问题本质、可证伪假设、相关工作和目标数据的详细整理统一维护在[当前研究入口](research/README.md)，不再另存不可修改的日期快照。

## 为什么 path tracing 还需要 `sample` 和 `pdf`

`evaluate(h, wi)` 只能回答给定方向上的散射值。Path tracer 还必须决定下一条随机光线往哪里走。匹配的神经 sampler 表示一个条件分布：

```text
q(wi | h)
```

可靠方案需要同时得到：

```text
wi = sample(h, random)
p  = pdf(h, wi)
f  = evaluate(h, wi)
throughput *= f * abs(dot(N, wi)) / p
```

sampler head 可以预测 GGX、vMF、spherical Gaussian 等可解析混合分布，也可以采用具有可计算密度的可逆方向模型。只让网络输出方向而无法计算实际 PDF，会破坏标准 MIS 和无偏权重的依据。

近似 sampler 不必与理想重要性分布完全一致；只要采样与 PDF 匹配，estimator 可以相对于 neural evaluator 保持正确。这里仍有两类不同误差：

- `evaluate` 与源材质 reference 的**表示误差**；
- 有限 samples 对 neural material 积分产生的**Monte Carlo 方差**。

因此 sampler 是 evaluator 成形后的第二个建模问题。它与 evaluator 共享 latent 和 `prepare`，但不能在 evaluator 尚未确定时先做独立的系统级“方差 kill test”。

## 环境光积分是独立研究问题

逐灯 neural evaluate 很直接；环境光需要计算整个半球积分。一个方向查询很快，并不自动意味着数十或数百次查询仍适合稳定实时显示。候选路线包括：

1. 少量确定性方向或预过滤光照基上调用 evaluator；
2. 使用匹配 sampler，配合时域累积和重建；
3. 增加与 evaluator 语义一致的 neural environment integration head；
4. 把 neural response 分解为可与环境光低秩/基函数快速积分的形式。

`integrate_*` 必须近似由同一个 `evaluate` 定义的积分，而不能在没有说明时变成包含阴影、可见性和场景 GI 的最终像素预测。后者属于另一项光传输研究。

## 研究阶段：基准优先，部署轨道按阶段收尾

当前不能直接执行多灯性能、sampler 方差和 UE 环境光质量的“kill test”，因为 neural representation、latent 和执行图尚未确定。研究按 `docs/research/experiment_framework.md` 的基准优先路线推进：先冻结语料与评测协议，再在稳定框架内比较 `docs/research/model_candidates.md` 的候选；本文的运行时合同保留为候选注册时的静态约束。以下阶段是证据范围与后续扩展顺序，不是前一项全指标通过后才能触碰后一项的瀑布 gate。

### 阶段 A：固定语义与 v1 基准

- 固定 `evaluate` 的方向、测度、颜色和余弦语义；
- 按材质难度分级冻结采样密度、split 与按源表示类型定义的泛化考核；
- 明确每个候选的 latent bytes、网络结构、precision 和单次调用图；
- 空间 latent/LOD 使用合同中已有的 UV 与 footprint 查询，并在缺少尺度/旋转覆盖时重新采集。

### 阶段 B：验证 neural representation 本身

按递进关系回答三个问题：

1. 一个小型 evaluator 能否在完整 `wo × wi` 域拟合单个未简化源材质；
2. 共享 decoder 加材质专属 latent 能否覆盖一组材质，并在给定 latent/MLP 预算下保持质量；
3. feed-forward compiler 能否为未见材质状态直接生成可用 latent，并保留参数编辑。

这里比较方向响应、能量诊断、连续 view/state sweep 和受控光照积分，不把 UE、多灯和 PT 方差混入模型选择。单材质表达力、共享表示和 source compiler 是三个必须分开归因的问题。逐 `(material, wo)` query group 独立拟合只能测一张方向切片的容量；它可以保留为诊断，但不能单独证明 view-conditioned neural material program 可行。

### 阶段 C：部署轨道

- 导出共享权重、CompiledMaterial/latent 和 Slang evaluator；
- 验证 Python/Slang 数值与坐标语义一致；
- 在 Falcor 中测量 `prepare`、单次 `evaluate`、state bytes 和方向响应图像；
- 与解析/closure 基线在相同 GPU 时间和内存条件下比较。

部署轨道在每个研究阶段收尾对当期最优部署档候选执行一次；里程碑候选再补真实 GPU、材质分歧和精度路径，形成“是否进入实时 Pareto”的决策依据。

### 阶段 D：扩展 matched sampler

- 选择具有可计算 PDF 的 proposal family；
- 训练或拟合 sampler head，并与固定 evaluator 共享 `prepare`；
- 先验证 sample/PDF 分布一致性和 MIS 正确性；
- 再比较固定 spp、固定时间下的方差与开销。

### 阶段 E：解决实时光照积分

- 测量多灯下 shared `prepare` 的摊销与 evaluator scaling；
- 为 HDRI/面光选择 deterministic、stochastic 或专用 integration head；
- 分开报告 evaluator 表示误差与积分近似误差；
- 再决定 UE/Substrate、Lumen 或其他路径需要完整方法、简化 fallback 还是缓存表示。

### 阶段 F：系统与工作流验收

- 未见材质、未见参数组合和空间变化资产；
- 编辑到新 latent 的延迟，必要时区分即时 compiler 与 cook refinement；
- Falcor/Slang 真实 GPU Pareto，以及 UE/Substrate 对照或正式接入；
- 同一资产在 deferred 与 PT profile 中的外观一致性；
- viewer 完整场景结果、固定输入的局部对照和可复现 capture。

阶段 D–F 是建立在 evaluator 已经通过阶段 B–C 之后的验收，不是现在凭空设定的前置 kill test。

## 当前 viewer 能证明什么

viewer 左侧是带有限深度上限的完整场景 path-traced reference，右侧是 deferred 实时方法。因此左右差图同时包含：

- 源材质与编译材质的局部散射差异；
- 实时环境/面积光积分近似；
- path tracing 与 deferred 在间接光、互反射和可见性上的系统差异。

它适合展示最终使用语境，不能单独用于训练或选择 evaluator。模型阶段应使用固定方向响应和匹配入射光/可见性的局部图像。

若场景只有一个不透明、反射型凸球和无限远 360° 环境图，没有其他物体、局部灯、透射或参与介质，那么场景级间接互反射基本消失，首个表面的出射光就是材质对环境的积分。此时两条管线的物理目标可以高度接近，但仍可能因材质近似、环境积分算法、采样噪声、mip/footprint、可见性边缘和数值实现而不完全一致。这个场景适合隔离环境积分，不等于一般场景中 deferred 与 path tracing 相同。

## 可形成的研究 claim

“小 MLP 能拟合 BSDF”或“神经 sampler 能加速 PT”都已有直接相关工作，不能单独构成项目贡献。更完整的目标 claim 是：

> 将保持原生语义和编辑能力的复杂材质编译为随机访问、有界成本的 neural material program；以共享的小型 evaluator 直接执行方向散射，为 path tracing 提供匹配的可计算密度 sampler，并为 deferred 动态灯光与环境光提供明确的有限成本积分路径；同一运行时资产在未见材质状态上保持质量、性能和工作流可用性。

其中最需要实验证明的新增价值是：跨材质的 compiler、random-access latent、deferred 与 PT 的共同 evaluator、matched sampler、环境积分和真实引擎 Pareto。Neural Dynamic GI 展示的 latent feature、随机访问小 decoder、tile/virtual-texture 与部署约束联合设计，是本项目的重要系统经验；材质方向函数、sampling/PDF 和光照积分则是需要另行解决的核心问题。

reviewer 很可能沿以下边界质疑，阶段设计必须能逐项回答：

- 如果只是每个材质单独 overfit 一个 MLP，统一 compiler 和编辑工作流在哪里；
- 如果网络只在 `prepare` 输出传统 closure，direct neural evaluator 的新增表达能力在哪里；
- 如果只有 path-tracing demo，deferred 多灯与环境光怎样消费同一资产；
- 如果 sampler 没有与实际方向生成匹配的 PDF，MIS 和 estimator 正确性如何成立；
- 如果 evaluator 查询次数随灯数增长，真实 shader batching、带宽和帧时间是否仍优于 Substrate/解析基线；
- 如果完整 viewer 差图混入 GI 和可见性，局部材质表示、积分近似与场景传输误差怎样分开。

因此论文证据链应按“表示容量 → shared latent → compiler 泛化 → shader 部署 → sampler/integration → 完整系统”建立，而不是用一张材质球图或单个总帧时间跨过中间结论。

相关工作的边界应明确记录：

- [Real-Time Neural Appearance Models](https://research.nvidia.com/labs/rtr/neural_appearance_models/) 已展示 latent appearance、neural BRDF decoder 和 importance sampler 在实时 path tracer 中的组合；本项目必须进一步证明通用 compiler、deferred 动态光照/环境积分和原生编辑工作流。
- [Neural Material Adapter](https://studios.disneyresearch.com/2026/07/01/neural-material-adapter-transforming-complex-materials-into-efficient-analytic-bsdfs/) 展示从高保真多层外观到高效解析 BRDF 的适配；本项目选择让运行时 evaluator 本身保持 neural，并研究它与实时积分及采样的共同资产。
- [RTX Neural Shading](https://github.com/NVIDIA-RTX/RTXNS) 提供在 Slang/DX/Vulkan shader 中执行小网络的工程基础，但 SDK 可执行不等于材质表示、编译和积分问题已经解决。
- [Neural Dynamic GI](https://openaccess.thecvf.com/content/CVPR2026/papers/Wu_Neural_Dynamic_GI_Random-Access_Neural_Compression_for_Temporal_Lightmaps_in_CVPR_2026_paper.pdf) 证明随机访问 latent、小 decoder、tile/virtual texture 和部署约束可以形成实际实时系统；它压缩的是动态 lightmap，不直接解决材质的方向函数和 sampler。
