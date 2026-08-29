# 共享 Slang scattering backend 可执行合同

## 1. Scope / Trigger

新增或修改 source reference、neural program、proposal、scene composer、package backend 或 path tracer 时适用。目标是让每个材质程序在同一接口下保留自己的求值、采样和 PDF 语义；统一的是调用合同，不是把所有材质归约成同一 closure 或 proposal。

## 2. Signatures

```slang
Backend.prepare(context, CompiledMaterial material) -> State
State.evaluate(float3 wiWorld, inout SampleGenerator sg) -> NclsScatteringEval
State.sample(out NclsScatteringSample sample, inout SampleGenerator sg) -> bool
State.pdf(float3 wiWorld) -> NclsScatteringPdf
```

Python 侧用于 path tracing 的 `ReferenceProgramDescriptor` 必须声明：

```text
PREPARE | EVALUATE | SAMPLE | PDF
```

## 3. Contracts

- descriptor 对完整 path-tracing capability fail closed；缺少任一必需入口时不得注册成 runtime reference。
- `evaluate()` 输出线性 RGB `f`，不含 cosine。公共 response adapter 对 reflection 与 transmission 都乘 `|n_s · wi|`。
- 连续事件满足 `sample.weight = evaluate(wi) * |n_s · wi| / pdf.forward`；`sample()` 与 `pdf()` 必须使用同一 proposal。delta 事件遵守 backend 自身的离散 measure，并显式标记 `Delta`。
- source-native API 的 sample tuple 是一个不可拆分的数值结果：方向、event、sample-path PDF 与 throughput weight 必须原样适配。极窄峰在掠射反射时可能依赖 sampler 内部尚未舍入的 half-vector；不得用已舍入 `wi` 调独立 `pdf/eval` 后重建 sample tuple，否则切向量相消会把稳定 native weight 放大许多数量级。独立 `pdf(wi)` 仍由同一 native proposal 提供并用于 NEE；数学 owner 相同不等于两个 float32 执行路径必须逐 bit 相等。
- `sample()` 返回 null、非有限量或连续事件非正 PDF 时，integrator 终止当前路径；不得切换 generic cosine/GGX proposal 后继续冒充同一 estimator。
- proposal、closure、resource layout 与 backend-specific `State` 都是程序私有实现。renderer 只能调用 canonical state 方法，不读取 source family、program key 或私有字段。
- heterogeneous scene composer 只能选择 concrete backend、准备资源无关的公共状态并把调用转发给对应 canonical state；不得在 composer 或 renderer 中重新实现某个材质族的 evaluate/sample/pdf。
- 受 Slang/DXC resource aggregate lowering 限制，MaterialX 等含 resource handle 的 state 把 handle 字段放在 value 字段之后。scene composer 的持久 state 只保存 resource-free prepared data，在调用点用全局绑定重建 concrete state；这是一种编译期布局约束，不是第二套 scattering ABI。
- neural package 额外实现稳定`NclsPackage*` host ABI；source reference不伪装成`ScatteringPackage@2`。训练、Falcor parity、package与viewer对同一program复用同一module closure，反射offset来自编译器，运行循环静态有界。

## 4. Validation & Error Matrix

| 条件 | 必须行为 |
|---|---|
| descriptor 缺 `prepare/evaluate/sample/pdf` 任一项 | 构造失败，不进入 renderer |
| `sample()` 返回 null、非有限方向/weight，或连续事件 PDF 非正 | 终止当前路径，不 fallback |
| project-owned sample weight 与 `evaluate * abs-cosine / pdf` 不一致 | GPU 数值测试失败 |
| `sample()` 与独立 `pdf()` 使用不同数学 proposal | sample→pdf/reverse-pdf 测试失败 |
| native sample tuple 与 native oracle 不一致 | 数值实现失败；恢复 source-native direction/event/PDF/weight，不用 independent query 重建 |
| native sample PDF 与独立 native PDF 只在已舍入极窄掠射方向漂移 | 分开验证 native sample identity 与 independent evaluate↔pdf；保留掠射/capture 回归，不把漂移误修成新 estimator |
| evaluate/pdf/weight 出现 NaN 或 Inf | 数值正确性失败，不允许 clamp 掩盖 |
| resource handle state 进入不可合法构造的 value aggregate | shader 编译失败；改为 resource-free composer state 并在调用点重建 |
| renderer 识别 source family/program key | 静态边界测试失败 |

## 5. Good / Base / Bad Cases

- Good：MDL car paint 从同一 target code 调用 evaluate/sample/pdf；MaterialX 用解析后的 diffuse/specular/anisotropy 构造自己的 matched proposal；OpenPBR 保留官方 `openpbr_sample` 返回的完整 tuple，并由官方 `openpbr_eval/openpbr_pdf` 处理独立 query；三者通过相同 state 接口进入同一 integrator。
- Base：Lambertian backend 的 cosine proposal 满足 sample→pdf 与 weight 恒等式，且仍通过完整四入口合同。
- Bad：renderer 发现某 backend 没有 sampler 后统一套固定 roughness GGX；或按 `surface.family` 分支调用材质专用自由函数。

## 6. Tests Required

- unit：descriptor fail-closed、reflection/transmission response measure、public symbol 与 renderer 静态边界。
- GPU：project-owned proposal 覆盖 evaluate/sample/pdf 有限性、sample→pdf、reverse PDF、连续事件 weight 恒等式与分量选择；source-native API 额外覆盖 sample tuple 对 native oracle 的 identity、independent evaluate↔pdf，以及极窄掠射方向与最终 capture tail。
- package：从自身绝对路径加载并编译真实 module closure，验证 `NclsPackage*` ABI 与反射布局。
- viewer：Release scene specialization 编译；真实 source capture 全 finite，并结合 sample-weight/PDF 尾部与空间邻域诊断高亮，不以 radiance clamp 通过。

## 7. Wrong vs Correct

```slang
// 错：integrator 按材质族重写采样语义。
if (surface.family == MaterialFamily::Mdl)
    scatter = nclsMdlSampleSurface(...);
else
    scatter = sampleFixedGgx(...);

// 对：所有程序只通过自己的 canonical prepared state。
let state = backend.prepare(context, material);
NclsScatteringSample scatter;
if (!state.sample(scatter, sg))
    return; // null/invalid event terminates this path
```

```slang
// 错：evaluate 来自材质，PDF 来自无关 proposal。
weight = state.evaluate(wi, sg).f * absCosine / fixedGgxPdf(wi);

// 对：sample/pdf/evaluate 都属于同一 backend。
let density = state.pdf(wi);
weight = state.evaluate(wi, sg).f * absCosine / density.forward;
```
