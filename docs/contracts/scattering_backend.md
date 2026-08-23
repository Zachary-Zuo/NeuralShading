# 散射后端合同

## 目的

本文定义 deferred、路径追踪、数据采集和拟合评测共同使用的散射语义。它不规定内部 packet 字段，因此 K2 LTC、其他解析表示、latent decoder 和直接 neural BSDF 都能实现同一接口。

Falcor 8.0/Slang 2024.1.34 的可编译合同位于 `shaders/ncls/contracts/`，Python 对应物位于 `src/ncls/core/scattering/`。下面的签名是冻结的语义设计；物理 buffer 对齐由单独的生成 ABI 决定。

## 整个材质接口与散射接口的边界

散射后端只是完整 deferred 材质接口的一部分。renderer 按以下稳定阶段处理材质：

```text
evaluate_visibility()      opacity / alpha test
evaluate_surface_frame()   shading normal、tangent、bitangent
evaluate_emission()        自发光
prepare_scattering()       生成不透明 ScatteringState
evaluate/sample/pdf        局部表面散射
```

位移在主可见性之前的几何阶段处理；interior/exterior medium 在路径跨越边界时交给 volume 阶段。它们都由 `MaterialProgram.outputs` 表达，但不能塞进 BSDF 返回值。

v1 中 opacity 固定为 1、emission 为 0、displacement 为空，surface frame 只来自几何切线和常量旋转。尽管如此，上述阶段边界从第一版就固定。法线图或 alpha texture 加入后只扩展公共材质求值器，不改变散射后端签名。

拟合后端只负责 `prepare_scattering` 及其后的散射操作。reference 和右侧方法必须共享 visibility、surface frame 和 emission 结果，避免把非散射差异误计为拟合误差。

## 方向和数值约定

- `wo`：从着色点指向相机或上一条路径顶点的出射方向；
- `wi`：从着色点指向光源或下一条路径顶点的入射方向；
- 两者都指向远离着色点的一侧；
- 世界空间和局部空间转换只能通过 `NclsShadingFrame` 完成；
- `evaluate()` 返回不含几何余弦的 BSDF `f(wo, wi)`；
- renderer 在积分时恰好乘一次 `abs(dot(shadingNormal, wi))`；
- PDF 相对于立体角；delta 事件的连续 PDF 为 0，并通过 event flag 表达；
- RGB 值为线性工作空间，必须有限；非 delta 的物理散射值必须非负。

Falcor `IBSDF` 当前把余弦包含在返回值中，并使用不同的参数命名习惯。项目必须通过一个集中 adapter 转换，不能让 Falcor 约定泄漏到数据合同或 Python API。

## 公共结构

```text
NclsShadingFrame
  normal
  tangent
  bitangent

NclsSurfaceInteraction
  position
  geometricNormal
  shadingFrame
  uv
  uvDx / uvDy
  materialInstanceId
  primitiveId
  frontFacing

NclsScatteringContext
  surface
  woWorld
  transportMode          Radiance / Importance
  componentMask
```

`geometricNormal` 用于 sidedness 和防止 shading-normal 漏光，`shadingFrame.normal` 用于实际 BSDF。backend 不得假定 tangent 与某个世界轴对齐。

事件标志至少保留：

```text
Reflection
Transmission
Diffuse
Glossy
Delta
FrontSide
BackSide
VolumeBoundary
```

v1 backend 可以只返回 `Reflection`，但不能重新分配标志位。

## 求值结果

```text
NclsScatteringPdf
  forward
  reverse

NclsScatteringEval
  f
  pdf
  eventFlags
  valid

NclsScatteringSample
  wiWorld
  weight
  pdf
  eta
  eventFlags
  valid
```

`sample().weight` 定义为该样本对路径吞吐的乘数：

```text
f(wo, wi) * abs(dot(Ns, wi)) / pdf.forward
```

delta 事件允许直接返回有限的离散事件权重。调用方不能通过重新求值 `f/pdf` 猜测 delta 权重。

`pdf.reverse` 为未来的双向方法、互易性诊断和可微渲染保留。无法提供反向 PDF 的 v1 backend 可以在 descriptor 中声明不支持；不能返回未经说明的近似值。

## Slang 语义接口

目标接口使用 Slang associated type 和编译期 specialization。热路径不使用运行时虚调用；viewer 切换方法时切换 program/pipeline variant。

```c
interface INclsScatteringState
{
    NclsScatteringEval evaluate<S : ISampleGenerator>(
        float3 wiWorld,
        inout S sampleGenerator);

    bool sample<S : ISampleGenerator>(
        out NclsScatteringSample result,
        inout S sampleGenerator);

    NclsScatteringPdf pdf(float3 wiWorld);
}

interface INclsScatteringBackend
{
    associatedtype CompiledMaterial;
    associatedtype State : INclsScatteringState;

    State prepare(
        NclsScatteringContext context,
        CompiledMaterial material);
}
```

`CompiledMaterial` 和 `State` 都是 backend-specific associated type。纹理、网络权重和其他全局资源由 backend 实例或 shader parameter block 绑定，不作为公共结构参数传递。

`State` 捕获当前着色点和 `woWorld`，所以后续逐灯查询不重复传入观察方向。`evaluate()` 接收 sample generator 是为了让随机游走参考解能在同一语义接口下做内部 Monte Carlo；确定性拟合后端忽略它。

如果 Slang 版本对某个泛型组合有限制，adapter 可以改写语法，但不能改变上述语义、方向和返回值。

## CPU/主机侧生命周期

每个拟合后端提供以下主机操作：

```text
describe() -> BackendDescriptor
compile_material(MaterialProgram, resources) -> CompiledMaterial
create_runtime(device, MethodBundle) -> BackendRuntime
prepare_frame(scene_revision, dimensions)
dispatch_prepare(gbuffer, output_state)
```

`CompiledMaterial` 是 view-independent 的静态结果，可以是规范化层栈、latent、纹理引用或其他方法数据。`dispatch_prepare()` 依据 GBuffer 中的着色点和观察方向生成 per-pixel state，也允许 backend 选择不落盘、直接在 lighting shader 中 inline prepare。

## BackendDescriptor

```text
BackendDescriptor
  backend_id
  backend_version
  scattering_contract_version
  supported_material_ir
  supported_capabilities
  state_storage_mode       inline / structured / raw
  state_stride
  state_alignment
  deterministic_eval
  bounded_execution
  shader_entry_points
  cost_model
```

`state_stride` 是方法描述，不是公共固定常数。renderer 依据 descriptor 分配资源，但从不解释内容。`inline` 模式允许 `state_stride=0`。

`cost_model` 至少记录 prepare 的网络/ALU 成本、每次 evaluate/sample 的成本、纹理查询数和状态 bytes/pixel。实际 GPU 时间由 viewer benchmark 补充。

能称为实时部署候选的 backend 必须声明 `bounded_execution=true`，且 state 大小、网络层数、lobe/atom 数和循环上界均与材质层数、灯光数之外的随机内部状态无关。尚未满足这一点的方法可以作为诊断 backend，但不能进入实时 Pareto 比较。

## 必需能力

所有可作为完整右侧方法的 backend 必须支持：

- `prepare`；
- `evaluate`；
- `sample`；
- `pdf`；
- 各向异性 shading frame；
- 有限值、事件标志和 capability 查询。

如果 neural evaluator 没有专用重要性采样器，它仍需提供一个无偏的通用 proposal 及正确 PDF；效率可以较低，但不能把 `sample/pdf` 从公共接口删除。

## 可选光照积分能力

以下是 capability，不是基础状态字段：

```text
AnalyticPolygonIntegration
PrefilteredEnvironmentIntegration
NeuralEnvironmentIntegration
DeltaEvents
Transmission
HomogeneousVolume
```

对应的专用 lighting adapter 可以提供：

```text
integrate_polygon_light(...)
integrate_environment(...)
```

缺少专用能力时，renderer 使用 `evaluate/sample/pdf` 的标准路径。不能因为某个 LTC backend 擅长面光积分，就把 LTC 参数暴露给公共 deferred pass。

## neural 实现方式

以下实现都不改变接口：

1. `prepare()` 中的小网络输出解析 closure 参数；
2. `prepare()` 输出 latent，`evaluate()` 中的小网络按 `wi` 解码；
3. `evaluate()` 直接查询 neural BSDF，`sample/pdf()` 使用另一个网络或通用 proposal；
4. 静态网络先把 `MaterialProgram` 编译为 view-independent code，再由 per-pixel 网络生成状态。

MethodBundle 决定使用哪种 shader 和权重。renderer 只看到 descriptor 和接口实现。

## reference adapter

每个源材质族的 reference adapter 实现同样的方向、事件和输出测度语义，但保留该源材质的原生参数、资源和求值算法。reference 可以是确定性的解析/查表实现，也可以像当前随机游走实现一样在 `evaluate()` 内部使用随机样本。

随机 reference 允许：

- `evaluate()` 内部使用随机样本；
- 每像素随时间累积；
- sample count 和方差由 reference 调用方记录；
- 运行成本不满足实时 backend 的固定预算。

reference 直接求值源材质 GT，不经过统一 approximation backend。它不生成通用 packet，也不参与 MethodBundle 的部署成本比较。不同材质族的 reference 不要求共享内部 IR、shader 或资源布局。

## 一致性测试

每个 backend 至少通过：

- Python/Slang 对相同物理状态的求值一致性；
- `evaluate`、`sample`、`pdf` 方向和测度一致性；
- 各向异性旋转测试；
- 有限、非负和 delta 事件测试；
- 白炉或能量诊断；
- 方法 bundle 加载后与训练端导出的响应一致；
- Falcor adapter 恰好处理一次余弦。
