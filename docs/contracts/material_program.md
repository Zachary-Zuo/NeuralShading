# MaterialProgram 合同

## 它是什么

`MaterialProgram` 是项目唯一的公共材质描述。它面向编辑、数据生成、训练输入和 viewer，不与 K2、LTC、latent 维数或任何拟合后端绑定。

迁移前的固定数组式 `LayerStack` 已被内部 `LayerStackIR` 取代。它仍是第一阶段研究范围的中心，但不承担顶层材质文件格式。

## 顶层结构

逻辑结构如下；具体 JSON Schema 位于 `src/ncls/core/material/schemas/material_program_v1.schema.json`，并由合同测试锁定。

```text
MaterialProgram
  schema_name             固定为 ncls.material-program
  schema_version          顶层容器版本
  program_id              内容寻址或稳定 UUID
  color_model             v1 为 linear-srgb
  nodes[]                 有类型 DAG 节点
  resources[]             纹理等外部资源描述
  outputs
    surface
    interior_medium
    exterior_medium
    emission
    opacity
    displacement
  metadata                不影响物理语义的名称、标签和 UI 信息
```

只有 `surface` 是 v1 必需输出。其他输出从第一版就存在，但可以为空；它们不能被塞进 `metadata`。

## 节点身份和版本

每个节点包含：

```text
Node
  id
  operation_namespace
  operation_name
  operation_version
  inputs
  parameters
```

完整操作身份是 `(namespace, name, version)`。新增操作不会提升 `MaterialProgram.schema_version`；改变已有操作语义必须增加该操作的 `operation_version`。只有顶层容器结构发生不兼容改变时才提升 `schema_version`。

加载器遇到未知操作或不支持的版本时必须返回明确的 capability error，不能忽略节点或套用相近操作。

## 类型系统

第一阶段注册以下端口类型：

```text
Float
Float2
Float3
Color3
Normal3
Spectrum             预留，v1 不实现
Interface
Medium
Surface
Emission
Opacity
Displacement
```

`Color3` 遵循程序顶层的 `color_model`，不能用来存储法线、方向或 IOR。`Normal3` 与普通 `Float3` 分开，避免后续纹理和坐标变换产生歧义。

节点注册表定义每个输入、输出和参数的类型、单位、范围、默认值与是否允许空间变化。图连接必须严格类型匹配，不进行隐式 `Color3`/`Float3` 转换。

## 参数来源

参数值统一使用 `ParameterSource`，从第一版保留以下类别：

```text
constant
texture
vertex_attribute
procedural
```

v1 运行时只要求实现 `constant`。其他类别可以被解析和验证，但不支持它们的 backend 必须拒绝编译。这样以后加入纹理不会改变节点或顶层合同。

纹理资源需要显式记录：

- 资源 ID 和相对 URI；
- 颜色空间或数据语义；
- channel swizzle；
- wrap/filter；
- UV set；
- 可选的 UV transform。

## v1 原子节点

第一阶段至少包含以下操作：

```text
ncls.interface.rough_dielectric@1
ncls.interface.rough_conductor@1
ncls.interface.diffuse@1
ncls.interface.sheen@1
ncls.medium.homogeneous@1
ncls.composition.layer_stack@1
```

类型专属参数不能复用含义不一致的字段：

- `rough_dielectric` 使用标量相对 IOR、`alpha_x/y` 和切线旋转；
- `rough_conductor` 使用 RGB `eta/k`、`alpha_x/y` 和切线旋转；
- `diffuse` 使用 RGB albedo；
- `sheen` 使用颜色和其明确版本的粗糙度定义；
- `homogeneous` 使用 `sigma_a/sigma_s/g/thickness`，单位和 v0 RGB 消光限制写入节点版本说明。

所有角度使用弧度。`alpha_x/y` 直接表示微表面分布参数，不进行隐藏的感知粗糙度平方。

## LayerStack 子图

`ncls.composition.layer_stack@1` 接收：

```text
interfaces: Interface[]
media: Medium[]
```

对于 N 个界面必须恰好有 N−1 个层间介质。v1 进一步限制：

- 1 ≤ N ≤ 8；
- 只计算从物体外部观察的反射；
- 最底层以上的界面必须能够透射，v1 因而只允许 `rough_dielectric@1`；
- 最底层必须是支持的不透明基底；
- 层间介质采用当前局部、无横向位移的 slab 假设；
- 有体散射时 RGB 总消光相同是 `homogeneous@1` 的范围限制，不是通用介质定律。

通过验证的子图规范化为 `LayerStackIR`。该 IR 可以使用固定 GPU 布局，但其 ABI 版本与 `MaterialProgram` 版本独立。

## 表面坐标和各向异性

材质程序不保存世界空间切线。各向异性方向来自着色点的标准切线坐标系，再叠加节点的局部 `tangent_rotation`。

后续法线图、切线图或 procedural frame 节点修改的是 `SurfaceInteraction` 使用的 shading frame，不改变散射接口签名。

## 规范化和哈希

规范化过程必须：

1. 验证 DAG 无环、节点 ID 唯一且所有引用存在；
2. 验证节点版本、端口类型、参数范围和资源；
3. 移除不影响物理语义的 metadata；
4. 对 map key、节点顺序和浮点序列化做 canonicalization；
5. 输出内部 IR 及其 SHA-256；
6. 记录原始程序哈希、规范化器版本和 IR ABI 版本。

family split 和数据去重都使用规范化后的物理语义哈希，不能使用文件路径或 JSON 原始字节顺序。

## 扩展规则

后续可以增加 `mix`、纹理、normal transform、thin film 或新的原子界面，而无需改变顶层合同。扩展必须同时提供：

- 节点注册信息和中文语义说明；
- 规范化规则；
- reference 支持或明确的 reference capability；
- 至少一个 backend 支持，或清晰的 unsupported error；
- schema、round-trip 和 shader ABI 测试。

非局部 BSSRDF、完整 volume、displacement 等使用预留输出连接独立 renderer 阶段，不伪装成局部 `Surface` 节点。
