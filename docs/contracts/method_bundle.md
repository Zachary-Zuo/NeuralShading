# MethodBundle 合同

## 它是什么

`MethodBundle` 是一个完整 neural material method 的可部署交付物。viewer、离线图像评测和性能测试只加载 MethodBundle，不直接加载训练目录、裸 `.pt` 或 backend-specific 参数数组。解析基线也使用同一容器，以便执行相同的完整性、成本和比较流程。

一个方法由以下部分共同定义：

```text
MaterialProgram/IR 支持范围
+ view-independent 材质编译器
+ runtime scattering implementation
    目标 neural method：material/spatial latent + shared evaluator weights
                       + per-pixel/per-hit prepare + neural evaluate head
    解析基线：私有解析参数 + analytic evaluate
+ 可选 matched sampler / lighting integration head
+ 权重和资源
```

只改变其中任意一项都产生新的 bundle identity。

## 目录布局

```text
method-bundle/
  manifest.json
  shaders/
  weights/
  resources/
  schemas/
  validation/
    metrics.json
    parity.json
```

纯解析方法允许没有 `weights/`。所有 URI 相对于 bundle 根目录，不能依赖训练机绝对路径。

## manifest 必需字段

```text
format_name                 ncls.method-bundle
format_version
method_id
display_name
created_at
source_git_commit

material_program_schema_versions
supported_ir_ids
scattering_contract_version
backend_id / backend_version
backend_descriptor
runtime_class               realtime / diagnostic

compiler
  kind                      analytic / parameter-network / latent / direct-neural
  feature_contract
  normalization_contract
  architecture_id
  weight_files
  precision

runtime
  platform
  graphics_api
  shader_model
  slang_version
  entry_points

capabilities
cost_claims
training_provenance
validation_provenance
content_hashes
```

`display_name` 只用于 UI；兼容和缓存使用 `method_id` 与内容哈希。

`runtime_class=realtime` 要求 backend descriptor 声明有界执行，并提供 `prepare/evaluate`、明确的 capability 和完整 cost model。neural bundle 还必须声明 evaluator 结构、精度、共享权重和 latent/state 预算。`sample/pdf` 只对声明 `ScatteringSampling` 或 `PathTracingCompatible` 的 bundle 必需。`diagnostic` 方法可以用于建模、开发窗口或离线分析，但 UI 必须明确标注，不能与实时方法混在同一性能排名中。

`capabilities` 至少显式记录方向求值、scattering sampling、path-tracing compatibility、环境光积分、面光积分、透射和非局部能力。入口存在不等于 capability 成立；loader 按 manifest 决定一个 bundle 能进入哪些 renderer path。

## feature 和 normalization

训练特征不能隐藏在 Python 函数中。bundle 必须精确描述：

- 输入的 canonical IR 版本；
- 节点/层 token 顺序；
- 连续参数的变换与范围；
- padding/mask；
- 坐标系和方向编码；
- color model；
- 输出 decoder 的参数约束。

neural evaluator 还必须描述 latent 的生成和过滤方式、`prepare` 形成的 view-conditioned state、`wi` 编码以及 evaluator 输出的物理量。若存在 sampler，bundle 必须说明 proposal family、参数约束和 PDF 测度；若存在 integration head，必须说明它近似的 evaluator 积分与允许的光照编码。

viewer shader 和 Python 训练端使用同一份生成合同。导出后必须对一组固定输入做 Python/Slang parity 测试。

## 静态编译与 per-pixel prepare

MethodBundle 可以实现两阶段执行：

```text
compile_material(MaterialProgram)
  -> view-independent CompiledMaterial

prepare(SurfaceInteraction, wo, CompiledMaterial)
  -> method-specific ScatteringState
```

目标 neural method 先把源材质编译为 material code、spatial latent texture 或二者组合；`prepare` 按着色点、footprint 与观察方向取得 filtered latent 并形成共享 state；`evaluate` 再对每个 `wi` 运行小型 MLP。逐资产优化 latent、feed-forward compiler 和 compiler + refinement 都使用这套生命周期，但必须在 provenance 中准确声明资产如何生成以及参数编辑是否需要重新优化。

共享 evaluator/sampler weights 属于 MethodBundle runtime；材质专属 latent/material code 属于 `CompiledMaterial`。若 bundle 设计包含材质专属权重，manifest 必须把它们标为 per-material asset、计入 compiled material bytes，并记录重新编辑/编译的成本，不能同时把它们当作免费共享权重。

`prepare` 的 feature contract 至少说明 latent texture 的坐标与通道、mip/LOD 选择、UV footprint、shading frame、`wo` 编码、shared trunk 输出和 state invalidation 条件。`sample` 可以使用 prepare 缓存的 proposal 参数，但实际随机数输入、方向生成和 PDF 仍由 sampling entry point 声明。

解析基线可以让 `CompiledMaterial` 直接保存规范化参数。带 physical core 的 neural method 可以在 state 中同时保存 core 所需数据、neural residual latent 和 sampler proposal 参数；这些字段仍完全私有。

这两个阶段是执行生命周期，不规定内部表示字段。

## 加载和兼容

viewer 加载顺序：

1. 验证 manifest schema 和所有内容哈希；
2. 检查平台、Slang、shader model 和散射合同版本；
3. 检查当前 MaterialProgram/IR 是否在支持范围内；
4. 注册 method-specific shader variant；
5. 编译或加载 `CompiledMaterial`；
6. 分配 descriptor 声明的状态资源；
7. 运行 bundle 自带的最小 parity probe；
8. 成功后才允许出现在右侧方法列表中。

不兼容 bundle 必须显示具体原因，不能退回相近方法。

## 成本信息

manifest 中的 `cost_claims` 是静态声明，至少包含：

- compiled material bytes；
- material/spatial latent bytes、mip/LOD 与过滤开销；
- scattering state bytes/pixel；
- prepare 的纹理获取、编码网络和推理精度；
- evaluator MLP 的层宽、层数、精度，以及单次 evaluate 的估计 ALU、网络和纹理查询；
- 可选 sample/pdf 的估计成本；
- 环境光、面光等专用积分器的固定 query 预算；
- 是否使用可变循环或数据相关分支。

`runtime_class=realtime` 的 bundle 还必须满足以下硬线；loader 依据 `cost_claims` 与 descriptor 校验，超线的 bundle 拒绝进入实时方法列表（仍可作为 `diagnostic` 加载）：

| 项 | 硬线 |
|---|---|
| 单次 `evaluate` | ≤ 2,000 MAC（标量 ALU）；声明 cooperative vector 执行且运行时确认支持时 ≤ 10,000 MAC |
| 单次 `prepare` | ≤ 10,000 MAC |
| `state_stride` | ≤ 64 B，或 `inline` |
| compiled material bytes | 均匀材质 ≤ 512 B；spatial latent ≤ 32 B/texel；包含全部烘焙的 material-static 参数 |
| 共享权重 | `evaluate` 权重 ≤ 32 KB；bundle 共享权重总量 ≤ 512 KiB |
| 环境光 / 面光 | 声明 `PrefilteredEnvironmentIntegration` / `AnalyticPolygonIntegration`，或固定 query 预算 ≤ 4 次 `evaluate`/像素 |
| 执行 | `bounded_execution=true`；`prepare/evaluate` 无数据相关循环、无 > 64 元素的函数级数组 |

依据工况（RTX 4090 级、1080p、材质着色 2 ms/帧、每像素 1 次 `prepare` + ≤ 4 次 `evaluate`）见 `docs/research/experiment_framework.md` §0.1。viewer 同时接受 `diagnostic` 与 `realtime`，并按 descriptor 分配私有 state/compiled material 资源；runtime class 不改变方法实现身份。

viewer benchmark 记录实测 prepare、1/8/32 等灯数下的 lighting scaling、环境/面光积分、总帧时间和显存。实测数据不回写 bundle 本体，而以 `(method_id, benchmark_scene_id, device_id)` 保存到运行报告。

## 当前 neural evaluator 部署记录

viewer 的方法 pass 已泛型化。compiled set 在导出时提供 `runtime_adapter`，MethodBundle 固化 shader module、反射生成的 offset、共享 FP16 权重、`CompiledMaterial` table、私有 state stride 与 parity probe：

- bundle 同时保存精确源 `MaterialProgram`、state ID 与 LayerStackIR hash；viewer 只有在 reference 材质 hash 完全一致时才允许选择；
- viewer 不解释 `CompiledMaterial` 或 state 字段，只按 descriptor 分配资源；
- 方法 module 用公共 alias 绑定 associated types，并通过 `INclsScatteringBackend` 提供 `prepare/evaluate/sample/pdf`；
- bundle 加载时必须通过同一 compiled set 的固定方向 GPU parity。

当前 03 轨道的 NVIDIA learned-frame LayerStack 离线预算适配方法与 core-frame candidate 都包含完整 matched GGX9 sampler。前者因真实运行成本标为 `diagnostic`，candidate 标为 `realtime`；分类不靠缩模，也不使用收敛后 quality 数值决定方法对应或训练状态。离线预算适配方法不等同于论文的 online training 复现。

## 训练 run 与 bundle 的边界

训练 run 可以包含 optimizer、last checkpoint、TensorBoard 和调试预测；MethodBundle 只能包含推理必需内容和验证证据。

导出命令必须从一个不可变 checkpoint 生成全新 bundle，计算内容哈希，并记录源 run/checkpoint。viewer 不允许直接监视正在被训练覆盖的 checkpoint。
