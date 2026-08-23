# MethodBundle 合同

## 它是什么

`MethodBundle` 是一个完整拟合方法的可部署交付物。viewer、离线图像评测和性能测试只加载 MethodBundle，不直接加载训练目录、裸 `.pt` 或某个 K2 参数数组。

一个方法由以下部分共同定义：

```text
MaterialProgram/IR 支持范围
+ view-independent 材质编译器
+ per-pixel prepare/decode
+ ScatteringBackend
+ 可选专用 lighting integration
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

`runtime_class=realtime` 要求 backend descriptor 声明有界执行，并提供完整 `prepare/evaluate/sample/pdf`。`diagnostic` 方法可以用于开发窗口或离线分析，但 UI 必须明确标注，不能与实时方法混在同一性能排名中。

## feature 和 normalization

训练特征不能隐藏在 Python 函数中。bundle 必须精确描述：

- 输入的 canonical IR 版本；
- 节点/层 token 顺序；
- 连续参数的变换与范围；
- padding/mask；
- 坐标系和方向编码；
- color model；
- 输出 decoder 的参数约束。

viewer shader 和 Python 训练端使用同一份生成合同。导出后必须对一组固定输入做 Python/Slang parity 测试。

## 静态编译与 per-pixel prepare

MethodBundle 可以实现两阶段执行：

```text
compile_material(MaterialProgram)
  -> view-independent CompiledMaterial

prepare(SurfaceInteraction, wo, CompiledMaterial)
  -> method-specific ScatteringState
```

纯解析方法可以让 `CompiledMaterial` 直接保存规范化参数。结构化网络可以先生成 material code，再由小网络按视角生成状态。直接 neural evaluator 可以让状态只持有 material latent 和观察方向。

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
- scattering state bytes/pixel；
- prepare 网络参数量和推理精度；
- evaluate/sample 的估计 ALU 与纹理查询；
- 是否使用可变循环或数据相关分支。

viewer benchmark 记录实测 prepare、lighting、总帧时间和显存。实测数据不回写 bundle 本体，而以 `(method_id, benchmark_scene_id, device_id)` 保存到运行报告。

## K2 的位置

当前 176-byte“精确顶层界面 + 两个 LTC 残差瓣”导出为普通的 `legacy-ltc-k2` MethodBundle：

- 它实现相同散射合同；
- 其 lobe 字段只存在于该 backend 内；
- viewer 和 deferred pass 不直接访问 lobe；
- 它的 6.73% median / 31.20% p90 只作为历史基线；
- 它不能决定其他 bundle 的状态大小、decoder 或 lighting 实现。

迁移后的 `legacy-ltc-k2` 后端私有 ABI 仍为 176 bytes，但使用独立 `LTK2` magic 和新 `NclsLayerInterfaceIR` 字段；不再复用名为公共 `ClosurePacket` 的类型。P1 参数网络已有与 PyTorch 逐层一致的 Slang compiler，导出时同时打包权重布局、feature/runtime schema、compiler shader、backend shader 和 parity probe，因此完整方法可标为 `realtime`。

这里的 `realtime` 只描述运行时完整且有界，不是质量结论。6.73% median / 31.20% p90 的长尾仍使 K2 保持 `legacy` 基线身份；后续方法可以选择不同 state 大小、解析表示或 neural decoder。

## 训练 run 与 bundle 的边界

训练 run 可以包含 optimizer、last checkpoint、TensorBoard 和调试预测；MethodBundle 只能包含推理必需内容和验证证据。

导出命令必须从一个不可变 checkpoint 生成全新 bundle，计算内容哈希，并记录源 run/checkpoint。viewer 不允许直接监视正在被训练覆盖的 checkpoint。
