# 原生 MDL Reference 技术设计

## 1. 架构边界

`mdl.program@1` 是独立的原生 source family。项目共享 source、query、batch、registry 与训练合同，但 MDL source 使用 MDL SDK 保留自己的语言、参数图和 closure 语义。

正式路径只有一条：

```text
vMaterials / project MDL fixtures
  ↓ MdlSourceCatalog + dependency hashing
canonical SourceSnapshot + typed editor
  ↓
MdlSdkCompilerBridge（项目源码，直接链接锁定 MDL SDK）
  ↓ ncls.mdl-compiled-artifact@1
MdlReferenceProgram + MdlGpuQueryRuntime
  ↓ dynamic HLSL module + project renderer runtime
当前 Falcor 8 / D3D12
  ↓ existing Falcor-CUDA shared buffers
offline EvaluatedBlock / live TrainingBatch@1
```

falcor2 不在上图中。它只存在于验证侧：

```text
frozen SourceSnapshot + frozen QueryPacket
  ├─ project/current-Falcor result
  └─ falcor2/MDL-SDK oracle result
       ↓
     parity report under artifacts/
```

验证需要两个 executable/runtime 是刻意的交叉检查，不是两条产品路径。oracle 不能被 provider 选择，也不能在项目实现失败时接管正式求值。

## 2. 固定依赖与所有权

### 2.1 上游固定

- 当前 Falcor：tag `8.0`，commit `9dc819c162b2070335c65060436041690b7937f8`，是唯一正式 GPU executor。
- MDL SDK：与 oracle 一致的 `2025.0.0-387700.1252` Windows x86-64 package；URL、size、SHA-256、license 与 runtime DLL 列表进入 reference manifest。
- falcor2 oracle：commit `d629c967fa800af81cf5c916bfb2a825b012f473`，其 submodules 由 commit 固定。
- neuralappearance correspondence：commit `305b4b9c12e679398c487603dd8245c3f348526c`，只登记公开配置与查询语义，不是运行时依赖。
- vMaterials：2.4.0 archive；首次正式下载时冻结 URL、ETag、Content-Length 与 SHA-256。

MDL SDK package 与 falcor2 clone 位于 ignored `external/`；vMaterials 位于 ignored `assets/source-materials/mdl-vmaterials2/2.4.0/`。根仓库只保存项目源码、fetch/build 脚本、patch-free pin、license/asset manifest 和 reference package。若新增固定上游，需要同步根 `AGENTS.md` 的锁定列表。

### 2.2 项目拥有的实现

- `tools/reference/mdl_sdk_bridge/`：小型 C++ executable，直接使用 MDL SDK API。
- `src/ncls/source_materials/families/mdl.py`：canonical snapshot、依赖闭包与 typed editor。
- `src/ncls/references/programs/mdl.py`：compiled artifact schema、implementation identity 与 capability。
- `src/ncls/data/providers/mdl.py`：统一 offline/live provider 和 current-Falcor runtime。
- `shaders/ncls/reference_backends/mdl_query.slang`：项目 query adapter。
- `shaders/ncls/reference_backends/mdl_runtime.slangh`：MDL HLSL renderer runtime，按官方 SDK runtime ABI 实现 texture/scene-data 等回调。
- `tools/reference/mdl_oracle/`：隔离的 falcor2 parity runner；只可被 integration/oracle test 显式调用。

项目不修改任何上游文件。若从 MDL SDK example 或 falcor2 适配 runtime 代码，保留原 copyright/SPDX，并在第三方通知中登记来源与修改。

## 3. Source identity 与参数编辑

### 3.1 Canonical snapshot

`SourceSnapshot.native_payload` 保存 canonical UTF-8 JSON：

```json
{
  "schema": "ncls.mdl-source@1",
  "pack_id": "nvidia.vmaterials@2.4.0",
  "module": "::vMaterials_2::Paint::Carpaint::Carpaint_Shifting_Flakes",
  "export": "Carpaint_Shifting_Flakes(...exact signature...)",
  "arguments": {
    "example": {"mdl_type": "color", "value": [0.8, 0.1, 0.05]}
  },
  "compilation_mode": "class",
  "mdl_language": "...discovered...",
  "mdl_sdk": "2025.0.0-387700.1252"
}
```

`resource_hashes` 使用 pack-relative URI，覆盖 root module、transitive imported user modules、textures、light profile/measurement declarations以及其他外部资源。标准库/内建 modules 通过 MDL SDK build identity 固定，不逐文件复制进 source snapshot。加载时拒绝绝对路径、`..`、symlink escape、缺失文件和 hash 漂移。

依赖发现以 MDL SDK module/dependency API 为准，而不是正则扫描源码。验收时从仅包含 manifest closure 的临时 search root 重新编译每个登记材质，验证没有漏记外部依赖。

### 3.2 Typed editor

bridge 的 `inspect` 命令输出中立 `ncls.mdl-inspection@1` JSON，包含 exports、exact signatures、参数类型、默认值、annotations、range/choice、resource 类型和可编辑原因。`MdlFamilyDefinition` 将其映射到项目 `ParameterDescriptor`：

- `bool/int/float/double`、`color`、`float2/3/4` 和有 choice metadata 的 enum 可编辑；double 在 snapshot 保留 MDL 类型，桥负责安全转换。
- 受支持的 `texture_2d` resource 以受约束的 pack-relative URI 编辑。
- matrix、array、struct、string、light profile、BSDF measurement 及 V1 未覆盖类型显示 read-only，并保留原因。
- `apply_edit()` 执行 stale snapshot、类型、finite/range、resource containment 与 hash 验证后，生成新 snapshot。

Python 不直接改 argument block bytes。每个新 snapshot 交给 MDL SDK 重新实例化和 class-compile；SDK 生成新的 argument block/layout，从而避免把某个 SDK build 的 offset 变成公共接口。

## 4. MDL SDK 编译桥

### 4.1 进程边界

V1 使用 project-owned executable 而不是把 MDL SDK C++ 类型暴露进 Python。这样桥的 ABI 小、生命周期明确，也不会把 MDL SDK headers/COM-like handles扩散到训练代码。桥提供：

```text
mdl_sdk_bridge inspect --request request.json --output inspection.json
mdl_sdk_bridge compile --request request.json --output-dir <cache-entry>
mdl_sdk_bridge native-evaluate --request request.json --queries queries.bin --output result.bin
```

`inspect/compile` 只发生在 source load/prepare/cache miss，不在每个 query 上启动进程。`native-evaluate` 只用于 fixtures 的验收，不是正式数据路径。

### 4.2 编译选项

authority 配置固定：

- class compilation；保留 authored arguments，不启用 learnable/mollification 变换。
- HLSL backend 请求 `internal_space=coordinate_world`；锁定的 MDL 2025 HLSL backend 会拒绝该 option，formal 与 falcor2 oracle 都使用相同的 backend 默认值。这个有效状态进入 compiler identity，不能在文档中把被拒绝的请求写成已启用配置。
- `use_renderer_adapt_normal=on`。
- `texture_runtime_with_derivs=off`，纹理查询固定为 `ExplicitLod(0)`；`num_texture_spaces` 与 `num_texture_results` 由 manifest 固定。该选择与冻结的 falcor2 oracle 查询合同一致，V1 不声明 derivative filtering。
- `fast_math`、optimization level、DF handle slot mode 与 spectrum conversion policy 进入 reference implementation identity。
- target functions 至少包含 `init`、`ior`、`thin_walled`、`surface.scattering` 和 `geometry.normal`；V1 不导出 emission/volume/displacement 的公共能力。

桥检查每个 SDK status/context message，并把 warning/error 结构化输出。任何未识别 target code resource、超过静态上限的 texture/DF handle 或 unsupported closure capability 都 fail closed。

### 4.3 Compiled artifact

ignored cache `build/mdl-reference/cache/<compile-key>/` 使用 `ncls.mdl-compiled-artifact@1`：

```text
manifest.json              # source ID、compiler/options、callable symbols、resource counts/hashes
generated.hlsl             # MDL SDK target code
argument-block.bin
ro-data/<name>.bin
textures.json              # MDL index、shape、gamma、path、dimensions、resource hash
bsdf-data/<index>.bin      # SDK 提供的 3D float tables
diagnostics.json
```

`compile-key` 由 source snapshot ID、bridge implementation hash、MDL SDK build 与完整 codegen options 组成。加载时逐项验证文件 hash。compiled artifact 是可重建执行缓存，不是 source GT，也不进 Git 或 HDF5。

## 5. 当前 Falcor 8 执行器

### 5.1 动态 shader module

current Falcor Python binding 支持：

```python
desc = falcor.ProgramDesc()
generated = desc.add_shader_module("ncls_mdl_generated")
generated.add_string(hlsl_source, virtual_path)
desc.add_shader_module().add_file(query_adapter)
desc.cs_entry("main")
compute = falcor.ComputePass(device, desc)
```

第一阶段 feasibility gate 必须先证明真实 MDL SDK 生成的最小 diffuse HLSL 能通过锁定 Slang `2024.1.34` 编译并执行；不能只用手写 HLSL substitute 通过门。

### 5.2 有界资源 ABI

query adapter 用 compile artifact 的 counts 生成 compile-time defines：

- `Texture2D<float4> gMdlTextures[ArrayMax<1, MDL_TEXTURE_2D_COUNT>.value]`
- `Texture3D<float> gMdlBsdfData[ArrayMax<1, MDL_BSDF_DATA_COUNT>.value]`
- sampler arrays、argument block 与 RO byte-address buffers
- input/output structured buffers

Python 按 MDL index 绑定 Falcor textures/buffers。所有 descriptor counts 有 manifest 上限并进入 program cache key；V1 不依赖 unbounded/bindless descriptors。current Falcor 自己的 `MaterialSystem.slang` 已证明同类有界数组是受支持的，但 feasibility gate 仍要覆盖 Python array binding、2D/3D texture 和 byte-address load 的最小实例。

2D texture 按 SDK 报告的 gamma、shape、wrap/crop/frame 语义加载；renderer runtime 实现 MDL 约定的查找函数。BSDF data texture 使用 SDK canvas 的原始 float table。UDIM、animated texture、cube/3D authored texture、light profile 与 measured BSDF 若未实现，编译时拒绝，不取第一帧或替换资源后继续。

### 5.3 Shading state 与 query convention

每个 query 显式构造 MDL shading state：

- geometric normal `(0,0,1)`，tangent_u `(1,0,0)`，tangent_v `(0,1,0)`；shading normal 初始与 geometric normal 一致。
- object/world transforms 为 identity，`meters_per_scene_unit=1`，object ID 和 animation time 固定并进入 query contract。
- `position`、`uv` 来自 `SurfaceSample`；V1 的纹理查询固定为 `SampleLevel(..., 0)`。统一接口仍携带 `uv_dx/uv_dy`，但 MDL V1 明确不消费它们，live batch 对应字段固定为零。
- 项目 `wo` 始终是离开表面、指向观察者的方向，项目 `wi` 是入射光方向。适配到 MDL evaluate data 的字段以 SDK 定义和 falcor2 oracle 为准，不根据局部变量名猜测。
- project output 为 `rgb-bsdf-times-absolute-shading-normal-light-cosine`。constant diffuse fixture 必须得到 `albedo / pi * abs(dot(n_s, wi))`，以此冻结方向与 cosine 约定。

V1 正式 query 为 reflection upper hemisphere。若材质包含 backface/transmission 定义，snapshot 仍保留原生定义，但 descriptor 不宣称已覆盖这些 query domain。

### 5.4 Runtime 生命周期

`MdlGpuQueryRuntime`：

1. 验证或生成 compiled artifact。
2. 用 current Falcor 8 创建 material-specific `ComputePass`。
3. 加载 argument/RO buffers 与 texture resources。
4. 预分配至少两个 shared-buffer slots。
5. `evaluate_torch()` 只更新 query buffers、dispatch，并返回同 CUDA device 的 view；不做 host sync/readback。
6. offline `evaluate()` 使用同一 runtime，在 `EvaluatedBlock` 边界读取结果。

材质参数或 source snapshot 改变时创建新的 immutable runtime/cache entry，不原地改变运行中的 compiled material。live lease 与 slot 生命周期沿用当前 `ReferenceLiveBatchSource` 合同。

## 6. Reference program 与统一数据流

`MdlReferenceProgram` 登记 `ncls.mdl-vmaterials2@1`：

- `compile_material()` 返回 renderer-neutral `MaterialPayload`，其中只保存 canonical source binding、compiled artifact identity 与有界 resource descriptors；不暴露 MDL SDK handle/argument offset。
- implementation identity 包含 bridge source hash、runtime/query shader hash、MDL SDK build、Falcor commit、Slang version 和 codegen options。
- capability 只声明 deterministic surface evaluate 与 UV spatial state；不声明 derivative footprint。

`MdlProvider` 使用现有 `BaseProvider`、`QueryPlan`、`EvaluatedBlock` 和 HDF5 collector。`MdlLiveReferenceBatchSource` 复用现有 direction/surface/query RNG 与 slot/lease 机制，不创建 MDL 专用 dataset schema、reader 或 training runner。CLI 只增加 source/provider 选择与 asset ID；不提供 `--renderer falcor2` 一类正式开关。

## 7. falcor2 oracle 隔离

`tools/reference/mdl_oracle/` 是进程外验证工具：

- 输入为版本化 `ncls.mdl-oracle-request@1`，内含 canonical source payload、resolved resource root、query packet、frame/state 常量与 `learnable=false`。
- 使用锁定 falcor2 + 同版 MDL SDK，按 neuralappearance/falcor2 官方集成执行。
- 输出版本化 response 与 provenance，只写 `artifacts/reference-parity/mdl/<run-id>/`。
- parity harness 分别启动 project runner 和 oracle runner；两个 Python module search paths 不混用。
- oracle output 不得被 source/provider/collector 读取，也不得成为训练 target fallback。

增加静态边界测试：扫描 `src/ncls/data/providers/mdl.py`、`src/ncls/references/`、live batch、collector 与 CLI 的 import/process launch closure，拒绝 `falcor2`、其 vendored SlangPy 或 oracle module。另有测试证明移除/重命名 ignored falcor2 clone 后，正式 MDL provider 仍能运行 fixtures。

## 8. 正确性证据

正确性分四层：

1. **ABI/编译 gate**：真实 MDL SDK HLSL 在 current Falcor 8/Slang 中编译；argument/RO/2D/3D resources 的最小探针可读。
2. **解析 fixtures**：constant diffuse、editable diffuse 与小型 checker texture 覆盖方向、cosine、参数更新、颜色空间、UV orientation、wrap/crop 与 `ExplicitLod(0)`。expected values 在看 GPU 结果前由定义冻结。
3. **MDL SDK native cross-check**：同一 bridge、同一 snapshot/query 对 fixtures 生成 native backend response，检查 HLSL renderer runtime 与 SDK native runtime 一致。
4. **falcor2 integration parity**：同一 vMaterials snapshot/query packet 对 car paint 与 patinated copper 做 disjoint formal parity；其余 shortlist 做 smoke。

falcor2 与项目路径共享 MDL SDK closure 实现，因此第 4 层主要发现 renderer state、资源、参数块、方向和 filtering 接线错误；第 2 层负责提供不共享这些接线的数学不变量。

calibration 与 formal query sets 使用不同 seed。容差记录 `source/scope/why_hard/failure_action`，由 float32 分析和 calibration 冻结；formal 只读取冻结值，不能覆盖。pilot 证据标记 diagnostic，不进入正式结论。

## 9. Fancy 资产登记

`references/mdl-vmaterials2-v1/assets.json` 登记：

- `Paint/Carpaint/Carpaint_Shifting_Flakes.mdl`
- `Metal/Copper_Antique_Brushed_Patinated.mdl`
- `Metal/Aluminum_Scratched.mdl`
- `Ceramic/Ceramic_Tiles_Glazed_Versailles.mdl`
- `Fabric/Velvet.mdl`
- `Wood/Wood_Tiles_Pine.mdl` 的 `Wood_Tiles_Pine_Mosaic` export

manifest generator 必须用 bridge discovery 固定 exact module/export signature、默认/authored arguments、完整资源闭包与 capability audit。若某材质使用 V1 unsupported resource/capability，不能静默挑一个更简单材质顶替；记录 evidence，判断是选择同 module 的公开 preset export，还是回到 planning 扩展 V1。

获取脚本 `scripts/fetch_mdl_assets.ps1 -VMaterials2 -AcceptNvidiaOmniverseTerms` 在任何网络或文件写入前检查 terms flag。archive、展开文件和 generated compiled artifacts 都不进根 Git。

## 10. 失败、回滚与兼容

- **Gate 1：Slang 兼容。** 若真实 MDL HLSL 无法由锁定 Falcor/Slang 编译，先定位语法/runtime ABI；不得改用手写近似 closure 或 falcor2 正式路径。
- **Gate 2：资源 ABI。** 若 current Falcor Python 无法绑定所需数组，在项目侧增加最小 native binding/plugin 方案并回到 design review；不修改 Falcor upstream。
- **Gate 3：texture filtering。** V1 formal/oracle 必须共同使用并登记 `ExplicitLod(0)`；任何未来 derivative filtering 必须升级 query capability 并重新验收，不能沿用 V1 parity 身份。
- **Gate 4：shortlist capability。** measured BSDF、light profile、UDIM/animation 或静态资源上限触发 fail closed，并登记具体 material/export/resource。
- **Gate 5：parity。** 解析/native 已过但 falcor2 formal parity 失败时按 integration defect 调查 frame、state、resource 和 backend options；不放宽门槛或切换正式 executor。

旧 source family、reference registry 和 HDF5 schema保持向后兼容。普通 unit/import 路径不要求 MDL SDK、vMaterials 或 falcor2；选择 MDL source 时缺失 MDL SDK/asset 给出明确诊断。正式 MDL GPU 测试通过现有 `scripts/run_falcor_python.ps1` 启动，oracle 测试使用单独 wrapper。

收尾时检查根仓库改动范围，并确认 `external/Falcor`、`external/MaterialX`、`external/OpenPBR`、`external/openpbr-bsdf`、`external/glm`、MDL SDK 与 falcor2 上游目录均保持干净。
