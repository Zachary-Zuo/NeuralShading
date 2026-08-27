# 原生 MDL Reference 与 falcor2 官方对照

## 目标与用户价值

在 NeuralShading 当前统一框架内新增保持原生语义的 MDL source family。正式数据路径由项目自有代码驱动 NVIDIA MDL SDK 编译材质，再由当前锁定的 Falcor 8 执行生成的 HLSL，最终通过既有 offline collector 与 live training 路径产生 `TrainingBatch@1`。

锁定的 falcor2 + 同版本 MDL SDK 只作为外部官方实现对照（oracle）：它接收与项目实现完全相同的材质、参数、资源和查询，生成 parity 证据，但不进入正式 provider、采集器、训练 runner 或产品 CLI。这样只有一条正式 reference 路径，同时保留足够强的交叉验证。

首版使用 NVIDIA vMaterials 2 提供当前展示集中缺失的 color-shifting paint、patinated brushed metal、scratch、glaze 与 sheen 外观。MDL 不得先蒸馏为 LayerStack、OpenPBR、MaterialX `standard_surface` 或 USDPreviewSurface 后再继承 MDL GT identity。

## “自己的正确 reference”的边界

本任务实现的是项目自有的 MDL reference integration，而不是重新实现 MDL 语言、标准库或全部 closure 数学：

- MDL SDK 是材质解析、class compilation、标准 closure 与 HLSL target code 的权威语义核心。
- 项目拥有 source identity、参数编辑、MDL SDK 编译桥、renderer state、资源 runtime、方向/纹理查询约定、Falcor 8 执行器和统一数据合同。
- falcor2 + MDL SDK 是独立 renderer integration 对照，不是正式运行时依赖。
- 因为两条实现共享 MDL SDK，falcor2 parity 不能被表述为独立数学模型证明；还必须用解析 MDL fixtures 和 MDL SDK native backend 检查共有错误。

若“自己的”被理解为完全移除 MDL SDK 并自行重写 MDL 编译器和 closure 库，则既不现实，也无法可靠继承 MDL 原生语义，不属于本任务。

## 背景与已确认事实

- 父任务 `08-27-reference-material-candidates` 已归档；正式报告位于 `.trellis/tasks/archive/2026-08/08-27-reference-material-candidates/research/report.md`。
- `NVlabs/neuralappearance` commit `305b4b9c12e679398c487603dd8245c3f348526c` 使用 falcor2 + MDL SDK 做 GPU 在线材质求值；公开 vMaterials 示例是 `Wood_Tiles_Pine_Mosaic`。
- falcor2 commit `d629c967fa800af81cf5c916bfb2a825b012f473` 的 MDL 集成展示了 HLSL target code、argument block/layout、只读数据段、2D texture、BSDF data 3D texture 与 DF handle 的完整接线。
- 当前 Falcor 8 没有现成 MDL material，但其 `ProgramDesc::ShaderModule::fromString()/addString()`、Python `add_string()` 与 `ComputePass(ProgramDesc)` 可以编译内存中的 material-specific HLSL；其 shader/material system 已使用有界的 2D/3D texture 和 buffer descriptor arrays。
- 当前项目已有 Falcor 8/D3D12 与 PyTorch CUDA shared-buffer 路径，因此 live target 不需要通过 falcor2 或 host memory 中转。
- MDL SDK 官方提供 HLSL/GLSL/PTX/native backend、class compilation、renderer integration 与 texture derivative 示例。项目可以直接集成 SDK，不需要复制 falcor2 renderer。
- `neuralappearance` 的 `learnable=true` 会改变 reference 行为；本任务权威路径与对照路径都固定 `learnable=false`。
- MDL SDK 为 BSD-3-Clause；vMaterials 2 content pack 受 NVIDIA Omniverse terms 约束。用户已允许首版下载并安装该 pack；获取脚本仍必须要求显式接受 terms，资产不进入根 Git。

## 单一正式路径

```text
SourceSnapshot
  -> 项目 MdlSdkCompilerBridge（MDL SDK 语义与 HLSL codegen）
  -> 项目 MdlReferenceProgram / MdlGpuQueryRuntime
  -> 当前 Falcor 8（唯一正式 GPU executor）
  -> EvaluatedBlock / TrainingBatch@1
```

验证时另行运行：

```text
同一 SourceSnapshot + 同一 QueryPacket
  -> 锁定 falcor2 + 同版 MDL SDK
  -> OracleResult（只进入 artifacts/parity，不回流正式数据）
```

## 需求

- R1：新增 versioned `mdl.program@1` source family。canonical snapshot 至少固定 pack/version/archive identity、module path、export signature、authored typed arguments、transitive module/resource hashes、MDL language与 MDL SDK build identity。
- R2：提供 typed parameter discovery/edit。标量、布尔、enum、向量/颜色与受支持 resource 参数可编辑；不支持的参数显示为 read-only。编辑产生新 snapshot identity，拒绝 stale patch、越界资源与类型不匹配。MDL argument block offset 不进入公共接口；每个新 snapshot 由 MDL SDK 重新构造权威 argument block。
- R3：项目自有 `MdlSdkCompilerBridge` 直接链接锁定的 MDL SDK，负责 module/export discovery、class compilation、HLSL target code、callable symbol、argument block/layout、RO data、texture/resource descriptor 与依赖闭包导出。桥接产物有版本化 schema 和内容 hash，放在 ignored build/cache，不作为 source GT。
- R4：正式 `MdlProvider` 只能依赖项目桥、当前 Falcor 8 和现有公共合同。`falcor2` 不得被正式 provider、batch source、collector、训练 runner 或产品 CLI import、启动或动态选择。
- R5：当前 Falcor 8 执行 project-owned query adapter、renderer runtime 与 MDL SDK 生成的 HLSL。V1 必须支持 shortlist 所需的 argument blocks、RO data、2D textures、BSDF data 3D textures、sampler/wrap/crop/gamma 语义；纹理过滤冻结为与 oracle 一致的 `ExplicitLod(0)`，不支持资源必须在编译时显式拒绝。
- R6：GPU reference 输出当前数据合同的线性 RGB `f * |n_s · wi|`。frame 固定为 geometric normal `+Z`、tangent `+X`、右手 bitangent `+Y`；`wo/wi` 角色、front/back face、IOR、position、UV、固定 LOD、scene units 与 transforms 全部进入 query identity。Lambertian analytic fixture 必须证明没有方向互换或重复 cosine。这不改变目标 neural runtime 的 `evaluate()` 输出线性 `f` 合同。
- R7：offline 与 live 共用同一个 `MdlGpuQueryRuntime` 和 shader。live 使用现有 Falcor/CUDA shared buffers，禁止 target 的 CPU/NumPy round-trip；offline 只在 `EvaluatedBlock`/HDF5 sink 边界读取 host 数据。
- R8：首版 capability 为 surface-BSDF evaluate-only。`sample/pdf`、emission、volume、displacement、measured BSDF 与 light profile 不进入 V1；任何使用必须明确 unsupported，不能以 distillation、baked maps 或其他 BSDF 冒充。
- R9：锁定 falcor2 commit 和它使用的 MDL SDK `2025.0.0-387700.1252` build。oracle 只由 parity 工具/测试显式启动，并只写 `artifacts/reference-parity/mdl/`。项目正式运行在没有 falcor2 的环境中仍可工作；发布 MDL reference 前 oracle parity 是验收门。
- R10：首批 vMaterials 2 modules 为 `Carpaint_Shifting_Flakes`、`Copper_Antique_Brushed_Patinated`、`Aluminum_Scratched`、`Ceramic_Tiles_Glazed_Versailles`、`Velvet`，另登记 `Wood_Tiles_Pine_Mosaic` 做 NVIDIA pipeline correspondence。正式数值验收至少覆盖无普通 2D texture 的 car paint 与带 texture 的 patinated copper，其余完成 discovery/load/evaluate smoke。
- R11：同步 schema/registry/CLI、构建与资产获取脚本、unit/GPU/integration/oracle 测试、reference package 中文说明和稳定文档。不得修改 Falcor、falcor2、MDL SDK、MaterialX 或其他上游源码；所有 external 工作树在验收后保持干净。

## 验收标准

- [ ] A1（需求交付；来源：用户）：正式数据流只有 `项目 MDL bridge -> 当前 Falcor 8 -> 统一 batch`。静态导入审计与运行证据证明 falcor2 不在正式 provider、collector、live batch、训练 runner或产品 CLI 的依赖闭包中。
- [ ] A2（需求交付；来源：用户）：当前框架能登记、加载和编辑原生 MDL material，并通过统一 offline/live producer 生成有效 `TrainingBatch@1`。
- [ ] A3（语义正确性；来源：`docs/material_scope.md`）：snapshot identity 保留 module/export/arguments/imports/resources/SDK；distill、bake 与 PreviewSurface 派生物不继承 MDL GT identity。
- [ ] A4（数值实现正确性；来源：MDL SDK）：fixture 和 shortlist 的 HLSL、argument block、RO data 与资源表全部来自锁定 MDL SDK；schema/hash 可复现，缺失或未知资源类型 fail closed。
- [ ] A5（理论/语义正确性；来源：解析材质定义）：constant diffuse、参数化 diffuse、程序化 checker texture fixtures 在预定义查询上满足解析期望；覆盖 `wo/wi`、cosine、frame、颜色空间、wrap/crop 和 `ExplicitLod(0)`。
- [ ] A6（数值实现正确性；来源：MDL SDK native backend）：解析 fixtures 的 Falcor 8 HLSL 结果与项目 bridge 驱动的 MDL SDK native execution 在 disjoint queries 上满足预先冻结的 float32 容差。
- [ ] A7（数值实现正确性；来源：falcor2 + MDL SDK oracle）：car paint 与 patinated copper 使用完全相同 snapshot/query packet 对 falcor2 oracle 做逐方向 parity；记录 falcor2 commit、MDL SDK build、query identity、容差的 `source/scope/why_hard/failure_action` 与结果。失败视为 integration defect，不把 falcor2 改成正式 fallback。
- [ ] A8（接口正确性；来源：项目接口）：live batch 全 tensor 位于同一 CUDA device，metadata 明确 `host_readback=false`；源码审计与 GPU test 均未出现 CPU/NumPy target round-trip。offline 产物只进入 `data/reference-responses/*.h5`。
- [ ] A9（接口正确性；来源：项目接口）：descriptor 只声明已实现的 evaluate/spatial 能力；derivative-footprint、公共 matched `sample/pdf`、emission、volume、displacement、measured BSDF 与 light profile 均明确 unsupported。
- [ ] A10（需求交付；来源：用户与 `docs/repository_policy.md`）：获取脚本要求显式 NVIDIA terms flag；manifest 固定 vMaterials 2 URL/version/size/SHA-256/license/module/export/resources，原包及展开内容不进根 Git。
- [ ] A11（需求交付；来源：用户）：五个 shortlist 与 `Wood_Tiles_Pine_Mosaic` 均完成 discovery/load/evaluate smoke，car paint 与 patinated copper 完成正式 parity。
- [ ] A12（质量交付；来源：项目规范）：相关 unit、GPU、integration、oracle、registry/tree validation 通过，稳定中文文档同步，Falcor/falcor2/MDL SDK/MaterialX 等上游工作树保持干净。

正式数值容差只能从 float32 误差分析和隔离 calibration queries 冻结；formal queries 与 calibration disjoint，formal 结果不得反向修改容差。

## 已冻结决策

- falcor2 + MDL SDK 是 validation oracle，不是项目正式 evaluator。
- 当前 Falcor 8 是唯一正式 GPU executor；不新增第二套 collector、batch schema 或 training runner。
- 项目直接依赖 MDL SDK 的编译/closure 语义，不依赖 falcor2 runtime；不重写 MDL 编译器和 libbsdf。
- MDL 使用 class compilation 保存原生可编辑参数，`learnable=false` 保持权威散射语义。
- V1 使用固定 `ExplicitLod(0)`，统一 batch 的 `uv_dx/uv_dy/mip_level` 固定为零并声明未被 reference 消费；不以这些兼容字段冒充 derivative texture filtering。
- oracle 结果只用于 parity，不得写入正式 HDF5、训练 target 或 source snapshot。
- 用户允许 vMaterials 2 成为 V1 必过验收资产；下载动作仍需显式确认 NVIDIA terms。

## 范围外

- 不在本任务训练 neural candidate、比较模型 quality 或发布 `ScatteringPackage`。
- 不接入完整 Omniverse scene、Base Materials、Automotive Materials 或全部 vMaterials catalog。
- 不提供完整 MDL surface/volume/emission/displacement runtime，也不在首版提供 matched sampler。
- 不实现一个脱离 MDL SDK 的自有 MDL 编译器或 closure 库。
- 不修改任何上游源码；若 current Falcor 8 无法正确执行 MDL SDK HLSL/resource ABI，不以 falcor2 正式运行时、host copy、PreviewSurface、distill 或 bake 降级并宣称完成，而是保留证据并返回规划。
