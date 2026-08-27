# 原生 MDL Reference 实施计划

## 0. 开始门与环境

- [ ] 用户批准本轮修订后的最终规划摘要后，运行 `conda run -n neural-shading python .trellis/scripts/task.py start 08-27-mdl-reference`。
- [ ] 加载 `trellis-before-dev`，重新读取项目、data/core 规范及本任务三个规划文件。
- [ ] 按 `.trellis/spec/project/dev-environment.md` 重新取证并在首次验证回复中声明机器状态。
- [ ] 记录根仓库与所有锁定 upstream 的 `git status --short`，识别现有用户改动；实现期间不覆盖无关修改。
- [ ] 在 task research 中登记本轮需求变化：原方案把 falcor2 作为正式 provider 已失效；新方案只把 falcor2 用作 oracle，formal rerun 尚未发生。

Rollback 0：没有用户对本轮重大架构变更的显式批准时保持 `planning`，不构建、下载或写实现代码。

## 1. Feasibility gate：MDL SDK HLSL 进入当前 Falcor 8

- [ ] 新增 project-owned 最小 `tools/reference/mdl_sdk_bridge/` CMake target 与 `scripts/build_mdl_reference.ps1`，只链接锁定 MDL SDK，不链接 falcor2。
- [ ] 新增项目 fixture `tests/fixtures/mdl/constant_diffuse.mdl`，bridge 以 class compilation 和冻结 HLSL options 生成真实 target code、argument block、RO/resource metadata。
- [ ] 用 current Falcor Python `ProgramDesc.add_shader_module().add_string()` 与 `ComputePass(desc)` 编译真实 generated HLSL + 最小 query adapter。
- [ ] 最小 dispatch 覆盖 argument buffer、RO buffer、一个 2D texture、一个 BSDF data 3D texture的 binding 探针；证明 Python shader-var array indexing 可用。
- [ ] constant diffuse 在法线、斜入射、不同 `wo` 的预定义 queries 上验证 `albedo/pi * abs(n·wi)`，冻结方向角色与 cosine 约定。
- [ ] 记录 Slang diagnostics、MDL SDK/Falcor/Slang identity、shader/resource counts 与 analytic comparison 到 `artifacts/reference-parity/mdl/feasibility/`。

Rollback 1：真实 generated HLSL、资源 ABI 或解析 diffuse 任一失败时先停止扩展 source/资产。不得用手写 closure、falcor2 正式 provider、PreviewSurface 或 bake 绕过 gate；保存诊断并回到 design review。

## 2. 锁定依赖、构建与资产脚本

- [ ] 新增 MDL SDK fetch/verify 脚本，固定 `2025.0.0-387700.1252` URL、size、SHA-256 和 license；目标必须是明确的 ignored `external/` 子目录，未知/非空目录 fail closed。
- [ ] 新增 falcor2 oracle fetch/build/run wrapper，固定 commit `d629c967fa800af81cf5c916bfb2a825b012f473` 与 recursive submodules；wrapper 与正式 current-Falcor wrapper 分离。
- [ ] 更新 `.gitignore`、`docs/repository_policy.md` 与根 `AGENTS.md` 的 upstream pin/clean-tree 合同。
- [ ] 新增 `scripts/fetch_mdl_assets.ps1`；`-VMaterials2` 必须同时提供 `-AcceptNvidiaOmniverseTerms`，且在确认 flag 前不得发起网络或写文件。
- [ ] 下载脚本使用可见进度、校验 archive hash，安全展开到 `assets/source-materials/mdl-vmaterials2/2.4.0/`；拒绝 path traversal、symlink escape 和覆盖未知文件。

Rollback 2：官方 package identity、license 或 hash 无法固定时不登记 active reference package；不从非官方镜像替代后宣称同一身份。

## 3. 完成 MDL SDK compiler bridge

- [ ] 定义并测试 `ncls.mdl-inspection@1`、`ncls.mdl-compile-request@1`、`ncls.mdl-compiled-artifact@1` 与 `ncls.mdl-native-query@1` schemas。
- [ ] 实现 MDL SDK lifecycle、plugin/search-path 初始化、module load、export/exact-signature discovery、annotations 和依赖/resource discovery。
- [ ] 实现 authored typed arguments 到 material instance、class compilation、HLSL link unit 与冻结 backend options。
- [ ] 导出 callable symbols、argument block/layout、RO segments、2D/BSDF-data texture descriptors/data、DF handle count 和结构化 diagnostics。
- [ ] 未识别 texture shape、measured BSDF、light profile、emission/volume/displacement requirement、animated/UDIM resource 或超过静态上限时 fail closed。
- [ ] 实现只供 fixtures 验收的 native backend query；它不暴露为正式 provider。
- [ ] compiled cache key 覆盖 source snapshot、bridge implementation、MDL SDK build 和完整 options；load 时逐文件验 hash。
- [ ] 为 bridge CLI、schema、dependency closure、path containment、determinism、stale cache 和 error diagnostics 添加 unit/integration tests。

Rollback 3：如果 project-owned executable 边界无法稳定表达 SDK lifecycle/资源，可提议最小 pybind/native service 变体并返回 planning；不得把 falcor2 C++ 类型泄漏到公共合同。

## 4. Source family、identity 与 reference package

- [ ] 新增 `MdlSourceCatalog` 与 `MdlFamilyDefinition`，注册 `mdl.program@1`。
- [ ] 用 bridge inspection 生成 canonical snapshot：module、exact export signature、authored typed arguments、class mode、language/SDK identity、pack identity 和完整 resource hashes。
- [ ] 映射 bool/int/float/double/color/vector/enum/texture2d editor；其余类型 read-only 并给出原因。
- [ ] `apply_edit()` 覆盖 stale snapshot、finite/range/type/resource containment；编辑后生成新 snapshot 与正确 invalidation。
- [ ] 新增 `MdlReferenceProgram` 与 `references/mdl-vmaterials2-v1/{reference.json,assets.json,README.md}`，注册 `ncls.mdl-vmaterials2@1`。
- [ ] implementation identity 覆盖 bridge、runtime/query shader、MDL SDK、Falcor/Slang 和 codegen options；compiled artifact 不成为 source identity。
- [ ] 更新 registry/tree/schema/serialization tests，证明旧 source/reference 向后兼容。

Rollback 4：dependency closure 或 exact export/argument identity 无法稳定重建时，reference package 保持 inactive；不得仅 hash root `.mdl` 文件。

## 5. Current-Falcor renderer runtime

- [x] 完成 `mdl_runtime.slangh` 的 MDL HLSL renderer ABI：LOD0 texture lookup、gamma、wrap/crop、frame、scene data、argument/RO access 和 BSDF data texture。
- [ ] 完成 `mdl_query.slang` 的 shading state、init/surface scattering evaluate、DF handle reduction 与项目 response measure 转换。
- [ ] 实现 `MdlGpuQueryRuntime`：dynamic program cache、2D/3D texture/buffer loading、material-specific defines、双 slot shared buffers 和 deterministic lifecycle。
- [ ] `evaluate_torch()` 只接受/返回同 CUDA device tensor，不执行 `.cpu()`、NumPy 或 host readback；`evaluate()` 只在 offline sink 边界回读。
- [x] 支持 `SurfaceSample.position/uv`；`uv_dx/uv_dy` 作为统一合同兼容字段保留但 V1 不消费，纹理过滤固定为 `ExplicitLod(0)`。
- [ ] 对 unsupported domain/resource/closure 进行编译期或 runtime-construction 拒绝；descriptor 不过度声明能力。
- [x] 添加 GPU tests：analytic fixtures、parameter edit、texture gamma/UV/wrap/crop/LOD0、BSDF data、determinism、slot reuse、same-device/no-host-readback。

Rollback 5：LOD0 filtering 或资源语义不正确时不运行 textured formal assets；修复或返回 planning。未来 derivative filtering 需新 capability 与新 parity 身份。

## 6. 统一 offline/live 数据流

- [ ] 新增 `MdlProvider`，复用 `BaseProvider`、`QueryPlan`、`EvaluatedBlock`、现有 spatial/direction helpers 与 HDF5 collector。
- [ ] 新增/注册 `MdlLiveReferenceBatchSource`，复用现有 route RNG、slot lease、metadata 与 `TrainingBatch@1`；不新增 dataset/reader/training runner。
- [ ] 在现有 CLI/provider factory 中增加 `mdl` 和 asset IDs；缺失 SDK/assets 给出可操作诊断，不影响普通 import/unit 路径。
- [ ] 添加静态边界测试，禁止正式 `src/ncls` provider/reference/live/collector/CLI import 或启动 falcor2/oracle。
- [ ] 添加“falcor2 clone 不存在时正式 fixture provider 仍运行”的 integration test。
- [ ] 执行单材质 offline smoke、HDF5 roundtrip、live batch、lease reuse 与短训练 consumer smoke；产物只写既定 `data/reference-responses/` 或 `artifacts/`。

Rollback 6：如果 formal path 需要 falcor2、第二套 collector 或 host target copy 才能运行，A1/A8 未满足，不能宣称接入完成。

## 7. vMaterials shortlist

- [ ] 在显式 terms flag 下获取 vMaterials 2.4.0，生成 archive/asset/package manifest 和 dependency closure。
- [ ] discovery 固定五个 fancy modules 与 `Wood_Tiles_Pine_Mosaic` 的 exact export/signature/defaults/resources。
- [ ] 对每个材质执行 capability audit；unsupported 项必须保留证据并按设计回到 planning，不静默 distill/bake/替换类别。
- [ ] 六个材质完成 project/current-Falcor discover/load/evaluate smoke；输出 finite、非负约束仅按对应 closure/domain 判断。
- [ ] car paint 与 patinated copper 完成 offline/live batch smoke，覆盖无普通 2D texture 与带 2D texture 两类资源路径。

Rollback 7：若 pack 下载、许可、hash、export 或 capability 不匹配，reference package 不设 active；保留 ignored assets 与 artifacts 诊断。

## 8. falcor2 官方 oracle 与正式 parity

- [x] 实现版本化 `QueryPacket`/`OracleResult`，明确 `wo/wi`、frame、position、UV、`ExplicitLod(0)`、scene units、IOR、`learnable=false` 与 response measure。
- [ ] 构建隔离的 falcor2 oracle runner；它只读取 request、写 artifacts，不 import 项目 formal provider，也不写 HDF5/training target。
- [ ] 用 analytic fixtures 做 oracle convention smoke，先排除方向/cosine 差异。
- [ ] 在 isolated calibration queries 上，根据 float32/backend 差异冻结 `abs+rel` tolerance，并记录 `source/scope/why_hard/failure_action`。
- [ ] 用 disjoint formal queries 对 car paint 和 patinated copper 做逐方向 parity；formal 不修改容差。
- [ ] 其余四个 fancy materials 完成 oracle load/evaluate smoke；`Wood_Tiles_Pine_Mosaic` 完成与 neuralappearance 配置的 module/export correspondence。
- [ ] 报告明确：falcor2 是独立 renderer integration，不是不同 closure 数学；解析/native 层与 falcor2 层共同构成正确性证据。

Rollback 8：parity 失败分类为 integration defect 或 protocol/design defect。不得把观察到的误差反写为更宽 hard gate，也不得让 oracle result 回流正式数据。

## 9. 文档、验证与收尾

- [ ] 更新 `references/README.md`、`docs/material_scope.md`、`docs/repository_policy.md`、相关 pipeline/reference 文档和中文使用说明。
- [ ] 文档明确“一条正式路径 + 一条验证 oracle”、SDK 语义依赖、V1 capability、许可/资产获取和失败诊断。
- [ ] 运行普通 unit：`conda run -n neural-shading python -m pytest <targeted unit tests>`。
- [ ] 运行 current-Falcor GPU/integration：`powershell -ExecutionPolicy Bypass -File scripts/run_falcor_python.ps1 -m pytest <MDL formal tests>`。
- [ ] 运行隔离 falcor2 oracle tests 与 frozen parity harness。
- [ ] 运行全量可承担测试、registry/tree/schema validation、`git diff --check` 与相关静态 import audit。
- [ ] 检查根仓库不包含 `external/`、`assets/`、`build/`、`artifacts/`、`reports/`、`data/` 误入文件；确认所有 upstream clean。
- [ ] 运行 `trellis-check`。失败需修复并重跑相关层，不能仅记录为通过。
- [ ] 更新任务 research/实现日志和开发者 journal；向用户汇总交付、验证、已知 V1 unsupported 能力与资产位置。
- [ ] 用户确认后由 `trellis-finish-work` 收尾；不自动提交或归档未经用户确认的任务。

## 验收映射

- A1：阶段 1、5、6、8 的 dependency/import/runtime 证据。
- A2/A3：阶段 4、6 的 source/editor/common-pipeline tests。
- A4：阶段 3 的 bridge artifact 与 fail-closed tests。
- A5/A6：阶段 1、3、5 的 analytic/native tests。
- A7：阶段 8 的 frozen falcor2 parity。
- A8/A9：阶段 5、6 的 shared-buffer 与 capability tests。
- A10/A11：阶段 2、7 的 asset/license/shortlist evidence。
- A12：阶段 9 的完整质量门与 upstream clean check。
