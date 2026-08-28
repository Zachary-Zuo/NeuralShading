# 技术设计：vMaterials preset catalog 与 capability audit

## 1. 设计目标与边界

任务只扩展 MDL source discovery、catalog、资源闭包和 runtime capability 判定。它不新增 family-specific producer，也不改变通用 online training/query 数据流。后续 neural config 仍通过 `mdl.program@1` locator 加载 snapshot；catalog 只负责产生权威 locator 与分组信息。

## 2. 持久化结构

### 2.1 `assets.json`

`references/mdl-vmaterials2-v1/assets.json` 继续保存正式代表入口，数量从 6 扩展到 11，每个 family 选择自身主 export。现有 6 条记录必须保持 source identity 和既有 audit 字段稳定；viewer 与 formal parity 仍使用各自显式白名单，不因 manifest 增长自动纳入新条目。

### 2.2 `families.json`

新增 `references/mdl-vmaterials2-v1/families.json` 与对应 JSON Schema。顶层记录 pack、module root、MDL SDK、11 个 family，以及去重后的资源 path→SHA-256 表。每个 family 记录：

```text
family_id
coverage_role
evaluation_role                 # parameterized / discrete-subfamilies / spatial-resource-control
module
source_path
primary_export
presets[]
```

每个 preset 记录：

```text
preset_id                       # family 内稳定 ID
export_name
exact_export
source_snapshot_id
parameters
source_module_paths
resource_paths                  # 引用顶层去重资源表
resource_signature
compiled_material_hash
surface_scattering_hash
geometry_normal_hash
cutout_opacity_hash
runtime_capability_audit
runtime_supported
unsupported_reasons
evaluation_subfamily
```

`resource_signature` 对实际闭包的 path/hash 有序集合取 canonical SHA-256；SDK compiled/sub-expression hashes 是结构证据，不替代 source/resource hashes。

## 3. SDK discovery 与 artifact

在 `ncls_mdl_sdk_bridge` 增加 `discover` 命令：加载一个绝对 MDL module，通过 `IModule::get_material_count()/get_material()` 输出 SDK 看到的 exact exported material definitions。Python `MdlSdkCompilerBridge.discover_module()` 验证 discovery document 的 schema、SDK build、module identity 与 export 唯一性。

现有 compile 路径增加以下只读审计字段：

- compiled material hash；
- `surface.scattering`、`geometry.normal`、`geometry.cutout_opacity` sub-expression hash；
- cutout capability：恒等于 1 或存在非不透明表达式。

target descriptor 纳入 `geometry.cutout_opacity`，使只被 cutout 使用的纹理也进入 target resource table。该 callable 不进入正式 shader 接口；`MdlReferenceProgram.compile_material()` 在构造 MaterialPayload 前检查 capability，非恒等于 1 时明确拒绝。

2D texture artifact 保留 SDK canvas 的 typed decoded payload。pixel type 通过集中映射进入 typed resource descriptor；`Rgb_16`/`Rgba_16` 使用 Uint16 normalized 语义，线性 `Rgba_16` 直接绑定为 `RGBA16Unorm`，不降成 8-bit。资源格式 support 与 closure/cutout support 是两条独立轴：即使 preset 因 cutout 暂不可执行，其 atlas 仍必须完整 materialize 并通过 payload/hash 校验。

为避免 172 次 inspection 重复落盘 4K decoded texture，inspection artifact 显式标记为 metadata-only texture payload；它仍记录 authoritative source path、尺寸、pixel type、gamma 与闭包。正式 `compile_snapshot()` 始终生成 decoded payload，viewer/parity 不得把 inspection artifact 当 runtime artifact。

emission、volume、displacement、measured BSDF、light profile 与未知 texture shape 保持现有 fail-closed 边界。`MdlCompiledArtifact.load()` 验证新增 audit/hash 字段，但允许加载“可 catalog、不可 runtime”的 cutout artifact；正式 runtime 支持检查由共享 helper 单独执行。

## 4. 生成流程

`tools/reference/generate_mdl_vmaterials_manifest.py` 改为 declarative 11-family specification，只硬编码 family id、role、module、primary export 与预期数量，不硬编码 172 个 export names。

流程如下：

1. 对 11 个 module 做 SDK discovery，验证逐 family 数量和总数 172。
2. 为每个 exact export 建立稳定 artifact key/目录。
3. 已存在目录先由 `MdlCompiledArtifact.load()` 完整验证；有效则复用，无效则明确失败，不覆盖可疑目录。
4. 未完成 entry 调 bridge inspect；用 `tqdm` 按 preset 更新 `completed/total`、elapsed、rate 与 ETA。
5. 从 artifact 构造 canonical snapshot、资源闭包、resource signature、capability 与分组。
6. 验证 11/172/164/8、唯一性、primary exports、原 6 条稳定性和所有资源 containment。
7. 先把两个 manifests 写到临时文件，重新加载/schema 验证后原子替换 tracked 文件。

生成器提供 check-only 模式：从已验证 artifacts 重建到临时位置并与 tracked manifests 比较，防止人工修改或非确定性输出。

## 5. Catalog 消费

在 source-material 层增加小型 catalog loader，职责限于：

- 验证 schema/version、family/preset/identity 唯一性和资源引用；
- 按 `family_id + preset_id` 查询记录；
- 结合调用方给出的 module root 生成统一 `mdl-export` locator；
- 默认拒绝把 `runtime_supported=false` entry 转成 runtime locator，显式 catalog/审计读取仍允许。

训练、reference 与 viewer 不增加 vMaterials 专用分支。后续 neural 配置可从 catalog 生成普通 locator 列表，再走现有 `SourceFamilyDefinition.load_snapshot()`。

## 6. Cutout 决策

16 个 Suede exports 分成 8 个 opaque 和 8 个 punched。所有 16 个都进入 catalog；punched entries 由 SDK capability 证据标为：

```text
evaluation_subfamily = "punched-cutout"
runtime_supported = false
unsupported_reasons = ["geometry.cutout_opacity"]
```

这不是删除 preset，而是把“可保留/可审计”和“当前 evaluator 可忠实执行”分开。后续 opacity 任务必须定义公共输出、组合/可见性语义、neural target、renderer 消费与 viewer 证据后，才能把这 8 个状态改为 supported。

## 7. 兼容性与回滚

- 不移动或修改 NVIDIA source tree。
- build/artifact 输出均位于 ignored 的 `build/` 或 `artifacts/`；生成中断不影响 tracked manifests。
- manifest 只在完整验证后原子替换；失败时保留旧 `assets.json` 和不存在/旧版的 `families.json`。
- viewer/parity 显式 ID 列表保持不变；新 5 个主入口只完成 source registration，不自动扩大 formal parity 结论。
- 若 bridge artifact 合同变更导致旧 cache 不兼容，以 compiler identity/cache key 产生新 artifact；不原地修改旧 artifact。

## 8. 受影响文件

- `tools/reference/mdl_sdk_bridge/main.cpp`
- `src/ncls/references/mdl.py`
- `src/ncls/references/programs/mdl.py`
- `src/ncls/source_materials/mdl.py` 或同层独立 catalog module
- `tools/reference/generate_mdl_vmaterials_manifest.py`
- `references/mdl-vmaterials2-v1/assets.json`
- `references/mdl-vmaterials2-v1/families.json`
- `references/mdl-vmaterials2-v1/schemas/*.schema.json`
- `tests/unit/test_mdl_source.py`
- 新增 catalog unit test与 MDL capability GPU test
- `.trellis/spec/data/mdl-reference.md`
- `references/mdl-vmaterials2-v1/README.md`
- `docs/research/mdl_vmaterials_neural_cohort.md`

## 9. 风险与控制

- **全量 audit 时间较长**：逐 preset 可恢复 artifact + `tqdm`，不增加 watcher/heartbeat。
- **manifest 体积膨胀**：顶层资源表去重；172 个 preset 只引用 path 列表。
- **SDK hash 被误当语义 identity**：compiled hash 只用于分组证据；canonical snapshot 仍由 source payload 与 source/resource hashes 决定。
- **cutout 被静默忽略**：artifact 可加载用于 catalog，正式 runtime helper 强制拒绝。
- **16-bit atlas 被 cutout 状态掩盖**：pixel format mapping 与 capability audit 分离；`Rgba_16` payload/typed binding 单独测试，禁止量化或省略正式 runtime payload。
- **现有入口漂移**：生成前后逐字段比较原 6 条，差异必须显式解释而不是自动接受。
