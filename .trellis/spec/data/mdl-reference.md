# MDL reference 集成合同

## 1. Scope / Trigger

新增或修改 `mdl.program@1` source、MDL SDK bridge、module discovery、compiled artifact、preset catalog、current-Falcor canonical backend 或 falcor2 parity 时适用。本合同防止 validation oracle 成为第二条正式路径，也防止把输入资源格式限制误报成 closure 输出限制，或恢复已删除的 MDL 专用 training/query 入口。

## 2. Signatures

```text
resolve_mdl_program_toolchain(overrides=None) -> MdlProgramToolchainDescriptor
create_mdl_program_provider(module_root, overrides=None) -> MdlProgramProvider
MdlProgramProvider.discover_module(module, *, output) -> ncls.mdl-module-discovery@1
MdlProgramProvider.inspect(module, material, arguments, *, output) -> metadata-only ncls.mdl-compiled-artifact@1
MdlProgramProvider.compile_snapshot(SourceSnapshot) -> ncls.mdl-compiled-artifact@1
MdlProgramProvider.native_evaluate(...) -> validation-only native packet
MdlVmaterialsCatalog.load(path, expected_bridge_sha256) -> ncls.mdl-vmaterials-family-catalog@1
MdlVmaterialsCatalog.verify_resources(module_root) -> None
MdlVmaterialsCatalog.locator(family_id, preset_id, *, module_root, allow_unsupported=False) -> mdl-export locator
MdlSourceFamily.load_snapshot(locator) -> SourceSnapshot
MdlReferenceProgram.compile_material(snapshot) -> MaterialPayload
ReferenceBackendCapability.open(...).evaluate/sample/pdf(...) -> typed GPU result
MdlMetalRegistry.load(path) -> ncls.mdl-metal-opaque-registry@1
MdlMetalNativeAssetCollection(registry, module_root) -> NativeAssetCollection@1
tools/reference/generate_mdl_vmaterials_manifest.py --artifact-root <ignored cache> [--refresh-artifacts|--check]
scripts/build_mdl_metal_registry.ps1 [-Refresh -InspectionRoot <ignored-artifact-dir>]
scripts/run_mdl_reference_parity.ps1 -Mode formal -AssetId <ids> -OutputDir <new artifacts path>
```

正式依赖方向只有：

```text
mdl.program@1 -> project MDL bridge -> locked MDL SDK target code
              -> NclsMdlGenerated typed source module + canonical mdl.slang
              -> generic ReferenceBackendSession / viewer
```

`external/falcor2`只能由`tools/reference/mdl_oracle/`和显式parity runner启动。

## 3. Contracts

- snapshot固定pack/version、module、exact export、typed arguments、module/resource hash与SDK build；argument block offset不是公共接口。
- MDL program provider 是 `mdl.program@1` 的内部 toolchain hook，不是公共 reference backend；其 Windows/Linux SDK library、plugins、bridge与archive hash全部来自根 `reference-backend-toolchains` manifest。
- C++ provider 只有一个 `SharedLibrary` 动态加载边界；Windows使用`LoadLibraryW/GetProcAddress`，Linux使用`dlopen/dlsym`并链接`${CMAKE_DL_LIBS}`。业务路径必须通过CLI显式传入SDK library与重复`--plugin`，不得按后缀猜平台。
- module discovery 必须通过 MDL SDK `IModule` 枚举 exact export；不得用正则表达式解析 MDL 源码来建立 authoritative preset 列表。discovery 记录 bridge executable SHA-256，并要求 export 唯一且稳定排序。
- compiled artifact 是 ignored cache，记录 compiler identity、compiled material/sub-expression hash 与所有文件 SHA-256；加载时拒绝缺失、额外或漂移文件。
- `inspect()` 只为 catalog 审计生成 `texture_payloads=metadata-only` artifact；`compile_snapshot()`、native parity、viewer 和正式 dispatcher 必须使用 `texture_payloads=decoded` artifact。metadata-only artifact 不得进入 runtime binder。
- `references/mdl-vmaterials2-v1/families.json` 是 11 个 family、172 个 authored preset 的版本化 catalog；记录 typed arguments、source/resource closure、runtime resource signature、compiled identity 与 capability audit。`assets.json` 只登记每个 family 的 primary export，不复制 172 条专用 producer。
- `metal-opaque-v1.json`是独立Metal source registry，不并入上述catalog：必须精确为837 authored / 692 opaque / 145 cutout-rejected / 178 opaque graphs / 52 texture sets / 64 authored schemas。每个leaf保留exact overload locator、typed descriptor、六组参数责任、recipe/metal/finish兼容关系；unknown、missing与cutout locator统一fail closed。
- Metal texture set最多9 slots；每slot保存source/content hash、pixel type、effective gamma、channel role、normal/mip/filter规则。MDL SDK BSDF-data table作为provider-owned静态slot登记；`Rgba_16`在registry、decoded artifact与tile collection全程保持uint16来源，不经Pillow降位。
- catalog locator 只返回通用 `mdl-export` source locator。默认拒绝 `runtime_supported=false` 的 preset；`allow_unsupported=True` 只允许审计/研究代码显式取 locator，不改变 runtime 能力。
- canonical backend必须完整实现`prepare/evaluate/sample/pdf`，`evaluate().f`为线性RGB且不含cosine。MDL target code的`bsdf_diffuse + bsdf_glossy`包含material-local shading-normal cosine；转换成公共`f`时必须除以输入`NclsShadingFrame`的light cosine，使renderer用同一输入frame乘回cosine后恢复MDL原生response。不得除以`init()`后的`state.normal`，否则normal-map材质会丢失cosine ratio。
- generated HLSL以`kind=slang-module-source`注入；argument block、RO segment、BSDF data、2D/3D texture与sampler均走通用typed binder。
- 多typed state按generated module与resource binding进入同一个execution group，argument/RO使用显式16-byte aligned offsets。decoded texture/BSDF payload以`FileResourcePayload`指向verified provider cache；group key使用content hash而非主机path，lazy group首次执行才读取。
- 输入像素格式支持与 closure 输出支持是两个独立 capability。V1 decoded texture 至少支持 `Sint8`、`Rgb`、`Rgba`、`Rgb_16`、`Rgba_16`、`Float32`、`Float32<2/3/4>`、`Rgb_fp`、`Color`；`Rgba_16` 必须以每 texel 8 bytes 保留为 `uint16`，并绑定 `RGBA16Unorm`，不得量化为 8-bit。无对应 sRGB hardware view 的 uint16/float texture 可先无损归一化并显式线性化为 float32。
- `geometry.cutout_opacity` 是输出/合成能力。当前 public evaluator 未实现它时，punched suede 必须以 `unsupported_reasons=["geometry.cutout_opacity"]` 失败关闭；这不允许 bridge 丢弃、跳过或降位其 `Rgba_16` cutout atlas。
- 正式JPEG decoder固定独立`external/stb` pin/hash。不得从falcor2 import、链接或复制runtime。
- 不存在`mdl_query.slang`、MDL 专用 query backend或MDL live batch source。训练只能通过source locator、canonical program、generic backend session与method source adapter。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| stale edit、类型/range错、resource越过pack root或hash漂移 | 拒绝source edit/compile |
| SDK library/plugin/target-code/bridge缺失 | generic doctor与provider preflight报告missing，禁止启动subprocess |
| discovery bridge hash、module 或 export 集漂移 | `MdlModuleDiscovery.load()` 拒绝 |
| catalog family/preset 重名、compiled identity 或 resource signature 漂移 | `MdlVmaterialsCatalog.load()` 拒绝 |
| catalog resource 缺失、越过 module root 或 SHA-256 不符 | `verify_resources()` 拒绝 |
| metadata-only artifact 进入 runtime | `require_runtime_supported()` 拒绝 |
| emission、volume、displacement、measured BSDF、light profile、未知 texture shape 或未知 pixel type | bridge/runtime fail closed |
| 已知 `Rgba_16`/float pixel type | 按原 bit depth 生成 typed payload；不得归入 unknown 或静默降位 |
| `geometry.cutout_opacity` 非常量 1 | catalog 标记 unsupported；正式 runtime fail closed，inspection 仍完整记录纹理元数据 |
| texture/RO/DF handle超过V1静态上限 | bridge/runtime构造失败 |
| artifact文件缺失、额外或hash不符 | `MdlCompiledArtifact.load()`拒绝 |
| 多source snapshot需要不同generated module | dispatcher拒绝，不能误绑第一个module |
| Metal registry count/identity/source closure、role或slot上限漂移 | registry/build check拒绝 |
| 同content resource位于不同state cache目录 | 以content hash判为同一binding；若descriptor不同仍拒绝group |
| formal路径import/启动falcor2 | 静态边界测试失败 |

## 5. Good / Base / Bad Cases

- Good：`Rgba_16` punched suede inspection 记录 `1024 * 1024 * 4 * 2` bytes 的 decoded runtime payload；typed binder 选择 `uint16/RGBA16Unorm`，随后仅因 public cutout 输出合同缺失而拒绝 evaluate。
- Good：692个opaque locators进入registry，训练只选择batch所属group并懒加载该组资源；52-set collection用memmap按tile+halo读取source或BSDF table。
- Base：constant diffuse只绑定argument/RO buffer，canonical evaluate/sample/pdf均可运行。
- Bad：看到 punched material 尚不可渲染，就跳过其 atlas、转成 8-bit，或把 preset 从 catalog 删除。
- Bad：为MDL复制一个query shader或producer；正式失败后启动falcor2生成target。

## 6. Tests Required

- unit：typed edit、path containment、artifact tamper、discovery sorted/exact export、catalog count/identity/signature、source/reference registry、formal/oracle import边界。
- unit/integration：Metal registry regeneration/source closure、692/145拒绝边界、52 descriptors、16-bit与BSDF tile、typed-state train/validation split和lazy file tamper；
- unit：用已知 16-bit pattern 断言 `_decoded_texture_binding()` 保留所有 bits，并返回 `dtype=uint16`、`format=rgba16-unorm`。
- current-Falcor GPU：generic backend session evaluate/sample/pdf、analytic diffuse、texture/RO绑定与slot生命周期；倾斜`geometry.normal` fixture比较MDL SDK native response与`public f × input-frame cosine`；真实 punched atlas 必须断言 payload byte count 为 `width * height * 4 * 2`，Falcor texture format 为 `RGBA16Unorm`。
- portability：Windows Release实际重编译；静态断言`SharedLibrary`同时含Windows/Linux loader、CLI plugin路径与`${CMAKE_DL_LIBS}`。Linux实际编译留在原生Linux gate。
- fail-closed：emissive fixture必须被bridge拒绝。
- fail-closed：punched suede 的 full artifact 必须先通过 decoded texture 校验，再因 `geometry.cutout_opacity` 被 runtime 拒绝。
- independent validation：MDL SDK native fixture parity；falcor2只产生隔离报告。
- manifest：生成后执行 `--check`，断言 11 families / 172 presets / 164 supported / 8 cutout-unsupported，且旧的 6 条 primary asset 记录逐字段不变。

## 7. Wrong vs Correct

```python
# 错：专用入口绕开统一binder
run_shader("mdl_query.slang", artifact)

# 对：generated module是MaterialPayload的一部分
session = create_reference_backend().open(mdl_reference, (snapshot,), ...)
```

```python
# 错：oracle成为fallback
try: return session.evaluate(...)
except RuntimeError: return falcor2_oracle.evaluate(...)

# 对：正式失败直接传播，oracle仅由显式parity命令运行
return session.evaluate(...)
```

```python
# 错：把尚未支持 cutout 输出误当成不支持 16-bit 输入
if artifact.capability_audit["cutout_opacity"]:
    skip_texture_payloads()

# 对：始终无损解码已知输入格式，输出能力在绑定后独立失败关闭
binding = decoded_texture_binding(pixel_type="Rgba_16", dtype="uint16")
artifact.require_runtime_supported()  # 仅在这里报告 geometry.cutout_opacity
```

```python
# 错：把Metal leaf复制进通用catalog或为其新增专用query/producer。
catalog.add_all(metal_exports)
MetalQuerySession(...)

# 对：registry只产生通用locator/typed states，后端仍执行canonical plan。
registry = MdlMetalRegistry.load(path)
plan = compile_single_program_plan(mdl_reference, states, query_recipe=recipe)
session = create_reference_backend().open(plan, query_capacity=capacity)
```
