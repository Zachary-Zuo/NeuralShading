# MDL reference 集成合同

## 1. Scope / Trigger

新增或修改 `mdl.program@1` source、MDL SDK bridge、compiled artifact、current-Falcor runtime、offline/live producer 或 falcor2 parity 时适用。本合同防止 validation oracle 变成第二条正式路径，也防止将未实现的 texture filtering/closure 能力写入公共 descriptor。

## 2. Signatures

```text
MdlSdkCompilerBridge.inspect(module, material, arguments) -> ncls.mdl-inspection@1
MdlSdkCompilerBridge.compile_snapshot(SourceSnapshot) -> ncls.mdl-compiled-artifact@1
MdlSdkCompilerBridge.native_evaluate(...) -> validation-only native packet
MdlProvider.evaluate(SourceState, SurfaceSample[], QueryPlan) -> EvaluatedBlock
MdlLiveReferenceBatchSource.next_batch(TrainingRouteRequest) -> TrainingBatch@1
scripts/run_mdl_reference_parity.ps1 -Mode formal -AssetId <ids> -OutputDir <new artifacts path>
```

正式依赖方向只有：

```text
mdl.program@1 -> project MDL bridge -> MDL SDK target code -> current Falcor 8
              -> unified EvaluatedBlock / TrainingBatch@1
```

`external/falcor2` 只能由 `tools/reference/mdl_oracle/` 和 parity runner 启动。

## 3. Contracts

- source snapshot 固定 pack/version、module、exact export、typed arguments、传递 module/resource hash 与 MDL SDK build；argument block offset 不是公共接口。
- compiled artifact 是 ignored cache，必须记录 compiler identity 和精确文件 SHA-256；加载时拒绝缺失、额外或 hash 漂移文件。
- V1 domain 是 front-facing upper-hemisphere surface-BSDF evaluate；response 为线性 RGB `f * |n_s · wi|`。
- V1 texture filtering 固定 `ExplicitLod(0)`；`TrainingBatch@1` 为兼容统一 schema 仍携带 `uv_dx/uv_dy/mip_level`，但三者必须为零，provenance 必须包含 `texture_filtering=explicit-lod0` 与 `uv_derivatives_consumed=false`。
- 正式 JPEG decoder 固定独立 `external/stb` pin/hash。可以与 oracle 使用相同 decoder 语义，但不得从 falcor2 import、链接或复制 runtime。
- internal query 可输出 MDL BSDF PDF 做诊断；公共 capability 不因此声明 matched `sample/pdf`。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| stale edit、类型/range 错、resource 越过 pack root 或 hash 漂移 | 拒绝 source edit/compile |
| emission、volume、displacement、measured BSDF、light profile 或未知 texture shape | bridge fail closed；不得只求 surface 后继续 |
| texture/RO/DF handle 超过 V1 静态上限 | bridge/runtime construction 失败 |
| compiled artifact 文件缺失、额外或 hash 不符 | `MdlCompiledArtifact.load()` 拒绝 |
| formal import/启动 falcor2 或 oracle module | 静态边界测试失败 |
| formal parity 超出冻结门槛 | 归类 renderer integration defect；不得放宽门槛或切换 executor |
| 缺 MDL SDK/vMaterials | 选择 MDL 时给出可操作错误；普通 import/unit 不受影响 |

## 5. Good / Base / Bad Cases

- Good：编辑 `texture_2d` pack-relative URI后生成新 snapshot，MDL SDK 重建 argument block，current Falcor 与 oracle 用同一冻结 query 做 parity。
- Base：无纹理 constant diffuse 只使用 argument block，在 current Falcor、MDL native 和解析 Lambertian 三方一致。
- Bad：provider 失败后启动 falcor2 生成 target；把 falcor2 result 写入 HDF5；或随机生成非零 mip metadata 但 runtime 固定 LOD0。

## 6. Tests Required

- unit：typed scalar/enum/range/texture edit、path containment、artifact tamper、schema/tree、formal/oracle import boundary。
- current-Falcor GPU：analytic diffuse、texture UV/gamma/wrap、parameter edit、RO/BSDF-data、same-device live batch、slot reuse、统一 HDF5 roundtrip、六材质 smoke。
- fail-closed：至少一个 emissive fixture 必须被 bridge 拒绝。
- independent validation：MDL SDK native fixture parity；car paint 与 textured copper 的 disjoint frozen falcor2 formal parity，报告须记录 bridge/Falcor/SDK/stb/falcor2 identity。

## 7. Wrong vs Correct

```python
# 错：oracle 成为 fallback，产生第二条正式 reference 路径
try:
    return formal_provider.evaluate(state, surfaces, plan)
except RuntimeError:
    return falcor2_oracle.evaluate(state, surfaces, plan)

# 对：正式失败直接传播；oracle 只由显式 parity 命令运行
return formal_provider.evaluate(state, surfaces, plan)
```

```python
# 错：runtime 不消费 derivative，却声明 uv-footprint
capabilities = ("evaluate", "spatial", "uv-footprint")

# 对：V1 descriptor 与实际 ExplicitLod(0) 一致
capabilities = ("evaluate", "spatial")
```
