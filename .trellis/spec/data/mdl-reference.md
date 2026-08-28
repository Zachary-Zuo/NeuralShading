# MDL reference 集成合同

## 1. Scope / Trigger

新增或修改 `mdl.program@1` source、MDL SDK bridge、compiled artifact、current-Falcor canonical backend或falcor2 parity时适用。本合同防止validation oracle成为第二条正式路径，也防止恢复已删除的MDL专用training/query入口。

## 2. Signatures

```text
MdlSdkCompilerBridge.inspect(module, material, arguments) -> ncls.mdl-inspection@1
MdlSdkCompilerBridge.compile_snapshot(SourceSnapshot) -> ncls.mdl-compiled-artifact@1
MdlSdkCompilerBridge.native_evaluate(...) -> validation-only native packet
MdlSourceFamily.load_snapshot(locator) -> SourceSnapshot
MdlReferenceProgram.compile_material(snapshot) -> MaterialPayload
ReferenceQueryDispatcher.evaluate/sample/pdf(...) -> typed GPU result
scripts/run_mdl_reference_parity.ps1 -Mode formal -AssetId <ids> -OutputDir <new artifacts path>
```

正式依赖方向只有：

```text
mdl.program@1 -> project MDL bridge -> locked MDL SDK target code
              -> NclsMdlGenerated typed source module + canonical mdl.slang
              -> generic ReferenceQueryDispatcher / viewer
```

`external/falcor2`只能由`tools/reference/mdl_oracle/`和显式parity runner启动。

## 3. Contracts

- snapshot固定pack/version、module、exact export、typed arguments、module/resource hash与SDK build；argument block offset不是公共接口。
- compiled artifact是ignored cache，记录compiler identity与所有文件SHA-256；加载时拒绝缺失、额外或漂移文件。
- canonical backend必须完整实现`prepare/evaluate/sample/pdf`，`evaluate().f`为线性RGB且不含cosine。
- generated HLSL以`kind=slang-module-source`注入；argument block、RO segment、2D/3D texture与sampler均走通用typed binder。
- 正式JPEG decoder固定独立`external/stb` pin/hash。不得从falcor2 import、链接或复制runtime。
- 不存在`mdl_query.slang`、MDL provider或MDL live batch source。训练只能通过source locator、canonical program与generic dispatcher。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| stale edit、类型/range错、resource越过pack root或hash漂移 | 拒绝source edit/compile |
| emission、volume、displacement、measured BSDF、light profile或未知texture shape | bridge fail closed |
| texture/RO/DF handle超过V1静态上限 | bridge/runtime构造失败 |
| artifact文件缺失、额外或hash不符 | `MdlCompiledArtifact.load()`拒绝 |
| 多source snapshot需要不同generated module | dispatcher拒绝，不能误绑第一个module |
| formal路径import/启动falcor2 | 静态边界测试失败 |

## 5. Good / Base / Bad Cases

- Good：编辑`texture_2d` URI后产生新snapshot，bridge重建artifact，generic dispatcher与native oracle在冻结query上parity。
- Base：constant diffuse只绑定argument/RO buffer，canonical evaluate/sample/pdf均可运行。
- Bad：为MDL复制一个query shader或producer；正式失败后启动falcor2生成target。

## 6. Tests Required

- unit：typed edit、path containment、artifact tamper、source/reference registry、formal/oracle import边界。
- current-Falcor GPU：generic dispatcher evaluate/sample/pdf、analytic diffuse、texture/RO绑定与slot生命周期。
- fail-closed：emissive fixture必须被bridge拒绝。
- independent validation：MDL SDK native fixture parity；falcor2只产生隔离报告。

## 7. Wrong vs Correct

```python
# 错：专用入口绕开统一binder
run_shader("mdl_query.slang", artifact)

# 对：generated module是MaterialPayload的一部分
dispatcher = ReferenceQueryDispatcher(mdl_reference, (snapshot,), ...)
```

```python
# 错：oracle成为fallback
try: return dispatcher.evaluate(...)
except RuntimeError: return falcor2_oracle.evaluate(...)

# 对：正式失败直接传播，oracle仅由显式parity命令运行
return dispatcher.evaluate(...)
```
