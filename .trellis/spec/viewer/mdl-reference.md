# Viewer MDL reference 与 catalog 合同

## 1. 适用范围

修改 MDL catalog、compiled artifact、动态 module、typed edit 或 package 交接时适用。viewer 消费正式 reference artifact；falcor2 仅为隔离 oracle，不进入启动或运行路径。公共 renderer 合同见 [conventions.md](conventions.md)。

## 2. 签名

```text
tools/reference/prepare_mdl_viewer.py --output <catalog.json> [--default-asset <asset-id>]
python -m ncls export <checkpoint> [--output <目录>]
scripts/launch_viewer.ps1 -Package <package> -Material <source/catalog>
loadMdlViewerCatalog(path) -> MdlViewerCatalog
loadMdlCompiledArtifact(entry) -> shared_ptr<const MdlCompiledArtifact>
selectMdlCatalogEntry(source, index) -> ReferenceSource candidate
ReferenceSource.hasLinkedMdlPackage() -> 当前 entry 是否包含 neural binding
applyMdlCatalogParameterView(source, parameter_view) -> ReferenceSource candidate
NclsViewer.applyLinkedMdlSource(candidate) -> reference + neural 原子提交
```

## 3. 输入输出合同

- 唯一 catalog schema 是 `ncls.viewer-material-catalog@2`。`catalog_id` 是除自身外全部字段的 canonical JSON SHA-256。Python 与 C++ 的 hash 文本必须一致，见 [json-identity.md](../core/json-identity.md)。
- `registry_uri/registry_identity` 可同时为 null；无 registry 时 count 反映实际 entries。catalog 不携带 checkpoint/compatibility 字段；taxonomy、graph/texture/schema identity 可为 null。源材质的 export/source snapshot、artifact 路径及 hash、renderer runtime identity 仍必需。
- entry 的 `package_root/package_id/program_id/asset_id/instance_id` 必须全有或全 null。`parameter_view` 独立可选；存在 package 不意味着存在 typed editor。允许多个 program；禁止以首个 entry 推断整个 catalog 的 linked 状态。
- 所有 URI 都相对 catalog 根且不得越界。runtime 先验证，选择 entry 时再严格验证精确 artifact 文件集合、各文件 hash、compiler/stb/SDK identity、V1 capability。不可变 catalog 用共享所有权；candidate 不复制整份编辑树。
- renderer 按 capability 选择。最新 hybrid 四入口包默认 reference PT 对 neural PT，支持任一侧改为 deferred。没有 typed compiler 就不显示 neural 参数编辑；诊断阶段字符串不决定 renderer。
- MDL 动态 `NclsMdlGenerated` 顺序为 target-code types → 项目 `mdl_runtime.slangh` → artifact generated HLSL；静态 `reference_backends/mdl.slang` 提供 canonical 四入口。runtime 不链接 MDL SDK DLL。
- 2D 纹理使用 bridge 解码的 pixel type、gamma、origin 和 payload；BSDF-data 使用 Float32 3D payload。argument/RO 按 16-byte row 上传。V1 固定 ExplicitLod(0)，不虚构对 UV derivative 的过滤支持。
- V1 每个 scene specialization 只允许一个不同的 MDL generated artifact；其他非 MDL 材质仍由各自 source 执行。MDL continuation 使用 target-code sample，MIS 使用同一 target-code PDF，`bsdf_over_pdf` 已含 `f*cos/pdf`，不得重复乘除。
- typed edit 以 authored argument block/base snapshot 为锚。完整 values map 同时构造 reference 和 neural candidate；type、enum、finite、int32、hard range、offset/size 必须验证，成功后一起提交。source-only editor 可独立存在。
- preset/edit 失败恢复 source/resources/pass、slot、runtime cache、freeze 和 accumulation；手动模式下不得偷偷重新启用 linked。capture 的编辑状态只在实际存在 parameter view 时生成。
- 旧 catalog/handoff reader 已移除；使用当前 prepare 入口重建，原 artifact 不就地升级。Windows 输出路径必须通过 closure 长度预算检查。

## 4. 验证与错误矩阵

| 条件 | 行为 |
|---|---|
| 旧 schema、未知字段、重复 source/binding identity、悬空 binding、越界 URI | 拒绝加载，提示当前 prepare 入口 |
| artifact 文件缺失/额外/hash 漂移，SDK/compiler/runtime 不匹配 | 拒绝 candidate |
| package binding 仅填部分，或 parameter view snapshot 不符 | 拒绝 catalog |
| emission/volume/displacement、未知纹理类型、纹理数超过 V1 限制 | 拒绝 artifact capability |
| scene 含两个不同 MDL generated artifact | 明确拒绝当前 specialization |
| package 无 sample/pdf 而请求 PT | Unsupported，不自动改模式 |
| typed values 缺 path、越界或 reference write 不完整 | 两侧 candidate 拒绝，保持原状态 |
| shader/resource 失败 | 保留原 binding，显示请求失败 |
| tensor 名称/shape/dtype 不符 | 在模型加载边界拒绝；短训或初始化不阻止部署 |
| sample absorb、非有限或连续 PDF 非正 | 终止当前路径，不换通用 proposal |

## 5. 正常、基础与错误案例

- 正常：一个 source-only catalog 配独立 hybrid package；两侧 PT，地面保持自己的 source，标题准确显示模式。
- 基础：六项 MDL reference catalog 没有 neural binding 和 typed view，仍可切换 source preset。
- 正常：catalog 某 entry 有 typed compiler 时联动编辑；另一个 entry 无 package 时仍可独立查看 source。
- 错误：按 display name 猜 package、以 catalog 第一个 entry 决定全局模式、shader 失败后把新 identity 写到旧画面上。
- 错误：通过 throughput clamp 隐藏 sampler/PDF 错配；必须检查极端权重及随 spp 收敛。

## 6. 必要测试

- `test_viewer_material_catalog.py`：v2 字段、canonical ID、source-only、可选 editor/binding、部分 binding 拒绝、旧 schema 拒绝。
- 当前 checkpoint 的导出与 source 准备共用 `ncls export`，图像使用同一个 `prepare_source_reference`。
- `test_training_checkpoint.py`：同一当前 reader、optimizer 状态保留和实际 tensor 结构检查。
- GPU：MDL native tuple、最新 hybrid 四入口和 quantized witness；不以 evaluator parity 替代 sampler 数值检查。
- Release/headless：真实动态 module、source/package 模式、identity/replay 与 finite EXR；only build script，结束后 Falcor clean。
- 修改 MDL 数学时补 car paint/ceramic 尾部回归；通用 viewer 改动不要求重跑旧大模型性能矩阵。

## 7. 错误与正确写法

```text
错误：catalog.entries.front().linked() -> 所有 entry 强制 neural deferred
正确：当前 entry 的 binding 决定 linked；当前 slot 的 capability 和 mode 决定 renderer

错误：先覆盖 active argument/raw buffer，再尝试 compile
正确：validate/build reference + neural candidate，全部成功才 commit
```
