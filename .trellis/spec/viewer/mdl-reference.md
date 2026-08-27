# Viewer MDL reference 合同

## 1. Scope / Trigger

修改 `NclsViewer` 的 MDL catalog、compiled artifact loader、material-specific shader module、GPU 资源、preset 切换或 capture identity 时适用。目标是让 viewer 显示与 offline/live provider 相同的正式 artifact，同时保持 falcor2 仅为隔离 oracle。

## 2. Signatures

```text
tools/reference/prepare_mdl_viewer.py --output <catalog.json> [--default-asset <asset-id>]
scripts/launch_mdl_viewer.ps1 [-Configuration Release] [-Width W] [-Height H]
loadMdlViewerCatalog(path) -> MdlViewerCatalog
loadMdlCompiledArtifact(entry) -> shared_ptr<const MdlCompiledArtifact>
selectMdlCatalogEntry(source, index) -> ReferenceSource
```

正式 shader module 的组合顺序固定为：

```text
MDL target-code types
-> project mdl_runtime.slangh
-> artifact generated.hlsl
-> MdlViewerAdapter.slang
```

## 3. Contracts

- catalog 固定 `ncls.mdl-viewer-catalog@1`、SDK build、六个 vMaterials asset、target-code types/runtime hash、snapshot id、artifact root 与 artifact identity。
- C++ loader 必须复核 artifact schema、V1 capability audit、compiler/stb identity、精确文件集合和每个文件的 SHA-256；不能只相信 catalog。
- 2D texture 使用 bridge-decoded payload、origin、pixel type 与 gamma；BSDF-data texture 使用 artifact 的 Float32 3D payload。argument block/RO data 按 16-byte row 上传。
- viewer 使用 `ProgramDesc::addShaderModule("NclsMdlGenerated").addString(...)`；generated HLSL 不进入根仓库，也不链接 MDL SDK runtime DLL。
- V1 同一 scene specialization 只允许一个 material-specific generated MDL program。MDL 路径延续必须调用同一 target code 的 `surface_scattering_sample`，环境光 MIS 必须调用同一 target code 的 `surface_scattering_pdf`；`sample.weight` 直接使用 SDK 定义的 `bsdf_over_pdf = f |n_s·wi| / pdf`，不得再次乘 cosine 或除 PDF。
- matched transport 是 viewer 内部正确积分尖锐 flakes/coat 的要求，不把训练/provider 的 source capability 从 `evaluate` 扩成公共 `sample/pdf`。纹理过滤仍为 `ExplicitLod(0)`。
- 禁止用 radiance/throughput clamp 修复 firefly。若同 replay 的孤立高亮随 spp 持续进入，先比较 source response、实际采样 PDF、MIS PDF 与 `bsdf_over_pdf` 的极端尾部。
- preset 切换必须先 validate/build，再原子替换 source/resources/pass。shader/resource 失败保留上一材质。
- capture 记录 `mdl_asset_id`、`mdl_compiled_artifact_sha256`、SDK 和 filtering。单边 capture 不是独立 image parity。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| catalog/schema/SDK 或 target/runtime hash 漂移 | 拒绝加载 |
| artifact 文件缺失、额外或 hash/identity 漂移 | 拒绝加载 |
| emission/volume/displacement capability、未知 texture shape/type、超过 16 textures | 拒绝加载 |
| 2D decoded payload 缺失或尺寸/origin 不符 | 拒绝加载，不从原图临时换 decoder |
| 同一 scene 出现两个不同 MDL generated artifact | 明确拒绝 V1 specialization |
| shader module 或 GPU resource 创建失败 | 保留上一有效 source/pass |
| falcor2 import、launch 或 runtime fallback 出现在 viewer 路径 | boundary test 失败 |
| MDL evaluate 配通用 cosine/GGX proposal，或 MIS 使用非 MDL PDF | 数值正确性失败；不得发布 viewer/capture |
| `sample` 返回 absorb、非有限方向/weight，或连续事件 PDF 非正 | 终止该路径；不得换 generic proposal 冒充同一 estimator |
| capture 全 finite 但孤立白点随 spp 增加 | 不能据此通过视觉门；执行 weight/PDF 尾部诊断，不得 clamp |

## 5. Good / Base / Bad Cases

- Good：car paint 的 evaluate/sample/pdf 来自同一 generated module；路径使用 `bsdf_over_pdf`，环境 NEE 用 MDL PDF 做 MIS，1024 spp 中 flakes 形成连续材质结构而不是随机白点。
- Base：默认 car paint 在 shaderball 上运行；其他 scene slot 继续使用 LayerStack fallback，不产生第二份 MDL generated program。catalog 选择 scratched aluminum 时仍记录同一 snapshot/artifact identity。
- Bad：viewer 直接读 `.mdl` 自行猜参数；从 falcor2 复制 shader/binary；用固定 roughness GGX 采样任意 MDL closure；hash 失败后显示旧图却把新 asset 写入 capture；为混合多 MDL program 文本重命名 generated symbols。

## 6. Tests Required

- unit/static：六项 catalog、unknown default、artifact/compiler/capability/hash检查存在、viewer/falcor2 boundary、公共 `PathSurface`、matched `sample/pdf` 路由与 LOD0。
- GPU adapter：固定 diffuse artifact 上验证 sampled direction/event 有效，sample PDF 等于 formal PDF，`bsdf_over_pdf == evaluate / pdf`；容差按 float32 formal query 冻结，不能根据结果调宽。
- Release：`scripts/build_viewer.ps1 -Configuration Release`，必须编译 C++ 和真实 string module入口，随后 Falcor clean。
- headless：car paint 与 glazed ceramic 各做 1024 spp shaderball capture；EXR shape正确、全 finite，manifest identity 匹配。对现场缺陷回归还要报告 max/high quantile 与基于局部邻域的孤立 firefly 数，不能只报告 finite。
- 视觉：交互窗口可切换六项 preset；car paint 与 glazed ceramic 随 spp 累计不持续增加孤立白点，真实 flakes、釉面高光与瓷砖图案仍保留。

## 7. Wrong vs Correct

```cpp
// 错：只读 generated.hlsl，忽略 artifact identity 与 renderer resources。
program.addShaderLibrary(artifactCode);

// 对：先验证完整 artifact，再组合正式依赖并绑定同一 argument/RO/texture payload。
auto artifact = loadMdlCompiledArtifact(entry);
program.addShaderModule("NclsMdlGenerated").addString(composedSource, virtualPath);
```

```text
错误：MDL shader 编译失败 -> 启动 falcor2 或显示 LayerStack fallback并标记 ready
正确：MDL shader 编译失败 -> 当前 switch 失败，保留上一有效材质并显示错误
```

```slang
// 错：closure 有自己的窄峰，却用无关的固定 GGX PDF 除 evaluate。
direction = sampleFixedGgx(0.2, rng);
weight = nclsMdlEvaluateSurface(...) / fixedGgxPdf(direction);

// 对：方向、weight 与 PDF 来自同一份 MDL target code。
NclsMdlSample sample = nclsMdlSampleSurface(..., xi);
direction = sample.directionWorld;
weight = sample.weight; // 已经是 bsdf_over_pdf
misPdf = nclsMdlPdfSurface(..., lightWorld);
```
