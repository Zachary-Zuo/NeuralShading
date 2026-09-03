# Viewer MDL reference 合同

## 1. Scope / Trigger

修改`NclsViewer`的MDL catalog、compiled artifact loader、material-specific shader module、GPU资源、preset切换或capture identity时适用。目标是让viewer显示与canonical MDL reference相同的正式artifact，同时保持falcor2仅为隔离oracle。

## 2. Signatures

```text
tools/reference/prepare_mdl_viewer.py --output <catalog.json> [--default-asset <asset-id>]
scripts/launch_mdl_viewer.ps1 [-Configuration Release] [-Width W] [-Height H]
tools/viewer/prepare_metal_catalog.py [--output-root <dir>] [--checkpoint <pt>] [--registry <json>]
scripts/launch_metal_viewer.ps1 [-CatalogRoot <dir>] [-AcceptNvidiaOmniverseTerms] [-SkipPrepare] [-SkipBuild]
loadMdlViewerCatalog(path) -> MdlViewerCatalog
loadMdlCompiledArtifact(entry) -> shared_ptr<const MdlCompiledArtifact>
selectMdlCatalogEntry(source, index) -> ReferenceSource
applyMdlCatalogParameterView(source, parameter_view) -> ReferenceSource candidate
NclsViewer.applyLinkedMdlSource(candidate) -> atomic reference + neural slot commit
```

正式 shader module 的组合顺序固定为：

```text
dynamic NclsMdlGenerated module:
  MDL target-code types -> project mdl_runtime.slangh -> artifact generated.hlsl
static project modules:
  reference_backends/mdl.slang -> SceneReferenceProgram.slang / reference_query.cs.slang
```

## 3. Contracts

- 原有六项入口固定为 `ncls.mdl-viewer-catalog@1`；Metal authored入口使用独立的 `ncls.viewer-material-catalog@1`，两者必须由同一 loader 明确分流，不能让未知 JSON 落回 LayerStack reader。
- `ViewerMaterialCatalog@1` 的 `catalog_id` 覆盖 registry、checkpoint、reference runtime、default export和全部entries的canonical JSON。每个entry必须绑定`export_id`、taxonomy、base `source_snapshot_id`、artifact、package/program/asset/instance identity和完整typed `parameter_view`；全catalog只允许一个`program_id`。
- Metal catalog writer从权威registry机械恢复全部692个opaque exports，保留145个cutout rejected计数但不把它们放进entries。writer只读取既有checkpoint并执行deployment compile；不得启动训练、改写checkpoint或手写preset/参数表。
- Metal catalog必须在reference compile、asset cook和package写出前，对全部692个typed editor view做一次轻量预检。MDL enum的authoring值`{"name": N, "value": I}`进入UI/runtime metadata前规范化为choice字符串`N`；各分量上下界相同的vector range规范化为共享scalar range。不能等到某条大payload已经写出后才由`InstancePayload`发现类型漂移。
- 692个entry有独立source/asset/instance identity，但重资产按52个`texture_set_id`分组：每个texture set只执行一次encoder-only cook，同哈希program/grid/reference payload通过NTFS hardlink物化，跨卷不支持hardlink时才复制。hardlink只改变存储，不改变package content hash或identity；内容映射只能接收已验证SHA-256的文件。
- native MDL runtime compile可在准备工具内有界并发，Windows原子发布若瞬时返回`PermissionError/WinError 5`可做短退避重试；cache key、最终manifest和artifact identity仍由正式provider拥有。中断后已原子提交的provider cache可复用，catalog staging本身仍不可加载。
- 已存在catalog的准备入口只验证catalog canonical identity、registry/checkpoint输入identity与结构，不遍历全部hardlink payload，也不为日志重新统计logical bytes；C++在实际选择entry时严格验证该entry的artifact/package并以candidate事务提交。
- `ReferenceSource`中的`ViewerMaterialCatalog@1`是大体积不可变元数据，候选选择、typed edit和事务rollback必须共享同一catalog对象，不能为每次候选深拷贝692条parameter view。shared neural program的GPU pass按实际slot mode懒创建；默认deferred不得预编译未使用的package path tracer。初始linked transaction若reference state已安装，也不得重复创建同一MDL scene pass或重复编译同一typed instance。
- quality-first evaluator的8×8 workgroup只是TDR上限，不是交互frame预算。交互deferred每帧最多提交一个workgroup，并以16×→8×→4×→2×→1×的coarse-to-fine stride逐级覆盖全屏；每级用代表性G-buffer sample填充对应block，UI显示当前stride/tile进度，任一相机/材质/灯光reset从16×重新开始。headless/capture固定`stride=1`并在导出前完成全部tile，不能把交互coarse preview写成正式结果。
- step 20000 checkpoint若descriptor identity与当前runtime不同，只允许在method key、component manifest以及全部tensor name/dtype/rank/shape完全一致时，显式标记`state-schema-compatible-preview`；UI/capture必须同时显示checkpoint/runtime descriptor和compatibility，不能伪装为exact或泛化结论。
- editable parameter当前允许bool/int/enum/float/float2/color。每个leaf同时含registry-derived `responsibility`、MDL argument `offset/size/type`和已有typed runtime mapping；`coordinates`、`frame`与metal/finish/aging/coating使用相同descriptor遍历，不按参数名增加viewer分支。
- linked edit以base authored argument block和base snapshot为锚；同一完整values map先生成`ViewerMaterialState@1`，再分别patch reference argument candidate与neural raw candidate并运行现有`nclsCompileMaterial`。values map必须恰好覆盖全部editable path，type、enum、finite、int32及hard min/max任一失败都拒绝。
- linked mode默认slot 0为reference path tracing、slot 1为匹配package的deferred evaluator。匹配只用export/snapshot/package/program/asset/instance identity；shared program GPU runtime只按`program_id`复用，display name和UI index不参与解析。
- preset切换或edit在reference resources/pass、neural package、raw upload与compiled state全部成功后才commit；失败恢复source/GPU/pass、两个slot、linked mode、freeze和accumulation状态。manual mode下的preset/edit不得偷偷重新启用linked mode。
- capture v4可增加`viewer_material_binding`与`viewer_material_state`字段；authoritative edit replay仍由viewer-scene中的完整source state恢复，随后按capture slot identity懒加载catalog内对应package，并用恢复出的同一完整parameter view重新运行每个匹配neural slot的instance compiler。旧v3/v4 reader保持兼容。
- Windows pinned Slang对过长的package closure路径会在相对include处失败；Metal catalog准备入口必须对输出根做路径预算预检，正式产物使用短的`artifacts/viewer/...`路径。
- C++ loader 必须复核 artifact schema、V1 capability audit、compiler/stb identity、精确文件集合和每个文件的 SHA-256；不能只相信 catalog。
- 2D texture 使用 bridge-decoded payload、origin、pixel type 与 gamma；BSDF-data texture 使用 artifact 的 Float32 3D payload。argument block/RO data 按 16-byte row 上传。
- viewer 使用 `ProgramDesc::addShaderModule("NclsMdlGenerated").addString(...)`；generated HLSL 不进入根仓库，也不链接 MDL SDK runtime DLL。
- V1 同一 scene specialization 只允许一个 material-specific generated MDL program。MDL 路径延续必须调用同一 target code 的 `surface_scattering_sample`，环境光 MIS 必须调用同一 target code 的 `surface_scattering_pdf`；`sample.weight` 直接使用 SDK 定义的 `bsdf_over_pdf = f |n_s·wi| / pdf`，不得再次乘 cosine 或除 PDF。
- runtime`ReferenceProgramDescriptor`必须公开并完整实现`prepare/evaluate/sample/pdf`，缺任一入口即fail closed。训练通过公共`ReferenceBackendSession`调用同一canonical state；MDL SDK compiler只作为该program的内部provider，不存在MDL专用公共backend、专用query shader或窄化的第二套capability plane。纹理过滤能力必须与typed resource/context实际传入的footprint一致。
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
| ViewerMaterialCatalog缺entry、重复identity、悬空binding、越界路径、runtime/artifact/package hash漂移 | 整体拒绝；writer不发布partial目录，viewer不改变当前comparison |
| linked typed values缺path、多path、未知path、越hard range或reference/runtime write不完整 | candidate拒绝；左右binding与accumulation identity保持原值 |
| capture replay的catalog/export/edit-state或package component identity漂移 | replay拒绝，不按display name或index寻找近似项 |
| Windows catalog输出根使closure路径超过固定预算 | prepare阶段报错并要求更短output root，不等到Slang link时才失败 |
| enum default仍为MDL对象，或vector normalization range未规范化 | 692-entry editor预检报材质名/export ID并终止；不得开始reference compile或asset cook |
| linked content的声明SHA-256与来源文件不符 | 准备阶段拒绝；不得把错误hardlink写入package |
| Windows并发cache目录原子发布瞬时拒绝 | 有界短退避重试；超过次数后保留既有cache/staging边界并明确失败 |
| 交互deferred在单个frame内遍历完整tile lattice，或每tile同步`submit(true)`直到全图结束 | viewer响应性门失败；改为跨frame单workgroup推进，headless才允许同步排空 |
| candidate复制完整catalog，或deferred启动同时创建未使用的package PT pass | 内存/启动门失败；catalog共享不可变所有权，program pass按slot mode懒创建 |
| falcor2 import、launch 或 runtime fallback 出现在 viewer 路径 | boundary test 失败 |
| MDL evaluate 配通用 cosine/GGX proposal，或 MIS 使用非 canonical MDL PDF | 数值正确性失败；不得发布 viewer/capture |
| `sample` 返回 absorb、非有限方向/weight，或连续事件 PDF 非正 | 终止该路径；不得换 generic proposal 冒充同一 estimator |
| capture 全 finite 但孤立白点随 spp 增加 | 不能据此通过视觉门；执行 weight/PDF 尾部诊断，不得 clamp |

## 5. Good / Base / Bad Cases

- Good：car paint 的 evaluate/sample/pdf 来自同一 generated module；路径使用 `bsdf_over_pdf`，环境 NEE 用 MDL PDF 做 MIS，1024 spp 中 flakes 形成连续材质结构而不是随机白点。
- Good：选择`brass/sheet` authored preset后左侧装载该export的canonical MDL artifact，右侧按component identity懒加载同entry package；修改`texture_scale`或round-corner参数时两侧从同一typed value提交。
- Good：quality-first evaluator在1600×900交互启动时先跨frame完成16×全屏反馈，再逐级精化；消息泵可在每个workgroup后处理输入。相同scene的headless capture直接以stride=1完成精确网格。
- Good：692个entry按`texture_set_id`分组物化；66个共享同一texture set的preset只cook一次大grid，但各自保留独立argument block、editor state和instance identity。
- Base：默认 car paint 在 shaderball 上运行；其他 scene slot 继续使用 LayerStack fallback，不产生第二份 MDL generated program。catalog 选择 scratched aluminum 时仍记录同一 snapshot/artifact identity。
- Base：关闭linked mode后保留旧manual slots/package选择器；继续选择catalog preset只更新reference且保持manual mode。
- Bad：viewer 直接读 `.mdl` 自行猜参数；从 falcor2 复制 shader/binary；用固定 roughness GGX 采样任意 MDL closure；hash 失败后显示旧图却把新 asset 写入 capture；为混合多 MDL program 文本重命名 generated symbols。
- Bad：按692个entry反复写出并重哈希同一百MiB级grid，或在每次已有catalog启动时递归统计logical bytes；这会让准备/启动时间与逻辑重复量绑定，而不是与唯一内容量绑定。
- Bad：虽然拆成8×8 workgroup，却在一次`onFrameRender`内循环11,250个tile并逐个同步submit；它只避免单dispatch触发TDR，仍会让Windows消息泵持续无响应。
- Bad：用preset display name或列表序号猜右侧package，或先覆盖active argument/raw buffer再运行compiler；失败时会造成左右错配且无法回滚。

## 6. Tests Required

- unit/static：六项 catalog、unknown default、artifact/compiler/capability/hash检查存在、viewer/falcor2 boundary、公共 `PathSurface`、matched `sample/pdf` 路由与 LOD0。
- unit/static：`ViewerMaterialCatalog@1` exact-field/canonical-id/path containment、duplicate/hanging binding、六类responsibility、bool/int/enum/float/float2/color、完整values map、hard range、program cache和candidate commit顺序；脚本静态断言无训练入口。
- exporter：先预检全部692个editor contracts；正式registry/checkpoint输出692 entries、145 rejected cutout、一个program、52个texture sets及692个独立asset/instance/reference identity。相同content digest必须`samefile`或跨卷安全复制；任一中途失败后output root不存在。
- exporter回归：至少包含对象形态enum default、共享scalar的`float2` range、Windows cache发布瞬时`PermissionError`、已有catalog快速返回，以及原报错的`medium_pitted_steel`严格artifact/package加载。
- viewer静态：断言交互deferred调用单tile入口、该入口无循环和同步submit、stride按16/8/4/2/1推进；headless仍调用完整tile入口且显式`gPreviewStride=1`。Windows真实进程在默认1600×900启动后至少连续采样30秒`Responding`，首次shader编译允许短暂峰值，但不能持续无响应。
- GPU canonical backend：固定 diffuse artifact 上验证 sampled direction/event 有效，sample PDF 等于 formal PDF，`bsdf_over_pdf == evaluate * |n_s·wi| / pdf`；容差按 float32 formal query 冻结，不能根据结果调宽。
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

```cpp
// 错：UI先更新reference，再按display name寻找一个neural package。
installReferenceSource(candidate);
selectPackage(candidate.displayName);

// 对：catalog identity解析完整binding；两侧candidate均成功后才发布。
auto program = ensureLinkedMdlProgram(entry);
installReferenceCandidate(candidate);
compileNeuralCandidate(program, candidate.parameterView);
commitLinkedComparison();
```

```python
# 错：每个entry重新写出并回读相同的大asset blob。
write_scattering_package(package_root, asset_payload=shared_asset)
ScatteringPackage.open(package_root)

# 对：writer仍按payload计算canonical hash，但相同digest复用已验证内容；
# 全量严格payload验证延迟到实际选择entry时执行。
write_scattering_package(
    package_root,
    asset_payload=shared_asset,
    linked_content_store=verified_objects,
)
ViewerMaterialCatalog.open(catalog_path, verify_payloads=False)
```

```cpp
// 错：交互frame同步排空完整网格；每个tile安全不代表frame可交互。
for (auto tile : allTiles) { execute(tile); submit(true); }

// 对：交互每frame推进一个tile并逐级精化；headless单独排空stride=1网格。
if (headless) executeAllTiles(/* stride = */ 1u);
else executeOneTile(progressiveCursor);
```

```slang
// 错：closure 有自己的窄峰，却用无关的固定 GGX PDF 除 evaluate。
direction = sampleFixedGgx(0.2, rng);
weight = mdlState.evaluate(direction, sg).f * absCosine
    / fixedGgxPdf(direction);

// 对：canonical state 的三个入口都落到同一份 MDL target code。
let mdlState = mdlBackend.prepare(context, material);
NclsScatteringSample sample;
if (!mdlState.sample(sample, sg)) return;
direction = sample.directionWorld;
weight = sample.weight; // 已经是 bsdf_over_pdf
misPdf = mdlState.pdf(lightWorld).forward;
```
