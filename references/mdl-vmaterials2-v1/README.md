# MDL / NVIDIA vMaterials 2 reference package

这一 package 登记项目第一条原生 MDL source family。MDL SDK 负责解析 MDL、class compilation、标准 closure 和 HLSL target code；项目负责 canonical source identity、typed 参数编辑、编译桥、renderer state、资源 ABI、方向与纹理查询约定，以及当前 Falcor 8 的正式 GPU 执行。MDL 材质不会先转成 LayerStack、OpenPBR、MaterialX `standard_surface` 或 USDPreviewSurface 后再继承 MDL GT identity。

正式数据路径只有一条：

```text
mdl.program@1 SourceSnapshot
  -> tools/reference/mdl_sdk_bridge
  -> ncls.mdl-compiled-artifact@1
  -> NclsMdlGenerated + shaders/ncls/reference_backends/mdl.slang
  -> ReferenceBackendCapability.open()
  -> generic ReferenceBackendSession
  -> 当前锁定 Falcor 8 / online evaluator 与 sampler routes
```

公共backend同时覆盖LayerStack、MERL、OpenPBR、MaterialX与MDL，并在Windows选择D3D12、Linux选择Vulkan。MDL SDK bridge仅是`mdl.program@1`的内部program provider；SDK library、plugins、target-code与bridge路径都由根`references/reference-backend-toolchains.json`解析，上层没有MDL专用backend或兼容入口。

锁定的 falcor2 + 同版 MDL SDK 只用于进程外 parity。它不得被 formal provider、collector、live batch、training runner 或产品 CLI 当作 fallback；oracle output 只进入 `artifacts/reference-parity/mdl/`。两条验证实现共享 MDL SDK closure 数学，所以 falcor2 parity 主要检查 shading state、方向、argument block、RO data、2D/3D 资源和 filtering 接线；解析 fixtures 负责提供不共享的数学不变量。

## vMaterials 2.4.0 首批 neural cohort

接下来的 neural 适配以 11 个 module、172 个 authored exports 为首批 cohort：

- 连续/参数化主族：`Ceramic_Tiles_Glazed_Versailles`、`Carpaint_Metallic`、`Carpaint_Shifting_Flakes`、`Effect_Pigment_Metallic`、`Velvet`、`Copper_Antique_Brushed_Patinated`、`Aluminum_Scratched`、`Retroreflective_Material`；
- 离散资源或 closure 子族：`Carbon_Fiber`、`Suede_Leather`；
- 空间资源与 NVIDIA neuralappearance 管线对照：`Wood_Tiles_Pine`。

入选 module 保留全部 authored presets；同一材质族训练一个支持原生参数的 neural program，冻结权重后再测试未见插值、未见端点和离散资源切换，不为每个颜色或粗糙度 preset 分别训练。完整分组、路径、preset 数量和评测边界见 [`docs/research/mdl_vmaterials_neural_cohort.md`](../../docs/research/mdl_vmaterials_neural_cohort.md)。

`assets.json` 现已登记 11 个 family 主入口；原有 6 个 viewer/parity 入口的 source identity、默认参数、资源 hash 与既有 audit 字段保持不变，新增 5 个入口只完成 source registration，不自动扩大 viewer/parity 白名单。

`families.json` 是 authored 状态的权威 catalog：由锁定 MDL SDK module API 枚举并逐 export class-compile，恰好包含 11 个 families、172 个唯一 `module + exact export` entries，以及 81 个去重后的 pack-relative source resources。每个 preset 保存 typed 默认参数、canonical snapshot、传递 module/texture 闭包、SDK BSDF-data content hash、resource signature、compiled/sub-expression identity 和 runtime capability audit。当前 164 个 opaque presets 可转成普通 `mdl-export` runtime locator；8 个 `Suede_Leather_*_Punched` 全部保留，但因 `geometry.cutout_opacity` 尚未进入公共输出合同而 fail closed。

catalog artifact 与 runtime artifact 分工明确：前者只 materialize 编译元数据和 SDK BSDF data，2D texture 记录 source path/type/dimensions/hash，避免 172 次重复落盘 4K decoded image；后者由 `compile_snapshot()` 完整解码并绑定纹理。inspection artifact 不能作为 runtime artifact。`Rgba_16` 等输入格式不是 unsupported capability：bridge 保留 4×Uint16 payload，generic Falcor binder 与 viewer 以 `RGBA16Unorm` 消费；punched preset 的唯一 unsupported reason 是 cutout 输出语义。

完整 vMaterials 解包目录保持在 ignored 的 `assets/source-materials/mdl-vmaterials2/2.4.0/`。研究分组写入 manifest，不移动 `Materials/vMaterials_2` 内的原始 module；最终精简包也必须保留 module、import 和纹理的原始相对路径。`assets.json` 继续记录当前已注册资产的 exact export、默认参数、source/resource hashes 与 runtime capability audit，但选中一个材质族不等于绑定或运行整个 vMaterials 包。

Base Materials 与 Automotive Materials 的 ZIP 已下载但没有完整解压，也不进入首批 reference、训练或 viewer 路径。Base 主要补基础覆盖且依赖未随 leaf wrappers 提供的 OmniPBR templates；Automotive 的当前 leaf materials 主要封装 `OmniUber_Automotive` / `OmniGlass`，还缺少可独立打包的完整 template 依赖闭包。两者的分析保留在上述研究文档中，等首批 parameter-aware neural program 稳定后再决定是否专项接入。

```powershell
.\scripts\fetch_mdl_sdk.ps1
.\scripts\build_mdl_reference.ps1 -Configuration Release
.\scripts\fetch_mdl_assets.ps1 -VMaterials2 -AcceptNvidiaOmniverseTerms
conda run -n neural-shading python tools/reference/generate_mdl_vmaterials_manifest.py `
  --refresh-artifacts --artifact-root build/mdl-reference/vmaterials-preset-audit-v1
conda run -n neural-shading python tools/reference/generate_mdl_vmaterials_manifest.py `
  --check --artifact-root build/mdl-reference/vmaterials-preset-audit-v1
```

## V1 合同与验收状态

V1 canonical backend 完整实现 `prepare/evaluate/sample/pdf`；`evaluate()` 返回线性 RGB `f`，不含 cosine，renderer 或 response adapter 在消费点显式乘输入 `NclsShadingFrame` 的 `|n_s · wi|`。MDL `geometry.normal` 或 normal map 引入的 material-local cosine 比值保留在等价 `f` 中，因此乘回输入 frame cosine 后逐值恢复 MDL 原生 response。纹理坐标使用原生 UV；当前 viewer catalog 仍固定 explicit LOD 0，online query 则按统一 shading context 传递 UV derivatives，不能为 MDL 增加专用 producer。

typed editor 已覆盖标量、布尔、颜色/向量、enum、SDK hard/soft range metadata 与受约束的 `texture_2d` 资源。每次编辑都产生新的 canonical snapshot，并由 MDL SDK 重新构造 argument block；资源 URI 必须位于 pack 内且 hash 会进入 snapshot identity。

当前不声明 emission、volume、displacement、measured BSDF、light profile 或非不透明 cutout；遇到未支持 closure/resource shape 或静态上限时必须 fail closed。输入 pixel type 单独按 typed format 审计，不能拿 capability unsupported 掩盖纹理降级。`ncls.mdl-vmaterials2@1` 已通过解析 fixture、MDL SDK native backend 对 current Falcor 8、以及隔离 falcor2 对 car paint/copper 的冻结 formal parity。可再生证据分别位于：

- `artifacts/reference-parity/mdl/native-fixtures-v2/report.json`：7 个 disjoint fixture query，response/PDF 最大相对误差分别为 `2.0645e-7` / `1.6723e-7`；
- `tests/gpu/test_mdl_native_crosscheck.py`：额外用倾斜 `geometry.normal` fixture 验证 `public f × input-frame cosine` 与 SDK native response 一致；
- `artifacts/reference-parity/mdl/windows-unified-backend-formal-framecosfix/report.json`：统一 backend 上的两种 vMaterials、264 个 query，全部通过原冻结门槛；carpaint/copper response 最大绝对误差分别为 `5.9605e-8` / `7.4506e-9`，PDF 全部通过。

正式纹理解码固定独立 `external/stb` commit `013ac3beddff3dbffafd5177e7972067cd2b5083`，`stb_image.h` SHA-256 为 `594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3`。这是项目 formal dependency；它不来自 falcor2，固定同一 decoder 语义是为了消除 JPEG 解码差异，而不是把 oracle 引入正式路径。viewer 现在验证并消费相同 artifact、decoded texture 与 V1 capability，已登记为 ready；独立 renderer `image_parity` 仍保持 pending，不由逐方向 parity 或单边 capture 代替。
