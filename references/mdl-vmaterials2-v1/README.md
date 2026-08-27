# MDL / NVIDIA vMaterials 2 reference package

这一 package 登记项目第一条原生 MDL source family。MDL SDK 负责解析 MDL、class compilation、标准 closure 和 HLSL target code；项目负责 canonical source identity、typed 参数编辑、编译桥、renderer state、资源 ABI、方向与纹理查询约定，以及当前 Falcor 8 的正式 GPU 执行。MDL 材质不会先转成 LayerStack、OpenPBR、MaterialX `standard_surface` 或 USDPreviewSurface 后再继承 MDL GT identity。

正式数据路径只有一条：

```text
mdl.program@1 SourceSnapshot
  -> tools/reference/mdl_sdk_bridge
  -> ncls.mdl-compiled-artifact@1
  -> shaders/ncls/reference_backends/mdl_runtime.slangh + mdl_query.slang
  -> 当前锁定 Falcor 8
  -> EvaluatedBlock / TrainingBatch@1
```

锁定的 falcor2 + 同版 MDL SDK 只用于进程外 parity。它不得被 formal provider、collector、live batch、training runner 或产品 CLI 当作 fallback；oracle output 只进入 `artifacts/reference-parity/mdl/`。两条验证实现共享 MDL SDK closure 数学，所以 falcor2 parity 主要检查 shading state、方向、argument block、RO data、2D/3D 资源和 filtering 接线；解析 fixtures 负责提供不共享的数学不变量。

## vMaterials 2.4.0 shortlist

- `Carpaint_Shifting_Flakes`：color-shifting paint，使用 RO data、3 个 SDK BSDF-data 3D table 和 scratch roughness texture。
- `Copper_Antique_Brushed_Patinated`：带 patina、brush normal、smudge 和 roughness 的铜材质。
- `Aluminum_Scratched`：多通道微观划痕和污渍金属。
- `Ceramic_Tiles_Glazed_Versailles`：glaze、mortar、normal 与 pattern 组合，使用 RO data。
- `Velvet`：sheen/velvet closure，使用 SDK BSDF-data table 和 2D imperfection/normal/color textures。
- `Wood_Tiles_Pine_Mosaic`：与 NVIDIA neuralappearance/falcor2 示例管线做 module/export correspondence。

`assets.json` 固定 archive URL、size、SHA-256、ETag、每个 exact export signature、默认参数、source/resource hashes 和 runtime capability audit。原包与展开资产位于 ignored 的 `assets/`，不进入根 Git。

```powershell
.\scripts\fetch_mdl_sdk.ps1
.\scripts\build_mdl_reference.ps1 -Configuration Release
.\scripts\fetch_mdl_assets.ps1 -VMaterials2 -AcceptNvidiaOmniverseTerms
```

## V1 合同与验收状态

V1 只对外声明 surface-BSDF `evaluate`，query response 是线性 RGB `f * |n_s · wi|`。内部 query 同时输出 MDL BSDF PDF 作为诊断字段，但不构成公共 matched `sample/pdf` 能力。纹理坐标使用原生 UV；纹理过滤固定为 `ExplicitLod(0)`，因此统一 batch 中的 `uv_dx`、`uv_dy` 与 `mip_level` 均为零，V1 不声明 footprint/derivative filtering。

typed editor 已覆盖标量、布尔、颜色/向量、enum、SDK hard/soft range metadata 与受约束的 `texture_2d` 资源。每次编辑都产生新的 canonical snapshot，并由 MDL SDK 重新构造 argument block；资源 URI 必须位于 pack 内且 hash 会进入 snapshot identity。

当前不声明 emission、volume、displacement、measured BSDF 或 light profile；遇到未支持资源或静态上限时必须 fail closed。`ncls.mdl-vmaterials2@1` 已通过三层验收并登记为 `active`：解析 Lambertian fixture、MDL SDK native backend 对 current Falcor 8、以及隔离 falcor2 对 car paint/copper 的冻结 formal parity。可再生证据分别位于：

- `artifacts/reference-parity/mdl/native-fixtures-v2/report.json`：7 个 disjoint fixture query，response/PDF 最大相对误差分别为 `2.0645e-7` / `1.6723e-7`；
- `artifacts/reference-parity/mdl/formal-stb-v6/report.json`：两种 vMaterials、264 个 query，全部通过冻结门槛，最大 response/PDF 相对误差不超过 `1.2201e-7`。

正式纹理解码固定独立 `external/stb` commit `013ac3beddff3dbffafd5177e7972067cd2b5083`，`stb_image.h` SHA-256 为 `594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3`。这是项目 formal dependency；它不来自 falcor2，固定同一 decoder 语义是为了消除 JPEG 解码差异，而不是把 oracle 引入正式路径。viewer integration 与 image parity 仍保持 `pending`，不由逐方向 parity 代替。
