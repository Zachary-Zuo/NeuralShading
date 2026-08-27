# 原生 MDL reference 实施报告

## 结论

`ncls.mdl-vmaterials2@1` 已作为 active ground-truth reference 接入当前统一框架。正式路径只有：

```text
mdl.program@1 SourceSnapshot
  -> 项目 MdlSdkCompilerBridge + MDL SDK 2025.0.0
  -> ncls.mdl-compiled-artifact@1
  -> 项目 runtime/query shader
  -> Falcor 8 @ 9dc819c
  -> EvaluatedBlock / HDF5 / TrainingBatch@1
```

falcor2 @ `d629c967` 使用同版 MDL SDK，只由隔离 parity runner 启动。静态边界测试证明 formal `src/ncls` 不 import/launch falcor2 或 oracle；formal parity provenance 也记录 `formal_provider_imported_falcor2=false`。

## 实施范围

- source：canonical module/export/argument/resource identity、typed bool/int/float/double/color/vector/enum/texture editor、range 与 pack containment；
- compiler bridge：class compilation、HLSL/native backend、argument block、RO data、2D/BSDF-data texture、stb JPEG decode、精确 artifact 文件 hash；
- fail closed：emission、volume、displacement、measured BSDF、light profile、未知 texture shape 和静态资源上限；
- runtime：current Falcor 8 offline evaluate、CUDA shared-buffer live batch、统一 HDF5 roundtrip；
- 资产：car paint、patinated copper、scratched aluminum、glazed ceramic、velvet、pine mosaic 六种 vMaterials 2.4.0；
- oracle：版本化 request/result/schema、冻结 calibration/formal gate、MDL SDK native cross-check。

## 合同修订记录

### Texture derivative → ExplicitLod(0)

- trigger：真实 MDL SDK HLSL 与 falcor2 effective backend state 均为 `texture_runtime_with_derivs=off`，oracle 使用 `ExplicitLodSampler(0)`；旧规划把 derivative support 写成已启用，和权威运行状态矛盾。
- invalidated evidence：旧规划中关于 `uv_dx/uv_dy`、`SampleGrad` 与 derivative-footprint capability 的文字；这些字段从未被 V1 shader 消费。
- scope impact：V1 保留 UV spatial state，但 capability 只声明 `evaluate/spatial`；live batch 的 `uv_dx/uv_dy/mip_level` 固定为零并记录 `uv_derivatives_consumed=false`。
- rerun required：重新执行全部 MDL GPU tests、MDL native parity 和 car paint/copper falcor2 formal parity；均已完成。冻结 tolerance 与 formal query set 未变。

### JPEG decoder identity

早期 formal/oracle 差异来自 Pillow 与 falcor2/stb 的 JPEG 量化差异，而不是 closure 或 frame。正式 bridge 改为独立锁定 `external/stb`，不依赖 falcor2 源码；commit 与 header hash进入 compiler/artifact/report identity。该修改后 response/PDF 达到近 bit-exact parity。

## 数值证据

- `artifacts/reference-parity/mdl/native-fixtures-v2/report.json`，SHA-256 `4d0097ebe34c0826a32f2ab226c90274237a013726726bae5b4af968d55c72f8`：7 queries，response/PDF 最大相对误差 `2.0645e-7` / `1.6723e-7`，通过；
- `artifacts/reference-parity/mdl/formal-stb-v6/report.json`，SHA-256 `3f1df5e7d9aa80dbe1d95be5ab380b5dac077e586c4d9dc7249bb6f686fd773e`：car paint + copper 共 264 queries，最大 response/PDF 相对误差 `1.2039e-7` / `1.2201e-7`，通过；
- 六材质 current-Falcor discovery/load/evaluate smoke 与 car paint/copper unified live `TrainingBatch@1` smoke 通过。

## 质量门

- MDL bridge Release build：通过；仅有 MDL SDK public header 的结构对齐 warning；
- unit：`89 passed`；
- MDL current-Falcor GPU：`11 passed`（9 个完整组合回归，加上新增的两种 fancy live 参数化 case）；
- PowerShell parser：9 个新增脚本语法通过；
- reference tree：6 packages 验证通过；
- `git diff --check`：通过；项目未配置独立 linter/type checker；
- Falcor、pbrt-v4、OpenPBR、openpbr-bsdf、GLM、MaterialX、falcor2、stb：固定 HEAD 且工作树 clean。

## 已知 V1 边界

viewer integration 与 image parity 仍为 `pending`；V1 不声明 derivative filtering、公共 matched sampler/PDF、emission、volume、displacement、measured BSDF 或 light profile。早期诊断使用的 ignored `external/libjpeg-turbo` 与两个 build 目录没有任何最终依赖；删除命令被执行环境策略拒绝，故本机可能仍保留这些可重建目录，但它们不进入 Git、manifest 或正式构建。
