# MDL Reference Viewer 实施报告

## 它是什么

当前 Falcor 8 `NclsViewer` 已能直接执行由锁定 MDL SDK bridge 生成的 `ncls.mdl-compiled-artifact@1`。正式路径为：

```text
vMaterials source
  -> 项目 MDL SDK bridge
  -> 带完整 identity/hash 的 generated HLSL 与资源
  -> 当前 Falcor 8 动态 string module
  -> viewer shaderball / capture
```

falcor2 没有进入 viewer 的 import、进程或运行时依赖；它仍只承担已隔离的外部 numerical oracle。

## 已实现范围

- 六种固定 vMaterials catalog：变色 flakes 车漆、古铜拉丝铜锈、微划痕铝、釉面 Versailles 陶瓷、velvet、pine mosaic 木材。
- C++ loader 对 schema、MDL SDK/compiler/stb identity、精确文件集合、SHA-256、V1 capability、资源类型/尺寸与 containment 做 fail-closed 验证。
- viewer 动态组合锁定 target-code types、项目 `mdl_runtime.slangh`、material-specific generated HLSL 与 `MdlViewerAdapter.slang`。
- argument block、RO data、2D texture 与 BSDF-data 3D texture 从同一 artifact 上传；V1 固定 `ExplicitLod(0)`，不冒充 derivative filtering。
- preset 切换采用 validate/build/swap；失败时恢复上一份 source、GPU resource、pass 与 metadata。
- capture manifest 记录 asset、snapshot、artifact、MDL SDK/compiler 与 filtering identity。
- source 对外仍只声明 `evaluate`；viewer 路径延续与环境光 MIS 内部消费同一 MDL target code 的 `sample/pdf`，不把它提升为训练/provider 的公共 sampler capability。

## 验证结果

- Release viewer build：通过；动态 material-specific module 已在真实 scene specialization 中编译执行。
- unit：`92 passed in 7.04s`。
- Falcor/D3D12 GPU：MDL viewer/formal adapter parity、texture runtime、CUDA live runtime、公共 PathSurface 共 `7 passed in 6.79s`。
- car paint、patinated copper、scratched aluminum 均完成 1024 spp headless capture；单 panel EXR 均为 `360 x 320 x 3` 且全有限。
- 三种 capture 的均值分别为 `0.6792268`、`0.1551895`、`0.4051748`，artifact short id 分别为 `1988478f2d16`、`cf8caf25c6be`、`eaeb2af00d7d`；目视分别呈现变色 flakes、青绿铜锈/裸铜斑驳和银白微划痕。
- `git diff --check` 通过；Falcor、MaterialX、pbrt-v4、OpenPBR、openpbr-bsdf 与 GLM 上游工作树均干净。

## 当前结论与边界

MDL `viewer_integration` 已具备可运行、可追溯的 ready 证据。由于尚未引入独立 renderer 的同场景 image oracle，registry 的 `image_parity` 继续保持 pending；当前三张 capture 是运行与视觉差异证据，不把它们写成 image parity。

交互式 viewer 已启动，默认是 car paint。右侧没有加载 neural package 时为空属于预期；本任务只接入 source reference，不生成或训练 approximation。

## Firefly 根因与修复

用户现场发现 car paint 与 glazed ceramic 的孤立白点随 spp 累计持续增加。固定 52 万方向的 GPU 诊断表明，原 viewer 用 fixed-roughness GGX 除 MDL evaluate 时，car paint/ceramic 最大权重分别达到 `3747/7018`；用同一 MDL PDF 后分别为 `75/17`。原因是 flakes 与釉面 coat 的窄峰不受固定 GGX proposal 覆盖，环境光 MIS 又错误使用该 proposal PDF。

修复没有增加 clamp：`MdlViewerAdapter` 现在调用 generated target code 的 `surface_scattering_sample/pdf`，路径直接使用 `bsdf_over_pdf`，环境 NEE 使用同一 MDL PDF。相同 replay 的 1024 spp car-paint capture 中，局部邻域定义的孤立 firefly 从 `193` 降为 `0`，最大通道从 `4112` 降为 `526`；剩余高值是连续高光/flake 区域。glazed ceramic 同门为 `2` 个边缘连续高光命中，目视球面不再出现随机白点。两份 EXR 都全有限。
