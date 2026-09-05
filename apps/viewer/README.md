# NclsViewer

`NclsViewer` 是 Windows/D3D12 的 `ScatteringPackage@2` 部署验证器。每个package以`ProgramRuntimeCache + AssetBinding + InstanceBinding`原子装入slot，program可按identity复用，asset/instance不会部分替换。界面固定包含两个宽度相同的`ComparisonSlot`；每侧独立选择`path-tracing`或`deferred`。加载、ABI、资源或capability失败只影响本侧，camera aspect和另一侧extent不变。

viewer 不解释 NVIDIA、LayerStack 或其他 program 的私有结构。C++ loader 校验 package 的三个 identity、module closure、typed buffer/texture/sampler descriptor 与 content hash，随后创建通用 `ScatteringBinding`。source 侧由 `SceneReferenceProgram` 选择 concrete canonical backend；积分器本身看不到 source family。NVIDIA latent 的两张 RGBA16F DDS mip chain 由 descriptor 绑定，不需要把 method shader 预编入 viewer。

MDL source reference 是 source 侧的动态 program，不是 neural package。`scripts/prepare_mdl_viewer.ps1` 复用正式 MDL SDK bridge 生成六种 vMaterials 的 hashed compiled artifact/catalog；viewer 再验证 SDK/compiler identity、精确文件集合和 V1 capability。动态 `NclsMdlGenerated` module 只组合 target-code types、项目 renderer callback 与 material-specific HLSL；静态 `reference_backends/mdl.slang` 直接实现 canonical backend，不存在 viewer adapter。falcor2 不在这条启动或运行路径中。

## 共享 renderer 与材质接口

package path tracer 在每个 scene hit 构造完整 scattering context，包括 position、shading/geometric frame、outgoing direction、material instance、UV 与 ray-cone footprint。续路径直接调用当前 slot binding 的 `prepare/sample/pdf`，直接光调用同一 state 的 `evaluate/pdf`；source scene path tracer 也只调用各 source 自己的 canonical state。两侧使用同一个 `PathTracer.cs.slang`，由 `SceneScattering` 在 active material ID 选择 source 或 package；底座、地面和其他对象保留自己的 source。两侧共享光照、遮挡、MIS 与累积行为。

共享 path tracer 在 primary surface 固定做 4 个 environment-light 样本与 4 个 BSDF path samples；power MIS 两侧分别使用 `4 * p_light` 与 `4 * p_bsdf`。BSDF sample 直接 miss environment 时累计带 MIS 的 direct contribution，命中几何时继续追踪完整 path suffix，因此 first-continuation 积分不会再退化成单样本。secondary surface 仍使用 4+4 environment direct MIS 与一条独立 continuation，避免 path tree 随 bounce 指数增长。所有 path material sample 都由 Falcor `UniformSampleGenerator` 产生，再通过同一 `ISampleGenerator` 交给各 backend 自己的 `sample()`；native direction/event/PDF/weight tuple 不被重建。环境 CDF 由与 GPU 双线性 radiance lookup 相同的 cell-integrated reconstruction 构造，避免亮 texel 过滤到相邻 cell 后仍报告暗 cell PDF。续路径 ray origin 根据实际 sampled direction 相对 geometric normal 的符号选侧，不依赖 reflection/transmission event label。该实现不使用 radiance/throughput clamp。

deferred 从 G-buffer 解码 material sentinel、geometric normal/front-facing、UV/gradient 和 frame，经过同一个 scene composer 调用 source/package `prepare/evaluate`。两侧使用相同的局部灯光查询；一次 prepare 复用多次 evaluate。它没有 GI 和场景阴影，不能用 PT 的全图阴影作为材质 parity。

source/package PT 都通过 `PathSurface.slang` 构造 scene surface。Falcor camera basis 含共同 focal-distance 尺度，primary ray-cone spread 使用 `cameraV/cameraW` 的长度比；输出 footprint 是 normalized UV derivative，可直接交给 MaterialX `SampleGrad` 或 neural latent filter。修改这一链路时必须运行 `tests/gpu/test_viewer_path_surface.py`，并用具有明显空间结构的 local-light capture验证，不能只检查 slot `ready` 或平均颜色。

## 构建

只使用项目脚本。它验证锁定 Falcor提交和干净 worktree，临时应用 `patches/falcor-viewer-overlay.patch`，构建结束后反向应用 overlay；`external/Falcor` 最终必须保持干净。

```powershell
.\scripts\build_viewer.ps1 -Configuration Release
```

准备并启动 MDL viewer：

```powershell
.\scripts\launch_mdl_viewer.ps1 -Configuration Release
```

新模型统一从 checkpoint 导出；输出路径由 run 管理。导出命令会打印 package 与 source/catalog 路径：

```powershell
conda run -n neural-shading python -m ncls export outputs/<config>/<run>/checkpoints/latest.pt
.\scripts\launch_viewer.ps1 -Package <package目录> -Material <source文件或catalog>
```

两侧模式用 `-ReferenceMode` / `-NeuralMode` 独立选择，默认 reference PT 与 neural deferred。训练中图像由公共 eval hook 调用同一 viewer，reference 默认 128 spp，实际值只由 YAML 决定。旧视觉证据留在 artifacts 原位置，历史 checkpoint handoff 已删除。

## capture v4 与回放

自动化比较使用 `ncls.viewer-capture@4`。headless capture 从 replay 的各 slot `target_spp` 读取采样目标，reference 默认 128 spp，可只改 YAML；未达到目标时不得导出 EXR，deferred slot 为确定性单次求值，不虚构 spp。双方 PT 的预算互相独立；training diagnostic 默认 reference PT 与 neural deferred，在每个 slot 分别记录 `mode/target_spp/spp`。研究对照是否 matched 由实验设计解释，不是 renderer 的固定数字门禁。`reference_samples_per_frame` 只控制 headless 每次 dispatch 的 batch，不进入交互状态。manifest 的 `slots[2]` 分别记录 `package_id`、`mode`、status 与 runtime/material/source identity；`*-slot-0.exr`、`*-slot-1.exr` 与 `*-difference.exr` 都固定为单 panel 的 `view_resolution`，difference 不复用双 panel composite 纹理。`source-reference` 是内建的权威 source transport 请求，其余值必须对应 `bundle_root` 下通过验证的 package。验证 neural mode 对称性时，可让两个 slot 绑定同一 neural package并分别选择 PT/deferred。

最小 slot 片段如下：

```json
{
  "format_name": "ncls.viewer-capture",
  "format_version": 4,
  "reference_integrator": "ncls.scene-path-tracer@1",
  "bundle_root": "../exports/example",
  "slots": [
    {"package_id": "source-reference", "mode": "path-tracing", "target_spp": 128},
    {"package_id": "<package-id>", "mode": "deferred", "target_spp": 0}
  ],
  "resolution": [1280, 720],
  "reference_spp": 128,
  "reference_samples_per_frame": 16
}
```

无窗口回放与捕获：

```powershell
external\Falcor\build\windows-vs2022\bin\Release\NclsViewer.exe `
  --replay artifacts\captures\request.json `
  --headless --capture artifacts\captures\result
```

同一 replay 把 slot 1 的 mode改为 `deferred`即可验证 deferred 路径。capture只报告实际 ready/unsupported/error 状态，不 fallback 到另一 program或 transport。

## 操作与诊断

交互模式保留 orbit、pan、dolly、材质/source编辑与 viewer scene保存。Comparison slots 中可换模式或 Swap sides。PT/deferred 均整 panel 更新，无 tile 扫屏；deferred 完成后缓存，实际输入变化才重算。标题不进入线性 capture。交互 PT 固定每次 dispatch 追加 1 spp，只要状态不变就持续累积，不受 headless capture 目标限制；任一相机、场景、材质、package或 mode变化都会把统一 sample sequence 重置到 0。普通运行把详细 shader diagnostics写入 `NclsViewer*.log`；需要完整控制台输出时使用 `--verbose-console`。headless 始终保留控制台日志，便于 CI 和 artifact审计。

固定路径 benchmark仍由项目脚本执行：

```powershell
.\scripts\benchmark_viewer.ps1 `
  -BundleRoot outputs\<config>\<run>\exports `
  -Preset configs\viewer-benchmark-v2.json `
  -OutputDirectory artifacts\benchmarks\viewer
```
