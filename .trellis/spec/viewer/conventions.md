---
name: viewer-conventions
description: viewer 的 pass 划分、MethodBundle loader 加载顺序与泛型化目标、四族 reference 分派、capture v3 / viewer-scene sidecar、光照方向约定、benchmark 输出位置
paths:
  - apps/viewer/**
  - configs/viewer-*.json
  - scripts/benchmark_viewer.ps1
---

# viewer 约定

## pass 划分（`docs/viewer_spec.md`「Pass 划分」）

```text
Falcor Scene
  ├─ SceneReferencePathTracer（ReferencePathTracer.cs.slang）：raw running mean + 亮度二阶矩
  │    └─ Denoise.cs.slang：display-only a-trous cross-bilateral
  └─ Raster SceneVisibility.3d.slang → Prepare.cs.slang → Approximation.cs.slang（deferred lighting）
raw + optional denoised + approximation → Composite.cs.slang（定量输入固定为 raw）→ tone map → UI
Parity.cs.slang：bundle 加载期 Python/Slang 固定方向 parity probe
```

- reference 独立追踪 primary ray；实时侧用 Falcor raster G-buffer；两者同 camera revision 与分辨率。
- reference 与实时侧共享 visibility、surface frame 与 emission，避免把非散射差异算成材质误差。
- 编译时按场景实际出现的 family mask 特化 reference shader；不改各族方程，不把它们统一成某个 backend ABI。

## 四族 reference 分派（`ReferenceSource.{h,cpp}`、`ReferencePathTracer.cs.slang`）

每个 Falcor material slot 保存独立 `ReferenceSource` 与 GPU 资源，命中后按 material ID 分派：LayerStack 原生随机游走；MERL 表查询 + 采样/PDF；OpenPBR resolved inputs + LUT；MaterialX `standard_surface` subset + 纹理 / normal map（mip 由 ray cone + UV Jacobian）。单界面 LayerStack、MERL、OpenPBR、MaterialX 用环境 importance sampling + BSDF sampling 的 MIS；深层 LayerStack 没有完整方向 PDF，环境路径保持 raw 有限深度估计。

## MethodBundle loader（`MethodBundle.{h,cpp}`）

- 加载顺序固定：manifest schema + 全文件 hash → 平台 / Slang / shader model / 散射合同版本 → IR 支持 → shader variant → `CompiledMaterial` → descriptor 声明的状态资源 → parity probe → 才进右侧方法列表。失败进 `BundleScanResult.failures` 并显示具体原因，不退回相近方法。
- 当前只接受 `diagnostic` 的 `film-m1-direct-neural@1`，backend 字符串、entry 名、`architecture_id` 硬编码在 `MethodBundle.cpp`；`Prepare / Approximation / Parity` 直接调 `nclsFilmM1*` 自由函数。目标（`p1_v2_plan.md` V4.2–V4.3）：按 `backend_id` 查表的 `BackendRegistry`（shader 路径、入口、state stride、layout 格式），pass 改为 `#include NCLS_METHOD_BACKEND_HEADER` + `typedef NclsMethodBackend` 经 `INclsScatteringBackend` 调用，`realtime` 时按硬线校验 `cost_claims`。新代码按目标结构写，不再往硬编码上叠。
- `kRequiredCapabilities`：deferred 为 `Prepare|Evaluate|AnisotropicFrame`；启用 PT + method 模式再要求 `Sample|Pdf`。
- viewer 需要分别计时 `prepare` 与 `evaluate`；只在 prepare 输出固定 closure 的基线不能称为验证了 direct neural evaluator。

## capture v3 与 viewer-scene sidecar（`docs/contracts/viewer_scene.md`）

- capture 至少记录 scene / HDRI / 源材质 / 相机 / 分辨率与 SHA-256、reference integrator ID、raw spp、`reference_scene_max_bounces`、`reference_layer_walk_max_depth`、噪声估计、raw/denoised 权威性、bundle、GPU timing、各文件角色。
- `ncls.viewer-scene@1` 逐 material slot 保存 family 与状态：LayerStack 内嵌 `MaterialProgram`；OpenPBR 保存具名 resolved 参数 + 原始 `.mtlx` provenance；MERL 保存测量表 URI/hash；MaterialX 保存文档 / 纹理 identity 与未被纹理驱动的 constant override。每个 binding 都要显式合法的 `source_asset_sha256` 与 `state_sha256`，不接受空字符串或"自动采用当前文件"。
- `--replay capture.json` 优先加载同目录 `*-scene.json`；material binding 数量必须与 Falcor slot 数一致、无重复。

## 交互与光照约定

- 方向光向量是 surface-to-light；矩形灯由 center + 两个 half-axis 定义，`normalize(cross(U, V))` 为发光法线；颜色按线性 RGB 编辑。
- UI 分组固定：Scene and camera / Material / Lighting（按灯型再分）/ Reference and display / Realtime method / Capture / Performance and status；禁用灯的子控件隐藏并说明。
- 交互模式把 Falcor/Slang 详细诊断写进 exe 同目录 `NclsViewer*.log`；`--verbose-console` 恢复；headless 始终保留完整控制台日志。

## benchmark（`scripts/benchmark_viewer.ps1`、`configs/viewer-benchmark-v1.json`）

复用 studio-v1 几何 / HDRI / 默认材质与 hash，固定三段相机路径、分辨率、reference 上限与帧数；输出进 `artifacts/benchmarks/viewer/`，含 preset / viewer / source / Falcor / scene / environment / material provenance；实测按 `(method_id, benchmark_scene_id, device_id)` 记录，不回写 bundle 本体。

## 反例

- 在 `NclsViewer.cpp` 里继续加第二个 `if (backendId == "...")` 分支。
- 用左侧源材质 reference 的 sampler 冒充被测方法的 sampling capability。
- 把 deferred-vs-PT 差图当 evaluator 表示误差的定量指标（它混入 GI 与可见性差异）。
- 手工改 `external/Falcor` 里的文件"临时试一下"。
