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
- loader 同时接受 `diagnostic` 与 `realtime`，不按 `backend_id`、entry 名或 architecture 写分支。bundle 的 `runtime.shader_specialization` 声明 module、反射生成的宏、compiled material/state stride 与 table index；`Prepare / Approximation / Parity` 用 `#include NCLS_METHOD_BACKEND_HEADER` 和公共 `NclsMethod*` alias，经 `INclsScatteringBackend` 编译期 specialization 调用。bundle 的 module 内容必须与当前 viewer runtime 逐字节同 hash。
- `kRequiredCapabilities`：deferred 为 `Prepare|Evaluate|AnisotropicFrame`；启用 PT + method 模式再要求 `Sample|Pdf`。
- viewer 需要分别计时 `prepare` 与 `evaluate`；只在 prepare 输出固定 closure 的基线不能称为验证了 direct neural evaluator。

### 通用 backend 初始化合同

#### 1. Scope / Trigger

当一个冻结 compiled set 要进入 MethodBundle/viewer，或方法的权重布局、`CompiledMaterial`、state packing 发生变化时触发。标准散射接口统一调用语义，但不统一方法私有布局；因此仍需一次资源初始化和编译期 specialization。

#### 2. Signatures

```text
ncls bundle export-compiled-set
  --compiled-set <dir> --preview-material <json> --parity <json>
  --output <new-dir> --display-name <text> --state-id <sha256>

runtime_adapter = {
  shader_module, shader_defines,
  compiled_material_stride, packed_state_stride,
  shared_weight_storage="float16-little-endian",
  backend_descriptor
}
```

方法 module 必须公开 `NclsMethodBackend / NclsMethodCompiledMaterial / NclsMethodPackedState / NclsMethodState`，以及 `nclsCreateMethodBackend / nclsPackMethodState / nclsUnpackMethodState`；前三个 pass 只使用这些公共名字。

#### 3. Contracts

- offset 必须来自 exporter 对实际参数布局的反射结果，作为十进制宏值写入 bundle；shader 内不得维护第二份 offset 表。
- viewer 把共享 FP16 权重绑定为 `StructuredBuffer<uint>`，把 compiled material table 按声明 stride 原样上传，不解释字段。
- `NCLS_METHOD_BACKEND_HEADER` 只允许 POSIX 相对 module；viewer 对 bundle 内 module 与自身 runtime copy 做 SHA-256 一致性检查。
- `prepare()` 返回的 transient State 可以持有 context/resource handle；per-pixel 只保存 `NclsMethodPackedState`，其 stride 必须等于 descriptor。
- runtime class 只描述成本分类，不改变同一 checkpoint/compiled set 的方法语义，也不得触发缩模替换。

#### 4. Validation & Error Matrix

| 条件 | loader 行为 |
|---|---|
| bundle 文件缺失、hash 不一致或 URI 越界 | 拒绝并记录具体文件/URI |
| module 与 viewer runtime hash 不同 | 拒绝并要求重建 viewer/bundle |
| define 名非大写数字下划线、值非十进制整数 | 拒绝，不能进入 shader 编译 |
| weight bytes 非完整 `uint`、material table 不能整除 stride、index 越界 | 拒绝资源初始化 |
| descriptor/state/material stride 不一致 | 拒绝，不猜测相近布局 |
| parity 编译或数值检查失败 | bundle 不进入方法列表；不得回退到另一 backend |

#### 5. Good / Base / Bad Cases

- Good：两个网络形态只靠不同 module/defines/stride 接入，同一 C++ 和 pass 源码加载；UI 如实显示各自 runtime class。
- Base：同一 module 的另一份 checkpoint 只改变权重、material table、内容 hash 与 method ID，不改 viewer。
- Bad：在 `MethodBundle.cpp` 加 `if (backendId == ...)`，或让 viewer 读取裸 `.pt` 并自行解释张量名。

#### 6. Tests Required

- unit：adapter 的 descriptor、反射 offset、material/state stride 与 capability；bundle 内容 hash 篡改必须失败。
- Falcor load-time：每个正式 bundle 编译 `Parity.cs.slang`，固定方向输出在独立测得的跨编译器 envelope 内。
- viewer headless：锁定 method ID 启动后 capture 断言 `approximation_available=true`、method/runtime class 一致。
- build：只经 `scripts/build_viewer.ps1`，结束后 `external/Falcor` 必须干净。

#### 7. Wrong vs Correct

```cpp
// 错：标准接口外再按方法解释资源。
if (method.backendId == "paper") uploadPaperWeights(method);

// 对：资源形状来自 bundle，调用语义来自公共接口。
createStructuredBuffer(method.compiledMaterialBytes, method.compiledMaterialCount, ...);
createMethodPass("Prepare.cs.slang", method.shaderModule, method.shaderDefines);
```

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
