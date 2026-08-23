# Windows 材质查看器规格

## 定位与当前结论

`apps/viewer/` 是 Windows/D3D12 原生材质查看器，用于把源材质 reference 和固定成本实时方法放到同一场景、相机与显示管线中观察。它不是训练 UI，也不是数据集 tile 浏览器。

左侧 reference 已升级为独立的完整场景 path tracer，不再复用 deferred G-buffer 后只计算首个表面的局部响应。当前实现追踪相机射线、Falcor Scene 几何相交、阴影、环境与解析光直接采样、材质散射、跨物体间接反弹和 Russian roulette，并累计线性 HDR raw 均值。LayerStack、MERL、OpenPBR、MaterialX 都在每次命中时以各自原生 reference 参与路径，不需要先转换为统一层模型。

这里的“完整场景 path tracer”描述 transport 覆盖范围，不代表无限深度。场景反弹和 LayerStack 内部随机游走分别有可记录的上限；raw estimator 只相对于这组有限上限求值。文档、UI 和 capture 不得把有限深度结果写成无限反弹的严格无偏解。

## 标准场景合同

默认场景固定为 `studio-v1`，入口是 `configs/viewer-studio-v1.json`。该 preset 锁定：

- MaterialX 1.39.4 的 shaderball GLB 及 SHA-256；
- Poly Haven Studio Small 03 1K EXR 及 SHA-256；
- 各向异性粗糙导体 MaterialProgram；
- 相机、环境、曝光、每帧 spp、场景反弹上限和层内随机游走上限；
- 上游 commit、许可证和资源 manifest。

版本化源资产位于被 Git 忽略但职责固定的 `data/source-materials/viewer-scenes/studio-v1/` 与 `data/hdris/polyhaven_1k/`。获取/验证由 `scripts/fetch_viewer_assets.ps1` 完成；runtime 副本位于 Falcor bin 目录的 `data/ncls-viewer/`。无 replay、无显式场景参数时只能从这个 runtime preset 启动，缺失或哈希不匹配必须报错，不能静默退回解析球。

用户仍可用 CLI 或文件对话框加载其他 Falcor Scene、HDRI 和源材质，以验证未见 mesh/材质；这属于显式覆盖，不改变默认验收场景。

## 画面与比较合同

- 左侧：raw reference 的显示版本，默认可显示有偏去噪预览；
- 右侧：显式选中的 `MethodBundle` 的 deferred 实时结果；未选择方法时 reference 使用全宽；
- 两侧共享场景、相机、材质 slot 绑定、灯光、环境旋转、曝光和 tone mapping；
- reference 独立追踪 primary ray，实时侧使用 Falcor raster G-buffer；两者使用同一 camera revision 和输出分辨率；
- raw reference 与 approximation 生成像素对齐的线性 HDR 图像，再统一 composite/tone map；
- 支持 split、线性绝对差、相对差和放大误差显示；
- 左右不能分别自动曝光，显示操作不能修改任一侧物理输入。

右侧目前没有 path-traced 全局传输，因此图像 difference 是 reference 与完整实时系统的视觉差异，会混入可见性和间接光差异。它适合发现系统级伪影，不是 closure 表示误差的定量替代；closure 的核心指标仍由固定方向响应数据计算。

## Reference estimator 与噪声语义

每帧追加可配置的独立 sample batch，raw texture 保存算术均值和亮度二阶矩。发生相机、场景、源材质、物理灯光/HDRI或积分器上限变化时清空累计；切换 MethodBundle、split、difference、共同曝光或去噪显示开关不清空累计。

环境贴图建立亮度 × `sin(theta)` 的二维 CDF。具有已知散射 PDF 的单界面 LayerStack、MERL、OpenPBR 和 MaterialX 使用环境 importance sampling 与 BSDF sampling 的 MIS；解析灯通过 shadow ray 做 next-event estimation。深层 LayerStack 保留原生随机游走语义，但当前没有完整层栈方向 PDF，不能对 HDRI NEE 做正确 MIS，因此主要依靠散射路径命中环境，预期收敛更慢。

默认去噪器是利用 raw 亮度方差、位置/深度、法线和 material ID 的多尺度 a-trous cross-bilateral filter。它有偏、只改变观看输入：

- comparison 和 difference 始终读取 raw；
- 噪声估计始终从 raw 二阶矩计算；
- `reference_linear` 始终指向 raw EXR；
- `reference_denoised_preview` 是单独的非权威 EXR；
- capture manifest 必须声明 `raw_authoritative=true` 和 `denoised_preview_authoritative=false`。

## Path tracer 与材质族

场景中的每个 Falcor material slot 保存独立 `ReferenceSource` 与 GPU 资源。path 命中后按 material ID 分派：

- LayerStack：顶层/多层界面、均匀 slab、体散射与吸收的原生随机游走；
- MERL：测量 BRDF 表求值与对应的采样/PDF；
- OpenPBR：resolved native inputs、颜色空间转换、采样/PDF 和 LUT；
- MaterialX：当前正式 `standard_surface` subset、纹理连接、normal map、采样/PDF；纹理梯度由 path ray cone 与三角形 UV Jacobian 得到。

编译时按场景中实际出现的 family mask 特化 shader，避免单一材质场景为未使用 family 支付全部分支成本。该优化不能改变各 family 的 reference 方程或把它们统一改写成某个 backend ABI。

## Pass 划分

```text
Falcor Scene
  ├─ SceneReferencePathTracer
  │    ├─ raw running mean + luminance moment
  │    └─ display-only a-trous denoiser
  └─ Raster PrimaryVisibility
       └─ ApproximationPrepare → DeferredLighting

raw reference + optional denoised display + deferred approximation
  → LinearComparisonComposite（定量输入固定为 raw）
  → SharedToneMapper
  → UI
```

## 相机、交互与重置

viewer 只有一个相机状态。orbit、pan、dolly、滚轮缩放、键盘移动和 UI 数值编辑都修改同一状态。移动时立刻显示低 spp 结果，停止后继续累计。UI 常驻显示 raw spp、累计时间、估计 mean relative standard error、reference GPU 时间、场景/环境标识和积分深度。

点击场景物体选择其 Falcor material slot；修改只作用于选中 slot。UI 按源材质族显示原生可编辑参数。MERL 通过更换测量文件修改，不能伪造并不存在的自由参数。

## MethodBundle

viewer 只列出通过 manifest、全文件 hash、平台、散射合同与 GPU parity 的实时 bundle。切换方法允许重建右侧 pipeline 和 backend-specific `ScatteringState`；不得把某个 backend 的 packet 布局提升为公共接口。

逐 tile direct fit 只对离散 `(材质, 观察方向)` 有效，不属于自由相机 MethodBundle。

## Capture 与 replay

capture v3 至少记录：

- scene、HDRI、源材质、相机、分辨率与 SHA-256；
- reference integrator ID、raw spp、scene/layer 深度上限、噪声估计；
- raw/denoised 的权威性和 bias 语义；
- MethodBundle、GPU timing 与 comparison 语义；
- raw reference、denoised preview、approximation、comparison、display 和 metrics 文件角色。

固定 studio-v1 是单材质场景，可以完整 replay。多 slot 场景会记录所有当前绑定，但在稳定的逐 slot 源材质序列化合同落地前必须标记 `scene_material_bindings_replayable=false`。

## 自动 benchmark

`configs/viewer-benchmark-v1.json` 与 `scripts/benchmark_viewer.ps1` 必须复用 studio-v1 几何、HDRI、默认材质及其 hash，并固定三段相机路径、分辨率、reference 上限和帧数。输出写入 `artifacts/`，包含 preset/viewer/source/Falcor/scene/environment/material provenance，不把单次运行结果提交到根仓库。

最终性能与质量门槛应依据首个正式 backend 的实测 Pareto 曲线制定；不能沿用旧 K2 状态布局作为预算，也不能用去噪预览掩盖 raw estimator 的收敛问题。
