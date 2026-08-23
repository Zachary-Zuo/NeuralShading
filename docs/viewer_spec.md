# Windows 材质查看器规格

## 实现状态

`apps/viewer/` 已实现本规格的第一版闭环，并在锁定 Falcor 8.0/D3D12 上用真实 P1 realtime `MethodBundle` 验证：bundle 全文件哈希、平台/合同检查、GPU parity、左右像素对齐捕获、逐字节 replay 和三段固定相机 benchmark 均已通过。构建、运行与捕获命令见 `apps/viewer/README.md`。

当前 reference 计算的是共享主可见性表面上的局部多层材质直接光照响应，不包含物体间投影、全场景间接光或穿物体传输；这与第一阶段“先不做投射物体”的范围一致，不应把它描述成完整场景 path tracer。

## 定位

viewer 是仅支持 Windows/D3D12 的原生交互应用，用来观察材质在真实几何、HDRI 和解析灯光下的效果。它不是训练 UI，也不是数据集 tile 浏览器。

实现依托锁定的 Falcor 8.0，但源码位于根仓库 `apps/viewer/`。构建通过根仓库维护的 CMake overlay 或等价方式接入 Falcor，不在 `external/Falcor` 留下未说明修改。

## 画面合同

- 左侧：随机游走参考解驱动的累积 path tracing；
- 右侧：当前选中 MethodBundle 的 deferred/实时结果；
- 两侧使用同一场景、相机、材质程序、几何法线/切线、灯光、环境旋转、曝光和 tone mapping；
- 两个 renderer 生成像素对齐的线性 HDR 图像，再由 composite pass 按垂直分割线选取，最后统一 tone map；
- 默认左/右各占 50%，分割线可拖动；
- 支持显示线性绝对差、相对差和放大误差图。

不能分别对左右自动曝光。任何仅影响显示的操作都不能改变其中一侧的物理输入。

## 相机和交互

viewer 只有一个相机状态。orbit、pan、dolly、滚轮缩放、键盘移动和 UI 数值编辑都修改这一个状态，因此左右天然同步，不做两个相机之间的追随复制。

第一版提供：

- 球体；
- shader ball；
- 一个带曲率、掠射和细节尺度的 hero mesh；
- 后续增加 glTF/USD 导入，不阻塞首个闭环。

## 参考累积

参考侧每帧至少追加一个可配置 sample batch。发生以下物理状态变化时清空累积：

- 相机或投影；
- 场景几何或实例变换；
- MaterialProgram 或其资源；
- 灯光、HDRI 或环境旋转；
- reference 积分器设置。

移动过程中显示当前低样本结果；停止输入后继续累积。UI 显示 spp、累计时间和估计噪声。

以下变化不清空 reference：

- 切换右侧 MethodBundle；
- 移动比较分割线；
- 切换差异显示；
- 修改共同曝光或 tone mapping。

## RenderGraph

当前 pass 划分：

```text
PrimaryVisibility / shared scene state
  ├─ ReferencePathTracer → ReferenceAccumulation
  └─ ApproximationPrepare → DeferredLighting

ReferenceAccumulation + DeferredLighting
  → LinearComparisonComposite
  → SharedToneMapper
  → UI
```

如果 Falcor 的路径追踪入口无法直接复用主可见性，允许 reference 路径独立追踪 primary ray，但必须使用同一相机和场景 revision。几何交点差异需有专项像素对齐测试。

## MethodBundle 切换

viewer 扫描用户指定的 bundle 目录。只有通过 manifest、hash、平台、散射合同和 parity probe 的方法才进入下拉列表。

切换方法允许重建右侧 shader pipeline 和状态 buffer。交互要求不是“任意 backend 共用同一二进制 packet”，而是“所有 backend 在同一 renderer 合同下可可靠切换”。

逐样本直接拟合结果不属于自由相机 MethodBundle，因为它只对离散 `(材质, 观察方向)` 有效。它可以在单独的锁定 tile 诊断工具中显示，不进入主 viewer 方法列表。

## MaterialProgram 编辑

第一版 UI 支持：

- 打开和保存 MaterialProgram JSON；
- 增加、删除、重排 LayerStack 内的界面和介质；
- 编辑 v1 常量参数；
- 显示节点版本、单位、范围和不支持原因；
- 在修改后同时重新编译 reference 与右侧方法；
- 显示当前规范化 IR 哈希和 MethodBundle 支持状态。

viewer 编辑的是 MaterialProgram，不直接编辑某个 backend 的 packet。backend 状态只允许在开发调试窗口中只读查看。

## 灯光

第一阶段至少提供：

- HDRI 环境和方位旋转；
- 方向光；
- 点光；
- 矩形面光。

右侧优先使用 backend 声明的专用积分能力；没有专用能力时使用标准 `evaluate/sample/pdf` 路径并明确显示 fallback。参考侧始终使用无偏路径采样。

## UI 信息

常驻或可展开显示：

- reference spp、噪声估计和 GPU 时间；
- 方法 ID、backend 版本和 checkpoint/run 来源；
- compiled material bytes、state bytes/pixel；
- prepare、lighting 和总 GPU 时间；
- 当前能力及 fallback；
- 场景、MaterialProgram、HDRI 和相机 preset ID。

## 比较和导出

viewer 支持保存：

- 左、右线性 EXR；
- tone-mapped PNG；
- difference EXR/PNG；
- 完整 capture manifest；
- GPU profiler 摘要。

capture manifest 必须足以通过命令行重放同一场景。截图不能只把参数写在文件名里。

## 自动 benchmark

交互 viewer 和无窗口 benchmark 复用 render graph 与 MethodBundle loader。`configs/viewer-benchmark-v1.json` 固定三段相机路径、hero 物体、分辨率、灯光、reference 设置和预热帧；`scripts/benchmark_viewer.ps1` 输出带 preset/viewer/source/Falcor 哈希的 JSON、CSV、日志和被 Git 忽略的图像。

最终验收门槛在首个新 backend 和 viewer 跑通后依据实测 Pareto 曲线确定，不能沿用 K2 的 176-byte 布局作为预算。
