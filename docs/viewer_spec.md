# Windows 材质查看器规格

## 实现状态

`apps/viewer/` 已实现第一版可运行骨架：bundle 全文件哈希、平台/合同检查、GPU parity、共享可见性、reference 累积、deferred prepare/lighting 和共同 composite 已接通。单次验证结果不在根仓库持久化；构建、运行与当前边界见 `apps/viewer/README.md`。

当前接入 viewer 的 reference 包括 LayerStack 随机游走、MERL 测量表、OpenPBR resolved native inputs 和 MaterialX `standard_surface` subset。它们计算共享主可见性表面上的局部材质光照响应，不包含物体间投影、全场景间接光或穿物体传输，不应描述成完整场景 path tracer。viewer 统一场景输入、输出图像和比较语义，不统一各源材质族的内部表示。

## 定位

viewer 是仅支持 Windows/D3D12 的原生交互应用，用来观察材质在真实几何、HDRI 和解析灯光下的效果。它不是训练 UI，也不是数据集 tile 浏览器。

实现依托锁定的 Falcor 8.0，但源码位于根仓库 `apps/viewer/`。构建通过根仓库维护的 CMake overlay 或等价方式接入 Falcor，不在 `external/Falcor` 留下未说明修改。

## 画面合同

- 左侧：当前源材质族的权威 reference 输出；LayerStack 使用随机游走累积；
- 右侧：当前显式选中 MethodBundle 的 deferred/实时结果；方法未指定或材质族不兼容时不创建伪结果，reference 改为全宽显示；
- 两侧使用同一场景、相机、材质程序、几何法线/切线、灯光、环境旋转、曝光和 tone mapping；
- 两个 renderer 生成像素对齐的线性 HDR 图像，再由 composite pass 按垂直分割线选取，最后统一 tone map；
- 默认左/右各占 50%，分割线可拖动；
- 支持显示线性绝对差、相对差和放大误差图。

不能分别对左右自动曝光。任何仅影响显示的操作都不能改变其中一侧的物理输入。

## 相机和交互

viewer 只有一个相机状态。orbit、pan、dolly、滚轮缩放、键盘移动和 UI 数值编辑都修改这一个状态，因此左右天然同步，不做两个相机之间的追随复制。

没有加载场景时提供球体、shader ball 和解析 detail hero fallback。加载场景时统一使用 Falcor `Scene` 和当前 importer 插件声明的格式，不再限制为单个 OBJ；主可见性同时保存 instance/material ID，单击物体选择对应材质槽。

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
PrimaryVisibility / Falcor Scene G-buffer
  ├─ FamilyReferenceIntegrator → ReferenceAccumulation
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

## 源材质与对象材质槽编辑

UI 按当前源材质族动态切换参数。第一版支持：

- 打开和保存 MaterialProgram JSON；
- 增加、删除、重排 LayerStack 内的界面和介质；
- 编辑 OpenPBR resolved native inputs；
- 编辑 MaterialX reference subset 中未由纹理连接占用的 literal 输入；
- 通过切换原始测量文件更换 MERL；
- 显示源材质族、原始文件、哈希和 MethodBundle 支持状态；
- 点击场景物体后，只修改该 Falcor material slot；各 slot 独立保存源材质与 GPU reference 资源。

viewer 编辑的是各源材质族的原生参数或其正式输入合同，不直接编辑某个 backend 的 packet。backend 状态只允许在开发调试窗口中只读查看。

## 灯光

第一阶段至少提供：

- HDRI 环境和方位旋转；
- 方向光；
- 点光；
- 矩形面光。

右侧当前使用固定成本的 deferred prepare/lighting：解析灯直接求值，环境光使用固定的确定性方向集合；没有 spp、跨帧样本累计或随帧变化的噪声。reference 侧直接使用该源材质族的权威求值和积分路径；随机游走 reference 使用随机采样，解析或测量材质 reference 不需要伪装成随机游走。场景阴影和全局间接光尚未接入，不能把当前两侧描述为完整场景传输。

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

capture manifest 必须记录场景、环境、相机、方法和 reference 设置。当前实现只序列化选中的源材质槽；在形成稳定的多 slot 源材质序列化合同前，不能宣称交互式多材质 capture 可以逐 slot 完整重放。截图不能只把参数写在文件名里。

## 自动 benchmark

交互 viewer 和无窗口 benchmark 复用 render graph 与 MethodBundle loader。`configs/viewer-benchmark-v1.json` 固定三段相机路径、hero 物体、分辨率、灯光、reference 设置和预热帧；`scripts/benchmark_viewer.ps1` 输出带 preset/viewer/source/Falcor 哈希的 JSON、CSV、日志和被 Git 忽略的图像。

最终验收门槛在首个新 backend 和 viewer 跑通后依据实测 Pareto 曲线确定，不能沿用 K2 的 176-byte 布局作为预算。
