# Viewer PT 空间材质根因修复设计

## 1. 边界

修复位于 viewer 的 scene surface interaction 构造和验收层。source reference、ScatteringPackage 与 method shader 继续消费现有 scattering contract；训练、checkpoint、MaterialX source adapter 和 NVIDIA runtime 数学保持不变，除非 probe 证明这些层也存在独立缺陷。

```text
Falcor scene hit / raster fragment
        │
        ├─ PT: shared hit decoder + ray footprint
        │          │
        │          └─ NclsSurfaceInteraction
        │                 ├─ source reference prepare/eval/sample
        │                 └─ package prepare/eval/sample/pdf
        │
        └─ raster: SceneVisibility UV/gradient
                   └─ NclsSurfaceInteraction
                          └─ package deferred prepare/eval
```

## 2. 根因取证设计

先增加只面向验证的 surface probe。probe 使用固定 camera、shaderball 和一个明确的 UV/LOD oracle，输出或 readback：

- material/instance/primitive identity；
- `uv`；
- PT footprint 或等价 texture-independent LOD；
- raster `ddx/ddy` 对应的 LOD；
- 选定纹理/latent mip 与采样结果。

probe 必须比较相同像素/相同 primary surface，避免用不同 jitter 或不同 transport 结果猜测。诊断资产放任务 scratch 或根仓库 versioned viewer fixture；只有长期回归所需的最小 fixture 进入根仓库。

## 3. 共享 PT surface helper

把 `ReferencePathTracer.cs.slang` 与 `PackagePathTracer.cs.slang` 中重复的 triangle hit 解码、basis 清理、front-facing、UV 翻转和 footprint 构造抽到 viewer-owned Slang helper。helper 输出纯 scene/surface 数据；reference/package 各自只负责 material prepare 与 transport。

设计约束：

- triangle mesh 走 Falcor 的权威 vertex fetch，并保留计算 footprint 所需的三角形 UV/world Jacobian；
- curve/SDF/displaced geometry 在缺少可信 UV differential 时必须有显式、保守且可诊断的 fallback，不能伪装成 triangle 精度；
- tangent fallback、normal orientation 与 `NclsScatteringContext` 约定只实现一次；
- OBJ 的 V flip 继续由 scene format policy 控制，不改变 GLB UV；
- PT footprint 的单位必须明确为 normalized UV derivative 或 texture-independent LOD，调用方不能重复乘入纹理尺寸。

具体计算选型由 probe 结果决定：若当前 ray-cone propagation 正确而 vertex UV 错，则修 vertex path；若 UV 正确而 LOD 单位/传播错误，则对齐 Falcor `RayCone::computeLOD()` / texture sampler 约定；若两者都错则同时修复。不得用经验 clamp 掩盖单位错误。

## 4. deferred 对齐

`SceneVisibility.3d.slang` 保留 raster `uv/ddx/ddy`。新增自动化比较把它作为 primary-hit screen-space oracle，而不是让 PT 直接依赖 G-buffer；真实 PT 仍可处理 secondary hit。共同 helper/测试负责字段约定，避免 transport 互相别名。

## 5. 回归 fixture 与指标

长期 fixture 采用小而高对比、可解析的 UV 图案和多 mip 内容，至少能独立识别：

- U/V 方向、wrap 与 V flip；
- 非常量 UV；
- 选择 mip 0 与粗 mip 的区别；
- normal/frame 方向（可用独立 probe，不把 normal map 颜色当 base color）。

测试分三层：

1. shader/GPU surface probe：直接比较字段和 LOD；
2. viewer headless local-light capture：隔离 transport，验证 reference/package 空间外观；
3. 真实 walnut/denim 收敛 evidence：用于视觉审阅，不用天然图像统计替代 fixture hard gate。

capture 增加的空间指标必须基于 raw linear output 或 probe buffer；tone mapping PNG 只做人类证据。

## 6. 兼容、回滚与风险

- 不改变 package manifest/ABI，优先只重编 viewer shader；如需新增 capture diagnostics，schema 以向后兼容的可选字段扩展并更新版本/测试。
- shared helper 改动会同时影响四种 source reference 与所有 package PT，必须跑 LayerStack、MERL、OpenPBR、MaterialX 和 NVIDIA 组合回归。
- 若修复暴露既有 capture 基线错误，以新 probe/oracle 为真相，旧 artifact 保留历史身份但不继续作为通过证据。
- 回滚点是旧 viewer commit；不修改或重写 200k checkpoint/package，若 package 无需重导出则保持其 identity 不变。
