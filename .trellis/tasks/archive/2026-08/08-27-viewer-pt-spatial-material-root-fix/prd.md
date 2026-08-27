# Viewer PT 空间材质根因修复

## 目标

从 surface interaction 数据合同出发，查明并根本修复 viewer 中 source reference PT 与 package PT 把空间材质呈现为近似常量外观的问题。修复后，reference PT、neural PT 与 deferred 必须在各自 transport 语义不变的前提下，读取同一命中点对应的 UV、frame 与 footprint，并能稳定呈现 MaterialX 原生纹理和 neural latent 的空间变化。

用户价值是让 viewer 真正成为部署与视觉对抗验收工具，而不是只证明 shader、package 和 slot 生命周期能够运行。

## 已确认事实

- 交互截图 `Snipaste_2026-08-27_15-13-01.png` 中，已经收敛的 `american_walnut_veneer` source reference PT 与 200k neural PT 均呈现平坦浅灰外观。
- 后台 512 spp capture `artifacts/nvidia-faithful/materialx-recorded-200k/viewer-reference-neural/diagnostic-512spp.json` 复现相同问题，因此不是低 spp 噪声。
- `american_walnut_veneer` 的 base color、roughness、normal 三张 4096×4096 纹理均由 viewer 日志成功加载；shaderball GLB 的两个 mesh 都有覆盖接近 `[0,1]²` 的 `TEXCOORD_0`。
- 用 `denim_fabric` source reference 做固定方向光诊断时，PT 只呈现纹理平均蓝色而没有原图中大尺度缝线，见 `artifacts/nvidia-faithful/materialx-recorded-200k/viewer-reference-neural/diagnostic-denim.json`。这说明纹理资源已被读取，但空间 lookup 退化。
- 同一 200k neural package 的 deferred capture 能产生明显空间变化，而 neural PT 与 source reference PT 都平坦。共同嫌疑位于 PT ray hit → `VertexData` → UV/footprint → texture/latent fetch 数据链，不在训练数据或 package 资源加载。
- 当前 reference PT 与 package PT 各自复制了 hit 加载、frame 整理和 isotropic ray-cone footprint 计算；deferred 另由 `SceneVisibility.3d.slang` 输出 raster UV/gradient。三条路径没有端到端一致性断言。
- 既有测试覆盖 latent fetch 算术、package parity、slot/capture schema 和 shader ready，但没有验证真实 scene hit 的 UV 空间变化、PT/deferred surface contract parity 或视觉纹理保真。
- 历史决策要求 viewer 做对抗性视觉检查，覆盖纹理细节、normal map、UV seam、footprint/zoom、alias 与 overblur；不得用去噪或单纯 ready 状态掩盖视觉差异。

## 需求

### R1：根因必须由数据证据闭环

- 在改动渲染结果前，建立可重复的 surface interaction 诊断，至少观测同一固定场景下 PT 与 raster/deferred 的 UV、有效 footprint/LOD 和 material ID。
- 区分并验证：命中 UV 是否变化、UV 是否与 raster 一致、footprint 是否有限且落在合理 mip、纹理/latent binding 是否采到预期资源。
- 最终任务记录必须说明直接根因、为何 source reference PT 与 neural PT 同时受影响、以及旧验收为何漏检。

### R2：共享且保持原生语义的 PT surface contract

- reference PT 与 package PT 的 triangle hit 解码、UV、geometric/shading frame、front-facing 和 footprint 构造必须复用一个 viewer-owned helper 或等价的单一实现，不再维护两份可漂移逻辑。
- helper 只能产生公共 `NclsSurfaceInteraction` 所需数据，不得把 MaterialX、NVIDIA 或其他 method/source family 私有状态提升为 viewer 公共接口。
- deferred 继续从 raster gradient 构造相同公共 context；需要显式验证 PT 与 deferred 的字段约定一致。
- 修复不能通过固定 UV、固定 mip、禁用过滤、替换原始纹理、提高对比度或 method/source family 特判实现。

### R3：正确的 footprint 与过滤行为

- primary hit 的 UV footprint/LOD 必须随分辨率、距离和表面投影合理变化，不能系统性退化到单一 texel 或最粗 mip。
- zoom 后应选择更细 footprint，远离或掠射时允许选择更粗 footprint；所有结果必须有限且静态有界。
- MaterialX reference texture 与 NVIDIA latent texture 继续遵守各自既有 sampler/filter 合同；修复不改变训练 recipe 或 package identity 的数学含义。

### R4：视觉与定量回归门

- 增加一个具有明显低频空间标记的 versioned viewer fixture 或诊断模式，使“UV/LOD 退化为常量”必然失败；不能只依赖天然低对比度 walnut 图像或人工肉眼判断。
- source reference PT 必须在固定 shaderball 场景中呈现 fixture 的空间标记，并对 `american_walnut_veneer`、`denim_fabric` 至少各生成一份收敛 capture 证据。
- neural package PT 必须呈现与 deferred 一致的空间布局；transport 差异只在匹配光照/局部求值条件下比较，不能把完整 PT 与有限 deferred 环境积分的差异误判为 evaluator 错误。
- capture 报告应持久化足以判定空间退化的定量指标，例如有效材质区域内的 UV 覆盖、LOD 分布、空间方差/与 oracle 的误差；门槛必须由 fixture 的明确期望定义。
- 交互 viewer 仍按时间持续累积；`spp` 只作为当前累计计数或 headless capture 复现预算，不成为交互停止条件。

### R5：架构和仓库边界

- 不修改 `external/Falcor`；如发现上游行为缺陷，只能在根仓库 viewer helper 中适配并以测试锁定。
- 不重训 200k 模型，除非根因证据证明训练或导出的 latent 本身错误；当前证据优先指向 viewer PT surface 数据链。
- 保持两个 `ComparisonSlot` 对称、50/50 extent、单侧失败隔离、source/package identity 与 capture v4 语义。
- 只提交本任务文件，不夹带用户现有 dirty 文件，不 push。

## 验收标准

- [ ] 有自动化 surface probe 证明固定 shaderball 可见材质区域的 PT UV 非常量，并与 raster/deferred UV 约定匹配。（类型：理论/语义正确性；来源：公共 scattering surface 合同与用户根因修复要求）
- [ ] 有自动化 footprint/LOD probe 覆盖分辨率或距离变化，结果有限、方向正确，且默认视角不会塌缩到最粗 mip。（类型：理论/数值实现正确性；来源：ray footprint 数学与现有 sampler 合同）
- [ ] reference PT 的高对比 fixture capture 通过空间 oracle；将 UV 固定或强制最粗 mip会使测试失败。（类型：需求交付；来源：用户要求根本修复与历史 viewer 对抗性检查决策）
- [ ] `american_walnut_veneer` 与 `denim_fabric` 的收敛 reference PT capture 能看到与原纹理对应的空间结构，而不是平均色。（类型：需求交付；来源：用户当前视觉问题；具体指标为 report-only，fixture hard gate 决定正确性）
- [ ] 200k NVIDIA package 的 PT 与 deferred 在匹配局部照明条件下保留相同空间布局，并通过由独立 fixture/oracle 冻结的数值容差。（类型：数值实现正确性；来源：同一 package math 与公共 context 合同）
- [ ] scoped unit/GPU/integration 测试、Release viewer build、headless captures、全量 pytest、`git diff --check` 通过。（类型：需求交付/实现正确性；来源：项目质量门）
- [ ] `external/Falcor` 和其他锁定上游保持干净。（类型：需求交付；来源：仓库政策）
- [ ] 根因、修复机制、旧验收缺口与新增防复发门记录到任务 research，并把长期合同更新到 viewer spec/稳定文档。（类型：需求交付；来源：用户根本检查要求与 Trellis 规范）
- [ ] 修复后的交互 viewer 已重新启动供用户查看，任务完成后归档并创建 scoped 本地 commit。（类型：需求交付；来源：用户持续执行与 viewer 查看要求）

## 不在范围内

- 改变 NVIDIA 200k checkpoint 的训练结论、重新选择模型或重新运行正式训练。
- 为了让效果更明显而修改 Poly Haven 原始资产、MaterialX 图或 tone mapping。
- UE 集成、通用 displacement/tessellation、环境积分新方法或其他候选研究。
- 将 source-family texture schema、NVIDIA latent layout 或 backend-specific state 暴露成公共 viewer ABI。

## 技术备注

- 当前最强假设是 PT hit 的 UV 或 footprint 与 raster path 不一致；实施阶段必须先用 probe 证实具体字段，不能直接以该假设代替根因证据。
- `reference_spp` 在 capture/replay 中保留为 headless 可复现预算；交互模式不读取它作为停止条件。
