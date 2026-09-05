# 技术设计

## 目标结构与边界

本任务完成 viewer 内一条通用材质执行路径：同一 slot 合同接收 source reference 或 neural package，同一 PT renderer 与同一 deferred renderer 按需调用既有四入口。最新 hybrid 是交付实例，代码结构不以 Metal 命名或识别模型 key。source 原生资源与数学留在 source adapter，neural 的 private state 留在 package module。

不拆子任务：标题依赖最终 slot 状态，deferred 与 PT 共用 scene binding/context，旧入口清理依赖新 producer/loader 完成；这些交付需要共同的 Release 和 capture 验收。主会话 inline 实现和检查，不派 agent。

## 1. 通用 viewer binding 与统一行为

继续使用 `INclsScatteringBackend/INclsScatteringState`、`ScatteringPackage@2` 和项目既有 source identity，不新增平行 scattering ABI，不把 reference 伪装成 neural package。

Host 将 slot 的已提交 binding、capabilities、mode、status、resources、timing、accumulation、display metadata 作为唯一运行状态。source/package reader 负责把原生输入解析成可绑定对象；slot lifecycle 只做 capability check、candidate resource/compile、atomic commit、reset 和 dispatch，不按左右位置或 method key 改行为。

- PT 要求 PREPARE/EVALUATE/SAMPLE/PDF 全部存在；deferred 要求 PREPARE/EVALUATE。capability 检查统一，缺失时报告 unsupported，不隐式切 renderer。
- source 与 neural 都可位于任一 slot，也都能使用所支持的 PT/deferred。默认 source PT 对新 hybrid PT；后续用户切换模式或 preset 不被强制恢复为 deferred。
- 两模式共用 source/package 资源绑定、sampler descriptor、active material ID 与 scene composer。已有可编辑 source 和 neural 各按真实 capability 提供编辑；linked edit 仍是双侧 candidate 成功后一起 commit，不能伪造新预算模型的 typed-edit。
- cache identity 至少包含 program/module、scene specialization、source binding set 和合法模式依赖；scene/material 变化不继续复用错误 specialization。failure 保留旧已提交画面与 metadata，另行显示失败请求。
- runtime 不依赖 PyTorch、Python 或 MDL SDK runtime DLL；generic 接口通过编译期 specialization 使用，避免热路径 existential dispatch。

## 2. 最新 hybrid 的完整部署

沿用 `src/ncls/learning/models/metal_budgeted_sampler.py` 与 `metal_budgeted_evaluator.py` 的现有数学：两路 specular 加 full-hemisphere proposal，GGX/Beckmann/uniform 分布 ID、CDF 分量选择/remap、折回上半球和双 preimage PDF、reverse PDF、fallback floor 均保持不变。full-hemisphere 是模型自身的显式分量，renderer 不在错误后换 sampler。

在 `metal_budgeted_asset.slang` 补 proposal adapter、prior、lobe clue、normalized weights/alpha/distribution；backend-private sampler module 实现 sample/pdf，wrapper 使用公共 frame/event/weight 合同。sample weight 来自同 prepared evaluator 的 `f * abs(cos) / p`，pdf 使用同一完整 mixture。

既有 packed state 已预留 12 个 proposal half，program blob 已包含参数，保持现有 160 B layout、固定两个 asset fetch 和 evaluator 数学。完整 prepare 的 MAC/延迟需要包括新增 proposal 计算并重新报告，不沿用 evaluator-only prepare 的计数。权重不变，不训练；若发现必须改变模型结构或训练语义，回 planning。

数值合同通过后声明 SAMPLE/PDF，更新 package witness、validator 和 handoff。生成新 program/package identity 与独立 output root，原 artifacts 不覆写。研究 diagnostic readiness 与 runtime PT capability 分开，GPU 正确不等于 formal 质量或泛化结论。

## 3. 共用 scene surface 与材质选择

以 `PathSurface`、`PathSurfaceMath` 与既有 source adapter 为基础建立公共 scene context。PT 从 ray hit 获取 surface，deferred 从 raster G-buffer 获取 surface；二者进入同一 context 适配后才调用材质 prepare。

G-buffer 的背景 sentinel 与 material ID 在读取边界解码一次；保留 geometric normal、shading normal、front-facing 和正确 tangent/frame。位置、wo/wi 坐标、V flip、normalized UV derivatives、filterRandom 均按统一合同传递。PT ray cone 与 raster ddx/ddy 允许有合理的采样 footprint 差别；在 matching witness 中提供完全相同输入，不能把不同 footprint 的图像直接当 parity。

scene composer 只在选中 material ID 替换为 neural，primary、secondary 和 deferred surface 都遵守此规则。其余对象使用相同 reference bindings。同步移除 `allMaterialsSupportedBy` 的“所有材质必须匹配同一 package”检查，改为只验证被替换的 binding identity。host 改动与 shader routing 同时完成，不能只放开拒绝而让 neural 覆盖全场景。

source-native 的颜色空间适配、surfaceInteraction、emission 和 native sample tuple 留在 backend/composer。共享的光照 response adapter 明确线性 f、cosine、照明乘法及色彩变换的边界；不重复乘 cosine、不把 tone mapping 烘进材质输出，也不删原生 OpenPBR 等色彩语义。

## 4. PT renderer

将现有 source/package PT 的重复 traversal/direct/environment MIS/continuation/RR 合并为一个公共实现；入口只实例化不同 scene binding，host 从同一 renderPath 调度。renderer 不识别 source family、method key 或 backend 私有字段。

同一 shadow ray 使用同一 origin offset、tmax 和 scene query。两侧共享 camera、geometry、lighting、bounce、environment CDF、sample identity 与 accumulation 规则；backend proposal 可不同，不能因此要求有限 spp 路径逐条相同。

同 opaque 几何的同一射线 visibility 应一致；不同 BSDF 可以改变反射、间接光、颜色串扰和最终阴影亮度。同 BSDF 控制用于检验 integrator，不拿 neural 拟合误差当 renderer 失败。

## 5. Deferred renderer 的材质正确性

保留一个通用 deferred renderer，同时支持 source 与 neural。单个 surface 只 prepare 一次，再以自己的方向 query budget 复用 state；query budget 是工作量，不能改变材质定义。资源纹理/latent 的 dtype、filter、address、mip、UV scale、颜色解码都来自正式 descriptor。

检查路径为：G-buffer decode → scene binding → canonical prepare → evaluate → response/cosine → 局部灯光 → linear HDR → display tone mapping。source/neural 和 PT/deferred 共用材质适配，不用各自补偿系数修图。失败/invalid response 按同一明确合同处理，不把 NaN/clamp 混成可用输出。

deferred 保留现有局部 direct/environment quadrature 的用途；本任务不增加 GI、secondary traversal、scene shadow system、复杂 MIS 或通过增加环境样本量逼近 PT。未遮挡的局部灯光/相同查询才用于材质 renderer parity；完整 PT 的阴影、GI 与有限 sample 差异单独解释。新增 reference deferred 使两侧可以直接比较相同局部照明下的材质。

## 6. 整帧调度与左右标题

PT 和 deferred 都单次 dispatch 完整 panel，删除 host tile 循环、每 tile `submit(true)`、跨 frame stride 精化及进度 UI。交互 PT 每次追加 1 spp，headless 按 remaining 截断；deferred deterministic 单次完成，状态未变可复用。GPU timing 覆盖整 pass。

保留固定 50/50 并排图像。在每个 panel 顶部显示不可独立拖走的轻量标题条，例如：

```text
Reference · MDL · PT · 128 spp        Neural · Metal hybrid · PT · 128 spp
Reference · MDL · Deferred           Neural · Metal hybrid · Deferred
```

标题的类型、方法和实际模式来自同一已提交 slot metadata；loading/unsupported/error 显示相应状态，失败保留旧画面时同时标明当前有效 binding 与新请求失败。标题不依左右角色，交换 slot 后自动交换内容。方法名过长可截断并用 tooltip 展示完整名，PT 显示真实 spp，deferred 不伪造 spp。

采用 UI overlay，不改变 render extent、viewport aspect 或线性 capture，不创建第二个 OS 窗口；现有设置面板调整默认位置避免挡住常驻标题。EXR/difference 保持纯图像，capture metadata 的类型/mode 与标题使用同一来源。UI 字体/渲染方式沿用 viewer/Falcor 能力，不引入新 UI 框架。

## 7. 历史实现和兼容层清理

用户已要求清理 viewer 相关兼容层，本次允许明确移除历史输入支持；现行 producer 必须先迁移，新输出闭环通过后删除旧 reader。既有用户 artifacts 不删除或原地改写，也不新增长期兼容 shim。

| 现状 | 收敛方式 |
|---|---|
| capture v3、method_id、`--method`、旧左右角色分支 | 只保留当前 slots[2] capture/replay 与 per-slot CLI；更新活跃 producer/consumer，旧输入明确拒绝 |
| viewer-scene v1 与旧 batch 字段处理 | 当前 scene schema 为唯一 reader/writer，headless batch 只归 capture |
| source-only MDL catalog 与绑定旧 step 20000 的 linked catalog 两套结构 | 以现有 ViewerMaterialCatalog 演进通用当前 catalog：entry 提供 source locator/artifact、identity、参数/capabilities 和可选 method bindings；source-only 是合法 entry，不再是 legacy reader。去除强制一份 catalog 必须对应全 692/单一旧 checkpoint/可编辑 neural 的假设，数量由实际 registry/cohort 声明。版本布局由实现冻结；本轮新 handoff 与当前 source 准备工具同步输出该格式 |
| `exact-diagnostic-evaluator-preview` 字符串直接驱动模式 | 正式 capability 决定能做什么，readiness 只记录研究状态；不能一见 diagnostic 就锁 deferred |
| source/package 两套 PT、独立输出/模式特例、重复材质适配 | 共用 slot lifecycle、scene binding 和每模式一个 renderer；backend-private native adapter保留 |
| progressive tile、quality-first/TDR 热路径、自动关 environment/bounce | 删除；诊断照明使用显式 scene/capture 配置，不改变默认启动行为 |
| full-width reference、approximation/旧 backend/K2 用户文案和 dead UI | 以当前 slot/method/mode 的中文说明和真实状态替换，删除无调用死代码 |
| native source adapter、ScatteringPackage@2、DDS 容器 header 和 resource/capability validation | 它们是现行语义/格式，不因包含 legacy/compatibility 字样就删除 |

清理范围覆盖 apps/viewer、必要的 src/ncls/viewer/schema/producer、tools/viewer、tools/reference 的 viewer 准备入口、scripts、相关 tests 与稳定文档；不扩张到训练模型历史目录和上游源码。

## 8. 验证与风险

先验证 sampler 独立 oracle，再验证通用 slot/state 和 matching surface/direction 的材质调用，接着检查相同 BSDF 的 PT visibility/transport 与 deferred 局部光，最后对新 hybrid 做真实 D3D12 package、PT/deferred/capture/UI 观测。当前其他 source 只做受影响 adapter 的必要轻量回归，不跑旧 full 或全候选性能矩阵。

正式新 hybrid PT capture 按既有 1024 spp 合同执行一次；其他数值控制采用小尺寸/有界样本。默认 1600×900 composite 用于整帧更新和 UI 检查，GPU median/p90、frame wall、首个有效更新延迟只作 report-only。慢则 profile 新模型热点，不恢复扫块或扩大训练。

标题截图只验证可见模式与几何布局；linear EXR 验证图像。新包允许正式 PT 调用不意味着模型空间细节已学好。历史输入移除会使旧 capture/catalog 需要由当前 producer 重新生成，这是用户清理范围内的兼容性变化，须在中文文档列明。
