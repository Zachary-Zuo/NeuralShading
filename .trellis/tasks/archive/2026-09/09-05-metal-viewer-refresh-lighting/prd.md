# Viewer 通用材质接口、模式标题与 PT/Deferred 一致性

## 目标与用户价值

用户希望 viewer 通过通用接口运行 source reference 与任意 neural package，并清楚显示两侧实际类型和渲染模式；最新 Metal 在同一场景光照下做双侧 PT 整帧对照，deferred 则以正确的局部材质语义提供另一种观察方式。2026-09-05 用户同意创建任务并规划，随后确认右侧默认也是 PT，并要求通用架构、左右标题、deferred 材质检查以及清理相关过时实现和兼容层。用户随后明确要求“开始实现”，按本范围进入实施。

## 实施前调查快照

以下文件与行号对应规划时的原实现；最终变更与验证记录见 research 和 artifacts。

- F1：旧 linked 入口强制左 PT、右 deferred，见 `apps/viewer/NclsViewer.cpp:1827`；新 Metal launcher 也明确设置右侧 deferred，见 `scripts/launch_metal_viewer.ps1:61`，并通过 `--evaluator-preview-lighting` 关闭环境、点光、矩形灯及场景 bounce，见 `apps/viewer/NclsViewer.cpp:703`。
- F2：交互 deferred 每 frame 只执行一个 8×8 logical tile，并按 stride 16→8→4→2→1 精化，见 `apps/viewer/NclsViewer.cpp:2301`、`:2331`。package PT 则整帧遍历 8×8 tile 且逐 tile `submit(true)`，见 `:2353`、`:2374`、`:2440`。旧任务 `.trellis/tasks/09-03-metal-authored-preset-viewer/implement.md` 记录这是为旧大模型交互卡死引入的调度。
- F3：deferred 没有 scene ray query，直接累加 sun/point/rectangle/environment，见 `apps/viewer/shaders/DeferredRenderer.cs.slang:98`；reference PT 有几何遮挡，见 `apps/viewer/shaders/ReferencePathTracer.cs.slang:128`。当前画面不能作为同 transport 的材质质量证据。
- F4：最新入选可部署候选为 `metal_budgeted_hybrid_v3`，Tungsten 单材质 step 2048，checkpoint SHA-256 为 `8a15a5945085bddc781c1e60cd434ffa78b3a791ceed05dbbe007f8e7fb8971e`。本机已有 `artifacts/viewer/metal-budgeted-ddp5-wrap-1d5f813-step2048/handoff.json`，记录 hybrid package `61447f0bb57979413cba4b72c3bf974cc18ae212b5a4bf22d519e5efcdc3d820`，只支持 prepare/evaluate/anisotropic-frame。它不是旧 step 20000 的 692 材质模型，也不是未被选中的后续 diagnostic profile。选择证据见 `.trellis/tasks/09-04-metal-neural-budgeted-redesign/research/single-material-selection.md:33`。
- F5：Python sampler 已有三分量 proposal、PDF、方向与 weight；Slang sample 返回 false、pdf 返回零，见 `shaders/ncls/backends/metal_budgeted/metal_budgeted.slang:37`。Slang prepare 未填 proposal state；现有 runtime blob 已携带 proposal adapter 权重和 compiler prior，见 `src/ncls/learning/metal_budgeted_runtime.py:23`、`:86`，因此可补部署实现，无需训练新模型。
- F6：reference PT 按 hit material ID 选择 binding；package PT 对所有 hit 使用同一 compiled material index，见 `apps/viewer/shaders/ReferencePathTracer.cs.slang:349`、`apps/viewer/shaders/PackagePathTracer.cs.slang:295`、`:326`、`:381`。进一步核对发现 host `allMaterialsSupportedBy()` 要求所有 scene source 都匹配同一 package（`apps/viewer/NclsViewer.cpp:1319`、`:2030`）；当前通常会拒绝不同地面材质的比较，并不是已经证明上次地面被替换。缺失的是只替换选中 material ID 的完整合同，不能只删 host 拒绝而保留 shader 的全场景替换。
- F7：已有 `INclsScatteringBackend/State` 定义四入口，见 `shaders/ncls/contracts/scattering_backend.slang:8`；viewer 仍按 source/package 分开创建 pass、检查模式、渲染和选输出。source deferred 被硬编码 unsupported，见 `apps/viewer/NclsViewer.cpp:2009`；`ComparisonSlot::bind()` 的 PT mask 只检查 SAMPLE/PDF，见 `apps/viewer/ComparisonSlot.cpp:13`。
- F8：deferred raster 输出 material ID 加一的 sentinel 编码，但 evaluator 原样传入，见 `apps/viewer/shaders/SceneVisibility.3d.slang:77`、`apps/viewer/shaders/DeferredRenderer.cs.slang:93`；G-buffer 未保留 geometric normal/front-facing，`PackageBackend.slang` 的 context 用 shading normal 代替 geometric normal 并固定 frontFacing=1。UV/frame/response contract 必须一并检查，不能只改调度。
- F9：历史入口包括 capture v3 reader、viewer-scene v1 reader、`--method`/method_id 单侧选择和旧 full-width/approximation 文案，见 `apps/viewer/NclsViewer.cpp:440`、`:520`、`:780`、`:3594`。MDL catalog reader 的 `legacy` 分支仍由新 Metal handoff 和 source-only 准备工具生产（`tools/viewer/prepare_metal_catalog.py:193`、`tools/reference/prepare_mdl_viewer.py:91`），需迁移生产者再删除，不能按名称直接破坏现行 source reference。

## 需求

- R1【对应 F4/F5】：以最新入选 hybrid checkpoint 为对象，完整部署已有 prepare/evaluate/sample/pdf，打开真实 PT capability。支持 PT 不等同于正式研究质量 ready；保持 checkpoint、source、profile 与 package identity 可追溯。
- R2【对应 F1/F2】：默认 reference 与新 Metal 均为 PT，每次交互 dispatch 整帧追加 1 spp，状态变化同时 reset；取消新模型路径的跨帧扫块及逐 tile CPU/GPU 同步。deferred 若由用户显式选择，也整帧计算，不再承担默认比较角色。
- R3【对应 F3/F6】：PT 两侧使用同一场景相交、几何可见性、光源、环境采样/MIS、bounce、相机与 surface 合同；只替换选中材质，其他 scene bindings 不变。不用关环境、关 bounce 或无阴影 deferred 代替完整 PT。
- R4：遵守 canonical scattering ABI、source 原生语义、固定 50/50 对称 slot、失败隔离与已有 edit 原子事务。不为新单材质 package 伪造 typed-edit 或 692 材质覆盖。
- R5：只测新 hybrid 和完成正确性所需的小型解析控制，不运行旧大模型、不重训、不启动 direct/旧 full 性能矩阵。
- R6【对应 F7；来源：用户新增通用架构要求】：基于既有四入口实现通用 viewer scene binding/adapter，source 与 neural 使用相同 capability、模式选择、资源生命周期、reset、timing、capture 和错误处理。PT 与 deferred 只按渲染模式组织算法；添加 neural method 不增加 viewer 枚举、method key 分支或专用 renderer。source 也能使用同一 deferred renderer。统一接口不抹平 backend 原生数学、资源或合法 capability 差别。
- R7【来源：用户要求左右显示类型】：固定并排视图的每侧顶部常驻标题，显示当前实际提交的类型（Reference/Neural）、方法名和 PT/Deferred；切换、加载失败和 unsupported 状态不显示虚假的模式或方法。左右可交换，标题不依赖左右角色。保留相同 viewport/aspect，不改成独立 OS 窗口。
- R8【对应 F8；来源：用户要求 deferred 材质相关无错误】：检查并修复 deferred 的 selected-material binding、material ID、position/wo/wi、geometric/shading frame、front-facing、UV/V-flip、normalized derivatives、texture/latent sampler/LOD、prepare 复用、线性 f 与 cosine、source 色彩适配和 tone mapping 边界；以相同局部光和匹配 surface query 对照 reference，隔离 renderer 错误与模型误差。不新增 GI、场景阴影系统或复杂环境积分来追求与 PT 全图一致。
- R9【对应 F1/F2/F9；来源：用户要求清理过时、错误及兼容层】：在 viewer、其 source/package/handoff 准备入口、capture/replay schema、测试与文档范围内清理历史格式读分支、单侧旧 CLI、重复 runtime/renderer、强制模式/lighting patch、旧 sweep/preview 状态和错误文案。现行生产者同步到统一入口后删除旧路径；当前公共 ScatteringPackage 与 native source/resource adapter 不属于历史兼容层。历史输入明确拒绝并提示重新生成，不新增长期双轨 reader；不删除用户既有 artifacts 或扩张到无关训练历史代码。

## 验收标准

- [x] AC1【需求交付｜来源：用户问题 1/2；映射 R1/R2/R3】：说明实际 launcher、model identity、更新调度和光照差异；区分代码确认与真实运行证据。
- [x] AC2【需求交付与接口正确性｜来源：用户追问右侧也应为 PT、shared-slang-backend 合同；映射 R1/R4】：最新 hybrid package 实现四入口；GPU sample→pdf、reverse PDF、weight 恒等式与 Python oracle parity 成立；capability 与真实实现一致。
- [x] AC3【需求交付｜来源：用户要求像 reference 一样更新；映射 R2】：默认左右 PT；相机、灯光、有效材质切换后两侧整幅更新，无逐 tile 扫屏；交互固定 1 spp/dispatch 并持续累积。记录默认 1600×900 composite 下的单侧 800×900 分辨率、GPU median/p90、frame wall 与首次有效更新延迟，这些数值为 report-only，不编造帧率 hard gate。
- [x] AC4【理论/语义正确性｜来源：只改变材质的比较要求、viewer 合同；映射 R3/R4】：同 opaque 几何、同 origin/direction 的 shadow visibility 一致；多材质场景中只替换选中 material ID。相同 BSDF 控制验证两侧 direct 与多 bounce estimator 的一致性。
- [x] AC5【实现正确性｜来源：项目 capture/runtime 合同；映射 R1–R4/R6/R8/R9】：受影响 unit/GPU、Windows Release 和新 hybrid matched PT capture 通过；最终 EXR finite、spp/mode/source/checkpoint/package/binding identity 可追溯；Falcor overlay 后 worktree 干净。parity 容差在运行正式 witness 前按 dtype/oracle/calibration 冻结，不依结果事后放宽。
- [x] AC6【架构与接口正确性｜来源：用户通用包装要求；映射 R6】：source/neural × PT/deferred 组合使用统一 slot lifecycle 和各模式共享 renderer，任一侧可绑定任一种受支持对象。PT 检查完整四入口，deferred 检查 prepare/evaluate；缺 capability 不回退、不伪装。小型测试 backend 无需增加 viewer method 分支即可走相同接口。
- [x] AC7【需求交付｜来源：用户要求清楚观察类型；映射 R7】：无需打开设置即可看到两侧 Reference/Neural、方法名、PT/Deferred；标题随已提交 binding/mode 原子变化，失败能区分旧有效画面与新请求。窗口缩放后仍与各 panel 对齐，linear EXR/difference 和相机 aspect 不受标题影响。
- [x] AC8【数值/语义正确性｜来源：用户 deferred 材质检查要求；映射 R8】：GPU witness 验证 ID sentinel 解码、frame/front-facing、UV/derivative、资源 sampler/LOD、线性 f/cosine/颜色；同一 evaluator 在匹配 surface/direction 下的 deferred 与 PT 调用结果一致。source/neural deferred 局部材质图像对照明确排除 GI/阴影/采样算法差异，不用完整 PT 图像误差衡量 deferred 正确性。
- [x] AC9【需求交付与接口正确性｜来源：用户清理要求；映射 R9】：形成基于实际调用者的删除/迁移清单并逐项完成；新 producer→loader→viewer→capture/replay 闭环可用，旧格式/旧 CLI 有明确拒绝行为，活跃代码和文档不再保留旧调度、重复模式路径或虚假状态；不为历史模型增加兼容测试和运行。

## 关键边界

- 几何遮挡一致不要求两个不同 BSDF 的最终图像逐像素一致。间接反射、颜色串扰、亮度以及有限 spp 噪声均可能不同；阴影与 radiance 分开验证。
- 不扩展训练、结构搜索、新模型质量或 692 材质泛化；不以 clamp、generic sampler fallback、单 bounce 或关闭环境掩盖缺陷。
- 不复测旧 full，不为其保留默认交互的性能防御调度；用户本次清理要求覆盖 viewer 相关历史 reader/CLI 的移除，旧文件不保证直接回放，但保留原文件并可从当前 source/checkpoint 重新生成新输入。
- 已有源材质可编辑性继续保留；本次不新增预算模型尚未部署的 typed-edit。
- 环境为完整 Windows：RTX 4090、neural-shading Conda 环境、Windows Falcor Python 构建均存在。实施按该环境运行；具体验证记录保存在当前任务 research 与 artifacts。

