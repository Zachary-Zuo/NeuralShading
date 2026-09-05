# 实施计划

## 当前阶段与执行方式

- [x] 用户同意创建任务，并确认右侧默认 PT。
- [x] 用户新增左右标题、通用 viewer 接口和行为、deferred 材质正确性与历史兼容层清理范围。
- [x] 调查 canonical ABI、slot mode/capability、G-buffer、source/package dispatch、producer/reader 调用者；更新 PRD 和 design。
- [x] 完成更新后的 PRD 收敛检查：需求/事实/验收映射完整，无阻塞的产品决策；本轮最终回复呈现更新摘要。
- [x] 用户明确要求“开始实现”，已 `task.py start`。
- [x] 实施前使用 `trellis-before-dev` 加载 project/core/viewer/data/learning 涉及层规则；记录 dirty 文件和环境。主会话 inline 实现与检查，不派 agent。

各阶段共用同一最终接口和场景合同，按以下顺序实施；不把孤立标题 UI 或 Metal sampler 作为全任务完成。

## 1. 冻结输入与通用合同（AC1/AC6/AC9）

- 核验 PRD 指定 hybrid checkpoint 的 SHA-256、profile、source snapshot 与现有 asset；不运行旧 full/direct，不重训。
- 将 design §7 的清理表逐项补全实际 producer/reader/CLI/test/spec 调用者及最终替代路径，清单保存在 task research；不按 legacy 单词盲删 native adapter。
- 定义已提交 slot binding/metadata、mode capability mask、scene binding table、render state/reset/capture 合同。
- 冻结统一 catalog 的必要 schema 变更：支持 source-only 和可选 neural bindings，保留 source-native identity/editability；更新当前 producer 与 read/write 测试，禁止伪造 692-entry 或新模型 typed-edit。
- 将实际 shape/offset、precision、sampler/frame/PDF/weight 容差和独立 oracle/calibration 依据登记在 task research，必须早于最终 witness。

Rollback A：接口设计暴露 source 原生语义无法保留时修正 adapter；不把不同源材质改写成统一解析 closure，不通过新增模型专用 viewer 分支绕过。

## 2. 最新 hybrid 四入口部署（AC2）

- 补 Slang proposal preparation、mixture sampler、完整 forward/reverse PDF 与 canonical wrapper。
- 覆盖三个分量、CDF remap、半球折回两个 preimage、grazing/invalid、frame、FP16 权重和 pack/unpack；sample weight 与同 evaluator/完整 PDF 一致。
- 更新完整 prepare cost、capability、package validation/witness 和通用 GPU tool，继续使用原 checkpoint 与 source assets。
- 通过数值正确性后才生成支持 PT 的新 package identity；readiness 不再通过 evaluator-preview 文案决定渲染模式。

Rollback B：parity 不通过则修部署，不改权重、不增加训练、不启用虚假 capability；不引入 renderer generic sampler fallback。

## 3. 公共材质 context 与每模式共享 renderer（AC4/AC6/AC8）

- 统一 host scene/source/package resource binding 和 candidate→commit，移除 source deferred unsupported 特判；PT capability 完整检查四入口。
- 建立统一 scene composer：仅替换 active material ID，其他对象保持 source；同时修改 host identity guard 和 primary/secondary/deferred shader routing。
- 补 G-buffer material sentinel decode、geometric normal/front-facing 与共同 frame 整理；UV flip/normalized derivatives/filterRandom/texture sampler 等只在定义的边界处理。
- 收敛 source/package PT 为单一公共 transport；原生 emission/色彩适配/native sample tuple 留在 adapter，验证相同 BSDF 的 direct/多 bounce/environment。
- deferred source/neural 共用 renderer 和 response adapter，保留一次 prepare 多次 evaluate；检查 f/cosine、颜色和 tone mapping 边界。
- GPU matching witness 比较相同 surface/direction 下 PT/deferred 材质调用，而非要求不同 raster/ray-cone footprint 的全图逐像素相等。
- deferred 只做局部材质正确性；不新增 GI/阴影系统、路径延续或为逼近 PT 增加环境预算。

Rollback C：source-native 数学或当前可编辑行为被共享实现损坏时修正合同，不以删除能力、clamp 或关灯通过验证。

## 4. 整帧更新、左右标题与入口收敛（AC3/AC7/AC9）

- PT/deferred 均全 panel dispatch；删除 tile/stride 状态、逐 tile submit/wait 和 UI 进度，GPU timing 覆盖整 pass。
- 交互 PT 恒为 1 spp/dispatch，变化同步 reset 且首样本保留；headless remaining 截断；deferred 保持 0 spp 和完成后缓存。
- 两侧顶部常驻标题从已提交 slot metadata 显示 Reference/Neural、方法名、PT/Deferred 和真实状态；模式/方法切换、交换侧、unsupported/编译失败、resize 同步。
- 标题采用 overlay，不改 panel extent/aspect，不进入 linear EXR/difference；设置窗口默认布局不遮住标题，不新增 OS 窗口或 UI 框架。
- 新 launcher 默认 reference PT 对 hybrid PT，取消自动 lighting override；支持当前 schema 的每 slot 指定与 UI 模式选择。
- 当前 catalog/handoff/scene/capture producer 全部迁移后，删除旧 reader、capture v3、scene v1、--method/旧 method_id 路径、重复 render/output 分支和过期文案；旧输入明确拒绝并说明重建入口，原 artifacts 不删除。
- 使用描述符和 capability 驱动新包，不按 checkpoint 旧阶段字符串锁定 renderer；保留当前 ScatteringPackage 与 native source adapter 的必要验证。

## 5. 检查与交付（AC2–AC9）

本会话已判定完整 Windows（RTX 4090、neural-shading 和 Falcor Windows 构建存在）。新会话实施时重新探针。所有 Python 使用 neural-shading，Falcor 测试经统一 launcher。

先执行受影响精确集合；新增 sampler/material/slot GPU tests 随实现加入命令，失败修复后只重跑相关集合。不得把全旧模型性能矩阵变成通用架构验收。

```powershell
conda run -n neural-shading python -m pytest tests/unit/test_metal_budgeted_runtime.py tests/unit/test_metal_budgeted_method.py tests/unit/test_metal_budgeted_profile.py tests/unit/test_metal_budgeted_package_gpu_tool.py tests/unit/test_viewer_slots.py tests/unit/test_viewer_studio.py tests/unit/test_viewer_material_catalog.py tests/unit/test_scattering_package.py -q
.\scripts\run_falcor_python.ps1 -m pytest tests/gpu/test_metal_budgeted_runtime_package.py tests/gpu/test_viewer_path_surface.py tests/gpu/test_viewer_path_sample_generator.py tests/gpu/test_viewer_path_environment.py -q
.\scripts\build_viewer.ps1 -Configuration Release
git -C external/Falcor status --short
git diff --check
```

- [x] 数学：新 hybrid sample/pdf/reverse/weight、PDF 归一化、quantized oracle、边界 witness。
- [x] 通用性：使用轻量测试 backend 和当前 source adapter 验证 source/neural × PT/deferred、完整 capability 拒绝、slot 交换、失败回滚；不增加 viewer method key 分支。
- [x] 材质输入：两个以上 material ID、背景 sentinel、非平坦 normal/tangent、正反面、V flip、normalized derivative、纹理 address/LOD；同材质同 query 的 PT/deferred parity。
- [x] PT：同 BSDF 控制的遮挡与多 bounce/environment；不同地面 binding 保留，最新 hybrid 双 PT 正式 capture 1024 spp 一次。
- [x] Deferred：source/neural 使用相同局部灯光和 surface 检查 finite 与 material contract；解释模型误差，不以 GI/阴影差异判定材质错误。
- [x] UI：默认 1600×900 两个标题清晰；切 PT/deferred、换侧、resize、加载失败标题一致；无 tile 扫屏。记录 GPU median/p90、frame wall、首个有效更新延迟和 spp，数值 report-only。
- [x] 输入输出：当前 producer→catalog/package→viewer→capture/replay 闭环；schema/identity/hash/mode 一致，旧 reader/CLI 确实移除而非藏在别名后。
- [x] 使用 `trellis-check` 检查受影响架构/规范，必要时读取 `trellis-update-spec` 同步 viewer/core/data/learning 的当前合同与迁移说明；不 push。

使用现有公共 exporter 生成同一 hybrid 的新四入口 package，不强制运行/重新构建 direct。产物写 `artifacts/viewer/metal-viewer-refresh-lighting/`；临时诊断脚本只写本 task `scratch/`。

Headless 使用实际生成的当前 source catalog 与 package ID：

```text
NclsViewer --material <current-catalog> --bundle-root <new-packages> --slot0-package source-reference --slot0-mode path-tracing --slot1-package <new-hybrid-id> --slot1-mode path-tracing --headless --capture <new-output> --reference-spp 1024
```

实施时以 exporter 输出替换占位符，完整命令和日志进入 artifacts，不使用旧包默认扫描根。标题可见性通过 UI 截图观察，EXR/difference 验证不受 overlay 污染。

## 当前收尾工作

代码与需求验收已完成：96 项相关测试、真实多材质 PT/Deferred/换侧控制、UI 切换/缩放/失败保留、最终 hybrid 双 PT 1024 spp、deferred 与逐位一致 replay。证据见 artifacts 下 current/validation.md 和 task research/closure.md。用户已批准提交并归档；本地实现提交 `19073e07c5398630000fc9536fbc46cf95e08e48`，按 trellis-finish-work 归档当前任务并记录 journal，不 push。
