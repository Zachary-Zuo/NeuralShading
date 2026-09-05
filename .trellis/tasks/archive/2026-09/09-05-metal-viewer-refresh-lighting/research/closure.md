# 接口收敛与验收证据

本任务保持已选 step2048 hybrid checkpoint 原始字节，部署已有四入口；未训练、未执行旧 full 或旧 direct checkpoint。单次图像、性能与数值结果集中在 `artifacts/viewer/metal-viewer-refresh-lighting/current/`，最终说明为该目录的 `validation.md`。

## 实际删除与迁移

| 旧实现或调用者 | 当前唯一入口与验证 |
|---|---|
| `ReferencePathTracer` / `PackagePathTracer` | 删除两个文件；`PathTracer` 调用 `SceneScattering`，小型同 BSDF 三材质场景测试 PT/Deferred 与左右交换 |
| package 覆盖全部 hit、host 全材质 identity 拒绝 | composer 只替换 active ID；其他材质保持原 source；host 只验证 active binding |
| deferred tile/stride 精化、PT 逐 tile submit/wait | 每个 ready slot 整 panel dispatch；交互 PT 固定 1 spp，deferred 缓存一次完整结果 |
| 新 launcher 右 deferred、自动 lighting override | 默认左右 PT，模式参数显式；场景原有 environment/bounce/lighting 保留 |
| Slang sample=false / pdf=0、proposal state 空缺 | 原 checkpoint 的 adapter/prior 参与 prepare；匹配三分量 sampler、完整 forward/reverse PDF、同 evaluator weight；GPU oracle 检查 |
| raster ID sentinel 原样使用、缺 geometric normal/front-facing | `SceneSurface` 统一 context；GPU witness 覆盖背景、多 ID、正反面、normal/tangent、V flip 与 normalized derivative |
| source deferred Unsupported、source/package 不同局部查询预算 | 共用 deferred renderer，1 environment + 1 rectangle query；无新增 GI 或 shadow traversal |
| catalog legacy/source-only/linked 双轨与固定 692-entry 假设 | `ViewerMaterialCatalog@2`，source entry 可带完整可选 package binding/typed view；MDL 与 Metal producer 同步迁移 |
| capture v3、viewer scene v1、`--method`、lighting preview CLI | 只接受 capture v4、scene v2、per-slot 参数；旧输入提示重建；原 artifacts 不删除 |
| capture 顶层单 method/approximation 状态、左右角色 EXR 别名 | `slots[2]` 和 `slot_0_linear` / `slot_1_linear`；输出 identity 与模式来自已提交 slot |
| JSON `dump()` 被当作跨语言 canonical hash | 共享 C++ canonical serializer；4096 随机 double、边界、Unicode、整数与 Python 逐字节一致 |
| runtime 修改后被迫修改训练 identity 或重新训练 | 独立 deployment snapshot 保留训练身份并检查 tensor 名称/shape/dtype；严格 resume/evaluate 规则不变 |
| GUI 隐藏 title bar 中的状态文字 | 两行显式 panel 标题，显示 Reference/Neural、PT/Deferred、真实 spp/状态和 family/profile |
| deferred 计时一直为 0、短 capture 在目标后空转 | 沿用 Falcor 的 in-flight frame 同步边界异步回收计时；headless 默认最少 1 帧，仍要求达到实际 PT spp target |

## 定位到的部署故障

1. 截图中的 deferred link failure 来自 `nclsFrameToWorld` 未直接 include `scattering/frame.slang`。LayerStack 的传递 include 掩盖了依赖，纯 MDL specialization 暴露错误。两个实际 MDL/hybrid deferred pass 已运行通过。
2. source pass 原先在启动时无条件构建 PT；现在根据实际模式按需创建，再在 scene/source specialization 变化时原子替换已创建的 pass。
3. 初版标题只把状态放进 `Gui::Window` 名称，但未启用 ShowTitleBar；真实截图确认后改为显式 text，扩大高度以避免 DPI 下滚动条。
4. 同 BSDF 测试原先把换侧前的 source 与换侧后的 package 要求逐位一致，出现 FP32 舍入差。修正为每个 renderer 与自身换侧前后的图像逐位一致；source/package 原有严格容差 `rtol=2e-5, atol=2e-6` 未放宽。

## 验证范围与边界

- 受影响 unit 与真实 viewer integration 集合：89 passed；最后 host/identity 集合：22 passed。数学、surface 和 shader 测试见最终 artifacts 中的验证记录。
- UI 已观察默认双 PT、逐侧切 Deferred、交换、1200×700 client 缩放、恢复 1600×900、无效 source drop 后保留旧 binding；截图在 `current/ui/`。一次无明确错误日志的旧窗口退出未复现，单独进程完整操作后以 exit 0 正常关闭，不把未知原因写成已定位 bug。
- 真实同 BSDF 控制含三 material ID、底座和地面；两模式左右在冻结容差内一致，每个 renderer 换侧前后逐位相同；不以不同 BSDF 的最终阴影亮度相同作为物理要求。
- source、runtime package 的 source identity 与 edited state hash 是不同层面的追溯字段；最终结果同时保存 catalog、checkpoint、artifact、package/program/asset/instance 与 scene binding，不通过修改 hash 验证来加载包。
- 几何 visibility 共用 renderer；不同材质仍会改变间接光和采样噪声。deferred 只验证局部材质，不要求与 PT 全图相等。
- 新 hybrid PT 的首次 shader 编译和持续渲染成本分开报告。完整四入口可部署不代表模型细节已拟合充分，也不代表正式泛化质量通过。
- task scratch 已局部忽略，避免临时 checkpoint、诊断脚本和日志进入归档提交；用户其他未提交文件保持原样。
