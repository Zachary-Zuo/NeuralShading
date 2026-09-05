# 实施记录

用户已指定本阶段提交归档，并将剩余实验转交服务器 24 小时自主研究；最终范围见 [closure.md](closure.md)。下文 `in_progress` 是移交前的记录。

用户已在规划审阅后明确回复「开始实施」。自 HEAD `e3f1c21` 开始按 implement.md 的 P1–P4 实施；文档中「本轮仅规划」与「尚未执行」属于上一轮交付状态，本记录逐项覆盖实际进度，不扩大既定诊断预算。

环境重新确认是完整 Windows：RTX 4090、`neural-shading` Conda 环境、锁定 Falcor Windows Python 扩展存在。Linux/NCCL 仍未在本机验证。

最新实施检查点见 [implementation-validation.md](research/implementation-validation.md)：raw 多 UV adapter → 共享资源/训练 → 流式 cook → 新 Slang/MethodBundle → viewer 已接通。划痕青铜完成初始化、恢复至 step 2、数值 validation、实际导出和 reference 128 spp/神经 deferred capture。全量 unit 366 passed，随后 C3/visual phase 回归 11 passed，新 spatial runtime/reference GPU 通过，C3 故障注入复验通过；任务仍为 `in_progress`。

下面保留早期实施记录；其中「尚未接通」等阶段描述由最新检查点覆盖。未完成部分是三个实际 source 的完整 D0、真实 matched summary-control 与 D1 指标/成本实验。summary 占位 profile 已明确拒绝，不能把 raw CNN 的同名运行当对照；没有启动六个质量 run。

P1 首批实现：共享 conditioning resource/binding、行筛选与拼接所有权；producer 在 rejection/paired 失败和成功路径释放 lease；adapter 显式返回资源；query 支持传入 filter random。Metal 原始像素接入尚在进行，此批不代表 P1 完成。

已执行：`conda run -n neural-shading python -m pytest tests/unit/test_conditioning_resources.py tests/unit/test_training_batch.py tests/unit/test_online_training_producer.py tests/unit/test_online_data_session.py tests/unit/test_mdl_fixed_source_adapter.py -q`。结果 **34 passed，3.53 s**。

新增 raw 固定解码、五语义 stem/融合/trunk/learned mip 与 texel QAT 后 bilinear 读取。CPU RF planner 按每层真实 canonical 地址拆分 seam，因此奇数、非方、1×N 的 tile 与完整图保持同一 stride phase。测试 `test_metal_asset_read.py test_metal_spatial_encoder.py`：**15 passed，2.55 s**；`test_metal_spatial_inputs.py test_mdl_metal_assets.py`：**9 passed，1.96 s**。

审读源 MDL 发现旧 UV 合同错误：Bronze 的 `vmat_transform` 是倒数 scale/原生旋转，scratch 与 normal 另用原始 coordinate_source；所有三个 diagnostic source 有 nonrepeat 多位置采样；Steel 另含 `vm_coord_post_scale`。用户随即明确不同 UV 分开编码。已按 design §9 修订，继续原生多 UV 路线，不采用有限域预烘焙。

C1 初始 state+零 delta、C2 稳定 softplus、C3 evaluator 有效标志传播、C4 正 Z 连续 frame 已落到 Python/Slang；独立正确性与 GPU 验证进行中，尚未登记通过。

后续验证：C1/C4 新 witness 与既有模型回归 **17 passed，2.06 s**；Slang softplus 正常数尾部/连续 frame 及现有 FP16/SNORM runtime package GPU 测试 **2 passed，130.44 s**。GPU 首次运行中 NumPy 小矩阵乘法触发本机 OpenMP runtime 冲突；改为独立逐元素点积 oracle 后通过，未设置忽略重复 runtime 的环境开关。

多 UV 组件：`native_uv.py` 从源 helper 调用读取原生坐标，按表达式分组；原生 nonrepeat 三位置 hash/坐标与独立 unsigned scalar witness 对照。`spatial_bundle.py` 为每组、每个原生 lookup 规划 raw RF；`spatial_asset.py` 分组读取并把 9×14 条件送 decoder。`test_metal_native_uv.py` **4 passed，1.94 s**；`test_metal_spatial_asset.py` **2 passed，2.67 s**，包括主/pair 共用编码图、更新后重新编码与非空间 lookup 条件。

新 layout v2 和 profile 已建立，Python prepare 为 137→32→32→24、proposal 为 80→16→13。C5 独立反向 prepare witness 通过；含 C1/C4 的 `test_metal_model_correctness.py` **4 passed，1.93 s**。共享 MethodDescriptor 增加 resource requirements，engine 校验 binding；`test_method_plugin.py test_training_engine.py` **6 passed，3.37 s**。新 layout 的 Slang 生成/部署、adapter raw 默认路径、cook、配置和 end-to-end 仍在实施，不能把旧 package 测试记作新多 UV ABI 通过。
