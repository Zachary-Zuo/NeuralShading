# 03 Neural Baseline 与候选选择实施计划

## 0. Planning Gate（2026-08-26 需求修订）

- [x] `01/02`均已提交归档，并读取公共数学、mollification corpus/entry、spec和最终验证证据。
- [x] 使用`trellis-session-insight`回查父任务历史并定位反复原因：旧实验观察值被提升成 Q1 硬门，原方法又被拆成原规模 diagnostic 与 `≤2k MAC` 缩模 baseline，导致训练结果无论如何都不能同时证明“忠实复现”和“过门”。
- [x] repository evidence确认SlangPy已在`environment.yml`锁定为0.43.1但当前环境未安装；这属于启动后环境同步，不引入Torch fallback。
- [x] 用户明确要求在原任务内修订，不创建新任务：复现成功只取决于方法正确实现与训练稳定收敛，不限制收敛后的绝对质量；原规模方法必须实现并沿通用 viewer 路径显示，缩模不再是本任务前置。

Final planning summary：先审计并实现原规模 NVIDIA learned-frame evaluator + GGX9 sampler，再实现 exact-core positive residual/LTC 候选。每个方法分别提交 implementation correspondence、convergence 和 parity/correctness 证据；最终 quality/cost 只做按材质结构分组的 Pareto 描述，不作复现 kill gate。数据只从`47ef…5a89` entry读取，`.875`回base-v5；03交付原规模 checkpoint/compiled assets，04/05 必须原样送入 MethodBundle/viewer。

## 1. 环境与双编译 Spike

- [x] 在`neural-shading`环境安装/同步锁定的`slangpy==0.43.1`并确认可导入；记录Agility SDK warning，不修改Falcor/Slang锁定提交。
- [ ] 更新并运行minimal MLP + learned-frame/core autodiff spike；验证Torch interop、gradient finite-difference `rtol≤1e-3`和吞吐。
- [ ] 冻结SlangPy/Falcor共同接受的语法、反射layout与weight read路径；为双编译加入常驻测试。

## 2. 方法对应关系审计

- [ ] 以 NVIDIA 一手论文/补充材料为权威，建立逐项 method-correspondence：latent/frame、方向 convention、decoder 输入、层宽/层数、激活、输出、loss/mollification、sampler 9 参数与 detach 边界。
- [ ] 只读审计 `D:\01_Workspace\Real-Time Neural Appearance Models`；记录可借鉴的结构与已发现的 GT/loss/sampler 风险，不复制其 Torch 前向，不把它当 correctness oracle。
- [ ] 将现有 `17→32→32→3`、view-conditioned frame、`z16`、`softplus`/output floor 与额外 appearance loss 逐项归类：删除错误 baseline 语义，或改成独立 adaptation ID；禁止继续沿用 NVIDIA baseline ID。
- [ ] 冻结原规模正式 config 与 smoke config。smoke 只缩训练时长，不缩模型；原规模结构、source schedule/optimizer 与所有适配项进入 config/checkpoint hash。

## 3. 单一 Slang Method Core

- [ ] 实现 NVIDIA 原规模 `z8 → 无bias的8→12非正交two frames`、`20→64→64→64→3 + exp(raw-3)` cosine-weighted response core和`11→32→32→32→9` sampler；`prepare(wo)`只重排原算术并缓存`wo`变换，frame不得依赖`wo`。
- [ ] 实现 response core → 公共 `evaluate(f)` 的显式 grazing adapter，并锁定 `evaluate()*wi.z == response`；不能改训练 target 后继续称无适配复现。
- [ ] 实现`core-frame-neural-v1` positive residual evaluator；复用01 common math/top interface，禁止prediction clamp/lobe-only。候选拥有自己的私有latent/state/网络身份，不用填充字段伪造与baseline同成本。
- [ ] 实现NVIDIA GGX9和LTC-K2 neural head decode；sample/pdf只调用01公共proposal，保留epsilon safety与显式null语义。
- [ ] 增Falcor kernels、SlangPy probes和独立数值oracle，覆盖frame边界、grazing、FP16 packing、finite/三态。

## 4. Data、Pipeline 与训练生命周期

- [ ] 新增entry-backed combined training store；严格校验base/supplement/entry ID，train curriculum与validation/test role分离，无目录fallback。
- [ ] 扩展pipeline/runner以传递training progress、target source和多阶段optimizer；旧pipeline行为/hash兼容测试必须通过。
- [ ] 新增thin parameter module与SlangPy autograd session；Torch只拥有loss/optimizer/oracle。
- [ ] 实现原 baseline joint evaluator/sampler lifecycle、response log-L1/mollification、对 latent detach 的 sampler KL；另实现用于2×2的冻结 evaluator matched sampler stage。candidate loss 与额外 linear/energy/peak 只放各自身份或 adaptation ablation。
- [ ] 新增 convergence report：逐step finite证据、初始化→checkpoint的paired validation改善、late-window divergence分析和checkpoint恢复复算；主 seed 即可出正式结果，额外 seed 只作异常诊断或补充，不得读取test。
- [ ] 新增原规模 direct、core candidate、两种sampler的smoke/formal configs与strict schema；移除以缩模网络冒充baseline的正式入口。

## 5. 数学与框架验证

- [ ] 运行unit：method correspondence、config/schema/layout/data routing/loss/detach/convergence/checkpoint identity/comparison/tamper。
- [ ] 运行SlangPy/Falcor双编译：evaluate/pdf/sample parity、gradient、FP16 packing和成本记账；成本不作复现通过线。
- [ ] 对两种sampler运行冻结的normalization/null、sample→pdf、histogram和同evaluator MC无偏协议。
- [ ] base v5与mollification corpus validator、reader`.874/.875`边界、六个upstream clean和repository policy通过。

## 6. 正式训练与 Matched 2×2

- [ ] 先运行所有配置smoke，确认run/checkpoint可恢复、curriculum source计数和test只读门。
- [ ] 训练原规模 NVIDIA evaluator；完成主 seed convergence report 后再运行一次 test，记录完整quality/static/RTX4090 query cost。已完成的第二个 baseline seed只作补充证据；低quality不触发改配置重跑。
- [ ] 训练exact-core positive-residual evaluator；用相同数据角色与预算类别生成独立convergence report，不要求与baseline具有相同宽度/latent bytes。
- [ ] 对两个best evaluator各训练GGX9/LTC-K2 head；生成四格sampler correctness、convergence与variance报告。
- [ ] 运行test/adversarial/dense、signed energy/core coverage、paired bootstrap≥1000、leave-one-out和offline-cook held-out-query workflow。
- [ ] 生成复现/比较 manifest、packed compiled-material set与Slang/Falcor parity evidence；分别写 implementation/convergence/comparison 状态，并保留按材质结构分组的非支配结果，不机械回退到A。
- [ ] 将既有 `formal-direct-v1/v2`、当前 core 诊断及其配置标为历史诊断：它们可用于定位输出 floor、autograd callable 与 loss 问题，但不能证明原方法复现。

## 7. MethodBundle/viewer 交接、Quality 与收尾

- [ ] 用中文更新`experiment_log.md`及相关候选/架构/学习文档，登记真实run/config/data/implementation/convergence/comparison identities，不把低质量写成复现失败。
- [x] 生成原规模 baseline 与 candidate 的 checkpoint/layout/compiled-material，并直接通过通用 MethodBundle 路径进入 viewer；加载的是同一份 FP16 packed asset 与 state table，没有替换为缩模。
- [ ] 使用`trellis-check`审计baseline忠实性、single-Slang、data flow、训练稳定性、2×2 matched、分组统计、成本记账与no fallback，并自修全部finding。
- [ ] 使用`trellis-update-spec`固化形成的learned evaluator/sampler、SlangPy和selection长期合同。
- [ ] 运行完整unit/GPU/SlangPy/reference gate、`git diff --check`、tracked artifact/HDF5审计与upstream cleanliness。
- [ ] 记录dirty归属和逻辑提交计划，创建scoped local commits；排除`SmileySans-Oblique.otf`，不amend、不push。
- [ ] 使用`trellis-finish-work`归档并记录journal，确认archive/commit provenance后才进入`04`。

## 8. 主要 rollback points

- SlangPy/Falcor兼容问题只允许修单一源/包装；不得自动启用Torch生产前向。
- 正式test只在配置与best checkpoint冻结后读取一次；训练设计变更需新config/run identity，不覆盖旧证据。
- implementation correspondence失败时回到方法语义，不靠增加训练步数掩盖；convergence失败时回到梯度/optimizer/data生命周期，不靠挑幸运seed掩盖。
- 实现正确且稳定收敛但quality较低时结束复现循环，保留结果并进入结构归因；不得再设置新的绝对quality门继续反复。
- sampler数学正确性失败不得靠放宽归一化/parity容差或私有fallback解决；这些是正确性门，不是材质质量门。
- checkpoint、compiled assets和报告只写ignored`artifacts/`；失败run保留manifest，不进入Git。

## 9. Viewer 交接实录（2026-08-26）

- viewer 已移除 `film_m1` backend/layout/state 硬编码。`Prepare / Approximation / Parity` 只读取 bundle 的 shader specialization、共享权重和 `CompiledMaterial` table，并经 `INclsScatteringBackend` 调用。
- core-frame candidate MethodBundle：`5d2169642fadf47ae56999a4191102e8905e0c0c768754c1bf7c0890129fe55d`，`runtime_class=realtime`，完整 matched GGX9 `sample/pdf`，位于 `artifacts/exports/unified-scattering-03-viewer/core-frame-native/`。
- NVIDIA learned-frame 原规模网络的离线预算适配 MethodBundle：`572d6b877eb5a293f9d6798ce44e83208334c9a8893998af483fefed047aeb1d`，`runtime_class=diagnostic`，保留 z8/frame/evaluator 与 native GGX9，位于 `artifacts/exports/unified-scattering-03-viewer/nvidia-paper-native/`。该目录名和 pipeline ID 是当时产物身份，不代表已经完成论文 online training 复现。
- 两个 bundle 的 load-time GPU parity 均通过；candidate headless capture 为 `artifacts/captures/unified-scattering-03-core-frame-viewer-smoke.json`，记录 `approximation_available=true`。NVIDIA 诊断产物的跨编译器 parity 容差由实测 envelope 得到，报告在 `artifacts/reports/unified-scattering-03/nvidia-viewer-parity-probe.json`；该容差只验证 SlangPy/Falcor/packed asset 一致，不验证 source-material quality 或论文训练 lifecycle。
- 可见 viewer 以共同 bundle 根目录启动，默认选择 candidate；UI 下拉可切换原规模 baseline。

## 10. 状态纠正（2026-08-27）

- 现有 NVIDIA run 是 25k-step、batch size 16 的离线 LayerStack 训练；论文是 GPU online reference、300k iterations、每次两个 65k batch。现有证据不得再写成“论文 baseline 已忠实复现”。
- `nvidia-original-convergence.json` 只证明数值有限、相对初始化改善且后期未发散；best checkpoint 在最后一步，late normalized slope `-0.168` 且 CI 全负，不能宣称训练已饱和。
- `nvidia-original-test-quality.json` 的 directional L1 state median/p95 为 `0.680/1.205`，能量误差 median/p95 为 `0.408/4.372`。viewer 底色差首先是 checkpoint quality 问题；load-time parity 反而证明 viewer 忠实执行了这个错误较大的模型。
- NVIDIA viewer smoke 的 `reference_scene_max_bounces=0`，右侧仍是 deferred。它不是 `Reference PT | Method PT`，也不能用来判断 method PT 是否正确。
