# 实施计划

## 执行原则

- 当前是一个 complex task；本文件与 `prd.md`、`design.md` 获得用户确认后才运行 `task.py start`。
- 不拆 child task。各 deliverable 共用同一 method/profile/layout identity，按下述 rollback point 串行冻结；依赖显式写在每阶段中。
- 全部 Python/pytest/pip 使用 `neural-shading` Conda 环境；Falcor Python 使用统一 launcher。
- 首次测试前重新执行 `.trellis/spec/project/dev-environment.md` 六项探针并在回复中报告环境状态。
- 不修改 `external/`；训练/checkpoint/package/report 写入 `artifacts/`；临时诊断脚本只写当前 task `scratch/`。
- 不恢复旧 v4 long，也不自动启动 692-source formal long 或旧 compact sweep。

## 当前实施断点（2026-09-05）

- Windows 端已完成新 profile、Python 数学核心、训练合同与 CPU 单测；不在本机执行 online reference、pilot、完整 runtime baseline 或旧 long。
- direct/hybrid pilot已组成同source、query、loss、optimizer、schedule、precision与encoder-only asset输入的matched pair；Linux DDP5旧`@1`、高吞吐`@2` v1、full-semantic`@3` v2与batch profile均已执行。v2共同step512证明完整semantic直连仍未保留one-texel响应；当前实现`@4` v3的Detail→frame semantic短路径并fresh重跑。
- product registry 已因 public CLI 必须先解析新 pilot 而提前切换到 `metal_budgeted`；这只是依赖顺序调整，不代表结构已经入选。Slang/package facet 仍须等待 Linux pilot 按预登记规则选择 hybrid 或 direct 后实现。
- 旧 full 只保留历史实现与显式对照用途；当前阶段不通过兼容 shim 让旧 run 冒充 canonical `metal`。
- pre-Linux `trellis-check` 已完成：checkpoint diagnostic readiness 由各方法 descriptor 的强校验 policy 声明，通用训练层不再按 Metal method key 分支；Windows 纯 CPU `tests/unit` 为 335 passed，layout/compileall/diff/Falcor clean 均通过。旧 `mdl-metal-full` fragment 保持历史语义，新全 cohort 使用独立 `mdl-metal-budgeted-full` identity。Linux handoff 在本任务 scoped commit 完成后重新生成，以冻结可直接同步的 commit identity。
- 用户于 2026-09-05 授权在当前 Linux 主机 GPU 5–9 上进行约 12 小时持续监督，并要求范围内 DDP/实现修复、及时本地 commit、neural 设计结论及 hybrid/direct 两个 Windows 可检视模型。本轮新增 `ddp5` 执行身份、事件驱动监督和双 diagnostic package 交付；它不恢复旧 full，不扩大 20k/192 B hard bound，也不把 5 卡 step 数冒充原单卡样本预算。
- 用户随后把通用吞吐架构和模型/loss审查提升为继续长训前的最高优先级，并授权早期step信息不足时按预登记条件继续实验。旧`@1` pair在共同step 128停止为before-profile；公共优化与新recipe验证后，hybrid/direct从fresh identity重新开始。

## Phase 0：冻结现状与旧基线

依赖：用户已确认 NVIDIA-class hard budget。

- [x] 保存当前 task 影响范围内的 `git status`，区分用户既有 dirty files；不改写或回退无关内容。
- [x] 核对旧 20k checkpoint/package、Tungsten source locator、viewer capture和 NVIDIA faithful package identity。
- [x] 先在旧 plugin 切换前冻结旧 full 静态成本和可复跑 package benchmark 输入。
- [x] 为 single-material probe 固定 source、参数 state、UV/footprint、方向 quota、train/validation seed和viewer crop identity。
- [x] 在 `research/protocol-freeze.md` 登记 calibration、pilot cap、选择规则、容差来源和 observed metric 的 report-only 身份。

Rollback point R0：如果旧 artifact/source identity 无法闭合，只修复/重建 diagnostic identity；不开始新模型，也不把不匹配图像用于比较。

## Phase 1：观测、loss 与 matched runtime 基础设施

依赖：Phase 0 protocol freeze。

### 1.1 通用进度与 metric

- [x] 在 method objective 中采用 `loss/optimization_total`、`loss/appearance`、`loss/proposal`、`loss/proposal_weight` 等标准 key。
- [x] `TrainingEngine` 只按标准 key选择可选 postfix，不按 `metal` 分支；保留 `row["loss"]` 为实际反向总目标。
- [x] 为负 proposal continuous NLL、非有限 loss和 tqdm 文案添加 unit tests。
- [x] 实现/测试逐通道 log/linear、chroma、per-channel peak 与 spatial-gradient metric；scale/epsilon来自隔离 calibration。
- [x] 增加 paired UV query 所需的通用 batch/recipe字段；不能增加 Metal 专用 producer loop或离线 response batch。

### 1.2 Runtime harness

- [x] 在 task `scratch/benchmark_scattering_runtime.py` 组合公共 package/source ABI，生成 coherent/divergent workload。
- [x] 实现 prepare-only、evaluate-only、prepare+N、sample-only、pdf-only 的 GPU timestamp测量、warm-up、批次同步和原始结果写出。
- [x] 在 Windows 对 optimized MDL、NVIDIA faithful和旧 full 只运行 bounded correctness/preflight smoke，确认公共 ABI、state adapter和 profiler lane；不把小批次 dispatch floor 写成性能结论。
- [ ] 新 package 完成后在 Linux/headless 一次性运行四控制 matched baseline，记录 precision、batch、packet、state/weight/asset bytes和CPU lifecycle时间。
- [ ] 将 `prepare`、read、asset和latency结果登记为 report-only；不在看到结果后自动升级为 hard gate。

Windows 保护线：Windows 只允许 `count≤1024`、`warmup≤2`、`measurements≤3` 的 harness smoke。完整 `65,536×32×100` matched measurement只在新 package 完成后由 Linux/headless 执行，不是 Phase 2/3 的准入条件。

Rollback point R1：若 Linux/headless harness无法保证相同 workload/sync/precision，停止 runtime结论，只保留静态账本；先修通用测量口径。Windows smoke只判断接口和timestamp采集是否成立。

## Phase 2：新 profile、layout 与 Python 数学核心

依赖：runtime harness bounded smoke已闭合；不依赖 observed baseline 达到某个速度，完整 matched runtime 延后至 Phase 5。

- [x] 新增 `metal_budgeted_layout_v1.json` 与 profile loader/generator，精确计算 dense MAC、reads、ProgramState/PreparedState stride和identity。
- [x] 静态断言 hybrid/direct `evaluate ≤20,000 MAC`、PreparedState `≤192 B`；超额构造必须失败。
- [x] 实现 responsibility-aware typed compiler，确定性 access/frame/resource字段绕过 learned guess。
- [x] 实现两-read asset representation、variant table和 `24→32→32→24` semantic decoder；监督字段必须进入 PreparedState。
- [x] 实现 stable two-frame + half/difference方向特征。
- [x] 实现 v1 `28→64→64→64→6` hybrid evaluator、同 shape direct control和最多双 analytic lobe。
- [x] 根据v1 step512审计实现v2完整24维semantic输入：`44→64→64→64→6`；保持160 B state、两次读取和20k MAC hard bound，并增加runtime消费回归。
- [x] 根据v2 step512与fixed-batch诊断实现v3 Detail→frame semantic无参数短路径；保持其他matched轴并增加decoder置零合同测试。
- [x] 实现三 component analytic proposal及 sample/pdf/weight identity。
- [x] 为 finite/nonnegative、grazing、half退化、Beckmann例外、RGB gate、state packing和all-parameter gradient写 unit/GPU tests。

Rollback point R2：任何 Python 主形态需要越过 20k/192B 才能闭合接口时，不放宽 bound；记录为 design defect并回 planning。数学核心未过 unit/GPU前不接 package。

## Phase 3：训练路径、asset cook 与单材质 pilot

依赖：R2 Python 数学和静态预算通过；Phase 1 metric/query可用。

- [ ] 新增 `metal_budgeted` MethodPlugin 的 model/data/objective/lifecycle/checkpoint/deployment facets；public key保持 `metal`。
- [x] 方法配置固定 `joint-response-fit → deployment-qat-refine`；appearance从step 1启用，curriculum只登记真实 direction/LOD/peak变化。
- [x] asset cook实现 encoder-only、bounded refinement、direct control三种独立 identity，共用部署shape。
- [x] pure compiler与optimized ProgramState control分角色训练/报告；不把teacher结果写成editability。
- [x] 先在 GPU 5–9 完成 budgeted objective 的 DDP step-0/8、rank mapping、checkpoint/resume与teardown smoke；真实缺陷按公共 DDP 合同修复并补测试。
- [x] 以`@1`共同step-128证据定位training/validation热点；实现窗口级validation packed reduce与bounded lookahead，补齐单机/DDP等价性和异常清理测试。
- [x] 用同source/query/model做旧/新执行路径bounded profile；新matched recipe固定每16 step report、depth/reference batch steps=2，并在per-rank batch 64/128/256/512中选出512这一吞吐—显存Pareto点。
- [x] 在新recipe fresh前审查direct/hybrid输出责任、appearance分项、proposal权重/梯度与calibration量级；v1共同step512确认direct并非零梯度，同时定位求值器只消费前8维semantic state的架构缺陷。
- [ ] 以`0/8/128/256/512/1024/2048`共同里程碑交替训练direct/hybrid DDP5 pair，保存分项loss、独立validation、rank stage/profile和完整日志；满足设计§15.7时才成对延长到最多4096。
- [ ] 按预登记规则比较microdetail、RGB/chroma、peak、energy和成本；输出 `research/single-material-selection.md`。
- [ ] 仅在direct/hybrid共同失败且完成failure classification后，允许启动≤4×主profile neural MAC的teacher diagnostic；不自动追加step/seed。

Rollback point R3：

- eager失败且属于representation/protocol：停止在Python层修改设计，不进入Slang；
- eager成功而QAT失败：只处理quantization；
- hybrid无净收益：选择direct并冻结新profile identity，不为保留设计而扩大模型；
- observed quality较低但实现正确：登记empirical outcome并请求下一步，不循环改模型直到好看。

## Phase 4：Canonical plugin、量化、Slang 与 package

依赖：R3 已选择 hybrid 或 direct 主profile，profile/layout/训练checkpoint identity冻结。

- [x] registry product module从旧 `metal_fused` 切换到 `metal_budgeted`；更新 method fragment correspondence，新 checkpoint拒绝旧resume。
- [ ] 实现 runtime parameter分类、FP16 pack、RGBA8 latent pack、ProgramState/PreparedState pack和asset variant resources。
- [ ] 生成Slang layout并实现asset fetch、semantic decoder、compiler、evaluator、sample/pdf；矩阵与敏感FP32策略和Python一致。
- [ ] 完成 eager FP32 → quantized Python → Slang exact/random probe；偏差容差在运行正式test前冻结。
- [ ] 编译 `ScatteringPackage@2` program/asset/instance，验证typed edit和asset swap原子更新。
- [ ] 为 hybrid/direct 两个 exact checkpoint 分别生成 evaluator-only diagnostic package/catalog；非入选模型同样可供 Windows 视觉比较，但不能声明 formal 或未验证的 `sample/pdf` capability。
- [ ] 旧 full代码只保留显式historical control所需边界；不继续作为product plugin，不增加旧checkpoint converter。

Rollback point R4：parity失败时只修改对应数值/packing层并更换implementation identity；不得重训来包住部署误差。

## Phase 5：代表性 cohort、runtime复测与部署证据

依赖：R4 package parity通过。

- [ ] 冻结一个 bounded representative cohort：标准matrix中的Base/Brushed/Scratched，加至少一个paint/crack或patina recipe和`Aluminum_Anodized` Beckmann例外；选择依据写入protocol，不宣称代表全692结论。
- [ ] 运行单GPU smoke、stop/resume、validation和QAT；检查required groups finite/nonzero/update。
- [ ] 汇总本轮 Linux 五卡 DDP regression，验证bucket/static graph、rank0-only checkpoint、resume、phase boundary与同序teardown；只报告该固定 topology，不扩成scaling研究。
- [ ] 用新package加入matched runtime harness，完成四控制 `prepare/evaluate/sample/pdf` 分解和static账本。
- [ ] Windows viewer输出reference/neural linear EXR与微细节/高光crop；运行deferred/PT和typed edit/asset swap。
- [ ] 以source state为单位生成bootstrap CI，observed quality/time/memory只作相对报告。
- [ ] 输出 `research/runtime-results.md`、`research/deployment-evidence.md` 和最终failure classification。

Rollback point R5：代表性cohort或runtime结果低于预期但正确时如实收尾；不自动进入全cohort long、更多seed、更多lobe或bit-rate sweep。

## Phase 6：规范、稳定文档与质量门

依赖：Phase 0–5 已交付或有明确empirical outcome。

- [x] 更新 `.trellis/spec/learning/online-training.md`，删除把canonical Metal锁死为旧 full shape的规则，换成新method/profile合同；旧规则保留在历史任务证据。
- [x] 更新 `docs/research/model_candidates.md`、`docs/research/experiment_log.md`、`docs/learning.md`、`docs/metal_linux_training.md`。
- [x] 检查 source/runtime/package/viewer 没有 method/source family专用upper-layer分支；本轮移除了 training readiness 对新旧 Metal method key 的分支，方法差异回到 descriptor policy。
- [ ] 运行相关unit、GPU、integration、layout generator `--check`、package validation、Release viewer build/capture、`git diff --check`和Falcor clean。
- [x] 运行 pre-Linux `trellis-check`；修复本任务引入的问题，不回退用户或其他任务改动。Linux GPU/integration/package/viewer 总检查仍随 Phase 3–5 结果执行。
- [ ] 逐项勾选PRD acceptance criteria并写最终摘要；不把未授权formal long列为任务未完成。

## Phase 7：剩余时间专项探索（条件执行）

依赖：hybrid/direct主训练与比较完成，两个Windows diagnostic package/catalog已生成并通过Linux侧验证，且12小时时间仍有余额。

- [ ] 从registry按机制选择最多3个exact locator：Beckmann例外、一个paint/crack/patina复合recipe、一个diffuse contamination或强结构纹理；记录选择依据，不按结果挑样本。
- [ ] 用入选profile做每材质单seed、最多256-step diagnostic probe，分方向/颜色/峰值/空间/analytic contribution报告。
- [ ] 仅当至少两个专项指向同一failure mechanism时，运行一个单seed、最多512-step的小型mixed cohort普适性检查。
- [ ] 输出`research/characteristic-material-probes.md`，把结构方向写成“失败假设→所需机制→预算增量→下一轮验证”，不自动实现多个新变体。

Rollback point R7：主交付有任何缺口或时间不足即不启动；专项结果不一致时保留分材质结论，不扩大实验直到得到统一故事。

## 计划验证命令族

精确测试文件在实现时按受影响范围确定，但只使用以下入口：

```powershell
conda run -n neural-shading python -m pytest tests/unit -q
scripts/run_falcor_python.ps1 -m pytest tests/gpu tests/integration -q
conda run -n neural-shading python -m ncls.cli train <budgeted-run.yaml> --stop-at-step <N> --output <artifact>
conda run -n neural-shading python -m ncls.cli validate <checkpoint> --batches <N> --device 0
conda run -n neural-shading python -m ncls.cli export <checkpoint> <package> --material-index <N>
conda run -n neural-shading python -m ncls.cli package validate <package>
scripts/build_viewer.ps1 -Configuration Release
scripts/benchmark_viewer.ps1 <frozen-arguments>
git diff --check
git -C external/Falcor status --short
```

Linux DDP只在原生Linux通过统一launcher执行；Windows不请求多GPU。

## Acceptance trace

| PRD验收 | 主要阶段 |
|---|---|
| NVIDIA-class预算与来源 | 0、2 |
| 单材质表达结果 | 0、3 |
| loss/metric/tqdm | 1、3 |
| matched runtime | 1、5 |
| 候选分析与消融 | research synthesis、3 |
| 新method/profile实现 | 2–4 |
| 冻结validation与Pareto证据 | 3、5 |
| unit/GPU/DDP/package/viewer | 2–6 |
| 五卡交替监督与两模型Windows交接 | 3–6、设计§15 |
| 剩余时间专项→普适性探索 | 7、设计§15.6 |
