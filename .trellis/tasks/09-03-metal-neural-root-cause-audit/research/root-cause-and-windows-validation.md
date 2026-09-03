# Metal 神经训练与部署根因及 Windows 验证

## 结论与当前边界

旧 20k checkpoint 的白模不是“训练得比较差”，而是部署了尚未开始接受 appearance 监督的 evaluator：当时前 20k step 只运行 `codec-warmup`，viewer 实际使用的 compiler、prepare、evaluator 与 proposal 仍逐位等于初始化状态。Linux 在 20k 后变慢也不是模型自然越训越慢，而是此时首次启用 online evaluator route 后，178 个 execution group 按 request 轮转，和容量为 8 的 LRU 组合成确定性的稳态 miss/evict/recreate。

共享路径现已改为从第 1 步运行 appearance 与 proposal 的端到端训练，并以 64-step group block、按需 `evaluate` pass 和窗口化 profile 消除旧结构性退化。Windows 已证明当前 exact identity checkpoint 能更新全部 13 个参数组、严格保存恢复、被 readiness 正确分级并经 package/viewer 呈现出相对初始化显著改善的输出。Windows 短训仍不是 120k formal 质量结论；Linux 目标机 full-cohort smoke、现场 4 分钟口径确认、120k long run 和最终 checkpoint 回传 Windows 尚未执行。

## 1. Root Cause Category

### 1.1 白模

- **Category**：B - Cross-Layer Contract（主因）；A - Missing Spec、D - Test Coverage Gap、E - Implicit Assumption（促成因素）。
- **Specific Cause**：训练层把 `phase_name=joint-appearance / phase_step=0` 当作“已进入下一阶段”的恢复游标；compiler/viewer 层只检查 tensor 存在、shape 与 finite，没有检查最终 capability 所依赖的参数组是否真的获得 finite/nonzero gradient 和 parameter update。viewer 又默认选择旧 20k checkpoint，并允许 `state-schema-compatible-preview` 用当前实现解释旧权重。
- **直接证据**：旧 5k 与 20k checkpoint 中六个 codec group 变化；`typed_compiler`、`optimized_state_teacher`、`prepared_model`、`angular_bank`、`analytic_core`、`hybrid_evaluator`、`proposal_sampler` 七组逐位不变，coverage 三项全 false、`last_audit_step=-1`。完整原始证据见 `initial-evidence.md`。

### 1.2 20k 后性能崩溃

- **Category**：B - Cross-Layer Contract 与 E - Implicit Assumption（主因）；D - Test Coverage Gap（促成因素）。
- **Specific Cause**：producer 的 round-robin 访问序列复用距离为 178，backend residency 上限为 8；“实现了 LRU”被错误地等同于“会命中”。每个 miss 还无条件构建训练不用的 `sample/pdf` pass。旧 metrics 每十步只记录最后一步 preparation、吞吐使用 run-global 累计值，进一步隐藏了中间冷构建。
- **直接证据**：旧 phase 1 的 10-step interval 出现约 45.06 和 24.76 秒/step，而对应记录行只显示约 2 秒 preparation；静态访问序列保证容量填满后近 100% miss。Windows full-cohort 探针在 block 调度下只在第一个 step和每 64-step group 边界创建一次 group，block 内全部命中。

### 1.3 调查中发现的同类问题

| 问题 | 分类 | 影响 | 处理 |
|---|---|---|---|
| validation 与 training 选择同一 group 序列 | B/E | holdout 与训练组相关，validation 还可能提前物化训练将访问的group | 固定 `validation_offset_blocks=104729`，保持独立seed与确定性恢复 |
| training/validation backend counter混在同一窗口 | C/D | 下一训练窗口错误归因validation成本 | 分成 `profile/reference_*` 与 `profile/validation_reference_*` |
| review使用run-global rate | D/E | 长程或phase局部退化被历史均值稀释 | 改用log-window step wall的rolling rate，并保留兼容字段 |
| MDL内容寻址目录并发发布遇到瞬时句柄拒绝 | D/E | Windows首次full-cohort物化可因 `WinError 5` 中止 | 共享发布逻辑有界指数重试；另一发布者胜出后严格加载验证 |
| complete phase显示按phase数组索引 | C | complete checkpoint越界，边界checkpoint又显示成“已训练phase” | 直接使用checkpoint的规范化`phase_name` |
| Metal launcher扫描不存在的`manual-packages` | C | 公共入口退化为reference-only package scan状态 | 改为catalog实际生成的`packages`目录 |
| `ready/finite/parity`被当作学习质量 | B/D | 两端可以一致地产生白模并仍显示ready | exact checkpoint固定viewer对照，同时报告readiness、线性误差和输出统计 |

## 2. Why Fixes Failed

1. **仅让 checkpoint 能被读取**：这是 surface fix。tensor key/shape 一致只证明内存布局可装载，不能证明 forward 语义相同，也不能证明相关参数接受过训练。
2. **只检查 finite 和 Python↔Slang parity**：这是 incomplete scope。它排除了 NaN 和实现分歧，却无法排除“一致地输出初始化白模”。
3. **把 phase 名称当能力**：这是错误 mental model。phase cursor 位于边界时描述的是下一步从哪里恢复，不是前一阶段之外的能力已经完成。
4. **给 backend 加 LRU 就认为性能有界且快**：这是错误 mental model。内存有界成立，但当复用距离远大于容量时，时间复杂度退化为每步重建。
5. **只记录 log step 的 preparation 和累计吞吐**：这是 tool/observability limitation。故障恰好发生在未记录的九个 step 中，现有日志不能区分模型、query、compile、resource、validation 或 checkpoint。
6. **用不同 implementation identity 的短训结果继续说明当前代码**：会形成 change propagation failure。本报告只把旧结果当历史线索；当前通过证据必须绑定当前 descriptor 和 checkpoint SHA。

## 3. Prevention Mechanisms

| Priority | Mechanism | Specific Action | Status |
|---|---|---|---|
| P0 | Architecture | `joint-coarse-to-fine` 105k + `qat-refine` 15k；asset/evaluator/sampler与最终runtime组从第1步启用 | DONE |
| P0 | Runtime contract | 统一readiness assessor；formal要求exact identity、`run_class=formal`、complete及全部required coverage | DONE |
| P0 | Runtime contract | diagnostic只能显式请求且固定为evaluator-only；删除shape-only兼容 | DONE |
| P0 | Architecture | `group-block-balanced@1`、record-count quota、64-step block、rank stride与独立validation offset | DONE |
| P0 | Architecture | training session只请求`evaluate`，不构建未使用的reference `sample/pdf` | DONE |
| P1 | Observability | 完整step/prepare窗口mean/p90/max、rolling/phase rate、cache/build/dispatch/rejection/memory与validation独立profile | DONE |
| P1 | Test coverage | 两phase首步13组gradient/update、proposal边界、resume、readiness正负例、group序列/operation lazy/profile隔离 | DONE |
| P1 | Integration | exact checkpoint→diagnostic package→Windows viewer固定reference/neural线性EXR对照 | DONE |
| P1 | Documentation | 更新learning/data/viewer spec和稳定Linux训练文档；将本类跨层错误加入thinking guide | DONE |
| P0 | Target-host evidence | Linux同commit full-cohort smoke、现场热点确认、120k formal及回传Windows部署 | TODO（只能在目标Linux执行） |
| P1 | Version control | 将本次spec/source/config/test作为一个identity原子提交 | TODO（等待用户授权commit） |

本项目没有 `src/templates/markdown/spec/`，因此本次不存在可同步的spec模板副本；项目内 `.trellis/spec/` 是实际规则所有者。

## 4. Systematic Expansion

- **Similar Issues**：其它分阶段方法也可能把“参数存在”误认为“能力已训练”；任何大于cache容量的轮转资源都可能有同样的thrash；任何只在log cadence取最后值的profile都可能遗漏尖峰；任何linked catalog入口都可能因目录命名漂移而静默退化。
- **Design Improvement**：capability必须由依赖参数组的训练证据派生，不从静态descriptor单独推断；训练调度的访问分布、复用距离和cache容量必须作为一个合同验证；validation是独立work stream，不与训练counter或group序列共享归因。
- **Process Improvement**：模型改动后的证据必须绑定descriptor/config/checkpoint hash；先做固定流优化证据，再做package parity，最后做viewer reference-neural质量证据，三层不能互相替代。
- **Knowledge Gap**：原设计混淆了codec重建、最终散射目标和部署可见结果。codec可作辅助正则，但viewer不直接显示codec，因而不能用它的20k训练量代表evaluator训练量。

## 5. Bayesian Evidence Update

调查开始时对“4分钟/step”的先验为：group冷构建/LRU thrash 55%，模型forward/backward自然变慢20%，I/O或checkpoint 15%，其它10%。发现178:8复用距离、20k正好启用evaluator route以及未使用pass构建后，结构性thrash上升为最高置信假设；Windows full-cohort block探针进一步证明block内无重建，但Linux绝对4分钟口径仍需目标机profile，因此不宣称已测得Linux加速倍数。

对白模的先验为：未训练/错phase 40%，checkpoint与runtime漂移25%，package/Slang错误20%，颜色/显示问题15%。逐位checkpoint比较和coverage把“未训练evaluator”提升为确定主因；old viewer同时存在implementation漂移，属于放大器。当前exact checkpoint在同一viewer中从16到544步显著降低reference误差，反驳了“共享训练或package链路完全没有学习信号”，但短训输出仍偏中性，因此不把“会学习”外推成“已达到formal材质质量”。

## 6. Windows 共享路径证据

### 6.1 当前 exact identity 学习与部署

- method descriptor：`4e390a4eb489d10bea687a819306549d5be63fd7528e2046416c9261971293bb`。
- 16-step checkpoint：SHA-256 `c94617b0e5c39d966ca5c455df084cddc15a0f1586c4fb57a224dfc1627e48bc`。
- 544-step checkpoint：SHA-256 `bb40309e737089c2d9f331b791980c5a460eea94c30531338ac14110fd09d569`；13/13参数组的finite/nonzero-gradient/update均为true。
- 544-step review：训练wall time `275.15 s`，rolling median `2.267 step/s`；hot-window step wall均值约`0.441 s`，p90约`0.469 s`，峰值显存`327,747,072 B`。
- 固定16-batch online evaluation成功，mean loss `0.67426044`。该聚合来自QAT complete checkpoint，不与viewer区域MAE混作同一指标。
- formal readiness正确拒绝该`run_class=profile` checkpoint；唯一原因是非formal run class，identity、complete与coverage均成立。diagnostic-evaluator readiness为true。
- fixed shaderball、同source、同scene、同descriptor的线性EXR：16→544 step的MAE为`0.422611 → 0.133556`，约下降68.4%；neural平均亮度`0.431380 → 0.125018`，reference为`0.047813`。这证明旧式“完全无学习白模”已消除，但544步仍偏中性且误差明显，不能冒充120k质量。

对应证据：

- `artifacts/metal-root-fix/windows-smoke-final/`
- `artifacts/metal-root-fix/windows-learning-probe-final/`
- `artifacts/metal-root-fix/viewer-diagnostic-final/`
- `artifacts/metal-root-fix/viewer-learning-probe-final/`

### 6.2 当前profile的冷/热成本解释

一次当前instrumentation cold step总wall约`6.391 s`，其中prepare约`2.049 s`、group build约`1.342 s`；build可进一步分成runtime compile `0.005 s`、pass build `0.190 s`、resource bind/upload `1.140 s`、slot build `0.007 s`。hot window不再创建group，reference `sample/pdf` request始终为0。

此前使用同一block/operation-lazy共享路径的692-source、178-group、584-step full-cohort探针显示：每个64-step block只在首步产生一次miss/create；resident逐步升至8，第九个group开始按边界一次evict，随后block内继续命中。该探针早于最新validation-offset descriptor，因此只作为调度/资源结构证据，不冒充当前method质量证据。

### 6.3 最终质量门

- `tests/unit`：225 passed。
- `tests/gpu tests/integration`：51 passed，使用统一Falcor Python launcher与RTX 4090。
- Release viewer重新配置/构建成功；overlay反向应用后`external/Falcor`保持clean，HEAD为锁定的`9dc819c162b2070335c65060436041690b7937f8`。
- config生成器通过：692 sources、16-step smoke、120000-step long，semantic fingerprint为`249202d2bf46810739a2a5891c0ccf9bb423f03348982b965b46d495a3855f01`。
- `compileall`与`git diff --check`通过；项目没有另行配置lint/type命令。生成JSON仅报告现有CRLF→LF提示，不是diff error。

## 7. Linux 剩余验证与停止条件

当前canonical semantic fingerprint为`249202d2bf46810739a2a5891c0ccf9bb423f03348982b965b46d495a3855f01`。Linux必须在本次修改形成的同一commit上：

1. 运行环境/配置preflight和692-source full-cohort smoke；
2. 用新增profile确认现场“4分钟”究竟是cold group、resource upload、dispatch、rejection、validation/checkpoint还是新的目标机问题；
3. 若仍出现逐step create/evict或吞吐随step系统恶化，停止long run并回到共享owner修复；
4. smoke与resume通过后启动唯一一次冻结的120k formal run；
5. 将最终checkpoint带回Windows做formal package、Slang与viewer验证。

不设置未经目标机实测的绝对秒数hard gate，也不通过减少source、batch/reference工作量制造加速。当前最重要的剩余风险是首次全量MDL编译/绑定仍有一次性成本，以及Linux Vulkan/driver/文件系统可能出现Windows未覆盖的新热点；这些由现有分项profile直接归因。

## 8. Knowledge Capture

- [x] `.trellis/spec/learning/online-training.md`：端到端phase、readiness、profile、group schedule、QAT与验证门。
- [x] `.trellis/spec/data/reference-query.md`、`mdl-reference.md`：operation-lazy、cache/profile和内容寻址发布。
- [x] `.trellis/spec/viewer/mdl-reference.md`：formal/diagnostic、exact identity和viewer文案。
- [x] `.trellis/spec/guides/cross-layer-thinking-guide.md`：checkpoint capability与LRU访问序列检查。
- [x] `docs/learning.md`、`docs/metal_linux_training.md`：中文稳定入口和Linux命令。
- [ ] 形成原子commit并在Linux目标机执行交接；当前任务保持active。
