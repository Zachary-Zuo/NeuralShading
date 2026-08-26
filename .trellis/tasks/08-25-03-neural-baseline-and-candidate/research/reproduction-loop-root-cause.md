# Baseline 复现反复的根因与修订

## 它是什么

本文记录 2026-08-26 对 `08-25-03-neural-baseline-and-candidate` 的回溯。结论不是“模型还差一点过线”，而是任务把三个不同问题混成了一个循环：方法是否正确、训练是否收敛、最终质量是否达到人为阈值。旧流程只要最后一个数值不满足，就回到实现和训练重跑，因此无法形成稳定终点。

## 已发生的运行

- `formal-direct-v1` 完成 6,750 steps，训练约 61 分钟；只走 mollification groups，没有进入 base-v5 阶段。
- `formal-direct-v2` 完成 20,000 steps，训练约 78 分钟；包含 mollification 与 base-v5 groups。
- 两次 run 都产生了有限 checkpoint 和可评测结果，但使用的是当前自定义 direct 形态，不是已经逐项审计完成的 NVIDIA 原方法。因此它们只能说明训练链路能运行，不能证明 baseline 已复现。
- 进程记录中存在约 7 小时 38 分钟没有训练工作的等待区间；随后 watcher 又排队了 core、paper、缩模 direct 和 sampler runs。等待链把“任务在推进”和“进程仍存在”混为一谈。

运行中还出现两类真实实现问题：

1. SlangPy Torch wrapper 会按公开 callable 身份缓存首次观察到的 active-gradient mask。冻结 evaluator 路径先 warm 后，sampler-only 阶段若复用 callable 身份，目标参数可能没有 `grad_fn`。这需要 role-specific wrapper identity，已经形成长期 spec；反复重启训练不能修复。
2. direct evaluator 后来加入最低输出比例 floor，说明训练诊断仍在改变函数形态。该改动也不是 NVIDIA 原方法的一部分，不能在变动中的实现上继续用质量门判断“复现是否成功”。

## 旧验收为什么不成立

旧 Q1 使用的 `0.045 / 0.10 / 0.013 / 0.15` 来自早期 P1 run 的观察值和已知 state 结果，其中部分数字甚至比被引用 run 的 observed p95 更严格。`docs/research/experiment_framework.md` 原本明确把质量线称为可修订参考而非 kill gate，后续计划却把它提升成跨 state 的硬验收。这是 spec drift，不是由不同材质的物理或感知特点推导出的合同。

旧任务还同时要求：

- 原规模论文结构只作 `diagnostic`；
- 另做 `≤2k MAC` 缩模形态作为正式 baseline；
- 缩模结果必须过上述绝对质量门；
- 未过门就继续在冻结预算内修改和重训。

这会系统性地产生循环：缩模形态并不等于原方法，过不了质量线又不能放宽或改变预算；即使训练正常完成，也既不能证明忠实复现，又不能结束任务。

## 数据采集为什么也形成了循环

`02` 的 directional mollification supplement 最终由 `27 个 v6 + 2 个 v7 + 1 个 v8` shard 组成，cap 从首轮 `512` paths/jitter/replica 逐次提高到 `524288`，合计记录 `86,283,124,736` 个 reference samples。每一版有独立 hash，旧文件未覆盖，最终 corpus 在 provenance 上可验证；但这不等于“一次冻结规则下的正式采集”。

真实执行把三件事混在一起：pilot 用于发现 estimator/预算、formal 用于生成发布数据、PRD 用于定义任务合同。每次 formal 达 cap 后都根据失败值推导下一 cap，再把 v1→v8 结果追加回 PRD，最终形成了事后合理化的版本链。问题不是 train 噪声可以不管，而是没有先用 pilot 确定足够大的 SPP 并一次冻结 formal plan，却让 `0.06/0.25` 等缺少清晰用户目标或理论来源的事后门持续驱动自动扩算。

正确边界是：先用不晋升的 pilot 测量噪声随 SPP 的下降并确定足够大的 SPP，再让用户确认固定 SPP 或带完整自适应停止规则的唯一 formal plan；formal 只执行一次该计划。SE/variance 用于验证 SPP 是否充分，而不是替代 SPP。若审计仍不充分就停止发布并返回 planning，不接纳噪声数据，也不自动提高预算。已发布 `f693…e4f3` 暂按其冻结 identity 读取，但 v6/v7/v8 组合不能作为以后数据采集的流程模板。

## 训练为什么慢且不能及时自诊断

证据链分为三段：

1. 早期 run 在 initial validation 与 post-curriculum base-HDF5 路径留下永久 `status=running` manifest；训练分区当时在优化循环内做小批随机 HDF5 读取，GPU 侧等待 CPU/I/O。后来把冻结 train 分区驻留内存后，25k joint diagnostic 才能在约 30.5 分钟完成；进一步调整后两次 formal 约 18–20 分钟。
2. 2026-08-26 复跑当前 post-curriculum hot-path：CPU data 中位约 `1.33 ms`、tensor transfer `0.64 ms`、forward `7.08 ms`，但同步测得 backward 中位约 `178.25 ms`、总 step `202.82 ms`。当前主要瓶颈已经从 HDF5 转成小 batch 的 SlangPy AD / 多 callable launch。配置 `batch_size=16` 对应每路 `16×64=1024` directions，远小于论文每路 65k 的训练 batch；这是未先做 batch scaling profile 的预算适配，不是原方法天然只能低利用率运行。
3. runner 只在 validation 边界更新 progress，训练和 validation 内没有连续的 `tqdm` 工作进度与 ETA，因此正常但缓慢、效率异常和真正停滞在外部看起来相同。一次串联 profile 超时后，外层命令退出但 `profile_validation_batching.py` 子进程仍存活；这是该 wrapper 的具体清理缺陷，不应据此把 PID、heartbeat、watcher 和进程树状态机提升为所有训练的统一准入要求。先前 watcher 约 7 小时 38 分钟无训练工作却继续排队，也说明增加监管层本身不能替代可见进度和性能分析。

正式训练应先用 smoke 确认正确性和显存，再让 train、validation 等长循环通过 `tqdm` 显示真实完成量、吞吐与 ETA。如果实测速率明显不合理或长时间没有 work unit 完成，再对对应慢段做 profile，定位 batch 过小、HDF5 I/O、SlangPy AD / callable launch 或同步边界等主导成本并优化。没有证据表明需要无人值守调度基础设施时，不先实现通用 heartbeat / watcher 系统。

## 为什么忠实复现要求最终变成了低预算离线训练

这不是论文含糊，也不是 RTX 4090/A6000 缺少能力，而是本项目执行顺序违反了原需求。论文一手材料给出的训练协议是 GPU 上在线生成 reference query，300k iterations、每次两个 65k batch，总计接近 400 亿个样本；现有配置却先冻结成 HDF5 curriculum，再以 25k steps、每 step 16 个 query group、每组 64 个方向训练。也就是说，每条 route 只有 2,560 万个方向查询；即便按 evaluator/sampler 两条 route 合计，也比论文名义样本量低约 760 倍。

形成该偏差的直接原因有四个：

1. method correspondence 只冻结了网络层数、宽度和 frame/sampler 参数化，没有把 online reference lifecycle、batch size、iteration count 与联合训练顺序作为同等强度的身份合同；
2. 为复用已经发布的 LayerStack HDF5 与降低 SlangPy 小 batch 运行成本，训练入口被改造成离线预算适配，但 pipeline ID 仍错误保留 `paper-v1`；
3. shader runtime 的软成本分类与训练复现混在一起，先追求可运行/可显示，再把未对齐的训练预算当作工程折中，而没有建立独立 adaptation identity；
4. 既有测试锁定了 schema、方向数学、打包 parity 和有限梯度，却没有锁定“论文训练协议不得降格”的 correspondence test，所以身份漂移没有及时失败。

因此，先前“忠实复现”的要求确实没有被完成。正确修复不是把 25k 简单改成 300k：当前 `Buffer.to_numpy()` + HDF5 replay 仍是离线链路。必须先实现 Falcor/Vulkan reference response 到 GPU tensor/loss 的直接传递，再以独立正式配置恢复论文的 online batch、iteration 和 joint evaluator/sampler lifecycle；在此之前现有 checkpoint 只能作为 LayerStack 离线预算适配诊断。

## 本质根因

- **任务合同缺失**：没有约束 hard gate 的来源，也没有把需求交付 / 理论正确性 / observed quality 分开。
- **顺序错误**：在 method correspondence、数据 acquisition policy 和 long-run performance preflight 冻结前就进入正式采集与训练。
- **文档职责漂移**：PRD 同时充当需求、实验日志和失败后的新计划，令范围与成本可以在连续执行中自行增长。
- **进度与性能观测错位**：长循环没有直接展示 work-unit 进度、吞吐和 ETA，遇到异常缓慢时也没有及时转入热点分析；额外 watcher 只放大了“进程仍在”的假象。
- **范围过宽**：baseline 复现、candidate、两 sampler 的 2×2、bundle 和 viewer 同时成为一个任务的收尾条件；局部正确结果无法形成终点，额外工作又遮蔽了最先需要完成的 joint baseline。

对应长期规则已写入 `project/research-execution.md`、`data/reference-and-corpus.md` 与 `learning/pipeline-and-evaluation.md`。下次继续 03 时应先给实际长循环补 `tqdm` 进度，用短时吞吐判断是否需要针对性 profile，并优化已经确认的主导热点；不再把 formal-run liveness 基础设施当作训练前置。也不能用新 quality 数值、额外 seed 或 watcher 队列替代未完成的 joint lifecycle。

## 对只读二手复现的结论

`D:\01_Workspace\Real-Time Neural Appearance Models` 提供了有用的结构对照：默认 latent 8、两个由 latent 提取的 frame、`z + T wi + T wo` decoder、`3×64` 最大 preset、`3×32 → 9` sampler，以及较长的正式训练配置。它只能作为二手线索，不能直接作为 correctness oracle。

只读审计发现其中至少有这些边界需要独立验证：layered BRDF GT 是单层 Cook–Torrance 的加权和，不是本项目 LayerStack random-walk reference；方向 softening 与已生成 target 的配对存在疑点；`dual_batch` 配置没有形成实际训练数据流；sampler objective 与 evaluator detach 边界需要重新核对；quick start 的短 schedule 和 Mitsuba sphere render 也不等于本项目正式 viewer 生命周期。因此本项目只参考其结构，不复制其 GT、loss、训练结论或 viewer 证据。

## 修订后的终点

本任务继续使用原目录，不创建新任务。复现状态只由以下证据决定：

1. method-correspondence 逐项证明实现对应原方法或明确登记为 adaptation；
2. loss、梯度、权重有限，预先冻结的主 seed 上 validation 相对初始化改善且后期无可信发散；额外 seed 只在轨迹异常或用户明确要求时补充；
3. checkpoint 可恢复，SlangPy/Falcor/packed asset 是同一实现；
4. sampler 的 PDF/null/sample 数学正确。

directional/energy/visual quality、sampler 方差、时间和内存继续完整报告，并按材质结构分组比较，但不再决定“复现成功”。原规模 baseline 必须进入 MethodBundle/viewer；超过软成本线只改变成本分类，不触发缩模替换。

## 进程处置

需求修订后已停止尚未开始产出、会继续训练缩模 direct/sampler 的两个 watcher；没有删除任何 artifacts。随后 `formal-core-v2` 在 3,750 steps 以 `status=failed` 结束，排队的旧 `formal-paper-v2` 没有启动；当前已无 03 正式训练进程。core 失败产物保留用于诊断，但现有 paper pipeline 也已被一手 correspondence 证明不忠实，因此不会继续训练。后续只有在原规模 baseline 结构、joint lifecycle 与 convergence report 冻结后才启动新的 formal run。
