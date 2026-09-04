# DDP5 持续监督记录

## 范围

本文件记录 GPU 5–9 上 hybrid/direct matched pair 的里程碑、实现缺陷、修复身份与实验解释。完整 stdout/stderr、checkpoint、metrics和机器采样保存在 `artifacts/metal-budgeted-pilot/ddp5-20260905/`，不提交到根仓库。

> 后续状态说明：下文在v3 step512处的“不再创建v4”是当时主pair未交付前的局部停止决定。主pair、双package和Windows handoff完成后，用户授权用剩余时间做专项探索；该探索另以fresh matched v4 pilot执行，结果见`characteristic-probes.md`，不反写早先里程碑的历史决策。

## Attempt 1：hybrid fresh step 0

- 时间：2026-09-05 01:44（Asia/Shanghai）
- 命令入口：`scripts/run_falcor_python.sh --gpus 5,6,7,8,9`
- resolved plan：hybrid DDP5，物理 GPU `[5,6,7,8,9]`
- 已完成：五个 rank 均创建 `NVIDIA RTX A6000` Vulkan device；train-only calibration完成；rank 0写出225,247 B checkpoint、SHA-256 sidecar与summary；进程退出后GPU全部释放。
- 失败：最终review读取0 B metrics时抛出`ValueError: training review requires metric rows`，Gloo控制面把rank-0错误传播给全部rank，torchrun整体退出1。
- 分类：training lifecycle implementation defect，不是NCCL reducer、reference、模型数值或资源问题。`stop_at_step=0`按冻结协议不执行optimizer step，因此空metrics合法；最终review路径错误地把正step约束用于step 0。
- 修复：metric loader新增默认关闭的`allow_empty`；CLI只有在final checkpoint `global_step == 0`时启用。正step空metrics仍失败，不改变formal readiness、method/data/query identity或训练数值。
- 回归：`test_training_review.py`覆盖step-0显式允许与默认拒绝；相关engine/launcher测试共同通过。

## Attempt 2：hybrid step 0 → 8

- 时间：2026-09-05 01:49（Asia/Shanghai）
- 恢复边界：Attempt 1修复后的hybrid DDP5 step-0 checkpoint。
- 已完成：五rank恢复相同checkpoint并取得第一步online evaluator/sampler batch；错误由所有rank一致报告，进程组退出后GPU释放，step-0 checkpoint未被部分覆盖。
- 失败：通用`validate_objective_outputs()`报告11个required component output缺失，包括asset/compiler/direction/evaluator trace与proposal identity。
- 分类：budgeted method implementation defect，不是DDP reducer或模型数值失败。objective已计算对应值，但只返回带`trace/`、`proposal/`前缀的诊断metric；descriptor要求的component output alias没有进入mapping。原unit只检查标准loss和gradient，没有执行通用conformance gate。
- 修复：保留原诊断metric，并为descriptor要求的component outputs增加detached alias；method unit直接调用`validate_objective_outputs()`。修复修改`metal_budgeted.py`并改变implementation identity，因此Attempt 1的两份step-0 checkpoint标为superseded，hybrid/direct都必须fresh重跑。

## Milestone 8：implementation `6b9f81c`

- hybrid/direct均从fresh DDP5 step 0开始，calibration的scale/P95/epsilon、reference execution plan与query stream一致；不同calibration hash只来自各自training config hash。
- 两侧0→8均完成，所有metric有限，六个required parameter group均已有finite/nonzero gradient与update，rank0-only checkpoint/review及teardown通过。
- peak allocated memory均为739,713,536 B/rank。hybrid/direct review median分别约2.928/2.903 step/s；首步含compile约3.6–3.9秒，后续tqdm显示约2–3 step/s。
- step 1→8 appearance：hybrid `3.6547→2.6996`，direct `4.7860→3.8209`；spatial gradient分别`0.3325→0.2541`与`0.3324→0.2534`。这是极短早期诊断，不形成结构选择。

## Milestone 128 与 before-profile：implementation `6b9f81c`

- 有效artifact根为`artifacts/metal-budgeted-pilot/ddp5-20260905-6b9f81c/`。hybrid/direct均从exact step-8 checkpoint恢复到128，写出rank0-only checkpoint、256条validation row、review和完整日志；五卡随后全部释放。
- hybrid/direct稳定训练step wall mean分别约`0.327/0.336 s`。其中batch prepare约`0.169/0.175 s`，forward+backward约`0.156/0.157 s`，两者几乎串行；`prefetch_depth=1`没有隐藏reference生产。
- hybrid/direct validation wall分别约`50.99/54.26 s`，validation reference session produce仅约`22.22/22.92 s`。剩余时间与当前每batch一次DDP report、descriptor control collective和逐scalar host读取相符，属于公共reporting/调度热点，不是模型显存不足或reference evaluate kernel本身占满全程。
- step128的256-batch validation appearance mean为hybrid `1.5644`、direct `3.8383`；peak分别`1.5740/5.2536`，spatial gradient几乎相同（`0.2856/0.2855`）。这是同一早期里程碑的明显差异，但在架构吞吐修复和fresh matched pair前不形成最终选择。
- 触发：用户要求先提高通用实验效率并在早期结果不足时继续。旧`@1` pair在step128停止，作为before-profile；engine等价优化可对其作性能回归，但任何cadence/prefetch/训练计划变化使用新identity并fresh运行。

## 通用吞吐修复与 batch geometry profile：commit `735cc56`

- validation在不改变256条逐batch全rank平均记录的前提下改为窗口级一次packed collective；旧hybrid checkpoint从128继续到256的真实验证由`50.986 s`降到`41.254 s`，约改善19%。同一窗口reference session produce约`21.8 s`，说明后续热点是reference生产和逐batch inference，不再是metric collective。
- 新`@2` recipe固定每16 step report、phase prefetch depth 2、`reference_batch_steps=2`。在同一hybrid/source/query/model上做fresh 64-step DDP5 probe；前三个稳定16-step窗口结果如下，均为diagnostic profile，不用于质量选择：

| per-rank batch | global batch | median step wall | median global work units/s | Torch peak MiB/rank | run elapsed（含calibration） |
|---:|---:|---:|---:|---:|---:|
| 64 | 320 | 0.2180 s | 2,935.7 | 705.3 | 41.84 s |
| 128 | 640 | 0.2392 s | 5,350.1 | 706.5 | 34.80 s |
| 256 | 1,280 | 0.2446 s | 10,465.2 | 708.9 | 33.66 s |
| 512 | 2,560 | 0.2192 s | 23,354.4 | 747.6 | 27.95 s |

- 512在本轮预登记候选中同时给出最高吞吐和很小的显存增量，选择为新pair共同batch；不因它尚未填满48 GiB显存而事后扩大profile cap。artifact位于`artifacts/metal-budgeted-throughput/735cc56/b{064,128,256,512}/`，每个目录含exact plan/checkpoint/review、完整训练log与2秒间隔GPU dmon。
- 相对旧`@1` step128稳定global work units/s约1,739，新512 probe约提升13.4倍；其中约8倍来自batch工作量增长，其余来自低频report和two-step生产。两种结构后续都必须从新config identity fresh开始，累计work units与global batch必须随step共同报告。

## step-128 后的模型/loss审查假设

- hybrid在step128→256的validation appearance由`1.5644→1.2286`，gate平方均值由`0.3305→0.4587`，positive RGB平方均值约`1e-5`；当前结构主要表现为“learned gate调制analytic lobes”，神经positive residual几乎未参与。这是结构行为证据，不先判为实现错误。
- direct step128 validation appearance为`3.8383`、peak为`5.2536`，明显弱于hybrid；它的positive RGB平方均值约`0.0091`，说明`bias=-5`后仍已开始离开零输出，并非零梯度。是否属于初始化/稀疏peak loss导致的优化问题，需要在新batch512 fresh pair的相同optimizer-update里程碑复查，不能用旧batch64早期值直接定论。
- direct与hybrid的proposal fallback分别约`0.826/0.495`（hybrid step128），虽然proposal参数只接收proposal loss，但其输入semantic state受各自appearance训练影响；差异是模型耦合结果，不是DDP reduce错误。后续同时审查proposal有效率、density NLL和fallback轨迹。

## 高吞吐 v1 pair：共同step 512

- artifact根为`artifacts/metal-budgeted-pilot/ddp5-efficient-c4dd62f/`；hybrid/direct使用per-rank batch512、global batch2560，并分别保存exact step512 checkpoint、256条validation row、review与完整日志。
- hybrid validation appearance在step`128/256/384/512`为`1.4469/1.1877/1.1570/1.1523`，peak为`1.3588/1.0719/1.0644/1.0559`；positive RGB trace约为零，gate由`0.3443`升到`0.7964`，analytic trace由`112.8`升到`1016.6`。它在约step384进入平台，主要依靠gate放大analytic core，没有形成有效neural residual。
- direct同期appearance为`3.8924/2.9156/2.4309/2.2204`，peak为`5.4313/3.4347/2.4702/1.9340`；positive RGB trace由`0.0091`升到`6.3744`，说明direct head确实在学习，不是`bias=-5`造成的softplus死区。linear项从step256后的`1.5471`升到`1.7215`，提示峰值改善伴随过冲风险。
- 两侧spatial-gradient始终约`0.281–0.283`，没有随appearance改善。代码审计显示semantic decoder输出24维并全部写入PreparedState，但neural evaluator只拼接`semantic_state[:8]`；后16维只能影响analytic状态，无法成为neural response输入。这是共享架构瓶颈，优先于继续把v1训练到1024。
- proposal轨迹不作为direct evaluator选择主指标：direct的proposal head只调整mixture weight，而semantic/analytic state随direct appearance共同变化，因此density NLL可以与appearance方向不同；当前没有DDP reduce或unused-parameter证据。

处理决定：保留v1 step512为对照，新增full-semantic v2身份，把方向输入从28扩到44、evaluate从10,368增至11,392 MAC，PreparedState 160 B和两次读取不变。hybrid/direct以`@3`从fresh共同边界重跑；不resume v1，也不先改loss。

## full-semantic v2：共同step 512与空间信号诊断

- artifact根为`artifacts/metal-budgeted-pilot/ddp5-full-semantic-2f91791/`。hybrid/direct均fresh到共同step512，DDP、gradient audit、checkpoint与teardown正常；稳定profile step wall约`0.23 s`，单段wall受reference consumer-wait长尾影响但无collective desync。
- 对同seed、同validation row做v2-v1 paired bootstrap：hybrid appearance差`-0.00083`、spatial差`-0.000017`且CI跨零，peak差`+0.00257`；direct appearance/log/linear/chroma分别改善`-0.00772/-0.06599/-0.06989/-0.00555`，但peak退化`+0.33945`、spatial退化`+0.000544`。完整semantic输入对direct平均响应有用，却没有恢复one-texel细节。
- task-local脚本重新生成8个online validation batch：target one-texel log梯度均值约`0.285`，hybrid/direct预测仅`0.0011/0.0038`。原始patch配对差异约`0.0134`，Detail/Context输出约`0.0013/0.0011`，semantic差异约`0.00033/0.00054`；模型基本输出空间常数。
- spatial-only梯度不是断图：hybrid的asset/semantic组L2约`0.00129/0.0293`，direct约`0.00409/0.0113`，direct directional组约`0.0109`。但固定同一batch以原学习率优化256次，误差只从约`0.329`降到`0.327/0.320`；十倍学习率512次也仅降到`0.295/0.269`。因此继续主recipe不能充分弥补当前信息衰减。

处理决定：v2 step512保留为信息连通消融，不继续到1024。v3把Detail四通道无参数residual到frame semantic前四维，其他matched轴和静态预算不变；两侧fresh重跑，之后不再按observed quality自动创建v4。

## Detail→frame semantic v3：共同step 512

- artifact根为`artifacts/metal-budgeted-pilot/ddp5-detail-frame-aac37e6/`；hybrid/direct均fresh到共同step512，DDP、checkpoint、gradient audit与teardown正常。
- 相对v2的逐validation-row paired bootstrap：hybrid appearance/log/linear/peak分别改善`-0.01213/-0.00755/-0.00849/-0.00742`，chroma轻微退化`+0.000105`；direct log/linear/peak改善`-0.01523/-0.00495/-0.00406`，但总appearance几乎持平并轻微退化`+0.00098`，chroma退化`+0.000824`。
- hybrid/direct spatial都退化约`+0.00033/+0.00032`，95% paired bootstrap CI均不跨零。短路径对平均响应和hybrid peak有净收益，却没有恢复one-texel细节；它不能作为空间问题已解决的证据。

处理决定：不再创建v4。v3作为当前matched结构继续到1024/2048，以获得成熟hybrid/direct checkpoint并完成部署；空间方向登记为下一轮role-separated Detail/Context编码、显式可量化高频latent和对应matched消融，不在本轮继续在线改结构或单纯提高loss权重。

## v3 成熟训练、phase resume 与结构选择

- hybrid/direct 均在同一实现、source/query、seed、global batch 2560 下完成 `0→1792` 的 `joint-response-fit`，并分别从 exact step-1792 checkpoint 恢复到 `deployment-qat-refine` 的 step 2048。两份 step-1792 checkpoint 都记录 `phase_name=deployment-qat-refine, phase_step=0`；恢复后的日志、review 与最终 checkpoint 证明 phase 边界不是只改显示文本。
- 每侧累计 `2048×2560×2=10,485,760` 个 evaluator/sampler route work units。所有五个 rank 的 required group 均保留 finite/nonzero gradient 与 update；没有 unused parameter、collective desync、非有限 metric、hook failure 或不完整 checkpoint。各训练段 dmon 的平均 SM utilization 约 `67%–76%`，周期性低谷对应每128 step一次的256-batch online reference validation，不支持继续以增大 batch 处理验证瓶颈。
- hybrid 的 validation appearance 在 step `1024/1152/1280/1408/1536/1664/1792/1920/2048` 为 `1.1308/1.1346/1.1432/1.1372/1.1301/1.1338/1.1560/1.1456/1.1441`。QAT 将末个 joint 点的回弹恢复到 `1.1441`，peak 为 `1.0672`，runtime FP16 weight MAE 约 `5.04e-5`；positive trace 仍约 `2.4e-6`，模型实际行为是 learned gate 调制 analytic core，且 analytic trace 存在明显长尾。
- direct 的同期 appearance 从 step1024的 `1.9473` 持续改善到 step1792的 `1.8415`，QAT step2048为 `1.8303`；最终 log/linear/chroma/peak/spatial 分别为 `0.6884/1.5920/0.00752/1.7450/0.28120`，runtime FP16 weight MAE约`5.73e-5`。它仍在学习，但与 hybrid 的主质量差距已经明确，不满足“结构选择不确定”这一4096扩展前提。
- step2048同序256条validation row的paired bootstrap中，`direct-hybrid` 的 appearance、log、linear、chroma与peak均为正，均值与95% CI分别为`+0.68615 [0.68013,0.69220]`、`+0.36807 [0.36633,0.36985]`、`+0.21976 [0.21307,0.22634]`、`+0.005928 [0.005911,0.005946]`和`+0.67777 [0.66522,0.69035]`。direct的spatial error小`0.001745 [0.001563,0.001930]`，但两者预测one-texel变化都远低于target，不能把该小差异解释成高频问题已解决。

处理决定：按预登记规则选择 `metal_budgeted_hybrid_v3` 为 canonical profile，保留 `metal_budgeted_direct_control_v3` 为 exact diagnostic视觉对照；不启动2048→4096。下一阶段先实现两者共形的FP16/RGBA8 runtime与evaluator-only Slang/package parity。由于deployment实现会改变method implementation identity，最终可交付checkpoint必须在deployment commit后fresh复跑，不能把当前训练identity静默贴到新runtime上。

## wrap修复后的最终pair与大batch选择：implementation `1d5f813`

- hybrid/direct在wrap边界oracle修复后从fresh step0交替运行到共同step2048；最终checkpoint SHA-256分别为`8a15a5945085bddc781c1e60cd434ffa78b3a791ceed05dbbe007f8e7fb8971e`和`4848b783407eba3a0127910dca370ea97f95f904416e974ad61867a3bbff2042`。两侧均`complete=true`，phase/QAT、五rank gradient/update audit、checkpoint与teardown通过。
- step2048 paired bootstrap的`direct-hybrid` appearance为`+0.68544 [0.67929,0.69144]`；log、linear、chroma、peak也都显著偏向hybrid。direct的spatial为`-0.001790 [-0.001967,-0.001617]`，但不改变两者均未恢复one-texel细节的判断。
- 新Windows handoff位于`artifacts/viewer/metal-budgeted-ddp5-wrap-1d5f813-step2048/`；hybrid/direct真实ScatteringPackage的边界GPU parity最大绝对误差分别为`1.87e-5/2.23e-4`。旧`2dc0965` handoff已显式标记失效。
- 后续batch profile在同一Tungsten/hybrid上比较per-rank `512/1024/2048`；steady median global work units/s分别约`21.1k/42.4k/87.7k`，peak显存约`0.73/0.94/1.37 GiB`。选择2048用于四项特征材质probe；它不改变已完成主pair的matched结论。
