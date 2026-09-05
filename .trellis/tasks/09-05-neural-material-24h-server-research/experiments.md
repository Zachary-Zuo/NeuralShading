# 24 小时实验候选与分支规则

## 共同协议

先读 [研究摘要](research/research-digest.md)。候选表是有依赖的待办库，24h 内不要求全部跑完。每次改变配置前提交实验登记；只有 D0 通过的 source 进入质量比较。旧任务 step 2 仅证明链路，旧 v4–v6 不作当前对照。

首轮使用已有三个 run 的 source locator：Tungsten、划痕青铜、开裂涂漆钢。采用新 campaign identity，按完整 raw receptive field（输出所依赖的原纹理区域，包含 halo/滤波支撑）分离 train 32、selection-validation 8、final-test 8 个 tile；选择与最终考核的 seeds 分别冻结为 2026090502/2026090503，首轮训练 seed 2026090501。先检查非重叠是否可行；若纹理尺寸/周期语义令数量不可行，必须在看到候选效果前修订数量与 identity，两臂同时更改并登记原因，不能虚构独立样本。

沿用 pilot 的 512 optimizer steps、B=128、pair 足迹 0/1/4 texel、16 个 reference footprint taps、16,384 个 train-only calibration、QAT 从第 0 步开始、关闭 proposal/visual 训练作为 D1 起点。以最终解析 YAML 和代码实际含义冻结，不能只凭名称认为 tap/query 数相同。首轮 matched 比较以同 source、相同查询 recipe/seeds/global batch/step/下游网络/read-plan 为单位；随机过程不能因模型初始化多消耗一次 RNG 而改变 reference 流。

每组预登记主指标：canonical 线性 f 的归一化 L1，以及同方向不同位置 response 差分 RMS 误差；并列报告绝对误差、log/chroma/peak、能量、无效/零值比例。按 source、mip/footprint、UV 边界、方向/grazing 分层。报告每组 matched 差值及至少 1,000 次块级 bootstrap 95% CI；块为独立 RF tile/texture-set，不能把相关 rays 当成独立样本。只有三个 source 时不把 source bootstrap 叫总体材质泛化；少数 tile 的 CI 要同时展示样本数与离散结果。跨 seed 确认报告 seed 间差异，不能只给单次 CI。

selection-validation 可指导下一实验；final test 的指标直到收束阶段才进入决策上下文。预先登记最终只比较哪些候选及主指标，多重探索和选择偏差明确写入报告。最终候选若反向利用 test 改进，该次 test 降级为探索证据。

## 实验卡

| ID / 优先级 | 假设与唯一变化 | 对照、预算与观测 | 进入/退出条件 |
|---|---|---|---|
| E00 / 必须 | native raw 输入、UV、reference 和部署读取确实一致 | 三个实际 source；D0 数值/亚 texel/跨 mip/边界/normal 顺序，reference 足迹点值及 16/64 taps witness；Linux 两步 train/resume、租约释放、峰值显存/吞吐 | 任何 native/GT/读取错误先修复；不以更多训练掩盖 |
| E01 / 必须 | 保持下游相同，raw CNN 比修正 summary 保留更多可学习空间信息 | 先实现真实 `metal_spatial_summary_control_v1`，两种 encoder 均去掉 asset-ID 可学习表，保持 UV、量化与读计划；3 source × 2 臂 × 512 steps | 完整 D0、真实 summary gate 后运行；两臂都完成才作 paired 结论 |
| E02 / 高 | 信号丢失发生在输入、编码、量化、prepare 或方向读出某一阶段 | 同 source、同方向/位置差分，记录 raw/stem/fusion/Detail-Context/SNORM/prepared/f 的 RMS、梯度范数/夹角、饱和；有限读出/optimized-code control 每臂最多 256 steps，GT 只在线驻留 | E01 有衰减或无收益时；读出与自由 code 仅作归因，不替代默认 encoder |
| E03 / 条件高 | correction 的正值约束限制对 core 的修正 | 固定 E、prepare、方向 hidden trunk、core 与 gate，仅换等形 `64→3` correction head；positive/signed 每臂先 256 steps，两种高频 source 优先 | 只有 residual 符号/饱和证据支持时进入；范围外 target、最终 clamp、held-out 误差均报告 |
| E04 / 条件高 | target remap 改善优化，不只是降低变换后 loss | 现有 log 对 cube-root，输出仍 canonical f；固定模型/query/预算，512 steps；零点 epsilon/导数处理在 train-only calibration 后冻结 | 梯度尺度/收敛证据支持时；canonical 指标无收益即记录，不只比各自 loss |
| E05 / 条件高 | filtering/LOD 的误差来源于支撑或读出，而非网络容量 | 0/1/4 足迹与 fractional LOD；learned derived mip 对独立 level control，或随机相邻 level 对确定性双读，单次只改一轴；512 steps | E00 过滤语义正确后；读数/显存改变单独计成本，不称完全 iso-cost |
| E06 / 中 | 局部训练辅助监督使 encoder 保留有用语义 | response-only 对一个 train-only auxiliary loss，512 steps；只用 reference/原生输入中确有定义的量，不杜撰 core/residual GT | E02 确认早期编码丢失时；辅助头不进入 runtime，报告额外训练成本 |
| E07 / 必须收束 | 最有希望的方法能保持新位置/参数/纹理上的表现 | 冻结 E/D；G1 final tile，原生参数 G2/G2s 与实际新 texture-set 的无 optimizer compile/response；选择前冻结 cohort | 不把同 texture-set 下不同 module 当新材质；样本不够就限制结论，不宣称 692 材质泛化 |
| E08 / 中 | 有界容量或坐标比盲目延长训练更有效 | 先方向坐标单轴，再单独 activation（含 LeakySmeLU）或 Detail/Context 宽度；每臂 512 steps；重新登记 MAC/reads/bytes/实测时间 | E02 指向读出/容量且满足 method constraints；不能把多项同时修改称归因 |
| E09 / 低 | 训练查询分配能更有效利用同一 GT 预算 | uniform 对 loss 或 loss×update 的 train-only bucket selector；512 steps，单独记录 unique Qref、network eval、selector 开销 | 只有 evaluator/GT 稳定且有足够余时；只保存 CPU recipe/EMA，不存 GT replay batch |
| E10 / 低 | 初始化方差足以解释当前排名不稳定 | 单初始化对 2/4 个小规模实例的逐次筛选，明确为项目改编；同时列 unique Qref 与总 network evaluations | 先有跨 seed 波动证据；不得把少数实例试验称 Taming 原配方复现或 iso-query 结论 |

## E00 / E01 必须补齐的实现

summary-control 当前主动报未实现。修正 summary 需要自身特征生成路径和独立测试：输入采用与 raw 相同的 native 解码/UV/group 语义、相同 conditioning 绑定、SNORM/cook/read-plan 和下游结构；允许旧 summary 表示的信息瓶颈，但不能继承旧 UV 错误或独立 asset-ID 表。用同 summary、不同二维图案的 fixture 证明 raw 学习层收到不同数据，并证明 summary 路径没有偷走 raw encoder 特征。

D0 检查实际颜色/roughness/normal 等 slot 的固定解码、顺序与组合（包括 normal 的 native 运算），cross-UV affine/nonrepeat/address、整数/非整数 LOD、full-image 与 halo tile、一致量化采样。比较 filtered GT 时应过滤 reference 响应；不能平均输入参数后求值就称正确 GT。64 taps 是 witness 对照，不自动成为所有训练的新预算。

先测 A6000 实际 B=128、真实资源/编码/GT/forward-backward 峰值显存、启动开销与稳态 step 时间，再排队；旧 B=8 preflight 峰值与 Windows step 2 总时间不能外推。OOM 时先隔离资源生命周期问题；若必须改 batch/tile，两臂在观察效果前同步改协议或 fresh run，并标记原实验无效/未完成。

## E03 的冻结形态

继承前继设计的受控归因实验：`r_positive = R*tanh(max(h,0))`，`r_signed = R*tanh(h)`，最终 `f=max(0,b+r)`。两臂 `weight=0,bias=0.01`，初始输出相同且非零梯度；R 为 train-only calibration 中每通道 `2*p99(abs(y-b))`，零通道以预登记 epsilon 处理。只更新 correction head，gate 真正冻结。

这是明确身份的 bounded positive/signed control，不能冒充旧 softplus 方法复现；旧候选文档对 signed+clamp 死区的警告仍需检查，此实验以相同非零初始化和饱和观测排除该混杂。有限 code/refinement 是 optimized-code control，不称表示能力上界。

## 调度和追加预算

0–2h 可分配 GPU 5/6/7 做三个 source D0，8/9 做独立短检查；E01 成熟后先排五个单卡作业，第六个补齐最早空闲槽。研究流程由依赖图与剩余时间决定，不承诺固定吞吐。每次启动记录预计耗时及估计依据，确认完整比较能在截止前完成。

E01 之后，先用 E02 判断值得修改哪一段：输入不变先修 native；raw 有信号而 latent 无信号查 E/QAT；latent 有信号而 f 无信号查 prepare/head/loss；粗 LOD 独差进入 E05；种子次序翻转优先做确认。可同时安排互不依赖的分支，Codex 串行解释结果。

单臂 cap 到达即结束。256/512 的探索后，只有存在预登记的收敛/方差理由且时间足够，才新建同预算 1,024/2,048-step 确认组；两臂同样追加，或 fresh run 并说明变化。不得不断延长落后臂直到成功。确认 seed 推荐 2026090601/2026090701，记录真实执行数；24h 内未完成则如实保留单 seed 局限。

用户已授权根据新证据创建 `R-###` 实验。每个新增项先写：来源事件/观察 → 可反驳假设 → 唯一变化与 matched control → 冻结 source/query/split → steps/query/GPU-hours/显存 cap → runtime 可部署性 → 接受/否定/停止规则。没有可隔离的假设时优先补分析或确认，不为占满 GPU 造无效实验。

## 成本与最终判据

训练时间拆分资源构建、reference query、encoder、forward/backward、评测/cook，不把首次 warmup 当稳态吞吐。报告 unique reference queries 和网络求值量，多实例共享查询不能混为一项。运行时记录实际 MAC、prepare state、instance/shared weights、latent bytes、UV/read 上限和该 source 的实际值。

在同卡、同精度、同 batch/query 分布、预热与同步协议下测 median/p90 查询耗时；prepare 单独测且声明复用次数。整帧 viewer GPU 时间不替代单次查询成本。自然成本不同的两臂报告 Pareto 观察及预算差异；不能仅凭同 512 steps 叫 wall-time 或 inference-cost matched。

收束时对选中的少数候选做一次阶段部署轨道（cook/export、Linux 可用的 Slang parity、真实成本）；未实现 matched sample/pdf 的候选不作 PT 方差宣称。形成结论所需的最小证据是可信 D0、至少一个完整 paired 比较及其局限；若前者失败，最终报告明确是输入/平台可行性结论，质量排名不可用。
