# Linux DDP 与 online pipeline 实施验证记录

## 已实施

- 新增公共`DistributedContext`与`DistributedObjective`。Linux多卡以phase-local`DistributedDataParallel`包装真实objective；目标机logging确认active graph固定且unused bytes为0后，使用`find_unused_parameters=False`、`static_graph=True`和`gradient_as_bucket_view=True`。旧的backward后逐parameter `all_reduce`已删除。
- NCCL data group负责DDP reducer和GPU tensor指标；Gloo control group负责run/phase/metric descriptor、小型rank state、rank-0 commit status与teardown error。model/resume/optimizer setup和phase transition等低频动作在进入下一次DDP collective前传播任一rank异常；训练热循环不因此新增barrier。
- scalar metrics按稳定descriptor打包为一次collective；stage汇总保留每个rank原值、min/mean/max、straggler rank和reference group 48-bit identity。累计global/local work throughput公式已修正。
- checkpoint先同步rank-local drain/RNG/query cursor错误，再只向rank 0 gather小型状态；仅rank 0编码完整model/optimizer并写periodic/final artifact，成功或异常通过control group广播。
- `NCLS_DDP_DEBUG=1`显式开启PyTorch 2.11存在的DDP detail与NCCL trace/dump/desync/timing诊断变量；默认性能run不启用。

## 本机证据

环境为完整Windows：RTX 4090、`neural-shading` Conda环境、Windows Falcor构建存在。它可验证公共逻辑与单卡GPU/Falcor路径，不能替代Linux/Vulkan/NCCL证据。

- `conda run -n neural-shading python -m pytest tests/unit -q`：最终复验为`297 passed in 33.91s`。
- 两进程Gloo测试覆盖DDP平均梯度`5.0`、SGD一步后参数`0.5`、packed metric、逐rank统计、phase-local active parameter、descriptor mismatch、rank-0及任意rank失败传播。
- 最终代码上的Windows Metal 16-step共享路径完成：`complete=true`、13组coverage由complete checkpoint gate通过、耗时`38.995s`，checkpoint SHA-256为`d4dae815729a175fe0c86355a28b7be14a94d03d468d63f2b35ea46d9dedeb53`。
- smoke首个training row为`steps_per_second=0.07645271`、每step 8 work units、`global_work_units_per_second=0.61162166`，证明新公式不再误写为`steps/s × world_size`。

## Linux目标机执行命令（实施前冻结）

```bash
NCLS_DDP_DEBUG=1 bash scripts/run_falcor_python.sh --gpus 0,1 -- \
  -m pytest tests/integration/test_distributed_training.py -q
bash scripts/run_falcor_python.sh -m ncls train \
  configs/training/runs/metal-linux-smoke.yaml --devices 0
bash scripts/run_falcor_python.sh -m ncls train \
  configs/training/runs/metal-linux-smoke.yaml --devices 0,1
```

随后对同一冻结source/query/model/batch配置执行matched scaling与fault injection，并保存DDP bucket、逐rank stage和NCCL诊断产物。以下章节记录命令执行后的实现与结果。

## Linux目标机首轮baseline（2026-09-04）

环境判定为Linux reference：Ubuntu kernel 5.15、10张RTX A6000、`neural-shading`环境和唯一Linux Falcor Python扩展可用。用户指定GPU5–9；探针时五张卡均约1 MiB占用、0%利用率，GPU0的外部满载不在本任务范围。

- 目标单元回归：`tests/unit/test_training_distributed.py`、`test_multi_gpu_launcher.py`、`test_training_runner_phase_graph.py`与`test_training_checkpoint_new.py`共`13 passed in 7.69s`。
- GPU5/6两卡synthetic NCCL在普通模式干净通过，两个rank各`1 passed`，日志位于`artifacts/09-04-linux-ddp-diagnosis/synthetic-nccl-gpu5-6-nodebug/`。
- `NCLS_DDP_DEBUG=1`时测试数值仍通过，但进程退出会打印PyTorch `ProcessGroupNCCL`未销毁告警。task-local最小probe证明：只创建NCCL/Gloo并销毁没有告警；绕过项目`DistributedContext`、直接使用PyTorch 2.11 `DistributedDataParallel`也复现告警。因此它归类为`TORCH_DISTRIBUTED_DEBUG=DETAIL` wrapper的诊断模式观察，不作为项目普通teardown失败；性能run必须关闭debug。probe源码已删除，原始日志保留在对应`teardown-probe-*` artifact目录。
- GPU5单卡`metal-linux-smoke`完成两phase共16 step：`complete=true`、`source_count=692`、gradient/update coverage完整、metric全finite、无hook failure；checkpoint SHA-256为`7002db05…ba254`，峰值显存`3,070,980,608` bytes，运行耗时`30.13s`。末步rolling速度约`0.994 step/s`，review的median为`0.873 step/s`；这些是pipeline改动前baseline，不是正式scaling结论。产物位于`artifacts/09-04-linux-ddp-diagnosis/metal-smoke-w1-gpu5/`。

首轮trace也确认已有优化只接通了一部分：Metal asset miss使用host worker与GPU residency，但生产CLI仍固定使用`SynchronousOnlineDataSession`；单卡review中累计host consumer wait为`2.138s`，而`ready_batches`与`reference_batch_steps`尚无统一生产session调用点。用户随后把完成data/reference pipeline与DDP实际提效纳入本任务，后续实现和第二轮profile以更新后的PRD/design/implement为准。

## 最终实现

- 生产CLI、训练engine与checkpoint validation统一改用`PipelineOnlineDataSession`。一个step一次提交全部named routes；session拥有有界pending/ready/acquired状态、严格logical ID、backpressure、共享lease lifecycle、失败取消和idle-only checkpoint。旧`SynchronousOnlineDataSession`、`next_batch()`及其export已删除，没有保留compatibility alias。
- `OnlineTrainingProducer`先为每个route request冻结独立generator，再由`ReferenceScheduler`把同execution group的连续evaluator request合并。随机样本identity只由`request.name/request.seed/request_index`决定，不包含`ready_batches`或`reference_batch_steps`；rejection top-up仍按logical request自己的generator推进，结果按ID切回typed batch。
- Metal host decode进入既有有界`HostPipeline`并可提前提交；prefetch不推进producer cursor，也不创建CUDA/Falcor资源。mip residency materialize、reference dispatch与lease仍由rank主进程拥有。Vulkan保持`global-sync`，本实现没有虚构同卡Falcor/model timeline重叠。
- Metal同一production的两步最多共同持有24个tile lease，working-set entry容量按真实最大并发从16调整为32；GPU字节预算仍由`GpuResidencyManager`硬限制。
- phase active graph在Linux DDP logging中稳定且`ddp_unused_parameter_bytes=0`，reducer收紧为`find_unused_parameters=False/static_graph=True`，不保留unused扫描兼容路径。
- 训练完成后曾稳定复现“checkpoint已经写出、解释器退出时segfault”。`reference_batch_steps=1`与`num_workers=0`仍复现，排除了packing和host worker；GC probe定位到进程Falcor device cache在解释器/原生runtime全局析构期才释放。现在session间仍复用device，但CLI在全部session关闭后、process group teardown前显式调用`close_reference_backend_devices()`；标准GPU进程随后均以0退出。
- data setup阶段也使用对称rollback：任一rank初始化失败并经control group传播后，各rank先关闭自己已创建的session或producer，再清空进程device cache，最后销毁process group；不让成功rank直接带着native owner退出。
- 新静态测试禁止engine/model/method/data plane读取`platform.system()`、物理GPU、Falcor API、内部GPU环境变量或backend key。Windows/Linux只在launcher、device/reference backend和capability层分化，上层使用相同step/session/batch/checkpoint合同。

## 数据面与恢复正确性

- `reference_batch_steps=1`与`2`的GPU5 matched run分别位于`matched-ref1-w1-gpu5/`和`matched-ref2-w1-gpu5/`。前两步loss均为`1.08488`、`1.08019`；加载checkpoint后model tensor逐项完全相等，request count相等，reference logical cursor都为2。
- unit覆盖两个logical request合并为一次backend call、输出逐tensor相同、rejection compaction/top-up、执行计划identity改变不改变sample、route set/顺序、backpressure、共享lifecycle、cancel/drain与resume cursor。
- `pipeline-teardown-w1-gpu5-v3/`的第2步checkpoint恢复到`pipeline-resume-w1-gpu5/`第4步并正常退出；续跑第3/4步loss为`0.998108/1.12305`，证明新session只在消费边界保存cursor。
- smoke每个rank的16个training step合计8次reference dispatch、16个logical step，`last_pack_steps=2`；phase、validation和stop边界没有被跨越。

## Weak scaling：固定per-rank workload

冻结配置为每rank每step `asset/evaluator/sampler=12/64/64`，即每rank 140 work units。性能run关闭`NCLS_DDP_DEBUG`，依次使用GPU5、5–6、5–7、5–8、5–9。稳态口径预先固定为step 10–16：先取每个step各rank wall max，再取7个step的median；global work/s为`world_size × 140 / median wall`。

| 卡数 | median rank-max step wall | global work/s | 相对单卡 | 并行效率 | 进程总耗时 | 每rank peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.914 s | 153.18 | 1.00× | 100.0% | 33.03 s | 2.86 GiB |
| 2 | 0.921 s | 304.13 | 1.99× | 99.3% | 36.48 s | 2.88 GiB |
| 3 | 0.962 s | 436.52 | 2.85× | 95.0% | 44.87 s | 2.88 GiB |
| 4 | 1.009 s | 554.88 | 3.62× | 90.6% | 50.24 s | 2.88 GiB |
| 5 | 1.094 s | 639.77 | 4.18× | 83.5% | 53.67 s | 2.88 GiB |

2卡和5卡global work/s均高于matched单卡，R13硬门通过。多卡DDP确实提高全局产出；5卡并非线性，剩余损失由rank-max step拉长、全局同步reference/data波动和reducer共同构成。这里的“跑满”指各rank持续完成同等工作、没有长期掉队或无来源等待并取得实际吞吐增益；本轮没有单独保存`nvidia-smi dmon`时间序列，因此不把瞬时GPU activity或100% utilization写成结论。

artifact目录依次为`final-scaling-w1-gpu5/`、`final-scaling-w2-gpu5-6/`、`final-scaling-w3-gpu5-7/`、`final-scaling-w4-gpu5-8/`、`final-scaling-w5-gpu5-9/`。五个run均`complete=true`、metric全finite、gradient/update coverage完整并正常退出。

## Strong scaling：固定global workload

观察性对照固定全局`asset/evaluator/sampler=12/64/64`，仅运行可精确等分的1/2/4卡；2卡每rank为`6/32/32`，4卡每rank为`3/16/16`。口径仍为step 10–16的rank-max wall median，不能与上表的weak scaling混为一列。

| 卡数 | median rank-max step wall | global work/s | 相对单卡 | 强缩放效率 | 每rank peak VRAM |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.914 s | 153.18 | 1.00× | 100.0% | 2.86 GiB |
| 2 | 0.635 s | 220.61 | 1.44× | 72.0% | 1.93 GiB |
| 4 | 0.497 s | 281.43 | 1.84× | 45.9% | 1.42 GiB |

结果说明固定global batch仍可提速，但小per-rank batch使模型/reference计算不足以摊薄reducer与控制成本，4卡收益明显低于线性。产物为`final-strong-w2-gpu5-6/`和`final-strong-w4-gpu5-8/`；二者均完整完成两phase并正常退出。

## 跨group长稳态与故障矩阵

- `metal-linux-long.yaml`在GPU5–9运行到第65步后按`--stop-at-step 65`正常提交checkpoint，训练活跃耗时`117.58s`、进程总耗时`125.46s`、每rank peak VRAM `3,100,845,056` bytes，checkpoint SHA-256为`f409ff3634e7c62c50b30c0733d54c1fcb060ffd63f85e796f16922c4ba1a1bc`。`complete=false`仅表示相对120000步formal计划的受控停点，不是运行失败。
- step 65窗口同时观察到2个execution group、3次dispatch和5个logical step：第64步group boundary与stop尾部使scheduler按边界拆包，没有把不同group误合并。该run无timeout、无segfault，rank-0 checkpoint写入约`0.255s`。
- GPU5/6的NCCL integration覆盖真实DDP平均梯度与一步optimizer、packed metric、逐rankstage/bucket，以及rank-1 data error、rank-0 write error、descriptor mismatch。追加的rank-1 `0.1s`数据长尾与rank-0 `0.1s`写入等待证明正常慢操作能由控制面有界协调；异常不发布success并由全部rank退出。
- 普通性能run中的`consumer_starvation`每两个logical step计1次，表示`global-sync`路径在ready为空时由consumer触发一次两步production，不是后台GPU producer饿死；其计数与`production_dispatches=8`、`logical_steps=16`严格对应，且没有无界等待。

## 最终质量门

- `conda run -n neural-shading python -m pytest tests/unit -q`：最终复验`303 passed in 22.16s`。
- `bash scripts/run_falcor_python.sh --gpus 5,6 -- -m pytest tests/integration/test_distributed_training.py -q`：两个rank各`1 passed in 4.9s`，torchrun正常退出。
- `conda run -n neural-shading python -m compileall -q src tests`：通过。
- `git diff --check`：通过。
- production源码搜索只剩静态否定测试提及`SynchronousOnlineDataSession`；没有旧类、`next_batch()`或同步compatibility入口。

本任务完成的是DDP/pipeline正确性、可恢复性、故障传播和目标机吞吐门。120000步formal训练尚未执行，也不是本次实现收尾所需证据；是否启动它由用户另行决定。
