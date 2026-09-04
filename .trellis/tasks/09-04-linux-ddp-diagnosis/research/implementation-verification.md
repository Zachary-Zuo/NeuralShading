# 阶段A+B实施与验证记录

## 已实施

- 新增公共`DistributedContext`与`DistributedObjective`。Linux多卡以phase-local`DistributedDataParallel`包装真实objective，启用`find_unused_parameters=True`和`gradient_as_bucket_view=True`；旧的backward后逐parameter `all_reduce`已删除。
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

## 待Linux目标机执行

```bash
NCLS_DDP_DEBUG=1 bash scripts/run_falcor_python.sh --gpus 0,1 -- \
  -m pytest tests/integration/test_distributed_training.py -q
bash scripts/run_falcor_python.sh -m ncls train \
  configs/training/runs/metal-linux-smoke.yaml --devices 0
bash scripts/run_falcor_python.sh -m ncls train \
  configs/training/runs/metal-linux-smoke.yaml --devices 0,1
```

随后对同一冻结source/query/model/batch配置执行1/2/3/4卡matched scaling与fault injection，并保存DDP bucket、逐rank stage和NCCL flight-recorder产物。通过前不宣称timeout已在目标机修复，也不把当前同步`OnlineDataSession`的data/reference scheduler写成已优化。
