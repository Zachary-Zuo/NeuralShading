# Linux DDP 整改实施计划

## 当前状态（2026-09-04）

- [x] Phase 0：只读审计、历史产物 matched 分析、风险与 timeout 因果链。
- [x] Phase 1：真实 DDP reducer 与 distributed objective。
- [x] Phase 2：指标、stage trace、checkpoint/finalization 对称化。
- [ ] Phase 3：Linux correctness、scaling 与 fault injection gate。
- [ ] Phase 4：接通 data/reference scheduler，并执行第二轮 Linux profile。

用户已批准阶段A+B，公共实现与Windows回归已完成。Phase 3已有两卡NCCL integration入口，但目标Linux机器尚未执行；Phase 4按批准边界继续延后。

## 1. Phase 1：真实 DDP reducer

### 改动

- 新增 `DistributedContext` 和 `DistributedObjective`，由公共 training 层持有。
- 用 phase-local DDP wrapper 替换 `_ddp_sync_gradients()`；保持单卡 objective 调用一致。
- 在 phase boundary 设置 parameter ownership、校验 descriptor 并一致地重构 wrapper/optimizer。
- NCCL 初始化传入 local `device_id`；保持 launcher/worker 现有物理设备映射。
- DDP wrapper只存在于`DistributedContext`与phase执行变量；optimizer、lifecycle和checkpoint始终持有underlying model，不引入散落的`.module`判断。

### 重点文件

- `src/ncls/learning/training/engine.py`
- `src/ncls/learning/training/distributed.py`（新增或等价的单一职责模块）
- `src/ncls/cli.py`
- `tests/unit/test_training_distributed.py`（新增）

### Gate

- 静态扫描不再存在 production per-parameter gradient `all_reduce`。
- 单卡 before/after 的 deterministic loss/parameter update 对齐。
- 两 rank synthetic test 对比“rank-local batch 拼接后的单进程 global batch”，active gradients 与一步 optimizer 后参数在规定 tolerance 内一致。
- 跨两个真实 phase 验证 unused/active parameter、wrapper lifecycle 与 resume。
- DDP logging 显示有限 bucket，而非 parameter 数量级 collective。

## 2. Phase 2：控制面与观测

### 改动

- metric descriptor + packed tensor reduction；修正 cumulative work/s。
- 增加 rank-local stage trace 与 rank min/mean/max/straggler 汇总。
- checkpoint 改为小型 rank state gather、rank-0 full snapshot/write、status broadcast。
- periodic/final checkpoint 统一 commit 协议；删除未消费的 `ddp_completed`，所有 rank 同序 teardown。
- 增加 debug profile：`TORCH_DISTRIBUTED_DEBUG=DETAIL` 与经 PyTorch 2.11 验证的 NCCL flight recorder 开关。

### 重点文件

- `src/ncls/learning/training/engine.py`
- `src/ncls/learning/training/distributed.py`
- `src/ncls/cli.py`
- `tests/unit/test_training_distributed.py`
- `tests/unit/test_training_runner_phase_graph.py`

### Gate

- metrics 每 cadence 使用固定数量 collective，名称/顺序不一致时首个 report 前失败。
- report 同时给出正确 steps/s、local/global work/s；用历史 fixture 防止公式回归。
- 非 rank-0 不创建完整 CPU model/optimizer checkpoint；rank-0 write delay/failure 可显式通知 peer。
- success/failure/interrupt 三种路径都按同一顺序 close data、control group、NCCL group。

## 3. Phase 3：Linux 目标机 gate

### 新增验证

- 建立 `tests/integration/test_distributed_training.py`，覆盖两卡 synthetic NCCL correctness、rank-0 artifact 与 bounded failure。
- 对 Metal smoke 执行 1/2/3/4 卡 weak/strong scaling，warmup 后跨至少一个 64-step group boundary。
- 保存 per-rank stage trace、DDP logging、NCCL diagnostics、resolved plan 与完整环境 manifest。
- 注入 rank data sleep、reference exception、rank-0 write delay/exception 和 phase descriptor mismatch。

### Gate

- correctness、checkpoint/resume、failure propagation 与 teardown 是 hard gate。
- speedup、GPU activity、显存、bucket overlap 是 observed report，不预设脱离机器的硬阈值。
- timeout 必须能定位为 data/reference、reducer/desync、checkpoint I/O 或 teardown 中的具体阶段和 rank；不能只得到一条无上下文 watchdog 消息。

## 4. Phase 4：data/reference scheduler

### 改动

- 用 pipeline-backed OnlineDataSession 替换生产 `SynchronousOnlineDataSession` 的固定入口。
- 接通 ready batch ring、host worker、residency lease、ReferenceScheduler 与 trace。
- Vulkan global mode 实现 group 内 packed reference dispatch；禁止同卡无 fence 的 reference/model overlap。
- group boundary 预热、host-only 下一组预取，并建立 same-group/cost-balanced scheduling 对照。

### Gate

- `reference_batch_steps=1` 与 packed 模式逐 logical step 的 request/RNG/top-up/target/resume 一致。
- queue/residency 有界，checkpoint/phase/stop drain 后 cursor 不超前。
- target Linux trace 证明 barrier/step 或 consumer starvation 有可解释改善；不把 Vulkan timeline 未重叠本身判为失败。

## 5. 验证命令

所有 Python 命令使用 `neural-shading` 环境。

### Windows 静态/unit gate

```powershell
conda run -n neural-shading python -m pytest tests/unit/test_training_launcher.py tests/unit/test_training_engine.py tests/unit/test_multi_gpu_launcher.py
conda run -n neural-shading python -m pytest tests/unit/test_training_distributed.py tests/unit/test_training_checkpoint_new.py tests/unit/test_training_events.py
conda run -n neural-shading python -m compileall -q src tests
git diff --check
```

Windows 只验证公共逻辑与单卡行为，不声明 NCCL/Vulkan 多卡通过。

### Linux correctness/smoke

```bash
bash scripts/run_falcor_python.sh --gpus 0,1 -- \
  -m pytest tests/integration/test_distributed_training.py -q
bash scripts/run_falcor_python.sh -m ncls train configs/training/runs/metal-linux-smoke.yaml --devices 0
bash scripts/run_falcor_python.sh -m ncls train configs/training/runs/metal-linux-smoke.yaml --devices 0,1
```

### Linux profile

对同一 frozen config 使用 `--devices 0`、`0,1`、`0,1,2`、`0,1,2,3`，分别产生 weak/strong scaling run identity。命令的 timeout/debug 环境变量、warmup、step、batch/work 单位和产物目录在目标机 preflight 后冻结到任务 research，不事后改口径。

## 6. 开始实现前检查

- [x] 用户已审阅 `prd.md`、`research/ddp-audit.md`、`design.md` 与本计划，并明确批准实施范围。
- [x] 重新运行 `trellis-before-dev`，加载 project/data/learning 规范。
- [x] 检查 dirty worktree，逐文件确认没有覆盖用户或其他任务改动。
- [x] Phase 1 只改 reducer/objective，不夹带 data pipeline 优化。
- [ ] 目标 Linux 机器、GPU 数量和新的 `artifacts/09-04-linux-ddp-diagnosis/` 目录可用。
