# Linux DDP 整改实施计划

## 当前状态（2026-09-04）

- [x] Phase 0：只读审计、历史产物 matched 分析、风险与 timeout 因果链。
- [x] Phase 1：真实 DDP reducer 与 distributed objective。
- [x] Phase 2：指标、stage trace、checkpoint/finalization 对称化。
- [x] Phase 3：Linux baseline correctness、scaling 与 fault injection gate。
- [x] Phase 4：接通唯一 pipeline-backed data/reference session，删除生产同步旧入口。
- [x] Phase 5：在GPU5–9完成第二轮correctness、fault、1/2/3/4/5卡weak scaling、1/2/4卡strong scaling及65-step跨group稳态门。

用户已批准阶段A+B以及扩展后的Phase 4/5；公共实现、原生Linux GPU5–9验证与规范沉淀均已完成。具体数值、artifact与剩余formal训练边界记录在`research/implementation-verification.md`。

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

- 扩展 `tests/integration/test_distributed_training.py`，覆盖两卡 synthetic NCCL correctness、rank-0 action、descriptor/rank failure 与 bounded teardown。
- 用 Metal smoke 验证两phase、checkpoint/resume与rank0-only artifact；用long config停在至少第65步，分离cold build、warmup和跨64-step group boundary的steady window。
- 对同一per-rank batch执行1/2/3/4/5卡weak scaling；只在route batch精确等分的1/2/4卡执行strong scaling。
- 保存 per-rank stage trace、DDP logging、NCCL diagnostics、resolved plan 与完整环境 manifest。
- 注入 rank data sleep、reference exception、rank-0 write delay/exception 和 phase descriptor mismatch。

### Gate

- correctness、checkpoint/resume、failure propagation 与 teardown 是 hard gate。
- 除R13明确冻结的“2卡与5卡global work/s高于单卡”外，GPU activity、并行效率、显存与bucket overlap均为observed report，不预设脱离机器的数值硬线。
- timeout 必须能定位为 data/reference、reducer/desync、checkpoint I/O 或 teardown 中的具体阶段和 rank；不能只得到一条无上下文 watchdog 消息。

## 4. Phase 4：data/reference pipeline

### 改动

- 将公共session合同改为有界`submit_step/acquire_step/release/drain`，engine只提交同phase、同group且不跨validation/checkpoint/stop边界的lookahead。
- 给唯一`OnlineTrainingProducer`增加批量step入口；先冻结每个logical request与RNG，再让`ReferenceScheduler`合并同group evaluator dispatch，最后按ID切回typed batch。
- 接通ready ring、现有`HostPipeline`、`GpuResidencyManager`、reference lease与`PipelineTrace`；host work可提前提交，CUDA/Falcor操作仍由rank主进程拥有。
- Vulkan `global-sync`模式只做packed dispatch与连续model消费，不并发同卡Falcor/model；stream-fence capability才允许真实`reference_inflight`。
- 在group boundary执行显式warmup/host-only预取；先以trace定位rank skew，只有证据支持时才增加确定性cost-balanced schedule，并把其identity纳入checkpoint。
- CLI切到新session后删除`SynchronousOnlineDataSession`、旧export与只验证旧类的测试；`num_workers=0`、`ready_batches=1`、`reference_batch_steps=1`由同一实现表达baseline，不建compatibility alias。
- 增加静态依赖测试：engine/model/method/data contract不读取OS、物理GPU、Falcor API或backend key；launcher/backend只向上提供`ExecutionContext`与capability。

### Gate

- `reference_batch_steps=1` 与 packed 模式逐 logical step 的 request/RNG/top-up/target/resume 一致。
- queue/residency 有界，checkpoint/phase/stop drain 后 cursor 不超前。
- target Linux trace 证明 barrier/step 或 consumer starvation 有可解释改善；不把 Vulkan timeline 未重叠本身判为失败。
- 生产入口只有一个session实现，Windows/Linux上层调用图一致；不保留旧同步compatibility路径。

### 重点文件与rollback

- `src/ncls/data/contracts.py`、`session.py`、`reference_scheduler.py`：拥有step queue、ready ring、lease与drain；若identity/释放语义不成立，回退本批全部session改动，不修改checkpoint猜测兼容。
- `src/ncls/learning/producer.py`、`training/engine.py`：统一批量生产与phase barrier；不得增加Metal或Linux专用分支。
- `src/ncls/cli.py`、`src/ncls/data/__init__.py`：一次切换并删除旧入口。
- `tests/unit/test_online_data_session.py`、`test_data_pipeline.py`、`test_reference_scheduler.py`、`test_training_engine.py`与Linux integration：逐层保护顺序、backpressure、fault、resume和跨平台边界。

## 5. Phase 5：目标机最终gate

### Gate

- 正常1/2/3/4/5卡weak-scaling run无timeout；2卡和5卡post-warmup global work/s高于单卡matched baseline。
- consumer starvation、ready depth、reference dispatch/barrier、rank min/mean/max与straggler能够解释剩余效率；GPU activity、显存与并行效率作为observed report。
- rank data sleep/reference exception、rank0 write delay/exception、descriptor mismatch均在有界时间内失败，产物不误标success。
- 两phase、stop/resume、rank0-only checkpoint/summary/review及同序teardown通过；1/2/4卡strong scaling单列，不与weak scaling混用。

## 6. 验证命令

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
bash scripts/run_falcor_python.sh --gpus 5,6 -- \
  -m pytest tests/integration/test_distributed_training.py -q
bash scripts/run_falcor_python.sh -m ncls train configs/training/runs/metal-linux-smoke.yaml --devices 5
bash scripts/run_falcor_python.sh -m ncls train configs/training/runs/metal-linux-smoke.yaml --devices 5,6
```

### Linux profile

weak scaling对同一frozen per-rank配置依次使用`--devices 5`、`5,6`、`5,6,7`、`5,6,7,8`、`5,6,7,8,9`。strong scaling只使用独立冻结且route batch可精确等分的1/2/4卡配置。性能run不设置`NCLS_DDP_DEBUG=1`；debug/fault run单独保存。timeout、warmup、step、batch/work单位和产物目录在preflight后冻结到任务research，不事后改口径。

## 7. 开始实现前检查

- [x] 用户已审阅 `prd.md`、`research/ddp-audit.md`、`design.md` 与本计划，并明确批准实施范围。
- [x] 重新运行 `trellis-before-dev`，加载 project/data/learning 规范。
- [x] 检查 dirty worktree，逐文件确认没有覆盖用户或其他任务改动。
- [x] Phase 1 只改 reducer/objective，不夹带 data pipeline 优化。
- [x] 目标 Linux reference 环境、GPU5–9和新的`artifacts/09-04-linux-ddp-diagnosis/`目录可用。
- [x] 用户审阅本次扩展后的PRD/design/implement并明确批准Phase 4/5实施。
