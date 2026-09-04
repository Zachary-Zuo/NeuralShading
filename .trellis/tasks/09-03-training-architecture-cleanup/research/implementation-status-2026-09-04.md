# 训练架构实施状态与验证证据（2026-09-04）

## 1. 已落地结构

- 用户入口改为 `ncls train/validate/export/eval`；run 只接受 `configs/training/runs/*.yaml`，公开 method key 为 `nvidia`、`metal`。
- `TrainingPlanResolver` 组合 base/method/data/recipe 并冻结输入文件、method descriptor、implementation 与六个 facet identity。
- NVIDIA 与 Metal 均由显式 `MethodPlugin` 接入；公共 engine/producer 不通过旧 implementation key 查找 method。
- `TrainingEngine` 只依赖 `OnlineDataSession`，固定 phase、step、validation、checkpoint 与 hook 生命周期；checkpoint 前显式 `drain()`。
- 新训练只写 `TrainingCheckpoint@1`。旧 v4 仅由 `LegacyCheckpointV4Importer` 生成只读 `EvaluationSnapshot`，不能 resume。
- `HostPipeline`、`GpuResidencyManager`、`ReferenceScheduler` 和 `PipelineTrace` 是共享 data-plane primitive。Metal source typed metadata 常驻 GPU，resident patch hit 路径不再把逐 step `asset_index/uv/mip` 回读 CPU。
- TensorBoard 和 visual eval 都通过 typed event/hook 接入。visual job 使用独立 probe identity、原子 spool 与可迟到 collector。

## 2. 1024 spp 开销修正

最初把 reference 与 neural 都作为 path-tracing slot 推到 1024 spp，真实运行超过 20 分钟仍未完成；这一 cadence 成本不可接受。随后冻结两种不同用途：

- 常规 `training-diagnostic`：reference 保持 1024 spp path tracing；neural 使用同相机/灯光的 deterministic deferred，`target_spp=spp=0`。
- 手工深度检查：可显式选择低 spp neural path tracing；双侧 1024 spp 只作极低频人工检查。

Windows RTX 4090 实测：旧 reference 1024 + neural PT 16 为 138.781 秒；新 reference 1024 + neural deferred 为 12.429 秒，约 11.2 倍加速。产物位于 `artifacts/training-architecture-cleanup/nv-v8/`。capture manifest 记录 `comparison_purpose=training-diagnostic`，slot 0 为 PT 1024/1024，slot 1 为 deferred 0/0；三个 linear EXR 均为 finite float32。`capture-difference.png` 的显示异常属于既有 PNG 导出问题，权威 difference EXR 正常，本次未把它误报为已修复。

## 3. Windows 验证

环境：完整 Windows；GPU=NVIDIA GeForce RTX 4090；`neural-shading` 环境与锁定 Falcor Windows build 存在。

- unit：`292 passed`。
- GPU：`47 passed`；新增 resident sampling 定向集合为 `6 passed`。
- integration/reference：`5 passed`。
- `compileall` 与 `git diff --check`：通过。
- viewer：Release overlay build 通过，结束后 Falcor 回到锁定 commit 且工作树干净。
- 新架构真实两步 smoke：`artifacts/training-architecture-cleanup/nv-v10/checkpoint.pt`，step 2 complete，checkpoint v1 SHA-256 为 `8975db57bf569b8599c65f533ced5ceed83e7e583e7026c23fabed89d4eb8693`；随后 `ncls validate --batches 1` 得到 finite mean loss `0.191624448`。
- TensorBoard event 可读取，train scalar step 为 `[1, 2]`，包含 loss、learning rate、吞吐、显存、GPU timing 与 reference profile；无 hook failure。

这些数值只证明当前 Windows 实现和生命周期，不是 formal 模型质量结论。

## 4. 尚未完成的目标机证据

当前没有原生 Linux/Vulkan/NCCL 目标机上下文，因此以下项目不得标为已验证：

1. `metal-linux-smoke.yaml` 的单卡 full-cohort before/after stage trace；
2. `--devices 0,1` 的真实 NCCL 两卡、rank device mapping 与 rank-0-only 输出；
3. host decode/transfer 与 model 的真实 timeline overlap；
4. `reference_batch_steps > 1` 的生产 packed dispatch 与 GPU batch ring；当前正式 YAML 仍锁定安全基线 1；
5. A6000 目标机的吞吐、GPU activity、barrier/step、peak residency 与新 long-run ETA。

这些是任务 AC11/AC11b 的剩余 gate。Windows 结果和历史约 52 小时 ETA 都不能替代；拿到目标机后按 `TESTING.md` 的 Linux 矩阵运行并把 artifact identity/observed profile 回填到本文件，再决定是否启用大于 1 的 reference batching。
