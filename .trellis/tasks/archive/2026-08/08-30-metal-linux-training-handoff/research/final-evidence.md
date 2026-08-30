# Metal Windows 验证与 Linux 单 GPU 交付证据

## 结论

`metal_fused_full_v1` 已在 Windows/RTX 4090 上以完整模型 shape、真实 GPU-resident online reference、真实 optimizer 和四阶段训练图完成数值/梯度/checkpoint-resume/部署闭环。为了缩短正确性验证时间，只压缩了 source/query/step 数；没有关闭组件、缩小网络、持久化训练 batch 或用 mock target 替代 reference。Linux 原生 smoke 与 long run 尚未在目标机执行，交付状态保持为 `pending-on-target-host`。

## 冻结身份

- full-cohort preflight identity：`4f402b95138a4d4f3ad5b65427d49b83da54783b9aac5bb71f07be95b5f9eb58`
- Metal registry identity：`fa6642e60d469231839756d749283b3d7d93e7163284c4094837770379dec8cc`
- Linux smoke/long semantic fingerprint：`59c84c80ecf19a2ade3642b869e2c9767f925b1f95c37d88c819bc98afbbdf7b`
- Windows smoke config SHA-256：`6319496097d81deed81838ec479fe1f8c0040ff13563c494dbe9985bf0993f92`
- Linux smoke config SHA-256：`a136a7a95e32acb6d529a7e45b59af47cd163091051043c7726b186e3755abe6`
- Linux long config SHA-256：`d4e5d32a7037df882fa986b5207405099d47101ce7b73372b2caf449524898e8`
- method descriptor SHA-256：`be8c4e721ce6eafd8b0a777fda732da6292f7471a13778eea73bcd43840fdfdc`

full-cohort preflight 闭合了 692 个 exports、178 个 execution groups、52 个 texture sets、64 个 schema table entries、4 个 texture roles、6 类 typed parameters 和 11 个 proposal components。Windows optimization 使用机械选出的 3 个 activation exports 与资产索引 `[6, 50, 22]`；它覆盖 20 个 required components，同时保持与全 cohort 相同的 method、route、loss、precision、optimizer 和模型 shape。

## 四阶段 optimization 与数值证据

最终 checkpoint 为 `artifacts/08-30-metal-linux-training-handoff/windows-four-phase-final.pt`：

- SHA-256：`7d6d31a297787639987dec60220f582a0a69d0ae7717d5ad89284ef8bcb56f6d`
- 大小：26,480,233 bytes
- 先运行到 global step 13（进入 `qat-refine`），再从同一个 checkpoint 恢复到 step 16；phase、query、optimizer 与 precision state 连续；
- 13 个 parameter groups 全部得到 finite、非零 gradient 和实际 update；training-only teacher 最后一次合法 audit 在 step 7，全部 runtime groups 最后一次 audit 在 step 15；
- 四阶段均无 NaN/Inf，QAT runtime trace 非零；QAT 以 FP32 master weights、FP16 STE runtime rounding 和既有 INT8 grid STE 执行，不是 phase alias。

随机 online 短跑只作为健康报告；proposal 段因窗口很短而有噪声，因此另按预冻结方法执行 fixed-query-stream micro-overfit。每个 repetition 都重新调用 authoritative reference，只复位 query state，不保存 response batch，`target_f` 始终在 CUDA 上。12 次 repetition 的末段减初段 loss delta 均为负：

| phase | loss delta |
| --- | ---: |
| `codec-warmup` | -0.12928905 |
| `joint-appearance` | -0.69964365 |
| `proposal-fit` | -0.42879959 |
| `qat-refine` | -0.13873275 |

证据 identity：`ea4f0eaa108f5953e32600f376e57c932fd81a098348965bb44693f46ae919c0`。最终 online evaluation（1 batch）记录 `mean_loss=2.21142364`；它是正确性诊断，不是收敛质量结论。

## profile 与 artifact 闭环

training review identity 为 `5e2e6a6b361d65ec60581d95f820c79aef55e53efd711762f8611c07997b150d`。跨 resume 的累计训练时间为 16.4854 s，logged median throughput 为 0.9952 steps/s，peak allocated memory 为 403,271,168 bytes。分项 median：

- online batch/reference submit wall：0.291815 s；
- forward GPU：0.133723 s；
- backward GPU：0.147494 s；
- optimizer GPU：0.015490 s；
- validation wall：0.376084 s。

最终 diagnostic package 位于 `artifacts/08-30-metal-linux-training-handoff/windows-four-phase-diagnostic-package`，package id 为 `949c29b7309bb4479a9bc21701c174c36e85e51ff18e16e47388a94acf43c6c8`。其 program/asset/instance/source snapshot 分别为 `7a92a2...`、`3bbfb0...`、`9c89db...`、`f3a065...`；package validator、sample/PDF invariants、typed edit 与 bundle replacement 均通过。

Release viewer capture 位于 `artifacts/08-30-metal-linux-training-handoff/windows-viewer-capture-final/capture.json`。两个 slot 均 ready，PT 1 spp/bounce 0 与 deferred 输出均生成 finite linear EXR；两路 absolute difference mean 为 0.0233725、max 为 1.022639。viewer GPU 时间约为 64,777.84 ms 与 37,208.22 ms，明确保留为当前诊断结果，不设置虚假的实时门槛。

## Linux 交付边界

`configs/learning/metal-fused-full-linux-smoke.json` 与 `configs/learning/metal-fused-full-linux-long.json` 都显式包含 692 个 exports 和 52 个 assets。两者的 semantic fingerprint 相同，只允许 budget/cadence 差异；long profile 为 120,000 steps（20k codec、70k joint、15k proposal、15k QAT）。命令与恢复流程见 `docs/metal_linux_training.md` 和最终 handoff manifest。

handoff 强制单进程、一个可见 GPU，不包含 DDP。目标机必须依次完成 source/config hash 检查、Falcor/Vulkan preflight、Linux-native smoke，成功后才允许开始 long run。long run 结束只生成 `ncls.training-review@1` 首轮审阅摘要，`automatic_followups=[]` 且 `next_action=user-review-required`；formal matrix、ablation、distillation、compact 与 Pareto 不会自动启动。

## 回归检查

- `tests/unit`：186 passed；
- relevant GPU：9 passed、35 deselected；
- Release viewer build：通过；
- `bash -n scripts/deploy_reference_linux.sh scripts/run_falcor_python.sh`：通过；
- Python `compileall`：通过；
- `external/Falcor`：clean；
- `git diff --check`：通过（仅 task JSON 的 CRLF→LF 提示）。

以上 Linux static checks 不能代替 Linux 原生 Metal smoke；该结果必须由目标机写入其 `artifacts/` 后才能升级状态。
