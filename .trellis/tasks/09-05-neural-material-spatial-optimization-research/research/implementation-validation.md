# 原生多 UV 实施检查点

本文是归档前的实际验证记录。用户随后要求按实现阶段提交归档，剩余 D0/D1 与质量研究已转交服务器任务，最终边界见 [closure](../closure.md)；下文的 `in_progress`、未提交和不归档是该检查点当时的状态。

本次继续推进 P1–P4，已把 raw encoder、训练资源、流式 cook 和新 Slang ABI 接通，完成划痕青铜 0→2 step 的公开入口 smoke。任务仍为 `in_progress`；这不是 D1 matched 质量结果，也没有执行 D1b/D2。

## 环境与代码范围

完整 Windows：RTX 4090、`neural-shading`、锁定 Falcor Windows Python 扩展。本次使用项目 Conda 环境；Linux/NCCL 未实机验证。起点 HEAD 为 `e3f1c21`，保留已有工作树修改和其它任务资料，没有 commit/push。

当前原始资源使用 `AdaptedConditioning` 的 resource/binding 合同。Metal CPU 预定 UV/RF cohort，GPU 生成 query；主/pair 共用编码图，下一次参数更新后重新编码。RF split 先冻结 8 个 held-out tile，再选 32 个训练 tile，保守 raw texel 占用集合不相交，非空间原生表保留为共享 source 条件。

按原生 UV 表达式分别编码；原生 nonrepeat 使用三位置坐标与权重。每组 Detail/Context 的真实四邻 SNORM texel 读取进入 prepare。cook 逐层流式推理，与训练 tile hierarchy 对照，覆盖 odd/non-square/1×N、不同原生尺寸、wrap/clamp 和多 slice lookup。

## 正确性与回归

| 检查 | 实际结果 | 证据范围 |
|---|---|---|
| 全量 `tests/unit` | 366 passed，49.31 s | 包含 resource 移交、raw adapter、RF split、C6 fixture、cook、配置及现有架构回归；日志 `scratch/unit-final.log` |
| 随后新增的 C3/visual phase 回归 | 11 passed，3.75 s | `test_metal_model_correctness.py` 与 `test_visual_eval_hook.py`；覆盖有效零/NaN、step 0/2/最终 phase 标签 |
| 新 spatial runtime + reference GPU | 2 passed，104.70 s | 两 UV 组、非方 Jacobian、fractional LOD、nonrepeat、FP16/SNORM、sample/pdf 与独立 reverse prepare；非对称 source 的绝对颜色和 16/64 点完整 response 平均 |
| 增加 C3 故障注入后的 spatial GPU | 1 passed，68.98 s | prepared semantic 注入 NaN 后 evaluate/sample 维持 invalid，半球外返回 invalid；日志 `scratch/gpu-spatial-final.log` |
| layout 生成一致性 | `generate_metal_budgeted_layout --check` 通过 | v2 JSON 与生成 Slang 同步 |
| Release viewer | 重试成功，Falcor 工作树干净 | 首次构建与 GPU 测试并发导致 DLL 被占用，退出测试后按原脚本重试；日志 `scratch/viewer-build-retry.log` |

此前本任务已通过 C1 初值/零 delta、C4 连续 frame、C2 正常数 softplus 尾部等测试，见 `progress.md`。GPU parity 未放宽已有容差。C6 单测使用真实 CPU cook、冻结模型和不在训练列表的 snapshot，但 adapter 为受控 fixture；实际另一资产的未见 source 质量尚未测量。

## 公共入口的真实 smoke

实际 run：`outputs/metal-spatial-probe-bronze-scratched/260905-220316-6185cc/`。

1. `train 0 --config configs/training/runs/metal-spatial-probe-bronze-scratched.yaml --stop-at-step 0` 完成 train-only 16,384 行 calibration 与初始化保存。
2. 从本 run 的 `checkpoints/latest.pt` 恢复到 step 2，optimizer/query 状态可恢复，训练报告正常结束。`logs/summary.json` 记录实际 elapsed 40.41 s；包含 RF 冻结、warm-up 和保存，不能当稳态吞吐。
3. `validate .../checkpoints/latest.pt --batches 1` 完成，mean loss=1.47487891。单 batch smoke 的 loss 不作为质量结论。
4. `export .../checkpoints/latest.pt --material-index 0` 成功，输出 `exports/step-00000002/material-0/`。
5. `eval .../checkpoints/latest.pt --config configs/training/runs/metal-spatial-stage-eval.yaml` 成功，输出 `eval/step-00000002-2d716c/`。

实际源有两个 nonrepeat UV 组，以及 BSDF/颜色非空间 lookup。4K 源的 raw preflight（不生成 GT）完成 GPU B=8 forward/backward，Torch peak allocation 3,484,646,912 B；不是 B=128 正式训练峰值。公开 smoke 的参数审计显示 asset encoder、typed compiler、semantic prepare、directional evaluator 有 finite/nonzero 梯度和实际更新；proposal 按配方停用，无 sampler 质量训练。

| 当前编译成本 | 划痕青铜实际值 |
|---|---:|
| prepare / evaluate dense MAC | 7,664 / 11,392 |
| packed prepared state | 176 B |
| instance | 768 B |
| runtime shared weights（含 word 对齐） | 38,744 B |
| 有效 latent texel 数据 | 190,141,776 B |
| prepare latent reads | 12 |

latent 字节数不含 DDS header 和空组占位纹理。保守 profile 上限为 9 UV 组、54 reads；实际单次查询时间尚未 benchmark，不能从 MAC 或 viewer 单帧推断实时成本。

## 图像与线性输出

capture 为 640×360，两个 panel 各 320×360，source reference 为 PT 128 spp，neural 为 deferred。两个 slot 均为 `ready`；显示 exposure=0、difference scale=8，完整相机、灯光、source/state/package/geometry identity 保存在 `capture.json`。

已打开 `capture-display.png`：step 2 的 neural 尚未表达参考中的青铜颜色与划痕，不能作为质量改善证据。source 与 neural 的渲染模式和过滤路径不同，此图只验证部署与 capture 生命周期。capture 记录 source MDL 为 explicit-lod0、未消费 UV derivatives；它不能替代 D0 的 dispatcher footprint witness。

`scratch/check_spatial_capture.py` 检查四个线性 EXR，读取为 float32，非有限值均为 0；具体 shape/range 见 `spatial-capture-check.json`。neural slot 仅一个 GPU timing sample，2.004992 ms 只作为本次 capture 观测，不作为查询延迟或 Pareto 数据。

该次 eval 产生的 package 正确记录 checkpoint step=2，但公共 visual evaluator 创建临时 checkpoint 时沿用了默认 `phase=initialization`。已修复后续 eval 的 phase 标签并通过 step 0/2/512 回归；没有覆盖本次原始 capture 或为了标签重复 cook/render。该 package 的空 gradient coverage 表示未透传诊断，不能解释为训练没有更新；训练导出包中的实际审计仍可追溯。

## 修复的边界问题

- scheduler 对 payload 的 release 包含 conditioning resource。移交 ready batch 前必须 retain 独立 owner；此前只看 adapter/dispatcher 的测试遗漏了这一层。新增完整 scheduler→producer→consumer 回归。
- evaluator-only 配方禁用 proposal 时，`Method.requirements(config)` 只要求启用的 route。未知 route 继续失败，不建立空 sampler 流。
- 当前 summary-control 的占位 profile 曾与 raw CNN 共用同一构造；现明确拒绝，防止无效 matched 对照。历史 hybrid recipe 也显式保留原 profile/correspondence，新入口拒绝静默套用 raw adapter。

## 下一次实施起点

1. 完成三个实际 source 的 D0 记录，补齐原生 normal/参数顺序和其它材质 raw/RF 的独立证据；保留 point/16/64 footprint 的差别。
2. 实现真正修正的 summary-control，与 raw 使用相同 UV grouping、read-plan、下游、量化、source/query 和训练预算；目前没有创建可运行的假对照。
3. 在上述条件具备后执行已冻结的 6×512 step D1，按 tile/query block 做 matched 指标与 bootstrap CI，并记录独立 cook/prepare/evaluate 成本。不得从本次 step 2 smoke 自动续成长实验。
4. 根据 D1 证据选择已批准的 D1b/D2 条件分支；Linux/NCCL、真实未见资产质量单列。任务不归档、不宣称 P4 完成。
