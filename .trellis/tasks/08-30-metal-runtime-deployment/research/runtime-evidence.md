# Metal runtime 部署证据

## 证据对象

本次 correctness run 固定到以下可追溯对象；它们位于忽略的 `artifacts/`，不作为根仓库源码提交：

- checkpoint：`artifacts/08-30-metal-runtime-deployment/full-method-runtime-verified-v7.pt`；SHA-256 `b630880451141adbe1669083de4f699e44f9b7e81e3bb837e95c73f59083f9db`；
- package：`artifacts/08-30-metal-runtime-deployment/package-v7`；package id `4eeadff233cfd88373dea186aeb4cc68106b8278db82daa521457e011690622e`；
- source snapshot：`3930c77f97b4514f29e19b671702ebfff415e7fe6af98ce40349c19e9b3cdbb4`；
- program / asset / instance id：`6e3773cb135cd15f6bcc30595ebd27831668a1fb2c4460fa551513781d415fea` / `c70137593f2012d777ff5ced7c8c30c6a318d478e60acd1abde41695e526f7a0` / `42feb26a113be372c2b60e06bf024f674b4a1f6c357dc41f554233052edec784`；
- viewer capture：`artifacts/08-30-metal-runtime-deployment/viewer-capture-v7-local-pt-deferred/capture.json`。

`scratch/audit_runtime_evidence.py`同时核对 package/checkpoint/capture 身份、训练数值、gradient coverage、linear EXR 和静态 MAC 记账。它不会把观察到的时间或图像差异升级为验收门。

## 完整方法与训练正确性

Windows correctness config 保留 `metal_fused_full_v1` 的全部结构和 13 个 parameter groups，只缩短 online work units：三阶段各 4 step，总计 12 step、60 work units。它不是缩模，也没有换掉 source、loss、compiler、evaluator 或 sampler。

- 13/13 parameter groups 都观察到 finite、non-zero gradient 和 parameter update；proposal group 最后审计于 step 11，其余 group 最后审计于 step 7；
- 13 条 training/validation records 的全部数值有限；最终 checkpoint 为 `phase_name=complete`；
- 峰值已分配 GPU memory 为 258,658,304 bytes；训练摘要记录 7.372 s wall time；
- proposal 的 sample→pdf identity 在训练与 validation 记录中均为 `0.0`，valid fraction 为 `1.0`；
- 这些短程 loss 只证明完整梯度路径可执行并发生更新，不宣称材质族已经收敛，也不作为质量结论。

## Runtime 计算与存储

以下为 full profile 的静态 MAC（一次乘加计为一个 MAC），不含 normalization、activation、transcendental、frame construction 和解析 BSDF 的 scalar 运算：

| 生命周期 | MAC |
|---|---:|
| typed compiler，每次 native typed edit 一次 | 6,974,528 |
| decoder，每个 semantic domain、每个 mip | 165,888 |
| 当前资产 7 domains × 2 adjacent mips | 2,322,432 |
| prepare heads，每个 shading point | 93,568 |
| `prepare()` 合计 | 2,416,000 |
| `evaluate(wi)`，每个方向 | 185,088 |
| `prepare()` + 一次 `evaluate()` | 2,601,088 |
| 已 prepare 后一次 `sample()` 的 neural 部分 | 185,088；proposal 本身另含解析运算 |
| 已 prepare 后一次 `pdf()` 的 neural 部分 | 0；仅解析 mixture PDF |

静态随机访问上限 106 次来自 `9 texture slots × 2 mips × (4 high-grid gathers + 1 low-grid filtered read) + 4 angular levels × 4 gathers`。它只记 texture/angular table 的随机访问，不把 890,965 个 FP16 shared parameters 的顺序 weight loads 混入“texture reads”。prepared state 的声明上限为 2,816 bytes。

当前质量优先 package 的逻辑存储为：

- `B_shared = 1,781,932` bytes（890,965 FP16 parameter elements + 2 bytes 对齐）；
- `B_asset = 439,710,464` bytes；
- `B_instance = 2,880` bytes。

当前 `B_asset` 和单次运行开销明显还不是最终 Pareto 点。这是后续训练看完质量后再做 codec/grid/结构消融和部署优化的基线事实，不是本任务临时设置的 kill gate。

## Slang、package 与 viewer

- Python quantized oracle 与 Slang 对完整 compiler、two-mip decoder、prepare/evaluate 进行 full-shape GPU parity；GPU 测试为 `2 passed`；
- `ScatteringPackage@2` loader 已验证 package，program/asset/instance 三层资源和 editable compiler ABI 均 fail closed；
- viewer program cache 以 `programId` 为键；slot 只持有 asset/instance binding；typed edit 在 candidate buffers 中上传 raw state、运行一次 compiler，成功后再原子替换；
- MDL compatibility 使用 canonical `sourceSnapshotId`，因为该身份包含 export、typed defaults 与 transitive resources；非 MDL family 仍使用 native asset hash；
- full Slang 循环使用有静态 `MaxIters` 的 `[loop]`，避免 DXIL 编译器展开完整大矩阵；没有删层、减宽或跳过权重；
- Windows viewer 对 package pass 统一使用 8×8 workgroup tile，并在 tile 间同步提交以规避 TDR；deferred 与 PT 共用同一实现，没有 Metal renderer branch。

真实 viewer capture 的每个 panel 为 160×240。slot 0 为 1 spp、scene bounce cap 0 的 package PT，slot 1 为 package deferred；只启用 sun，两个 slot 均为 `ready`。三张 linear EXR 均为 finite RGB：slot 0 最大值 1.937240，slot 1 最大值 1.930397，绝对差均值 0.023942、最大值 1.206252。观察到的 GPU 时间分别为 64,749.746 ms 和 37,207.513 ms；这说明当前质量优先实现非常慢，不能被误读成实时性能结果。

## 可复跑命令

```powershell
conda run -n neural-shading python .trellis/tasks/08-30-metal-runtime-deployment/scratch/audit_runtime_evidence.py --package artifacts/08-30-metal-runtime-deployment/package-v7 --checkpoint artifacts/08-30-metal-runtime-deployment/full-method-runtime-verified-v7.pt --metrics artifacts/08-30-metal-runtime-deployment/full-method-runtime-verified-v7.metrics.jsonl --capture artifacts/08-30-metal-runtime-deployment/viewer-capture-v7-local-pt-deferred/capture.json
conda run -n neural-shading python -m ncls.cli package validate artifacts/08-30-metal-runtime-deployment/package-v7
scripts/run_falcor_python.ps1 -m pytest tests/gpu/test_metal_runtime_package.py -q
scripts/build_viewer.ps1 -Configuration Release
```

## 最终质量门

- `conda run -n neural-shading python -m pytest tests/unit -q`：180 passed；
- `scripts/run_falcor_python.ps1 -m pytest tests/gpu/test_metal_runtime_package.py -q`：2 passed；
- `conda run -n neural-shading python -m ncls.cli package validate artifacts/08-30-metal-runtime-deployment/package-v7`：package 与三层 identity 验证通过；
- `scripts/build_viewer.ps1 -Configuration Release`：Release executable 重编译通过；
- `scratch/audit_runtime_evidence.py ...`：package/checkpoint/capture identity、13/13 gradient groups、finite metrics 与 linear EXR 全部通过；
- `git diff --check`：通过；`git -C external/Falcor status --short`：无输出，overlay 已撤回且上游工作树干净。
