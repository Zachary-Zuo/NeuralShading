# 01 可复用散射数学原语实施计划

## 0. Planning Gate

- [x] 父任务产品、范围、兼容性与风险边界已冻结，本 child refinement 未改变这些决策。
- [x] 仓库公式 inventory 已完成；NVIDIA 官方 supplemental Listing 3/4 已核对并记录在 `research/math-inventory.md`。
- [x] PRD 已完成 requirement convergence 与 lossless convergence pass；blocking open questions 为空。
- [x] `design.md` 已冻结公共所有权、sample/null 状态、数学变换、迁移顺序和阈值先于结果的测试协议。
- [x] 连续执行 metadata 与父任务一致，final planning summary 已记录在本节；满足 `task.py start` 的既有授权边界。

Final planning summary：目标是先建立唯一 Falcor-free math 层并证明 sample/PDF/null 与 LayerStack reference 不漂移；范围包括公共公式、NVIDIA 9 参数 proposal、analytic control core、旧调用者迁移与 GPU 数学 oracle；不包括训练、bundle/viewer 和旧身份删除。关键风险是变换/Jacobian 与 RNG 漂移，分别由 fixed quadrature/histogram/re-evaluation 和迁移前后固定 seed probe 阻断。所有 AC 均有可执行 gate，无 deferred technical TODO。

## 1. 开发环境与基线

- [x] 按 `.trellis/spec/project/dev-environment.md` 判定为完整开发机：RTX 4090、`neural-shading`、Falcor Python、原生 Windows 均存在；Falcor HEAD 为 `9dc819c1…` 且工作树干净。
- [x] 在任何产品代码改动前运行固定 seed LayerStack reference probe；入口为 `scratch/reference_baseline.py`，输出为 `artifacts/validation/01-scattering-math-reference-baseline.json`。
- [x] 记录当前基线：`test_legacy_ltc_k2.py` + `test_scattering_contract.py` 为 11 passed；legacy GPU parity + reference physics/shard 为 5 passed。

## 2. 公共 math 模块

- [x] 新建 `shaders/ncls/scattering/{common,frame,cosine,ltc,ggx,fresnel,mixture,nvidia_proposal}.slang`，并在 `docs/architecture.md` 登记目录职责。
- [x] 实现统一 `NclsDirectionalSample` 的 continuous/null/invalid 状态。
- [x] 实现 canonical cosine、tilted cosine、LTC sample/PDF/response 与 K=3 mixture select/remap/full PDF。
- [x] 迁移 GGX `D/G/lambda` 与 VNDF，增加预抽 `u` pure overload。
- [x] 按官方 Listing 3/4 实现 NVIDIA raw decode、non-centered anisotropic GGX NDF 和带固定 safety cosine 的完整 proposal。

## 3. 调用者迁移

- [x] `reference/sampling.slang` 只保留 RNG/HG/reference adapter；旧 public function name 继续从公共层可见。
- [x] `reference/interfaces.slang` 使用公共 frame/GGX/Fresnel 与 pure sample，同时保持每个 branch 的 RNG 消费顺序。
- [x] `legacy_ltc_k2.slang` 调公共 frame/cosine/LTC，packed ABI、descriptor 和 contract behavior 不变。
- [x] `lobe_residual_mlp.slang` 删除 LTC basis 公式副本并调公共层；不补旧 sampler TODO。
- [x] 新建 `ltc_k2_analytic_control_core.slang`，提供 exact-top + K2 LTC evaluate core，不注册 fallback。

## 4. GPU Oracle 与回归

- [x] 新建 `tests/gpu/kernels/scattering_math.cs.slang` 和 `tests/gpu/test_scattering_math_gpu.py`，直接包含生产 math source。
- [x] 完成 fixed quadrature normalization、sample histogram、`sample.pdf == pdf()`、mixture component frequency、VNDF normal/null、boundary finite 测试。
- [x] 增加 analytic control compile/evaluate probe，并保持 legacy Python/Slang parity。
- [x] 重跑固定 seed LayerStack probe，与改动前 artifact 对比；运行 reference physics/shard smoke。

## 5. 验证命令

所有 Python/pytest 只通过项目环境，Falcor 测试只通过项目 wrapper：

```powershell
conda run -n neural-shading python -m pytest tests/unit/test_legacy_ltc_k2.py tests/unit/test_scattering_contract.py -q
.\scripts\run_falcor_python.ps1 -m pytest tests/gpu/test_scattering_math_gpu.py tests/gpu/test_legacy_ltc_k2_gpu.py tests/integration/reference/test_reference_physics_gpu.py -q
conda run -n neural-shading python -m pytest tests/unit -q
.\scripts\run_falcor_python.ps1 -m pytest tests/gpu -q -k "sampling or scattering or reference or legacy_ltc"
git -C external/Falcor rev-parse HEAD
git -C external/Falcor status --short
```

若完整 GPU suite 的既有无关失败出现，必须用 focused reproduction 判定归属并记录；本任务引入或触及的 gate 不得留待后序 child。

实现结果（2026-08-26）：focused unit 为 11 passed，完整 unit 为 85 passed；final reviewer 补入独立 Listing 3/4、LTC Jacobian、sample state 与 analytic identity oracle 后，公共数学 GPU 为 20 passed，和 legacy/reference 合并为 25 passed，filtered GPU gate 为 25 passed / 1 deselected，完整 GPU 为 26 passed。固定 seed baseline 的 before/after SHA-256 均为 `312C75235232FCB0E14959506AFC2D098270C3FA5988274BF6D7C9607D03161C`，因此 mean/variance/sample count 字节级一致。合并 GPU 收集曾因 Torch 与 Windows BLAS 同进程下的 `np.dot` fatal abort 失败，测试已改为不调用 BLAS 的 Gauss-Legendre recurrence 与逐元素求和，保持阈值不变后通过。`scripts/build_viewer.ps1 -Configuration Release` 成功构建并复制全部公共 math 与 analytic control shader，overlay 反向应用后 Falcor 仍为锁定提交且工作树干净。

## 6. Quality、Spec 与提交

- [x] 使用 `trellis-check` 复读 core/data Quality Check，审计数据流、公式唯一性、include 依赖和测试 oracle 非同源性。
- [x] 使用 `trellis-update-spec` 判断 `shaders/ncls/scattering/` 所有权、null-event 与 pre-drawn RNG 规则是否需要长期固化。
- [x] 记录 dirty-path 归属与逻辑提交计划：一个 `refactor(core): 建立可复用散射数学层` work commit，包含 01 task/research/scratch、公共 math、reference/legacy 调用者、GPU oracle、CMake、architecture 与 core/data/project spec；唯一未识别 dirty 文件 `SmileySans-Oblique.otf` 明确排除。
- [ ] 直接创建 scoped local commit，不 amend、不 push；随后运行 `trellis-finish-work` 归档并确认 archive/commit provenance。

## 7. Rollback Points

- 公共 pure module 首先独立新增；迁移每个调用者后立即跑 focused compile/parity，便于定位而不复制旧公式。
- normalization/Jacobian 失败时回到对应 distribution 设计，禁止用 rejection/resampling 或放宽阈值绕过。
- reference probe 漂移时撤回最后一个调用者迁移单元并核对 random draw 次序；不改 reference GT 语义。
