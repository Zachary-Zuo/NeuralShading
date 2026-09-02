# 测试说明

所有 Python 命令使用唯一 Conda 环境 `neural-shading`。需要导入 Falcor Python 模块的测试必须通过平台 launcher 启动：Windows 使用 `scripts/run_falcor_python.ps1`，Linux 使用 `scripts/run_falcor_python.sh`。

## CPU 单元测试

```powershell
conda run -n neural-shading python -m pytest tests\unit -q
```

覆盖 `MaterialProgram`/`LayerStackIR`、CorpusPlan 密度与 G2/G2s split、`reference-shard` v5、corpus reader 的矩形 batch、整球 T mixture、散射合同、`quality-v1` 四层报告、source-aware reciprocal 指标、state-block 配对 bootstrap、training/pipeline v1 schema，以及当前部署回归 fixture 的私有状态。

### P1 v2 框架接入（`p1_v2_plan.md` P2.8，本地只做静态检查）

```powershell
conda run -n neural-shading python -m pytest `
  tests\unit\test_deployment_budget.py `
  tests\unit\test_pipeline_contract.py `
  tests\unit\test_training_config.py `
  tests\unit\test_p1_audit.py `
  tests\unit\test_quality_evaluation.py `
  tests\unit\test_reference_shard.py -q
```

期望：全部通过，无 skip。重点断言：

- `test_deployment_budget.py`：注册表 10 个 pipeline 都带 `runtime.deployment_candidate`；只有 `lobe-residual-k2-v1` 为 `True`，其 `parameter_costs(None)` 满足 `C_eval_macs ≤ 2000`、`C_prepare_macs ≤ 1e4`、`state_bytes_per_pixel ≤ 64`（实际 48）、`B_asset ≤ 512`（128）、`B_evaluate_weights ≤ 32 KB`（0）。
- `test_training_config.py`：14 个 `configs/learning` 配置全部解析；`checkpoint_selection` 默认 `median_then_p95` 且旧 hash 不变；tail guard 回放 P1 v1 M2-S history 选 step 7500 而非 4500。
- `test_p1_audit.py`：`_pipeline_core_probe` 对 M1 返回 `(None, False)`，对 M2 返回 `(callable, True)`。
- `test_quality_evaluation.py`：`quality-v2` 与 `quality-v1` 只差 `checkpoint_selection` 块。

兼容性复核（远程有 P1 v1 checkpoint 时）：`learn evaluate`/`audit-p1` 加载 `artifacts/runs/p1-*-seed-20260824/checkpoints/best.pt` 仍须通过 `pipeline_sha256` 与 `training_config_sha256` 校验（`deployment_candidate` 与默认 `checkpoint_selection` 均不进哈希）；M2-S 的 audit 报告仍含 `analytic_core_energy_ratio`、`deadzone`、`deadzone_correlations`，M1 不含。

运行时边界（静态检查覆盖不到）：`training/runner.py` 在 `tail_guard` 下的 best.pt 替换发生在「当前 validation 把至今最小 p95 压低到使已保存 best 被剔除」时，用 `configs/learning/smoke/lobe-residual-k2-p1-smoke.json` 之外的任一 M2 smoke 配置加 `"checkpoint_selection": "tail_guard"` 跑 8 步确认 `run_manifest.json["best_validation"]` 含 `selection`、`step`、`value` 三个键。`lobe-residual-*` 配置本次只能解析，`create_model/predict_f` 抛 `NotImplementedError`（等 P2.5）。

### 取消容量档位（`capacity` 改为可选、`compare` 去掉 `--vary`；本地只做静态检查）

```powershell
conda run -n neural-shading python -m pytest `
  tests\unit\test_training_config.py `
  tests\unit\test_quality_evaluation.py `
  tests\unit\test_deployment_budget.py `
  tests\unit\test_pipeline_contract.py -q
```

期望：全部通过，无 skip。重点断言：

- `test_training_config.py`：`TrainingConfig(pipeline, stage, model)` 不带 `capacity` 可构造，`capacity is None`，`to_dict()` 中无 `capacity` 键，JSON 往返相等；四个 `lobe-residual-*` 配置解析后 `capacity is None`；P1 v1 配置（`film-evaluator-*`、`analytic-residual-*`、`per-state-teacher-l-v1`）仍带 `"S"/"M"/"L"` 且 `resolved_sha256` 不变（`test_checkpoint_selection_defaults_to_legacy_and_keeps_its_hash`）。
- `test_quality_evaluation.py`：`compare_quality_reports` 的报告 `matched.fixed_training_fields == {"steps", "seed", "dataset_selection"}` 三键；baseline 与 candidate `seed` 不同时抛 `ValueError`，消息含 `matched training fields`；报告里没有 `varied_training_fields` 键。
- `test_deployment_budget.py`、`test_pipeline_contract.py`：P1 v1 家族内部仍按 `P1PipelineSpec(family, capacity)` 注册 7 个 pipeline，本次未改动，用作回归。

CLI：`conda run -n neural-shading python -m ncls.cli learn compare --help` 不再列出 `--vary`；带 `--vary capacity` 调用应报 argparse `unrecognized arguments`。

兼容性复核（远程有 P1 v1 checkpoint 时）：`lobe-residual-*` 配置删去 `"capacity": "S"` 后 `resolved_sha256` 变化，这些配置在此前没有正式 run，无需迁移；P1 v1 checkpoint 的 `training_config_sha256` 校验仍须通过。

静态分析覆盖不到：`runner.py` / `evaluator.py` / `p1_audit.py` 把 `config.capacity` 写进 provenance，`capacity=None` 时应落成 JSON `null`，确认 `run_manifest.json` 与 quality 报告仍能被 `_read_report` 的 hash 校验接受（`allow_nan=False` 对 `null` 无影响，但需实跑一次 smoke 配置确认）。

## P1.0 SlangPy autodiff spike（仅远程，`p1_v2_plan.md` Phase 1）

```powershell
conda env update -n neural-shading -f environment.yml   # 新增 slangpy==0.43.1
conda run -n neural-shading python scripts\spike_slangpy_autodiff.py `
  --device-type cuda --groups 16 --directions 256 --iterations 50 `
  --output artifacts\spikes\slangpy-autodiff.json
```

`--device-type` 取 `spy.DeviceType` 成员名（`cuda` 失败时脚本自动回退默认设备并记录 `device_fallback`）。通过判据（脚本同时写入 JSON 的 `pass` 字段）：

| 项 | 判据 |
|---|---|
| `lobe_gradients` | 对 amplitude/inverse_scale/shear/angle 的梯度与 `torch_eval.eval_ltc_residual` 的 float64 autograd 最大相对误差 ≤ `1e-3`（前向同） |
| `mlp_gradients` | 64 宽 MLP 的权重梯度与 float64 有限差分（24 个随机权重、步长 1e-3）最大相对误差 ≤ `1e-3`；前向与 Torch 镜像 ≤ `1e-3` |
| `throughput` | batch 16 × 256 前向+反向：`torch_m1s_forward_backward_ms / slang_forward_backward_ms ≥ 0.5`（M1-S 在同一脚本内用随机权重的 `ConditionedSharedEvaluator` S 档现场计时） |

需要回填到 `docs/research/p1_v2_plan.md` P1.0 行 / §5 的字段（都在输出 JSON 里）：

- `versions.slangpy`、`versions.slangc`（wheel 内 `slangc -v` 的输出）与 `versions.version_attributes`：即「slangpy 版本」与「slangpy 携带的 slang 版本」。
- `module.weight_tensor` / `module.weight_read`：哪一种可微权重张量写法（`GradOutTensor` / `GradInOutTensor` / `DiffTensor`，`.get({i})` 或 `[{i}]`）被接受；`module.attempts` 里失败候选的编译错误就是「与 Falcor Slang 2024.1.34 的语法差异清单」的第一批条目，回填后同步改 `shaders/ncls/backends/lobe_residual/lobe_residual_mlp.slang` 的 `NCLS_LOBE_RESIDUAL_WEIGHTS_T` / `NCLS_LOBE_RESIDUAL_WEIGHT_READ` 默认注释。
- `torch_interop`：`Tensor.from_torch/to_torch/from_dlpack` 与 `TorchModule` 是否存在，决定 P2.5 `session.py` 是零拷贝还是经 numpy。
- `throughput.*`：写入 `experiment_log.md` 的 Slang 实测行。

失败时：`module` 全部候选编译失败 → 脚本以 `RuntimeError` 退出并打印每个候选的诊断；任一 `pass=false` → 按 P1.1 备选登记到 `experiment_log.md`。

## S2.1 Slang core（本次只写不编译）

`shaders/ncls/backends/lobe_residual/{lobe_residual_mlp,lobe_residual_core,lobe_residual_pack}.slang` 只依赖 `contracts/layer_stack_ir.slang`、`reference/{interfaces,sampling}.slang`。编译验证属于 P2.7 双编译探针；在此之前可用 Falcor 自带 slangc 做冒烟：

```powershell
external\Falcor\build\windows-vs2022\bin\Release\slangc.exe `
  shaders\ncls\backends\lobe_residual\lobe_residual_pack.slang -target dxil -profile cs_6_5 -entry none 2>&1 | Select-Object -First 40
```

需要确认的写法：`struct : IDifferentiable` 自动合成 Differential、`[MaxIters]`、`no_diff` 前缀于 `StructuredBuffer` 下标、`f32tof16/f16tof32`、`out float[64]` 作为可微参数。`Pdf/Sample` 目前是 S2.3 之前的桩（返回 0/false）。

## Slang/GPU 与随机游走参考

```powershell
.\scripts\run_falcor_python.ps1 -m pytest `
  tests\gpu tests\integration\reference -q
```

覆盖 Python/Slang ABI、方向和余弦语义、`prepare/evaluate`、sampling-capable backend 的 `sample/pdf`、各向异性、解析 diffuse、互易性、多层执行、统计量，以及各源材质 reference 的真实执行。P0 的正式大语料按 family/role 分 shard，不再用“全部材质写入同一 HDF5”作为架构测试。

### 统一 Reference Backend：Windows与Ubuntu/A6000均已实机验证

Windows公共入口只编译/检查toolchain并运行无source-assets的仓库fixture probe：

```powershell
.\scripts\build_reference_backend.ps1 -Configuration Release
.\scripts\run_falcor_python.ps1 -m ncls.cli reference doctor
.\scripts\run_falcor_python.ps1 -m ncls.cli reference probe
```

固定MDL状态的两步online training在用户已复制vMaterials资产后执行：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls.cli learn train `
  configs/learning/nvidia-rta2024-mdl-effect-pigment-smoke.json `
  artifacts/training/mdl-windows-smoke/checkpoint.pt
.\scripts\run_falcor_python.ps1 -m ncls.cli learn evaluate `
  configs/learning/nvidia-rta2024-mdl-effect-pigment-smoke.json `
  artifacts/training/mdl-windows-smoke/checkpoint.pt --batches 1
```

Linux支持仍由实际构建和probe决定，不把这次结果外推到其他发行版。2026-08-29已在Ubuntu 22.04.5、10张RTX A6000、driver 550.78、glibc 2.35、GCC 11.4.0上完成GPU 0验证；锁定的Falcor commit为`9dc819c162b2070335c65060436041690b7937f8`。连续运行部署时，后一份成功报告为`artifacts/deployment/reference-linux/20260829T125648Z/report.json`，全部external、Falcor build和MDL provider均为`reused`：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/deploy_reference_linux.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/deploy_reference_linux.sh
```

部署脚本可获取根manifest锁定的`external/`源码和MDL SDK binary package、创建或更新已有Conda中的`neural-shading`环境并完成编译；不安装Conda/driver、不使用`sudo`，也不下载、移动或写入`assets/`。Linux额外固定安装`cuda-compat=12.8.1`；当宿主driver主版本低于570时，launcher从该Conda环境加载CUDA 12.8用户态compatibility库，使SlangPy运行时编译的PTX与项目固定toolchain一致。570及以上driver不启用这层覆盖。

多GPU机器只通过`CUDA_VISIBLE_DEVICES=<单个物理序号>`选择GPU；launcher把同一序号交给Falcor Vulkan，Torch/SlangPy使用重映射后的`cuda:0`。用户自行复制资产后先构建validation-only的OpenPBR C++ probe，再运行完整GPU集合：

需要在多张卡上并行跑相互独立的实验时，使用 fan-out launcher；这不是 DDP，不共享模型、optimizer 或 checkpoint。每个输出路径应包含`{gpu}`占位符：

```bash
bash scripts/run_falcor_python.sh --gpus 2,3,4 -- \
  -m ncls.cli learn train configs/learning/metal-fused-full-linux-smoke.json \
  artifacts/metal-linux-training/gpu{gpu}/checkpoint.pt
```

```bash
conda run -n neural-shading cmake -S tools/reference/openpbr_probe \
  -B build/openpbr-probe -G Ninja -DCMAKE_BUILD_TYPE=Release
conda run -n neural-shading cmake --build build/openpbr-probe \
  --config Release --target ncls_openpbr_probe --parallel 16
CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls.cli reference doctor
CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m pytest \
  tests/gpu/test_reference_query_dispatcher.py \
  tests/gpu/test_reference_backend_contracts.py \
  tests/gpu/test_mdl_native_crosscheck.py \
  tests/gpu/test_mdl_hlsl_feasibility.py \
  tests/gpu/test_merl_reference_gpu.py \
  tests/gpu/test_openpbr_reference_gpu.py \
  tests/gpu/test_layer_stack_ir_gpu.py -q
CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls.cli learn train \
  configs/learning/nvidia-rta2024-mdl-effect-pigment-smoke.json \
  artifacts/training/mdl-linux-smoke/checkpoint.pt
CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls.cli learn evaluate \
  configs/learning/nvidia-rta2024-mdl-effect-pigment-smoke.json \
  artifacts/training/mdl-linux-smoke/checkpoint.pt --batches 1
```

本次完整GPU集合结果为`20 passed`；固定MDL训练完成`bootstrap -> finetune`两步并能重新加载checkpoint执行一批evaluate。完整边界见`docs/reference_backend_deployment.md`。

## Neural material 方法的分阶段门槛

目标 evaluator 尚在建模阶段，测试随能力形成而增加，不能预先把未实现的 sampler、环境积分或 UE 性能写成已存在门槛：

1. 建模原型：完整 `wo × wi` 覆盖、方向/余弦测度、finite、符合 source/color-space 的数值范围、动态范围和 reference 响应误差；
2. 共享 decoder/latent：latent identity、跨 view 共享、材质 split 与 optimized-latent 对照；
3. compiler：未见材质状态、参数编辑和 compiler-vs-optimized-latent 差距；
4. Slang 最小部署：Python/Slang parity、bundle hash、prepare/evaluate 分项成本；
5. matched sampler：sample/PDF 分布一致性、MIS 测度和固定时间方差；
6. integration head：与同一 evaluator 的高样本环境/面光积分对照；
7. 系统阶段：多灯 scaling、viewer capture 和 Falcor/UE 式工作负载。

P0 数据先验证 corpus manifest 与全部 shard：

```powershell
conda run -n neural-shading python -m ncls.cli data validate-corpus `
  artifacts\corpus\layer-stack-v1.json
```

首次 dense slice 完成后还要执行分辨率审计，并把报告中的逐 state 晋升清单回填 CorpusPlan：

```powershell
conda run -n neural-shading python -m ncls.cli data audit-dense `
  artifacts\corpus\layer-stack-v1.json `
  --output artifacts\corpus\layer-stack-v1-dense-audit.json
```

模型正式结果只由 `quality-v1` sanity 判为有效或无效；主指标、scorecard、诊断量和成本均不使用独立 kill gate。候选比较必须使用 `learn compare` 的 matched state-block paired bootstrap。

## pbrt-v4 外部交叉验证

首次配置和构建 probe：

```powershell
cmake -S tools\reference\pbrt_probe -B build\pbrt-probe-current `
  -G "Visual Studio 17 2022" -A x64
cmake --build build\pbrt-probe-current --config Release `
  --target ncls_pbrt_probe --parallel 12
```

比较锁定 pbrt-v4 与 Falcor 随机游走参考解：

```powershell
.\scripts\run_falcor_python.ps1 tools\reference\pbrt_compare.py `
  --pbrt-exe build\pbrt-probe-current\Release\ncls_pbrt_probe.exe `
  --samples 65536 --batches 8 --max-depth 32
```

默认 suite 同时覆盖 diffuse-clear、conductor-clear、conductor-absorbing 和 conductor-scattering，并包含不同方位的各向异性 conductor 切片。

## OpenPBR、MERL 与 MaterialX 源材质

获取固定上游和原始资产：

```powershell
.\scripts\fetch_reference_sources.ps1 -All
conda run -n neural-shading python scripts\fetch_source_materials.py merl
conda run -n neural-shading python scripts\fetch_source_materials.py polyhaven
```

构建 OpenPBR CPU probe，以及 MaterialX 官方 viewer、上游 runtime 和独立 float parity probe：

```powershell
cmake -S tools\reference\openpbr_probe -B build\openpbr-probe `
  -G "Visual Studio 17 2022" -A x64
cmake --build build\openpbr-probe --config Release `
  --target ncls_openpbr_probe --parallel 12
.\scripts\build_materialx_reference.ps1 -Configuration Release
```

运行原生身份、参数编辑、实表查表、图依赖和 shader generation 回归，以及三个材质族的离线呈现：

```powershell
conda run -n neural-shading python -m pytest `
  tests\unit\test_reference_registry.py `
  tests\unit\test_openpbr_material.py `
  tests\unit\test_merl_material.py `
  tests\unit\test_materialx_catalog.py `
  tests\integration\reference\test_source_material_references.py -q

conda run -n neural-shading python tools\reference\analytic_material_preview.py openpbr
conda run -n neural-shading python tools\reference\analytic_material_preview.py merl
conda run -n neural-shading python tools\reference\materialx_preview.py
```

三个新增源材质族都由 Falcor viewer 直接呈现，而不是只停留在 Python adapter。MERL 与 OpenPBR 做逐方向数值 parity；MaterialX 的空间纹理契约使用共同相机线性 HDR 图像 parity：

```powershell
.\scripts\build_viewer.ps1 -Configuration Release
conda run -n neural-shading python tools\reference\materialx_parity.py --suite
```

MaterialX suite 先生成一次 `common-sphere.obj`，让上游 renderer 与 Falcor 光栅管线加载同一路径，并在报告/capture 中核对几何 SHA-256；随后用无纹理核心 probe 检查 closure 公式，再验证全部 8 个原始 4K 材质。验收门槛由 `references/acceptance.json` 版本化；逐次运行的完整指标和图像统一写入 `artifacts/validation/materialx-parity/suite/`，不在 Git 中持久化。

## Windows viewer

```powershell
.\scripts\build_viewer.ps1 -Configuration Release

external\Falcor\build\windows-vs2022\bin\Release\NclsViewer.exe `
  --bundle-root artifacts\exports --headless --frames 32 `
  --width 320 --height 240 --capture artifacts\captures\smoke
```

固定相机路径 benchmark：

```powershell
.\scripts\benchmark_viewer.ps1 `
  -BundleRoot artifacts\exports `
  -Preset configs\viewer-benchmark-v1.json `
  -OutputDirectory artifacts\benchmarks\viewer
```

验收时还应将 capture manifest 传回 `--replay`，确认左右 EXR、显示 PNG、difference 和材质快照逐字节一致；篡改任一 bundle 内容哈希后，锁定方法 ID 的 headless replay 必须以非零状态失败。

## 仓库边界与静态检查

```powershell
conda run -n neural-shading python -m compileall -q src tests tools
git diff --check
git -C external\Falcor status --short
git -C external\pbrt-v4 status --short
git -C external\OpenPBR status --short
git -C external\openpbr-bsdf status --short
git -C external\glm status --short
git -C external\MaterialX status --short
```

所有上游工作树必须为空。`build/`、`data/`、`artifacts/`、`external/` 和缓存不得进入根仓库。

## Metal full-cohort Linux单GPU待验证

当前Windows证据只覆盖RTX 4090上的3-export stratified正确性run，不能替代以下原生Linux gate。目标机必须先具备`assets/source-materials/mdl-vmaterials2/2.4.0/Materials`与锁定的Linux Falcor/MDL构建：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/deploy_reference_linux.sh
conda run -n neural-shading python tools/learning/build_metal_training_configs.py
conda run -n neural-shading python tools/learning/preflight_metal_fused.py \
  --output artifacts/metal-linux-training/full-cohort-preflight.json
CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls.cli learn train \
  configs/learning/metal-fused-full-linux-smoke.json \
  artifacts/metal-linux-training/smoke/checkpoint.pt
conda run -n neural-shading python -m tools.learning.build_metal_linux_handoff \
  --output artifacts/metal-linux-training/handoff.json
```

期望：config check报告`source_count=692`、四个phase与`distributed=false`；preflight报告692 exports、178 groups、52 texture sets、20 required components；smoke review为complete，全部metric finite、gradient/update coverage完整，并包含nonzero `runtime_fp16_quantization_trace`及profile分项。静态分析覆盖不到Linux Vulkan adapter选择、A6000实际VRAM/ETA与692-source session资源，因此这些值只能由目标机review回填。

smoke成功后才按[Metal Linux训练交接](docs/metal_linux_training.md)以long config自身的`--stop-at-step 16`checkpoint启动/恢复120000步训练。训练结束只审阅自动生成的review、基础evaluate和代表性package；不自动执行formal、追加seed、消融或Pareto。
