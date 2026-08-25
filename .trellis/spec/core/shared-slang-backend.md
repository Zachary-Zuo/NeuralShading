---
name: core-shared-slang-backend
description: 散射合同的实现规则与"每个方法一份 Slang"架构：INclsScatteringBackend、Falcor-free core、SlangPy 训练 / Falcor 测试 / viewer 共用同一源、权重布局由反射生成、parity 退化为双编译探针
paths:
  - shaders/ncls/backends/**
  - shaders/ncls/contracts/scattering_backend.slang
  - shaders/ncls/contracts/scattering_contract.slang
  - src/ncls/core/scattering/**
  - src/ncls/core/representations/**
  - src/ncls/bundle/**
  - src/ncls/learning/slang/**
  - tests/gpu/**
  - apps/viewer/shaders/**
---

# 散射合同与共享 Slang 后端

> P1 v1 的教训：一个模型写四处（Torch 前向、手写 Slang 复刻、手写 exporter、viewer 硬编码），每处都在膨胀，parity 测试成为必需品。目标结构（`docs/research/p1_audit.md` §6.2、`p1_v2_plan.md` E1）：每个方法一份 Slang，训练、GPU 测试、viewer 三处 `#include` 同一文件。

## 散射合同要点（`docs/contracts/scattering_backend.md`）

- `wo` / `wi` 都指向远离着色点的一侧；世界 ↔ 局部只经 `NclsShadingFrame`；`evaluate()` 返回**不含**几何余弦的 `f`，renderer 恰好乘一次 `abs(dot(N, wi))`；PDF 相对立体角，delta 事件 PDF 为 0 并由 event flag 表达；RGB 线性、有限、非 delta 值非负。
- Falcor `IBSDF` 把余弦包含在返回值里且命名不同，必须经集中 adapter 转换，不能泄漏到数据合同或 Python API。
- Slang 接口按能力分层、编译期 specialization，热路径无运行时虚调用：

```slang
interface INclsScatteringState { NclsScatteringEval evaluate<S : ISampleGenerator>(float3 wiWorld, inout S sg); bool sample<S>(out NclsScatteringSample, inout S sg); NclsScatteringPdf pdf(float3 wiWorld); }
interface INclsScatteringBackend { associatedtype CompiledMaterial; associatedtype State : INclsScatteringState; State prepare(NclsScatteringContext context, CompiledMaterial material); }
```

- `CompiledMaterial` 与 `State` 是 backend 私有 associated type；renderer 只按 `BackendDescriptor` 分配资源，从不解释内容。`sample/pdf` 的可调用性只由 capability 决定（`REQUIRED_REALTIME_CAPABILITIES = Prepare | Evaluate | Sample | Pdf | AnisotropicFrame`，`src/ncls/core/scattering/contract.py`）。
- `State` 只在材质状态、footprint、shading frame 与 `wo` 都未变时复用；`prepare()` 不消费选方向的随机数；`sample()` 用 `State` 里的 proposal 参数生成方向，`pdf()` 算同一 proposal 的密度。
- 合同范例：`shaders/ncls/backends/legacy_ltc_k2/legacy_ltc_k2.slang` 的 `LegacyLtcK2State` / `LegacyLtcK2Backend`（cosine proposal，pdf 诚实匹配）。

## 每个方法一份 Slang

```text
shaders/ncls/backends/<name>/
  <name>_mlp.slang    原语：配置轴宏、Params（权重缓冲 + 偏移字段）、dense 层、lobe 解码
  <name>_core.slang   Falcor-free：[Differentiable] prepare / evaluate，Pdf / Sample（随机数由调用者传入）
  <name>_pack.slang   State 打包（half）
  <name>.slang        合同包装：struct <Name>Backend : INclsScatteringBackend，从 ISampleGenerator 取随机数后调 core
```

- **训练与评测**：SlangPy 加载 core，反射 `Params` 布局，`torch.autograd.Function` 包装 `bwd_diff`；loss 与 optimizer 留在 Torch，`training/runner.py` 只把 `pipeline.predict_f` 换成 Slang 调用。`create_model` 返回只持有 params 与 latent 张量的薄 `nn.Module`，**无 Torch 前向**。
- **GPU 测试**：Falcor Python 与 SlangPy 同时编译同一 core，evaluate 数值 `rtol 2e-5`（`p1_v2_plan.md` P2.7）；任一编译器失败即阻止提交。`ISampleGenerator` 测试实现见 `tests/gpu/kernels/legacy_ltc_k2.cs.slang`。
- **viewer**：`#include` 同一文件；`Prepare / Approximation / Parity` pass 通过 `INclsScatteringBackend` 泛型调用，由 bundle 声明的 module 路径与类型名 specialization（V4.3 目标；当前 `film_m1` 直接调 `nclsFilmM1*` 自由函数绕过合同，是待迁移债务）。
- **权重布局**：由 Slang `Params` 反射生成，Slang 内不写偏移常量；bundle 只存张量与反射出的 layout，删除手写 exporter。
- **Torch 的角色**：`src/ncls/core/representations/legacy_ltc_k2/torch_eval.py` 只作 evaluate / pdf 的 parity oracle；不再新增 Torch 生产前向。

## Slang 版本

Falcor 8.0 锁定 Slang 2024.1.34；SlangPy（`environment.yml` 固定 `slangpy==0.43.1`）携带更新的 slang。core 只用两边都验证过的写法；语法差异清单由 P1.0 spike 回填到 `p1_v2_plan.md`，`lobe_residual_mlp.slang` 的 `NCLS_LOBE_RESIDUAL_WEIGHTS_T` / `NCLS_LOBE_RESIDUAL_WEIGHT_READ` 宏承接训练侧的可微张量写法。不升级 Falcor 的 slang。

## MethodBundle 边界（`docs/contracts/method_bundle.md`）

- viewer 与评测只加载 `MethodBundle`，不加载训练目录或裸 `.pt`；导出必须从不可变 checkpoint 生成全新 bundle 并算内容哈希。
- `runtime_class`：`realtime` 要求 `bounded_execution`、`prepare/evaluate`、完整 capability 与 cost model，且满足硬线；否则 `diagnostic`，UI 必须标注、不进实时排名。
- 共享 evaluator / sampler 权重属于 bundle runtime；材质专属 latent / material code 属于 `CompiledMaterial`；材质专属权重必须标为 per-material asset 并计入 bytes。
- 加载顺序固定（manifest + hash → 平台 / Slang / 合同版本 → IR 支持 → shader variant → CompiledMaterial → 状态资源 → parity probe → 才进方法列表）；不兼容显示具体原因，不退回相近方法。

## 反例

- 在 Python 里再写一遍 MLP 前向"方便调试"。
- 在 Slang 里写 `static const uint kPrepareWeight0 = 0u;` 之类的手工偏移。
- backend core `import` Falcor 模块，导致 SlangPy 无法编译。
- 只输出方向不给 pdf 却声明 `ScatteringSampling`。
