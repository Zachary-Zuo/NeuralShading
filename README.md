# NeuralShading

NeuralShading 研究如何把多种保持原生语义的源材质族编译为统一、随机访问、运行成本有界的 neural material program。目标方法用小型 MLP 直接实现逐方向 `evaluate(wo, wi)`，让同一份编译材质进入 deferred、hybrid ray tracing 和 path tracing 的材质求值位置。源材质可以是纯数学模型、可编辑材质图、程序材质、高分辨率纹理或测量外观；每个材质族保留自己的权威 reference，不要求 GT 先被分解成某种层参数或固定 closure。

项目的正式架构分为三段：CorpusPlan 驱动的 reference 语料、统一 neural evaluator 训练/评测、Windows/D3D12 部署验证。三段只通过 `MaterialProgram`、`reference-corpus`/矩形 HDF5 shard 和 `MethodBundle` 交换数据。LayerStack、MERL、OpenPBR 与 MaterialX 已有保持原生语义的 provider；P0 首先完整冻结 LayerStack v1 语料，其他材质族随后接入同一 corpus 合同。

目标运行时分成三个清晰阶段：`compile_material()` 生成 view-independent latent 资产；`prepare()` 在每个 raster pixel 或 ray hit 获取、过滤并编码 latent、footprint 与 `wo`；小型 evaluator MLP 对每个 `wi` 直接输出散射。Path tracing profile 在 evaluator 成形后再增加匹配且具有可计算密度的 `sample()/pdf()`；环境光和面光积分作为后续独立能力研究。

当前采用基准优先顺序：先生成 LayerStack v1 corpus 并冻结 `quality-v1`，再在着色器预算内比较 evaluator 候选，之后进入 compiler、Slang、sampler 与 Falcor/UE 式系统验收。现有解析 backend 只承担部署回归 fixture、成本对照和可选物理 core/sampling proposal，不注册为研究候选。

源材质 reference 已扩展为五个 active package：LayerStack 随机游走、pbrt coated 独立验证、OpenPBR 1.1.1、MERL 测量 BRDF，以及 8 个原生 MaterialX/Poly Haven 4K 纹理材质。它们从 `references/registry.json` 统一发现，但各自保留原始参数、测量表或图/纹理 GT。

## 最短启动路径

创建唯一环境并安装项目：

```powershell
conda env create -f environment.yml
conda run -n neural-shading python -m pip install -r requirements-torch-cu128.txt
conda run -n neural-shading python -m pip install -e .
```

解析 LayerStack v1 语料计划：

```powershell
conda run -n neural-shading python -m ncls.cli data plan-corpus `
  --config configs\corpus\layer-stack-v1.json `
  --shard-root data\reference-responses `
  --output artifacts\corpus\layer-stack-v1-plan.json
```

正式采集需要锁定的 Falcor Python；命令支持 verified-file 断点续采：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls.cli data collect-corpus `
  --config configs\corpus\layer-stack-v1.json `
  --shard-root data\reference-responses `
  --output artifacts\corpus\layer-stack-v1.json

conda run -n neural-shading python -m ncls.cli data validate-corpus `
  artifacts\corpus\layer-stack-v1.json
```

P1 已完成 M1 S/M/L、matched M2 S/M/L、per-state teacher 与 M3 response-space oracle 的正式比较；统一使用 `ncls learn train/evaluate/compare/benchmark`、`ncls learn oracle-m3` 和 [`quality-v1`](docs/learning.md)，不启用迁移前 pipeline 或 config。M1-M 是通过全部主参考线的质量起点，M1-S 是更快的 Pareto 端点；详细结论见 [`experiment_log.md`](docs/research/experiment_log.md)。P1 的 30-state selection 与完整 LayerStack v1 共用同一单-state 方向密度，只缩减当前阶段研究的 state 范围；reference 样本预算按排名、训练和诊断用途分别冻结。

## 文档入口

- [这个问题是什么、如何进入 UE 式实时管线](docs/realtime_material_compilation.md)
- [架构与边界](docs/architecture.md)
- [源材质族与统一神经材质程序](docs/material_scope.md)
- [Reference package 固定入口](references/README.md)
- [数据采集](docs/data.md)
- [训练、评测与导出](docs/learning.md)
- [Windows viewer](apps/viewer/README.md)
- [稳定合同](docs/contracts/)
- [分层测试](TESTING.md)
- [仓库边界](docs/repository_policy.md)
- [当前研究问题、相关工作与实验路线](docs/research/README.md)

Falcor 8.0、pbrt-v4、OpenPBR、openpbr-bsdf、GLM 和 MaterialX 位于被 Git 忽略的 `external/` 独立克隆，固定提交见 `AGENTS.md`。本项目源码不会把未说明修改留在上游工作树。
