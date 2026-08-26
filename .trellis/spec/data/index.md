---
name: data-index
description: 源材质接入与数据采集层入口：ReferenceProvider、CorpusPlan → reference-shard v5 → reference-corpus manifest、reference shader、references/ package；开发前检查与质量检查
paths:
  - src/ncls/data/**
  - src/ncls/source_materials/**
  - src/ncls/references/**
  - references/**
  - shaders/ncls/reference/**
  - shaders/ncls/data/**
  - configs/corpus/**
  - tools/reference/**
  - tests/integration/reference/**
  - docs/data.md
  - docs/contracts/reference_dataset.md
---

# 源材质接入与数据采集

> 这一块的职责：保存或导入源材质资产和原生参数状态，通过该族的 reference 得到方向响应或图像 GT，写入可恢复、可验证的数据集。它只依赖公共核心。

## 正式入口

```text
CorpusPlan（configs/corpus/*.json）
  → ncls data plan-corpus
  → ncls data collect-corpus（LayerStack 在 Windows 经 run_falcor_python.ps1，在 Ubuntu 经 run_falcor_python.sh）
  → 按 family / role / direction_count 拆分的矩形 reference-shard v5（data/reference-responses/）
  → reference-corpus manifest（artifacts/corpus/*.json）
  → ncls data validate-corpus / audit-dense
```

代码：`src/ncls/data/{profiles,collector,dataset,corpus}.py`；provider 在 `src/ncls/data/providers/`（LayerStack、MERL、OpenPBR、MaterialX）；reference shader 在 `shaders/ncls/reference/`，采集 compute 入口在 `shaders/ncls/data/`。

详细规则见 `reference-and-corpus.md`。

## 开发前检查清单

- [ ] 已读 `core/material-interface.md`：新材质族保留原生语义，不反演成 LayerStack。
- [ ] 新 reference 已在 `references/registry.json` 与 `references/<package-id>/` 登记身份、实现、资产清单、许可证与 capability 状态。
- [ ] 采样密度、split、噪声预算只在 `CorpusPlan` 里声明；不给采集器加可任意拼接的 profile 参数。
- [ ] 已区分 pilot 与 formal：pilot 只确定成本 / 精度 / 吞吐，formal plan 在采集前一次冻结；达到 formal cap 后不会自动生成更高 cap 的 vN 继续追门。
- [ ] 改 shard / corpus 字段时同步 `src/ncls/data/schemas/*.json`、`docs/contracts/reference_dataset.md`、`tests/unit/test_reference_shard.py` / `test_corpus_plan.py`。
- [ ] 已判定开发机状态：正式采集只在“完整 Windows”或“Linux reference”状态可做；viewer 与 D3D12-only `falcor` marker 测试仍只在完整 Windows 状态运行。

## 质量检查

- [ ] `ncls data validate` / `validate-corpus` 通过：hash、role 完整性、方向不跨 role 重合、state metadata 不跨 shard 漂移。
- [ ] 采集器对已存在且不一致的文件报错而不是覆盖；续采只在 hash / state 集合 / 计划完全一致时复用。
- [ ] 正式 corpus 的每个 shard 都来自同一份预先批准的 acquisition policy；plan 变更后的结果使用新身份，不能与旧 plan 混合后称为一次冻结采集。
- [ ] reference 噪声预算按用途分层：validation/test 主 response 通过精度门；train 主 response 使用 pilot 推导并在 formal plan 中冻结的足够 SPP，SE/variance 用于审计该 SPP 是否充分；diagnostic / reciprocal 只落盘 SE。没有把诊断 role 的噪声改成训练目标。
- [ ] 新族的 `ReferenceDescriptor` 声明了 `incident_domain`、`capabilities`、`implementation_sha256`。
- [ ] 一次性采集报告进 `artifacts/`，没有把 `.h5` 以外的东西写进 `data/`。
