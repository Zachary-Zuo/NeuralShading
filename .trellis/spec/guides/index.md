# Thinking Guides

> **Purpose**: Expand your thinking to catch things you might not have considered.

---

## Why Thinking Guides?

**Most bugs and tech debt come from "didn't think of that"**, not from lack of skill:

- Didn't think about what happens at layer boundaries → cross-layer bugs
- Didn't think about code patterns repeating → duplicated code everywhere
- Didn't think about edge cases → runtime errors
- Didn't think about future maintainers → unreadable code

These guides help you **ask the right questions before coding**.

---

## Available Guides

| Guide | Purpose | When to Use |
|-------|---------|-------------|
| [Code Reuse Thinking Guide](./code-reuse-thinking-guide.md) | Identify patterns and reduce duplication | When you notice repeated patterns |
| [Cross-Layer Thinking Guide](./cross-layer-thinking-guide.md) | Think through data flow across layers | Features spanning multiple layers |

---

## Quick Reference: Thinking Triggers

### When to Think About Cross-Layer Issues

- [ ] Feature touches 3+ layers (API, Service, Component, Database)
- [ ] Data format changes between layers
- [ ] Multiple consumers need the same data
- [ ] You're not sure where to put some logic
- [ ] You are adding an event kind, JSONL record, RPC payload, or config field
- [ ] UI / command code starts casting raw payload fields directly

→ Read [Cross-Layer Thinking Guide](./cross-layer-thinking-guide.md)

### When to Think About Code Reuse

- [ ] You're writing similar code to something that exists
- [ ] You see the same pattern repeated 3+ times
- [ ] You're adding a new field to multiple places
- [ ] **You're modifying any constant or config**
- [ ] **You're creating a new utility/helper function** ← Search first!
- [ ] Two files read the same untyped payload field with local casts
- [ ] Multiple branches update the same derived state from `kind` / `action`

→ Read [Code Reuse Thinking Guide](./code-reuse-thinking-guide.md)

### When Verifying AI Cross-Review Results

- [ ] Reviewer claims "user input can be malicious" → Check the actual data source (internal manifest? user config? external API?)
- [ ] Reviewer flags "missing validation" → Is the data from a trusted internal source?
- [ ] Reviewer says "behavior change" → Read the code comments — is it intentional design?
- [ ] Reviewer identifies a "bug" in test → Mentally delete the feature being tested — does the test still pass? If yes → tautological test

**Common AI reviewer false-positive patterns**:
1. **Trust boundary confusion**: Treating internal data (bundled JSON manifests) as untrusted external input
2. **Ignoring design comments**: Flagging intentional behavior documented in code comments as bugs
3. **Variable misreading**: Not tracing a variable to its actual definition (e.g., Map keyed by path vs name)

**Verification rule**: Every CRITICAL/WARNING finding must be verified against the actual code before prioritizing. Budget ~35% false-positive rate for AI reviews.

---

## Pre-Modification Rule (CRITICAL)

> **Before changing ANY value, ALWAYS search first!**

```bash
# Search for the value you're about to change
grep -r "value_to_change" .
```

This single habit prevents most "forgot to update X" bugs.

---

## How to Use This Directory

1. **Before coding**: Skim the relevant thinking guide
2. **During coding**: If something feels repetitive or complex, check the guides
3. **After bugs**: Add new insights to the relevant guide (learn from mistakes)

---

## Contributing

Found a new "didn't think of that" moment? Add it to the relevant guide.

---

**Core Principle**: 30 minutes of thinking saves 3 hours of debugging.

---

## 本项目的触发点（NeuralShading）

上面的通用指南保留原文；以下是这个仓库里最常出现的"没想到"场景，命中任一项就去读对应指南或 spec：

### 跨层（→ Cross-Layer Thinking Guide + `core/index.md`）

- [ ] 改了 `abi/*.json`、`schemas/*.json` 或 `docs/contracts/` 中任一字段——Python dataclass、生成的 `.slang`、C++ loader、测试四处必须同步
- [ ] 改了 `evaluate()` 的输出语义、余弦归属、方向约定或 PDF 测度——数据采集、quality-v1、Slang backend、viewer 四处同时受影响
- [ ] 改了 `MethodBundle` manifest 字段——Python exporter、`manifest.py`、`MethodBundle.cpp`、`docs/contracts/method_bundle.md` 同步
- [ ] 给 backend 加了新 capability——ABI 枚举、descriptor、bundle 校验、viewer `kRequiredCapabilities` 同步
- [ ] 改了 shard / corpus 字段——writer、reader、validator、learning reader、`reference_dataset.md` 同步

### 复用（→ Code Reuse Thinking Guide + `project/code-organization.md`）

- [ ] 要写一个 BSDF 公式、VNDF 采样或 LTC 响应——先看 `shaders/ncls/reference/{interfaces,sampling}.slang`、`legacy_ltc_k2.slang`
- [ ] 要写 loss——先看 `src/ncls/learning/pipelines/appearance_loss.py`
- [ ] 要写 canonical JSON / hash / 原子写——先看 `data/profiles.py`、`training/runner.py`
- [ ] 要在 Torch 里写模型前向——停下：前向只写 Slang（`core/shared-slang-backend.md`）
- [ ] 要复制一个 pipeline / provider / pass 文件再改——抽共用函数，不复制

### 本项目特有

- [ ] 我准备把某个诊断脚本放进 `scripts/` 或 `tools/`——它是一次性的吗？一次性的放 `.trellis/tasks/<task>/scratch/`
- [ ] 我准备保留一个旧路径"以后再删"——不允许；同一任务递归迁移并删除，或在报告里列待迁移清单提醒用户
- [ ] 我准备宣称"已验证"——先看 `project/dev-environment.md` 判定本机状态

### 关于通用指南中的 Trellis 专属小节

Cross-Layer 指南里的「Cross-Platform Template Consistency」「Generated Runtime Template Upgrade Consistency」「Mode-Detection Probe Checklist」「Event Log / Projection Boundary」，以及 Code Reuse 指南里的「Template File Registration (Trellis-specific)」，描述的是 Trellis CLI 自身仓库的情况，与本项目无关；本项目的对应规则以 `.trellis/spec/project/` 与各层 `index.md` 为准。
