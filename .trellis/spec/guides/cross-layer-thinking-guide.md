# Cross-Layer Thinking Guide

> **Purpose**: Think through data flow across layers before implementing.

---

## The Problem

**Most bugs happen at layer boundaries**, not within layers.

Common cross-layer bugs:

- API returns format A, frontend expects format B
- Database stores X, service transforms to Y, but loses data
- Multiple layers implement the same logic differently

---

## Before Implementing Cross-Layer Features

### Step 1: Map the Data Flow

Draw out how data moves:

```
Source → Transform → Store → Retrieve → Transform → Display
```

For each arrow, ask:

- What format is the data in?
- What could go wrong?
- Who is responsible for validation?

### Step 2: Identify Boundaries

| Boundary              | Common Issues                     |
| --------------------- | --------------------------------- |
| API ↔ Service         | Type mismatches, missing fields   |
| Service ↔ Database    | Format conversions, null handling |
| Backend ↔ Frontend    | Serialization, date formats       |
| Component ↔ Component | Props shape changes               |

### Step 3: Define Contracts

For each boundary:

- What is the exact input format?
- What is the exact output format?
- What errors can occur?

---

## Common Cross-Layer Mistakes

### Mistake 1: Implicit Format Assumptions

**Bad**: Assuming date format without checking

**Good**: Explicit format conversion at boundaries

### Mistake 2: Scattered Validation

**Bad**: Validating the same thing in multiple layers

**Good**: Validate once at the entry point

### Mistake 3: Leaky Abstractions

**Bad**: Component knows about database schema

**Good**: Each layer only knows its neighbors

### Mistake 4: Every Consumer Parses The Same Payload

**Bad**: A command reads JSONL events and casts fields inline:

```typescript
const thread = (ev as { thread?: string }).thread;
const labels = (ev as { labels?: string[] }).labels;
```

This looks local, but it means every consumer owns a private version of the
event contract. The next field change will update one command and miss another.

**Good**: Decode once at the event boundary, then export typed projections:

```typescript
if (!isThreadEvent(ev)) return false;
return ev.thread === filter.thread;
```

**Rule**: For append-only logs, JSON streams, RPC payloads, or config files,
create one owner for:

- event / payload type definitions
- type guards and normalization from `unknown`
- metadata projections used by UI commands
- reducers that replay state from the source of truth

Rendering code may format fields, but it must not redefine the payload contract.

---

## Checklist for Cross-Layer Features

Before implementation:

- [ ] Mapped the complete data flow
- [ ] Identified all layer boundaries
- [ ] Defined format at each boundary
- [ ] Decided where validation happens

After implementation:

- [ ] Tested with edge cases (null, empty, invalid)
- [ ] Verified error handling at each boundary
- [ ] Checked data survives round-trip
- [ ] Checked that consumers import shared decoders / projections instead of
      casting payload fields locally
- [ ] Checked that derived state points back to the source event identifier
      (`seq`, `id`, `version`) instead of inventing a second cursor

## External GPU Runtime Boundary

当一个 logical batch 跨越 PyTorch/CUDA 与 Falcor/D3D12 时，同步 fence 只解决读写顺序，不能替代 runtime 的 frame 生命周期：

- 先画出 `CUDA producer → wait_for_cuda → Falcor dispatch → wait_for_falcor → CUDA consumer`，并标明 buffer 所有权和 lease 释放点。
- 使用 runtime 官方支持的传输入口，不把可映射 shared memory 等同于可无限期原地覆写的稳定 API。
- 把 `end_frame()` 放在全部 route lease 释放后的 iteration 边界；tiled dispatch 是同一 frame 内的实现细节。
- 除数值 parity 外，增加跨越多个 iteration 的完整 forward/backward/optimizer soak；单 kernel 或少量 dispatch 只能证明局部互操作。
- 长跑若在 semaphore/command-buffer 交接点失败，先检查 frame rotation、deferred release、transient heap 与 shared-buffer transfer path，再把它归因于模型或上游驱动。

## GPU Surface / Differential Boundary

当数据从 camera/scene hit 跨到 source reference、package backend 或 texture sampler 时，先把每个量的单位和隐含尺度写出来：

- [ ] camera basis 是归一化方向、image-plane slope，还是共同乘有 focal distance？方向上的 `normalize()` 是否掩盖了 footprint 中仍存在的尺度？
- [ ] ray cone 是世界空间长度还是角度；triangle Jacobian 是线性尺度还是 `log2`；传给 backend 的 gradient 是否为 normalized UV？
- [ ] texture/latent dimension 由 surface 层还是 sampler 层加入？只允许一个 owner，不能重复乘除尺寸。
- [ ] reference PT、package PT 与 deferred 是否由同一 surface contract 产生字段，而不是三份“看起来相同”的公式？
- [ ] semantic evidence 是否包含 raw UV/LOD 或明显空间结构？resource loaded、slot ready 和平均色只能证明 lifecycle。

具体实现与测试入口见 `../viewer/path-surface.md`。

## Typed GPU Texture Shape Boundary

typed payload从program跨到Falcor texture binder时，`shape`的spatial axes固定在前，channel只允许在最后：

- scalar `texture2d`：`[height, width]`；RGBA `texture2d`：`[height, width, 4]`；
- scalar `texture3d`：`[depth, height, width]`；RGBA `texture3d`：`[depth, height, width, 4]`。

binder必须从前置spatial axes统一推导`width/height/depth`，不得用依赖rank的负索引猜测；上传前同时验证rank、payload element count与format bytes-per-texel。新增texture kind/dtype时，unit test至少覆盖scalar与带channel两种rank，真实GPU test还要包含非相等的`depth/height/width`，避免轴交换被立方体尺寸掩盖。

## Scattering Estimator Ownership Boundary

当 source/neural backend 的散射状态跨到 renderer 时，“存在同名接口”不等于 renderer 已遵守合同。必须沿真实调用图确认 estimator 的所有组成量属于同一 owner：

```text
backend.prepare → state.evaluate / state.sample / state.pdf → MIS / throughput
```

- [ ] runtime descriptor 是否 fail closed 要求 `prepare/evaluate/sample/pdf`，而不是缺入口后由 renderer 补 generic proposal？
- [ ] `evaluate().f`与renderer乘回的cosine是否属于同一个input shading frame？source内部若改变normal，是否把`|N_source·wi| / |N_input·wi|`保留在等价`f`中，而不是在source侧除以`N_source`、renderer侧再乘`N_input`？
- [ ] continuous sample 的实际方向分布、reported PDF、`evaluate` 与 `weight = f·|n_s·wi|/pdf` 是否来自同一 backend？
- [ ] 上游 API 自带 eval/sample/pdf 时，是否把 direction/event/PDF/weight 当作不可拆分的 native sample tuple 验证？极窄掠射方向若因已舍入 `wi` 重建 half-vector而出现独立 query 漂移，是否保留 native tuple，并分别验证 independent evaluate↔pdf，而不是用后者重建 sample？
- [ ] direct-light MIS 与 BSDF-hit MIS 是否调用同一 state PDF，并在 multiple-sample NEE 下使用相同的 `n·p_light`？
- [ ] multiple-sample MIS 是否显式区分 `n_light·p_light` 与 `n_bsdf·p_bsdf`，并让两侧各自除以自己的 sample count？
- [ ] direct BSDF pool 与 continuation 的所有权是否从路径空间检查？如果多个 downstream strategy 都继承一条上游 continuation，仅增加 downstream samples 不能消除共同的 throughput 尾部。
- [ ] primary BSDF sample 是否同时拥有两种互斥结果：miss environment 是 direct strategy，hit geometry 是完整 path suffix？若另取一条 primary continuation，必须证明它没有重新引入单样本上游瓶颈或 direct 双计。
- [ ] environment importance CDF 是否与真正的 GPU radiance filter 使用同一 reconstruction？point CDF 配 bilinear lookup 会让 PDF 与被积函数支持域错位。
- [ ] renderer 是否完全不识别 source family/program key？heterogeneous composer 只能做 concrete canonical state dispatch，不能重写材质数学。
- [ ] query/data capability 与 runtime scattering capability 是否分别命名？训练 batch 不传 sampler，不代表 runtime 可以私下绕开 canonical sampler。
- [ ] invalid/null sample 是否终止路径？fallback、radiance clamp 与 throughput clamp 都会掩盖 estimator 所有权错误。
- [ ] tail 诊断是否同时包含 sample→pdf/weight 数学门、HDR 环境峰值、空间邻域与随 spp 的 RSE？`finite` 或单张显示图不能区分真实窄高光与随机 firefly。
- [ ] contribution AOV 是否按路径前缀与 strategy 两级拆分？整图 top residual 可能被合法连续高光污染；应先固定用户可见问题区域，再检查 downstream strategy 是否因共同上游 throughput 而相关。
- [ ] ray-origin side 是否由实际 sampled direction 与 geometric normal 决定，而不是依赖可能与 smooth-shading 几何关系不同的 event label？

具体 ABI、错误矩阵与测试入口见 `../core/shared-slang-backend.md` 和 `../viewer/conventions.md`。

## Viewer Sampling Lifecycle Boundary

当同一 PT 同时用于交互 viewer、headless capture 和 benchmark 时，先把 estimator identity、dispatch batch 与 termination target 分开：

- [ ] 交互是否每 dispatch 追加一个新的 `globalSample`，并在状态不变时持续累积，而不是继承 capture spp cap？
- [ ] headless 的 `reference_spp` 和 batch 是否只存在于 replay/options，未进入 UI 或 viewer scene？
- [ ] reset 是否只由相机/材质/灯光状态实际变化触发？reset 后当前 sample 是否保留，而不是用 dragging/preview flag 算完再丢弃？
- [ ] source/package 是否都以 `globalSample = accumulatedSpp + sampleIndex` 派生同一 path suffix identity？
- [ ] 测试是否分别覆盖交互无 cap、headless remaining 截断和相同 target 下 raw output identity？

具体 owner、错误矩阵与断言点见 `../viewer/conventions.md` 和 `../viewer/capture-harness.md`。

## Large Linked Deployment Catalog Boundary

当source registry、typed editor、reference artifact与neural package组成数百条viewer catalog时，不能把entry数量当成重资产数量：

- [ ] 在任何native compile、GPU cook或package写出前，是否遍历全量entry并运行typed editor validator？抽查前N条不能覆盖稀有enum/vector schema。
- [ ] authoring值是否在边界处规范化为runtime/UI类型，例如MDL enum对象转choice字符串、共享vector range转runtime支持的标量range？
- [ ] 是否分别统计entry identity与唯一重资产identity（如`texture_set_id`）并按后者分组cook？
- [ ] 相同content hash是否复用不可变payload，同时保持每个source/asset/instance的canonical identity？logical bytes不能用来估算hardlink后的物理占用。
- [ ] 已有catalog的启动路径是否只做轻量identity/结构检查，并把大payload严格验证放在实际entry加载事务中？不要为了打印统计再次遍历全目录。
- [ ] Windows并发生产者发布内容寻址目录时，是否处理`exists → atomic replace`之间的瞬时竞争/拒绝，并保持有界重试和不可加载partial边界？

具体合同与测试入口见 `../viewer/mdl-reference.md`。

---

## Cross-Platform Template Consistency

In Trellis, command templates (e.g., `record-session.md`) exist in **multiple platforms** with identical or near-identical content. This is a cross-layer boundary.

### Checklist: After Modifying Any Command Template

- [ ] Find all platforms with the same command: `find src/templates/*/commands/trellis/ -name "<command>.*"`
- [ ] Update all platform copies (Markdown `.md` and TOML `.toml`)
- [ ] For Gemini TOML: adapt line continuations (`\\` vs `\`) and triple-quoted strings
- [ ] Run `/trellis:check-cross-layer` to verify nothing was missed

**Real-world example**: Updated `record-session.md` in Claude to use `--mode record`, but forgot iFlow, Kilo, OpenCode, and Gemini — caught by cross-layer check.

---

## Generated Runtime Template Upgrade Consistency

Some generated files are both documentation and runtime input. In Trellis,
`.trellis/workflow.md` is parsed by `get_context.py`, `workflow_phase.py`,
SessionStart filters, and per-turn hooks. Template changes must be validated
against both fresh init and upgrade paths.

### Checklist: After Modifying A Runtime-Parsed Template

- [ ] Identify every runtime parser that reads the template, not just the file
      writer that installs it
- [ ] Check whether relevant syntax lives outside obvious managed regions
      such as tag blocks
- [ ] Verify fresh `init` output and a versioned `update` scenario that writes
      the older `.trellis/.version`
- [ ] Add an upgrade regression using an older pristine template fixture, then
      assert the installed file reaches the current packaged shape
- [ ] Update the backend spec that owns the runtime contract

---

## Versioned Documentation Boundary

Versioned documentation is a cross-layer boundary: source paths, `docs.json`
version routing, and the rendered version selector must all describe the same
release line.

### Checklist: Before Editing Versioned Docs

- [ ] Identify the target release line: stable, beta, or RC
- [ ] Verify the edited MDX path matches that line:
  - stable: `docs-site/{start,advanced,...}` and `docs-site/zh/{start,advanced,...}`
  - beta: `docs-site/beta/**` and `docs-site/zh/beta/**`
  - RC: `docs-site/rc/**` and `docs-site/zh/rc/**`
- [ ] Verify `docs.json` navigation points the version label to the same paths
- [ ] Grep the opposite tree for release-line-specific terms before committing
- [ ] Treat beta content appearing under root release paths as a source-path bug,
      not a rendering bug

**Real-world example**: A beta-only task workflow change documented
`prd.md` + `design.md` + `implement.md`, task-creation consent, and Codex
mode banners under root `start/` and `advanced/` paths. The docs site then
served 0.6 beta behavior under the Release selector. The fix was to restore root
release docs, move the 0.6 content to `beta/` and `zh/beta/`, and add a grep
audit for beta markers against the root release tree.

**Real-world example**: Codex inline mode changed workflow platform markers from
`[Codex]` / `[Kilo, Antigravity, Windsurf]` to `[codex-sub-agent]` /
`[codex-inline, Kilo, Antigravity, Windsurf]`. Fresh init was correct, but
`trellis update` only merged `[workflow-state:*]` blocks and preserved stale
markers outside those blocks. Result: upgraded projects got new hook scripts
but old workflow routing, so `get_context.py --mode phase --platform codex`
could return empty Phase 2.1 detail.

---

## Mode-Detection Probe Checklist

When a CLI auto-detects a mode by probing a remote resource (e.g., checking if `index.json` exists to decide marketplace vs direct download):

### Before implementing:

- [ ] Probe runs in **ALL** code paths that use the result (interactive, `-y`, `--flag` combos)
- [ ] 404 vs transient error are distinguished — don't treat both as "not found"
- [ ] Transient errors **abort or retry**, never silently switch modes
- [ ] Shared state (caches, prefetched data) is **reset** when context changes (e.g., user switches source)
- [ ] **Shortcut paths** (e.g., `--template` skipping picker) must have the same error-handling quality as the probed path — check that downstream functions don't call catch-all wrappers

### After implementing:

- [ ] Trace every path from probe result to the mode-decision branch — no fallthrough
- [ ] External format contracts (giget URI, raw URLs) are tested or at least documented as comments
- [ ] Metadata reads consume a complete response or use a streaming parser — never parse a fixed-size prefix as full JSON
- [ ] When reconstructing a composite identifier from parsed parts, verify **all** fields are included and in the **correct position** (e.g., `provider:repo/path#ref` not `provider:repo#ref/path`)
- [ ] Verify that **action functions** called after a shortcut don't internally use the old catch-all fetch — they must use the probe-quality variant when error distinction matters

**Real-world example**: Custom registry flow had 8 bugs across 3 review rounds: (1) probe only ran in interactive mode, (2) transient errors fell through to wrong mode, (3) giget URI had `#ref` in wrong position, (4) prefetched templates leaked across source switches, (5) `--template` shortcut bypassed probe but `downloadTemplateById` internally used catch-all `fetchTemplateIndex`, turning timeouts into "Template not found".

**Real-world example**: Agent-session update hints fetched npm `latest` metadata with `response.read(4096)` and then parsed it as complete JSON. The `@mindfoldhq/trellis` package metadata exceeded 4 KB, so the JSON was truncated, parse failed silently, and the first session injection showed no update hint. Fix: read the complete response before parsing, and add a regression where `version` is followed by an 8 KB metadata tail.

---

## Cross-Platform Template Consistency

In Trellis, command templates (e.g., `record-session.md`) exist in **multiple platforms** with identical or near-identical content. This is a cross-layer boundary.

### Checklist: After Modifying Any Command Template

- [ ] Find all platforms with the same command: `find src/templates/*/commands/trellis/ -name "<command>.*"`
- [ ] Update all platform copies (Markdown `.md` and TOML `.toml`)
- [ ] For Gemini TOML: adapt line continuations (`\\` vs `\`) and triple-quoted strings
- [ ] Run `/trellis:check-cross-layer` to verify nothing was missed

**Real-world example**: Updated `record-session.md` in Claude to use `--mode record`, but forgot iFlow, Kilo, OpenCode, and Gemini — caught by cross-layer check.

---

## Generated Runtime Template Upgrade Consistency

Some generated files are both documentation and runtime input. In Trellis,
`.trellis/workflow.md` is parsed by `get_context.py`, `workflow_phase.py`,
SessionStart filters, and per-turn hooks. Template changes must be validated
against both fresh init and upgrade paths.

### Checklist: After Modifying A Runtime-Parsed Template

- [ ] Identify every runtime parser that reads the template, not just the file
  writer that installs it
- [ ] Check whether relevant syntax lives outside obvious managed regions
  such as tag blocks
- [ ] Verify fresh `init` output and a versioned `update` scenario that writes
  the older `.trellis/.version`
- [ ] Add an upgrade regression using an older pristine template fixture, then
  assert the installed file reaches the current packaged shape
- [ ] Update the backend spec that owns the runtime contract

**Real-world example**: Codex inline mode changed workflow platform markers from
`[Codex]` / `[Kilo, Antigravity, Windsurf]` to `[codex-sub-agent]` /
`[codex-inline, Kilo, Antigravity, Windsurf]`. Fresh init was correct, but
`trellis update` only merged `[workflow-state:*]` blocks and preserved stale
markers outside those blocks. Result: upgraded projects got new hook scripts
but old workflow routing, so `get_context.py --mode phase --platform codex`
could return empty Phase 2.1 detail.

---

## Mode-Detection Probe Checklist

When a CLI auto-detects a mode by probing a remote resource (e.g., checking if `index.json` exists to decide marketplace vs direct download):

### Before implementing:
- [ ] Probe runs in **ALL** code paths that use the result (interactive, `-y`, `--flag` combos)
- [ ] 404 vs transient error are distinguished — don't treat both as "not found"
- [ ] Transient errors **abort or retry**, never silently switch modes
- [ ] Shared state (caches, prefetched data) is **reset** when context changes (e.g., user switches source)
- [ ] **Shortcut paths** (e.g., `--template` skipping picker) must have the same error-handling quality as the probed path — check that downstream functions don't call catch-all wrappers

### After implementing:
- [ ] Trace every path from probe result to the mode-decision branch — no fallthrough
- [ ] External format contracts (giget URI, raw URLs) are tested or at least documented as comments
- [ ] Metadata reads consume a complete response or use a streaming parser — never parse a fixed-size prefix as full JSON
- [ ] When reconstructing a composite identifier from parsed parts, verify **all** fields are included and in the **correct position** (e.g., `provider:repo/path#ref` not `provider:repo#ref/path`)
- [ ] Verify that **action functions** called after a shortcut don't internally use the old catch-all fetch — they must use the probe-quality variant when error distinction matters

**Real-world example**: Custom registry flow had 8 bugs across 3 review rounds: (1) probe only ran in interactive mode, (2) transient errors fell through to wrong mode, (3) giget URI had `#ref` in wrong position, (4) prefetched templates leaked across source switches, (5) `--template` shortcut bypassed probe but `downloadTemplateById` internally used catch-all `fetchTemplateIndex`, turning timeouts into "Template not found".

**Real-world example**: Agent-session update hints fetched npm `latest` metadata with `response.read(4096)` and then parsed it as complete JSON. The `@mindfoldhq/trellis` package metadata exceeded 4 KB, so the JSON was truncated, parse failed silently, and the first session injection showed no update hint. Fix: read the complete response before parsing, and add a regression where `version` is followed by an 8 KB metadata tail.

---

## When to Create Flow Documentation

Create detailed flow docs when:

- Feature spans 3+ layers
- Multiple teams are involved
- Data format is complex
- Feature has caused bugs before

---

## Event Log / Projection Boundary

Append-only logs are cross-layer contracts. A single event travels through:

```
CLI input → event writer → events.jsonl → reader → filter → reducer → display
```

### Checklist: After Adding A New Event Kind Or Field

- [ ] Add the event kind to the central event taxonomy
- [ ] Add a typed event variant or type guard at the event layer
- [ ] Add normalization helpers for array/object fields that come from
      user input or JSON
- [ ] Keep `seq` / `id` assignment in the event writer only
- [ ] Make filters and reducers consume the typed event guard, not local casts
- [ ] Make display code consume reducer output or typed events, not raw JSON
- [ ] Add at least one regression that proves history replay and live filtering
      use the same filter model

**Real-world example**: Thread channels added `kind: "thread"`, `description`,
`context`, labels, and `lastSeq`. The first implementation replayed thread
state correctly, but several commands still re-parsed event payload fields with
local casts. The fix was to make the core event layer own `ThreadChannelEvent`
and `isThreadEvent`, make `reduceChannelMetadata` the only channel metadata
projection, and make `reduceThreads` the only thread replay reducer.
