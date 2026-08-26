---
name: viewer-index
description: Windows viewer 层入口：Falcor overlay 构建、左侧场景 path-traced reference 与右侧 MethodBundle、capture/replay 与 viewer-scene sidecar、benchmark；开发前检查与质量检查
paths:
  - apps/viewer/**
  - patches/**
  - scripts/build_viewer.ps1
  - scripts/benchmark_viewer.ps1
  - scripts/fetch_viewer_assets.ps1
  - configs/viewer-*.json
  - docs/viewer_spec.md
  - docs/contracts/viewer_scene.md
  - docs/contracts/method_bundle.md
---

# Windows viewer

> `apps/viewer/` 是 Windows/D3D12 原生查看器：左侧是各源材质族原生 reference 的完整场景 path tracer（有限深度），右侧是显式选中的 `MethodBundle`。它是部署与系统验证工具，不是训练 UI、不是模型结构搜索工具。它只依赖公共核心与 Falcor，不依赖训练代码或 PyTorch。

## 构建与运行方式

- 只用 `scripts/build_viewer.ps1`：先 `fetch_viewer_assets.ps1` 验证固定资产，再检查 `external/Falcor` 在锁定提交 `9dc819c…` 且工作树干净，临时应用 `patches/falcor-viewer-overlay.patch`（只增加一个 `add_subdirectory()`），构建后在 `finally` 反向应用并再次确认干净。
- 项目 shader 依赖显式列在 `apps/viewer/CMakeLists.txt`；`external/openpbr-bsdf` 与 `glm` 作 include 目录；studio-v1 资产复制到 Falcor bin 的 `data/ncls-viewer/`。
- 默认场景 `configs/viewer-studio-v1.json`（shaderball + Poly Haven HDRI + 各向异性粗糙导体），资产 hash 只约束默认启动，不拦截显式加载别的 scene / HDRI。
- 只能在"完整"开发机状态构建与运行（`project/dev-environment.md`）。

详细规则见 `conventions.md`。

## 开发前检查清单

- [ ] 我改的是 viewer 私有实现，还是公共核心的合同？后者去 `core/`。
- [ ] 新 backend 通过 `INclsScatteringBackend` 与统一 `NclsMethod*` 类型别名接入，不在 pass 里直接调 backend 自由函数，也不在 `MethodBundle.cpp` 硬编码 backend 字符串或 architecture。
- [ ] 新 shader 已加进 `apps/viewer/CMakeLists.txt`。
- [ ] 改 capture / viewer-scene 字段时同步 `docs/contracts/viewer_scene.md` 与 `docs/viewer_spec.md`。
- [ ] 不改 `external/Falcor`；需要改上游先写 `patches/` 并在 `AGENTS.md` 说明。

## 质量检查

- [ ] 构建后 `git -C external\Falcor status --short` 为空。
- [ ] raw reference 始终是 comparison / difference / 噪声估计的权威输入；去噪只改观看，manifest 标 `raw_authoritative=true`。
- [ ] 左右共享场景、相机、材质 slot 绑定、灯光、曝光与 tone mapping；不能分别自动曝光。
- [ ] 材质或物理光照变化自动解除 freeze、清空累积；切换 bundle / split / difference / 显示开关不清空。
- [ ] headless `--capture` 后 `--replay` 逐字节一致；篡改 bundle 内容哈希后锁定方法 ID 的 replay 非零退出。
- [ ] 新增 bundle 能力（如 PT + method）时 UI 明确标注 `diagnostic` / `realtime`，不混入同一性能排名。
