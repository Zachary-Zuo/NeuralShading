# 实施计划：MDL Reference Viewer 集成

## 1. Artifact 与 catalog

- [x] 实现 C++ `MdlReferenceArtifact` 的 schema/identity/hash/containment/capability 验证与资源描述。
- [x] 增加正式准备器，复用六种 vMaterials source state 与 `MdlSdkCompilerBridge` 生成 ignored viewer catalog。
- [x] 增加 catalog/artifact tamper 与 falcor2 boundary unit tests。

## 2. Falcor 8 runtime

- [x] 抽取/新增 viewer MDL adapter，使用公共 `PathSurface` 构造 MDL shading state。
- [x] 用具名 string shader module 组合 target-code types、项目 runtime、generated HLSL 和 adapter。
- [x] 上传 argument block、RO data、2D/BSDF-data texture 与 formal sampler；绑定到 reference pass。
- [x] 扩展 `ReferencePathTracer` 的 evaluate 与 viewer-internal matched sample/pdf 分支，但不声明公共 MDL sampler capability。
- [x] 增加同 artifact/同方向的 viewer adapter 与 formal query GPU parity。

## 3. Viewer UX 与可追溯性

- [x] `ReferenceSource`/scene state 支持单一活动 MDL artifact 和六项 catalog dropdown。
- [x] 切换时使用 validate/build/swap 原子流程，失败保留旧 binding。
- [x] replay/capture 写入 asset、snapshot、artifact、compiler/SDK/filtering identity。
- [x] registry 将 `viewer_integration` 改为 ready，`image_parity` 保持 pending。

## 4. 启动与质量门

- [x] 增加 `scripts/launch_mdl_viewer.ps1`，只通过 `scripts/build_viewer.ps1` 构建并启动可见 Release 窗口。
- [x] `conda run -n neural-shading python -m pytest tests/unit -q`
- [x] 运行 MDL viewer GPU parity 与真实 headless capture；检查 EXR finite、尺寸、1024 spp 和 manifest identity。
- [x] `scripts/build_viewer.ps1 -Configuration Release`
- [x] `git diff --check`，确认 `external/Falcor` 与所有锁定 upstream 干净。
- [x] 启动交互式 viewer，默认选择 car paint，交给用户现场观察并切换六种 preset。

## 高风险点与回滚点

- material-specific generated symbols 会冲突，因此 MVP 限制同时一个 MDL program；不要用重命名 generated code 规避。
- shader module 编译或资源创建失败时不替换当前 source/pass；保留上一有效材质。
- 不修改 `external/Falcor`；构建 overlay 退出后若不干净立即停止。

## 5. Firefly 根因修复

- [x] adapter 抽取唯一 MDL shading-state 构造，并增加 target-code `sample/pdf` 入口。
- [x] MDL 路径延续改用 `bsdf_over_pdf`，环境光 MIS 改用同一 MDL PDF；不增加 radiance/throughput clamp。
- [x] 增加 GPU regression，验证 sampled direction 的 `evaluate/pdf` 与 `bsdf_over_pdf` 一致。
- [x] 重建 viewer，复跑 car paint 与 glazed ceramic 1024 spp capture，比较修复前后极端尾部和空间离群点。
- [x] 更新 viewer MDL 稳定合同，记录“matched transport 是显示尖锐 closure 的正确性要求”。
