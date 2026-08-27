# MDL Reference Viewer 集成

## 目标与用户价值

把已经通过 MDL SDK native cross-check 和 falcor2 numerical parity 的 `mdl.program@1` compiled artifact 接入当前 Falcor 8 `NclsViewer`。用户启动项目自己的 viewer 后，可以在固定 shaderball、相机和灯光下切换六种 vMaterials，直接观察 car paint、铜锈、划痕铝、釉面陶瓷、velvet 和木质 mosaic 的原生 MDL surface response。

## 已确认事实

- 正式依赖只能是 `mdl.program@1 -> project MDL bridge -> MDL SDK target code -> current Falcor 8`；falcor2 仅为隔离 numerical oracle。
- `ncls.mdl-compiled-artifact@1` 已包含生成 HLSL、argument block、RO data、2D/BSDF-data texture、compiler identity、capability audit 和所有文件的 SHA-256。
- Falcor 8 C++ `ProgramDesc::addShaderModule().addString()` 支持把运行时组合的 MDL target code 作为具名 shader module 编译，不需要改上游 Falcor。
- 当前 viewer 的 source path tracer 只有 LayerStack、MERL、OpenPBR、MaterialX 四个 family；registry 中 MDL 的 `viewer_integration` 和 `image_parity` 均为 pending。
- MDL V1 只声明 surface-BSDF `evaluate` 和 `ExplicitLod(0)`；它没有公开 matched `sample/pdf`、emission、volume 或 displacement。

## 需求

1. viewer 从经过完整 identity/hash/capability 验证的 compiled artifact 加载 MDL；缺文件、额外文件、hash 漂移、SDK/compiler identity 不符或超出 V1 capability 时必须 fail closed。
2. viewer 动态组合锁定 MDL target-code types、项目 `mdl_runtime.slangh`、material-specific generated HLSL 和 viewer adapter，并由当前 Falcor 8 执行；不得 import、启动或 fallback 到 falcor2。
3. `ReferencePathTracer` 必须继续使用公共 `PathSurface` 取得位置、frame 和 UV。MDL evaluate 输出沿用已经冻结的线性 RGB `f * |n_s·wi|`；MDL 路径延续与环境光 MIS 必须消费同一 target code 的 `surface_scattering_sample/pdf`，但不把它提升为项目公共 source capability。
4. 提供六种 vMaterials 的 viewer catalog 和交互选择；默认启动 `carpaint-shifting-flakes`，切换材质时原子重建 artifact binding、shader pass 和 accumulation，失败保留上一份有效材质。
5. viewer runtime 不依赖 Python、PyTorch、训练目录或 falcor2。项目 launcher 可以在启动前通过现有 Python source/bridge API生成 ignored compiled cache 和 catalog。
6. 启动入口必须使用 `scripts/build_viewer.ps1` 构建 overlay；完成后 `external/Falcor` 恢复干净。
7. capture/replay 记录 asset id、source snapshot id、compiled artifact SHA-256、MDL SDK/compiler identity 和 V1 filtering capability，使看到的图像能追溯到同一正式 reference artifact。

## 验收标准

- [x] 需求交付（用户本次请求）：一条项目脚本能准备六种 artifact、构建 Release viewer 并启动可交互窗口，默认显示 MDL car paint shaderball。
- [x] 需求交付（用户本次请求）：viewer UI 可在六种 vMaterials 间切换，至少 car paint、patinated copper、scratched aluminum 的可见响应不同且输出有限。
- [x] 语义正确性（MDL V1/项目 artifact 合同）：artifact tamper、unsupported capability 和 falcor2 boundary 有 unit/static regression；失败不会静默显示旧 family 或 oracle 结果。
- [x] 数值实现正确性（同一正式 artifact 的已有 Falcor query runtime）：MDL viewer adapter 的方向/frame/response 在 GPU probe 中与 formal query runtime 对同一 artifact、同一方向一致；容差沿用已冻结 formal parity 的 float32 误差策略，不根据本次结果调宽。
- [x] 需求交付（可观察、可追溯）：Release build 通过，真实 headless capture 生成带完整 identity 的 manifest；shaderball EXR 尺寸、spp 和 finite pixel 检查通过。
- [x] 工程合同（viewer/upstream 规范）：构建和启动后 `external/Falcor` 工作树干净；registry 仅把 `viewer_integration` 改为 ready。没有独立 renderer image oracle 前，`image_parity` 保持 pending。
- [x] 缺陷修复（用户现场反馈 / 数值正确性）：car paint 与 glazed ceramic 不再因 fixed-GGX/MDL PDF 错配持续积累 firefly；matched sample 的 `weight/pdf/event` 有 GPU regression，修复后 1024 spp capture 的极端尾部与空间离群点相对修复前诊断显著下降，且不使用 radiance clamp。

## 范围外

- 不把 falcor2 或其 MDL renderer 复制进正式 viewer。
- 不把 MDL matched `sample/pdf` 公开为训练/provider 的 source capability；viewer 可以内部消费 artifact 已生成的同一 BSDF `sample/pdf`。仍不支持 emission、volume、displacement 或 derivative filtering。
- 不实现 viewer 内任意 MDL module 浏览器、完整 typed parameter editor，或同一 scene 内多个不同 generated MDL program 的同时混合。
- 不训练或编译 neural material，也不把 MDL 反演成 LayerStack/OpenPBR/MaterialX。

## 阻塞问题

无。MVP 的 UX 是六项固定 catalog 下拉框，默认 car paint；更通用的 module browser 和 typed editor 延后。

## 现场缺陷修订（2026-08-27）

- trigger：用户观察到 car paint 与 glazed ceramic 的白色噪点随 spp 累计持续增加，明确要求定位根因并修复。
- invalidated evidence：原 1024 spp smoke 只验证 finite 与可运行；它没有检查高动态范围响应下 estimator 的权重尾部，因此不能证明 fixed-GGX proposal 适合 MDL flakes/coat。
- scope impact：只把 MDL viewer transport 改为同 target code 的内部 matched `sample/pdf`，并修正环境光 MIS；不改变 artifact identity、source evaluate 数据合同或公共 capability。
- rerun required：adapter GPU parity、matched sample/PDF GPU regression、Release build，以及 car paint/ceramic 同 replay capture 的尾部和视觉检查。
