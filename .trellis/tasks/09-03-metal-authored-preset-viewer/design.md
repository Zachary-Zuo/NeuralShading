# Metal authored preset viewer 与自动匹配设计

## 1. 数据流

```text
metal-opaque-v1.json
  ├─ 692 taxonomy + typed descriptors
  ├─ canonical MDL runtime artifacts
  └─ step 20000 compile_program/asset/instance
                    ↓
        ViewerMaterialCatalog@1
                    ↓
     loader → identity resolver
                    ↓
        LinkedMaterialController
       ├─ MDL argument candidate
       └─ neural instance candidate
                    ↓ atomic commit
            two existing slots
```

准备工具运行于 Python/Falcor 构建环境；最终 viewer 只读取 catalog/files，不加载训练代码、PyTorch 或 MDL SDK runtime。

## 2. Catalog writer/loader

writer 接受显式 registry、checkpoint 和 output root：

1. 验证 registry/checkpoint/config identity；
2. 重新构造 692 个 authored snapshots，并在重工作前预检全部typed source parameter views；
3. 用正式 provider有界并发准备 MDL runtime artifacts；
4. 对 checkpoint 执行一次 `compile_program`，按52个`texture_set_id`分组cook assets，为每个 authored snapshot 编译独立initial instance；
5. 写 taxonomy、typed edit descriptors、reference/method bindings、files/hashes；
6. 对catalog结构与identity self-load后原子安装；大payload由viewer选择entry时严格加载。

缺任一 entry、重复 identity、悬空 binding、路径越界、额外/缺失文件或 hash 漂移时整体失败，不发布 partial catalog。运行产物写入 ignored `artifacts/`/`build/` 目录。

相同SHA-256的program/grid/reference payload在同卷上使用hardlink，跨卷才复制；这只优化物理存储和写入，不改变各entry的source/asset/instance/package identity。已有catalog入口不递归读取或统计全部payload。

C++ loader 复用现有 `ScatteringPackage@2` typed blob/resource/sampler 校验和 `ProgramRuntimeCache`/`AssetBinding`/`InstanceBinding` 类型。catalog 自己只增加 entry taxonomy、reference binding、edit descriptors 与 component table；不复制宽松 loader。

## 3. 参数编辑

每个 entry 保存两组由 exporter 生成的 writes：

- `reference_writes`：parameter path、MDL type、argument-block offset/size、enum name/value；
- `method_writes`：复用现有 `ncls.typed-material-editor@1` 中 token/raw/derived writes 与 runtime compiler descriptor。

UI 先按 `SourceParameterView` 规范化值，再把同一值写入两个 candidate。支持类型固定为 registry 当前实际出现的 bool/int/enum/float/float2/color；出现 resource 或未知类型时 catalog 生成 fail closed。

`ViewerMaterialState@1` 对 catalog/registry、base snapshot 和按 path 排序的 normalized values 做 canonical JSON hash。它是 viewer edit identity，不替换 `SourceSnapshot.snapshot_id`。Reset parameter 恢复 authored default；Reset material 清除全部 edits。

## 4. Resolver 与原子提交

resolver 以 `export_id + base source_snapshot_id + program_id + asset_id + instance_id` 建立左右 binding，display name/index 只用于 UI。

controller 保存 committed/candidate：

1. 选择 entry或收到 edit；
2. validate typed value并生成 edit-state；
3. 创建/复用 reference program/resources，复制并 patch argument block；
4. 创建/复用 neural program/asset，复制并 patch raw buffer，dispatch `nclsCompileMaterial`；
5. 完成 capability/shader/resource validation；
6. 同时替换两个 slot并 reset accumulation。

失败仅更新 error text并释放 candidate。linked mode 隐藏/禁用 per-slot material/package editor；manual mode仍走现有逻辑。

## 5. UI/capture

Material panel 提供级联 filter、搜索、typed groups、reset、checkpoint/phase 与短 identity。普通 preset 切换不显示路径或 hash 输入；详细 identity 放在可展开诊断区。

默认 studio/launcher 指向 catalog。step 20000 linked mode 默认右侧 deferred/evaluator；manual PT 可留在 Advanced 并显示 `joint-appearance` 状态。

capture/replay 保存 catalog/registry、export/base snapshot、normalized edit values/edit-state、reference artifact、checkpoint/program/asset/instance 与 slot mode。新字段需要 schema 升版时保留旧 reader，不改变 renderer 的两 slot 对称结构。

## 6. 文件所有权

- Python schema/writer：`src/ncls/bundle/` 或相邻 deployment 模块；Metal mapping 放既有 method exporter附近。
- 准备入口：`tools/reference/` 或 `tools/viewer/`，脚本不含手写 preset 数组。
- C++：新增 `ViewerMaterialCatalog.*`、`LinkedMaterialController.*`；`NclsViewer.cpp` 只做生命周期接线和 UI。
- 测试：unit schema/export/identity、C++ static/loader、GPU argument/raw compiler probes、viewer headless/replay。

不修改 learning producer/config/model、source adapter、Metal shader数学、reference backend数学或 scattering ABI。

## 7. Rollback

catalog/transaction 任一层失败时保留当前六项 MDL catalog与单 package入口。不得通过手写 692 项、放宽 identity、修改模型或启动训练绕过 viewer integration 问题。
