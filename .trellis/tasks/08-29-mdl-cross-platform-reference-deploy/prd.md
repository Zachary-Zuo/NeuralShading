# 统一跨平台 Reference Backend 与 Linux 编译部署

## Goal

把五个 canonical ground-truth reference 材质族的 Windows/Linux 执行、依赖准备、构建、预检与部署收敛成统一、可发现、可验证的公共 backend capability。上层只调用材质无关的 backend/session 接口；LayerStack、MERL、OpenPBR、MaterialX 与 MDL 的原生语义继续由各自 `ReferenceProgramDefinition` 负责，平台和 toolchain 差异只存在于公共 backend 或 program 内部 provider。提供一个 Linux 编译部署入口，在不下载任何材质资产的前提下获取并校验锁定代码/SDK、建立项目环境、构建 FalcorPython 与必要 provider，并输出可追溯报告。

## Background and confirmed facts

- 当前语义主链已经是 `SourceFamilyDefinition → ReferenceProgramDefinition → ReferenceQueryDispatcher → OnlineTrainingProducer`。五个正式 program 已共享 `RuntimePayload`、`MaterialPayload`、typed resource binder 与 `prepare/evaluate/sample/pdf` query schema；不需要再发明一套 closure/evaluator 接口。
- 当前缺口位于执行与部署层：`ReferenceQueryDispatcher` 自行 import Falcor、选择 D3D12/Vulkan并创建 device；测试和工具仍有 D3D12硬编码；依赖获取、Falcor/MaterialX/MDL构建与launcher由多条平台脚本拼接。
- MDL 是唯一需要额外编译 provider 的正式 program。现有 bridge 固定 Windows SDK目录、`.dll/.exe` 与 Win32 loader；它应成为 MDL program内部 provider，而不是公共 backend 的抽象中心。
- LayerStack无外部 program provider；MERL运行时消费源测量表；OpenPBR运行时需要锁定 LUT/source；MaterialX source解析需要锁定 MaterialX；它们都通过同一 Falcor/Slang execution backend执行。
- `ncls.pbrt-coated-crosscheck@1` 与 falcor2 MDL oracle 是 independent-validation executor，不进入 canonical dispatcher或online producer。
- `assets/`、`external/`、`build/` 与 `artifacts/` 不进入根仓库。用户会自行迁移 `assets/`；部署脚本只允许获取代码/toolchain，不获取 source material、纹理、测量表或vMaterials package。
- Linux目标发行版尚未冻结；后半段必须在实际Ubuntu/A6000服务器记录并验证真实OS、driver、glibc、compiler、Vulkan、Conda和build身份。

## Frozen decisions

- 公共 backend只覆盖五个进入 `ReferenceProgramDefinition` registry的ground-truth program：LayerStack、MERL、OpenPBR、MaterialX、MDL。
- pbrt crosscheck与falcor2 oracle继续作为独立验证工具；不为了容纳CPU外部进程而稀释GPU online-reference接口。
- 公共接口不出现MDL SDK、`.dll/.so/.exe`、Visual Studio/GCC或某个材质族的参数。MDL compiler是MDL program内部toolchain provider。
- 不保留无必要的旧调用兼容层。项目自有调用方和tests直接迁移到新接口；没有外部稳定合同的旧Falcor薄函数、Windows-only常量和旧脚本入口直接删除。
- Linux deployment可以联网获取并校验锁定的source/toolchain依赖；不得下载、展开或修改任何`assets/`内容。
- deployment不调用`sudo`，不安装Conda或driver。系统prerequisites只探针并fail fast；已有Conda时可以创建/更新唯一的`neural-shading`环境。
- deployment成功不依赖`assets/`是否存在。真实资产query、MDL training与研究验收在用户复制资产后用独立命令执行。
- 本任务仍包含固定、空间均匀MDL snapshot的`ncls learn train` smoke；不实现multi-snapshot或参数泛化。
- Windows公共架构与回归先完成；Linux/A6000后半段继续同一任务。Linux gate完成前任务保持未完成。

## Requirements

### R1. 公共 backend capability

- 定义唯一公共 `ReferenceBackendCapability` 与版本化 descriptor/report。descriptor表达platform id、Falcor revision、graphics API、build/runtime roots、Slang/runtime identity和backend build identity；report表达ready/missing/error，不包含材质专属字段。
- 公共 factory按Windows x86-64或Linux x86-64解析backend；未知OS/arch明确拒绝，不做D3D12/Vulkan fallback。
- 上层通过backend capability创建query session。session接收任意registered `ReferenceProgramDefinition`和对应snapshots，统一提供`evaluate/sample/pdf/end_iteration/close`；upper producer/runner/CLI不import Falcor、不判断OS、不拼build路径。
- program-specific `compile_runtime/compile_material`继续保留为必要插件接口；公共backend只消费typed payload，不知道某材质是解析式、测量表、纹理图或generated MDL code。

### R2. 五个reference program覆盖与边界

- LayerStack、MERL、OpenPBR、MaterialX与MDL都必须通过同一backend factory/session构造，不得有family-specific dispatcher、producer、query shader或platform branch。
- backend doctor枚举canonical registry并为每个program输出统一status；program可通过公共preflight/provider hook报告内部依赖，但公共report只使用版本化requirement/status结构。
- stable reference registry、source descriptors与代码注册表必须一致；independent-validation package不得被backend discovery误识别为canonical program。
- 测试必须静态证明正式上层不存在D3D12、Vulkan、`.dll/.so/.exe`与`sys.platform`判断；这些细节只允许在platform resolver、launcher/build脚本和program内部toolchain provider中出现。

### R3. MDL内部跨平台provider

- 定义版本化MDL toolchain manifest，登记同一锁定SDK build的Windows/Linux archive URL、size、SHA-256、library/plugins、target-code types与bridge build layout。
- portable bridge维持同一source target、CLI、schema和codegen options；Windows使用`LoadLibrary/GetProcAddress`，Linux使用`dlopen/dlsym`，动态加载封装为RAII `SharedLibrary`。
- MDL program只调用内部provider的discover/inspect/compile/native-evaluate语义操作，不读取平台路径或文件后缀。
- artifact/cache记录semantic identity与platform build identity。跨平台artifact/cache/checkpoint在没有显式portability contract时不得静默复用。
- 删除仅为旧调用保留的`MDL_SDK_DIRECTORY`、直接bridge构造和路径参数兼容；所有项目调用方改用provider factory或program公共hook。

### R4. Linux编译部署

- 提供`bash scripts/deploy_reference_linux.sh`作为唯一Linux编译部署入口。它可以获取锁定external源码、Falcor Packman依赖、stb与Linux MDL SDK，但不得调用任何source-material fetcher或写入`assets/`。
- 执行顺序为：系统/网络/Conda探针 → root/external revision与dirty检查 → `neural-shading`环境create/update → 锁定依赖安全获取 → Falcor setup/FalcorPython build → portable MDL bridge build → 公共backend doctor/device/compile smoke → deployment report。
- archive必须验证size/SHA-256、单一root、path containment、symlink/traversal与partial状态；existing target只有通过真实probe才复用，不通过时fail closed，不自动递归删除或覆盖。
- external clone存在时必须验证固定commit和clean；不得reset/checkout覆盖用户修改。缺失时只获取manifest登记的revision。
- 脚本不冻结Ubuntu字符串；记录`/etc/os-release`、kernel、arch、glibc、GPU/driver、Vulkan、compiler、Git、Conda、Falcor、SDK和backend identity，以真实build/smoke决定结果。
- 结果写入`artifacts/deployment/reference-linux/<run-id>/report.json`与中文摘要；第二次运行安全复用正确环境/download/build并再次成功。

### R5. Identity、training与部署后验证

- query/checkpoint identity包含reference program descriptor、source snapshots与backend build identity；Windows/Linux未证明可移植时resume失败。
- NVIDIA method为`mdl.program@1`注册明确的`fixed-uniform` adapter：只接受单snapshot、1×1、无空间2D texture依赖的typed常量参数；离散资源只进入identity，不伪造连续特征。
- 新增`effect-pigment-metallic`两步smoke config，沿用generic producer/runner/checkpoint与同一CLI；只证明data flow、finite loss/gradient、materialization与checkpoint load，不声明参数泛化。
- deployment自身只运行不依赖source assets的backend/device/compile probe。用户复制`assets/`后，Windows/Linux质量gate再对五个program执行代表性真实snapshot query，并执行MDL training smoke。

### R6. 文档、测试与迁移

- 更新project/data/learning/viewer specs、稳定中文文档、reference packages与TESTING说明，描述公共backend、内部provider、部署输入边界、资产迁移后命令与Windows/Linux支持状态。
- Windows viewer继续消费canonical payload/backend，不增加Linux viewer；viewer构建结束后`external/Falcor`必须clean。
- 现有项目自有调用方一次性迁移，不留下deprecated alias、双registry或Windows/Linux两套公共API。

## Acceptance Criteria

- [ ] **AC1｜需求交付｜来源：用户“所有reference材质使用公共后端”**：五个canonical program均由同一`ReferenceBackendCapability`创建同一session类型；producer、runner、CLI无family/platform分支。
- [ ] **AC2｜语义正确性｜来源：现有reference合同**：公共backend只消费`RuntimePayload/MaterialPayload`，`prepare/evaluate/sample/pdf`的输入输出、linear `f`、PDF、event与lease合同不变。
- [ ] **AC3｜需求交付｜来源：用户“不留不必要兼容层”**：项目内不存在旧Falcor platform薄入口、Windows-only MDL目录常量、上层direct bridge构造或deprecated alias；所有调用方与tests已迁移。
- [ ] **AC4｜需求交付｜来源：用户“部署只是编译”**：Linux deployment不调用source-material fetcher，不下载/展开/修改`assets/`，且在`assets/`不存在时仍可完成依赖获取、环境准备、FalcorPython/MDL bridge构建和compile/device smoke。
- [ ] **AC5｜需求交付｜来源：用户同意联网获取构建依赖**：缺失external/source/toolchain可按版本化manifest获取；已有错误revision、dirty tree、hash drift、unsafe archive或partial target均fail closed并给出可操作诊断。
- [ ] **AC6｜需求交付｜来源：用户权限边界**：deployment不调用`sudo`、不安装Conda/driver；缺少系统prerequisite时在构建前失败并写report。
- [ ] **AC7｜语义正确性｜来源：platform/backend合同**：Windows解析为D3D12、Linux解析为Vulkan；未知OS/arch拒绝。五个program的backend/session identity显式包含实际platform build，未证明的跨平台cache/checkpoint不能resume。
- [ ] **AC8｜数值实现正确性｜来源：现有GPU/query invariants**：Windows与实际Ubuntu/A6000在用户迁移资产后，对五个program完成代表性`evaluate/sample/pdf`、same-device CUDA tensor、双slot lease与无host response readback测试；已有family-specific parity门不因本任务放宽。
- [ ] **AC9｜语义正确性｜来源：MDL provider合同**：同一bridge source在Windows/Linux构建并产生同schema artifact；缺SDK/library/plugin/bridge、unsupported capability与artifact tamper均被拒绝。
- [ ] **AC10｜需求交付｜来源：用户同意training闭环**：同一版本化config和CLI在Windows与Ubuntu/A6000完成固定`effect-pigment-metallic`两步training smoke，loss/gradient有限、materialization发生、checkpoint可加载且identity完整。
- [ ] **AC11｜需求交付｜来源：用户延后确定Linux版本**：Linux report记录实际OS/GPU/driver/compiler/glibc/Vulkan/Conda/Falcor/SDK/backend身份；支持声明不外推到未验证发行版。
- [ ] **AC12｜语义正确性｜来源：仓库policy**：Windows viewer Release与既有reference回归通过，任何构建后`external/`固定clone保持clean，生成物只进入`build/`或`artifacts/`。

## Out of Scope

- pbrt coated crosscheck、falcor2 MDL oracle或其他validation-only executor的公共backend化。
- 任何source material、纹理、测量表、vMaterials或viewer runtime asset的下载、同步与许可接受。
- Linux viewer、Windows viewer跨平台迁移或UE集成。
- 把不同source family归约成LayerStackIR、共同closure词汇或MDL中间表示。
- MDL multi-snapshot resource binding、连续参数采样、parameter-conditioned representation与未见参数状态泛化。
- Windows checkpoint直接在Linux resume的portability承诺。

## Deferred items and risks

- Linux MDL binary archive内部`.so`布局和锁定Falcor在目标Ubuntu版本上的实际ABI必须由后半段真实extract/build验证；若需要修改upstream，停止并另行批准根仓库patch方案。
- deployment不证明真实资产已迁移完整；资产完整性由各reference package manifest在部署后验证阶段负责。
- Windows阶段产生的platform build identity不会成为Linux验收替代证据。
