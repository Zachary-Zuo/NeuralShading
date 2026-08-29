# 统一跨平台 Reference Backend：技术设计

## 1. 设计目标与非目标

本任务统一的是五个canonical reference program下面的执行与部署能力，不统一它们的原生材质语义。公共依赖方向为：

```text
SourceFamilyDefinition
  -> ReferenceProgramDefinition.compile_runtime/compile_material
  -> typed RuntimePayload / MaterialPayload
  -> ReferenceBackendCapability.open(...)
  -> ReferenceBackendSession.evaluate/sample/pdf
  -> OnlineTrainingProducer / tools / tests

ReferenceBackendCapability
  -> Windows: Falcor D3D12 + Slang
  -> Linux:   Falcor Vulkan + Slang

MDL ReferenceProgramDefinition (内部)
  -> MdlProgramToolchainProvider
  -> Windows DLL bridge / Linux SO bridge
```

LayerStack、MERL、OpenPBR、MaterialX和MDL继续保有不同的source schema、编辑合同、资源和program实现。pbrt/falcor2不进入这张图。

## 2. 公共运行接口

### 2.1 Backend descriptor与factory

在`ncls.references.backend`定义唯一公共平台能力：

```python
@dataclass(frozen=True)
class ReferenceBackendDescriptor:
    backend_key: str                 # ncls.falcor-reference-backend
    version: int
    platform_id: str                 # windows-x86_64@1 / linux-x86_64@1
    falcor_revision: str
    slang_revision: str
    device_api: str                  # d3d12 / vulkan
    build_root: Path
    python_module_root: Path
    runtime_library_root: Path
    semantic_identity: str
    build_identity: str

@dataclass(frozen=True)
class ReferenceCapabilityStatus:
    requirement_id: str
    category: str                    # execution / program-provider / environment
    status: str                      # ready / missing / invalid
    detail: str

@dataclass(frozen=True)
class ReferenceBackendReport:
    descriptor: ReferenceBackendDescriptor
    statuses: tuple[ReferenceCapabilityStatus, ...]
    ready: bool

class ReferenceBackendCapability:
    descriptor: ReferenceBackendDescriptor
    def doctor(self, programs=discover_reference_programs()) -> ReferenceBackendReport: ...
    def augment_environment(self, base: Mapping[str, str]) -> dict[str, str]: ...
    def open(
        self,
        definition: ReferenceProgramDefinition,
        snapshots: Sequence[SourceSnapshot],
        *, query_capacity: int,
        device: torch.device | str,
        slot_count: int = 2,
    ) -> ReferenceBackendSession: ...

def create_reference_backend(...) -> ReferenceBackendCapability: ...
```

只有factory/resolver读取OS/arch并选择build layout/API。unknown platform、错误architecture或缺build直接拒绝，不做fallback。

### 2.2 Session合同

把现有`ReferenceQueryDispatcher`实现收进backend session；query/result/lease数据合同保留。公共名称为`ReferenceBackendSession`，提供：

- `evaluate(query, wi, seeds, evaluation_samples)`；
- `sample(query, seeds)`；
- `pdf(query, wi, seeds)`；
- `end_iteration()`与`close()`；
- `reference_program_identity`、`backend_identity`、`device`。

session仍只有一套generic query shader和typed binder。它不根据family分支，只对payload descriptor的`structured-buffer`、`texture2d/3d`、sampler和`slang-module-source`做通用绑定。

`OnlineTrainingProducer`改为接收可选backend capability（测试注入）或调用factory，再由`backend.open()`获得session。runner/CLI不知道Falcor。所有工具和GPU tests迁移到同一入口；删除`import_falcor()`、`create_falcor_device()`与旧dispatcher构造兼容alias。

### 2.3 Identity

identity分三层：

1. program semantic identity：program source/shader、source contract、capabilities与bounded execution；
2. backend build identity：platform、Falcor/Slang revision、device API、实际runtime/FalcorPython build probe；
3. source identity：snapshot ids与resource hashes。

`reference_program_identity`由三层共同计算。checkpoint保存backend descriptor/build identity；Windows/Linux在没有独立portability contract时不能互相resume。program key/version/capabilities保持平台无关，build identity允许不同。

## 3. Program接口与内部provider

`ReferenceProgramDefinition.compile_runtime()`和`compile_material()`是五族真正的公共插件接口，继续保留。新增两个默认hook：

```python
def provider_descriptor(self) -> ReferenceProgramProviderDescriptor | None: ...
def preflight_provider(self) -> tuple[ReferenceCapabilityStatus, ...]: ...
```

默认返回无provider/ready；只有确实依赖额外构建产物的program覆盖。公共backend doctor只看到generic status，不访问provider具体字段。

### 3.1 五族映射

| Program | program内部工作 | 公共backend工作 | deployment资产要求 |
|---|---|---|---|
| LayerStack | pack原生layer/slab记录 | Slang/Falcor执行 | 无 |
| MERL | 把已加载测量表形成typed buffer | Slang/Falcor执行 | 无；不下载表 |
| OpenPBR | resolved inputs与锁定LUT | Slang/Falcor执行 | 无；不下载材质 |
| MaterialX | 原生document解析与typed纹理payload | Slang/Falcor执行 | 无；不下载Poly Haven |
| MDL | 内部provider生成HLSL/argument/RO/texture payload | Slang/Falcor执行generated module | 无；SDK是toolchain，不是材质资产 |

真实snapshot仍要求各自资产存在并通过package manifest；这是部署后query阶段的输入合同。

### 3.2 MDL provider

MDL内部定义`MdlProgramToolchainProvider`协议与一个subprocess bridge实现。source family和MDL program只通过factory取得provider；不公开legacy路径参数或Windows常量。

`MdlToolchainDescriptor`记录SDK root/library/plugins/target-code types/bridge以及semantic/build identity。toolchain manifest同时记录Windows/Linux官方archive身份。bridge动态加载封装：

```cpp
class SharedLibrary {
public:
    explicit SharedLibrary(const fs::path& path);
    ~SharedLibrary();
    void* symbol(const char* name) const;
};
```

Windows实现`LoadLibraryW/GetProcAddress/FreeLibrary`，Linux实现`dlopen(RTLD_NOW|RTLD_LOCAL)/dlsym/dlclose`；MDL database、class compilation、HLSL codegen、native validation与resource decode完全共享。plugin路径由platform toolchain record解析，不在C++业务代码写死后缀。CMake在Linux链接`${CMAKE_DL_LIBS}`并把Release产物放到manifest规定位置。

artifact schema保留一版公共结构并扩展`compiler_identity`的semantic/build/platform字段；cache key包含snapshot、semantic identity和build identity。旧cache自然miss，不删除。

## 4. Build/deployment manifest

新增根级`ncls.reference-backend-toolchains@1` manifest，作为runtime resolver和deployment orchestrator的共同真相。它登记：

- Falcor revision、Slang revision、Windows/Linux build profile与runtime layout；
- external Git providers：Falcor、MaterialX、OpenPBR、openpbr-bsdf、glm、stb的URL/revision/submodule策略；
- binary provider：Windows/Linux MDL SDK archive URL/size/SHA-256/root/layout；
- 五个program key到provider requirement ids的映射；
- 明确的`asset_policy: external-only-no-source-assets`。

manifest/schema拒绝duplicate ids、unknown program、unsafe relative path、非HTTPS binary URL、错误hash和未登记platform。source-material URL不得出现在该manifest。

## 5. Linux compile deployment

唯一入口：

```bash
bash scripts/deploy_reference_linux.sh
```

### 5.1 权限和输入

- 原生Linux x86-64、NVIDIA driver、Vulkan loader、Git、curl/tar、C/C++ toolchain、CMake prerequisite与Conda由管理员预装；
- 脚本不调用sudo，不安装Conda/driver；
- 可以联网获取manifest登记的external/code/SDK；
- 不读取source-material manifest以发起下载，不调用`fetch_source_materials.py`/`fetch_mdl_assets.ps1`，不写`assets/`；
- assets不存在是合法部署状态。

### 5.2 执行阶段

1. 在任何下载前探针OS/kernel/arch/glibc/GPU/driver/Vulkan/compiler/Git/Conda/network并创建report目录；
2. 检查root与已有external clone，错误revision或dirty直接失败；
3. 若缺失则按manifest shallow fetch固定revision并验证clean；初始化锁定submodules；
4. 创建或非破坏性更新`neural-shading`，安装锁定CUDA 12.8 PyTorch与editable project；
5. 安全获取/验证Linux MDL SDK与stb；
6. 执行Falcor`setup.sh`和FalcorPython Linux/Vulkan build；
7. 构建portable MDL bridge；
8. 运行backend doctor、Falcor Vulkan device probe、五个program runtime-module compile probe和MDL project fixture compile；
9. 输出JSON/中文报告和下一条launcher命令。

所有阶段用真实probe决定`fresh/reused/failed`，不用stamp。再次执行允许新建报告并复用正确产物。partial、incomplete target和hash drift只报告恢复路径，不自动删除。

compile probe不创建伪造source material，也不宣称真实资产query已通过。部署后的真实验收单独运行，并要求用户已复制资产。

## 6. Windows迁移

- 新增Windows公共backend build/doctor入口，读取同一manifest并构建Falcor/MDL provider；viewer与测试调用该入口或backend factory。
- targeted fetch/build helper可以作为实际provider实现保留，但旧用户入口若只做转发且没有独立用途则删除；不建立deprecated alias。
- 所有生产代码、tools和tests更新为backend factory/session；D3D12 hardcode只允许在明确测试platform resolver本身的fake或底层platform实现中存在。
- viewer仍是Windows/D3D12产品，但其reference program/payload身份与headless backend共用；不把viewer UI纳入公共backend。

## 7. Fixed MDL snapshot training

`nvidia.mdl-fixed-uniform@1` adapter只接受一个`mdl.program@1` snapshot，拒绝2D spatial texture依赖和不支持typed参数。按稳定name/type/channel规则把float/int/bool/enum/color/vector编码成1×1 native feature；范围归一化规则和snapshot进入layout identity，资源hash只进入identity。

`effect-pigment-metallic` smoke使用1×1 latent、step 0 bootstrap和step 1 materialization/finetune。target仍来自backend session中的canonical MDL program；response不host readback或持久化batch。短run只把有限性、梯度、lifecycle和checkpoint load作为gate。

## 8. 验证策略

### Windows gate

- unit/static：manifest、resolver、doctor、五program registry、identity、archive安全、MDL provider、adapter与config；
- public backend对五个program的代表性GPU query与lease/CUDA interop；
- MDL bridge Release、native/formal parity、资源类型覆盖；
- MDL两步training/checkpoint；
- viewer Release与Falcor clean。

### Linux/A6000 gate

- deployment在无assets副本环境运行一次并重复运行一次；
- 确认report明确`assets: not-managed`；
- 用户复制资产后，运行与Windows相同的五program backend query集合和MDL training smoke；
- Linux MDL fixture/native query与Windows冻结query比较，容差只用既有门或formal前独立calibration；
- 报告实际环境、时间和显存为observed metrics，不作为hard gate。

## 9. 失败、回滚与任务生命周期

- 公共backend迁移失败时，以新增factory/session为rollback point；不靠兼容alias维持两套路径。
- MDL provider失败不影响其他四program接口；回滚provider registration，不回滚公共backend。
- Linux build失败保留`artifacts/`report/log，不修改external；若必须patch upstream，停止并回planning。
- deployment不得以下载资产、换材质、放宽parity或增加训练预算绕过失败。
- Windows gate通过后任务仍保持`in_progress`；只有实际Ubuntu/A6000 deployment和资产后验收通过才完成。

## 10. 关键取舍

- 复用现有`ReferenceProgramDefinition`作为材质插件边界，避免第二套“统一材质接口”。
- backend统一平台执行，不统一source语义；MDL复杂性留在内部provider。
- deployment允许获取构建依赖但不管理资产，换取职责清晰和可重复编译；完整reference正确性由用户迁移资产后的独立gate证明。
- 不提供兼容alias，换取单一架构；代价是本任务必须一次性迁移全部项目调用方与tests。
