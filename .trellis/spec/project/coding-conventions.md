---
name: project-coding-conventions
description: Python / Slang / C++ 在本仓库的实际代码事实：frozen dataclass 与 __post_init__ 校验、中文 docstring、无 logging、生成的 ABI 文件、ncls 前缀、Falcor-free core、测试分层与 marker
paths:
  - src/**
  - tests/**
  - tools/**
  - scripts/**
  - shaders/**
  - apps/viewer/**
---

# 编码约定（写的是现状，不是理想）

## Python（`src/ncls/`、`tests/`、`tools/`、`scripts/`）

- 每个模块以 `from __future__ import annotations` 开头；导入顺序 future → 标准库 → 第三方 → `ncls`。
- 值对象一律 `@dataclass(frozen=True)`，校验放在 `__post_init__`，失败 `raise ValueError("<英文消息，说明期望什么>")`。范例：`src/ncls/learning/pipelines/base.py` `LearningPipelineDescriptor`、`src/ncls/data/contract.py` `ReferenceDescriptor`。不可变：返回新对象，不原地改传入参数。
- 输入类型用 `Mapping` / `Sequence`，返回具体类型；`str | None` 而不是 `Optional`；内置泛型 `list[int]`。
- 没有 `logging`，库代码不 `print`；只有 `src/ncls/cli.py` 向用户输出。失败即抛异常，不静默 fallback、不 `nan_to_num`。
- docstring 用中文、只写非显而易见的事（生命周期、为什么不进 hash、哪个文档定义了它），代码自明处不写；行内注释同样中文。
- 身份用 SHA-256：canonical JSON 固定 `ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")`；文件写入用临时文件 + `os.replace`（`training/runner.py` `_write_json_atomic`、`training/checkpoint.py`）。
- 公共合同的 schema 是 `src/ncls/**/schemas/*.json` 与 `abi/*.json`，由测试锁定；改字段先改 JSON，再改 dataclass，再改测试，再改 `docs/contracts/`。
- 路径只用 `pathlib.Path`；项目根与产物目录从 `src/ncls/paths.py` 取，不自行拼接 `artifacts/`。
- 常量元组放模块顶部（`SPLIT_NAMES`、`QUERY_ROLE_NAMES`、`CHECKPOINT_SELECTIONS`）；枚举用 `IntEnum` / `IntFlag` 且数值来自 ABI JSON（`core/scattering/contract.py`）。
- CLI 子命令 help 用中文；配置只接受完整 schema，命令行只能覆盖已声明字段。

## 测试（`tests/unit` / `tests/gpu` / `tests/integration`）

- 普通函数式测试，文件级 docstring 说明"锁定哪个合同 / 哪条规则"（`tests/unit/test_deployment_budget.py`）。
- 需要 Falcor 的测试：`falcor = pytest.importorskip("falcor")` + `@pytest.mark.falcor`，kernel 放 `tests/gpu/kernels/*.cs.slang`；需要下载资产或构建 probe 的集成测试在缺失时 `pytest.skip("<原因>")`，不伪造数据。
- 断言具体数值与 hash（`test_training_config.py` 锁旧 hash 不变）；新增 marker（如 `slangpy`）同步写进 `pyproject.toml` 的 `[tool.pytest.ini_options].markers`。
- 数据用 `tests/fixtures/`；GPU parity 容差单列，half 打包后的容差与 float32 分开。

## Slang（`shaders/ncls/`、`tests/gpu/kernels/`、`apps/viewer/shaders/`）

- include guard `NCLS_<路径大写>_SLANG`；自由函数 `ncls*`，结构 `Ncls*`，常量 `NCLS_*`；相对路径 `#include`。
- `shaders/ncls/contracts/scattering_contract.slang` 与 `layer_stack_ir.slang` 由 `src/ncls/core/{scattering,material}/abi_layout.py` 从 `abi/*.json` 生成，文件头写着"不要手工编辑"——改 ABI 先改 JSON 再重新生成。
- backend core 文件（`shaders/ncls/backends/<name>/<name>_core.slang`）必须 Falcor-free：只 `#include` `contracts/` 与 `reference/`，不 `import` Falcor 模块；合同包装（`<name>.slang`）才 `import Utils.Sampling.SampleGeneratorInterface`。
- 配置轴用 `#ifndef NCLS_<NAME>` 宏给默认值（`lobe_residual_mlp.slang`）；固定循环 `[unroll]`；可微函数标 `[Differentiable]`，IR 与方向输入 `no_diff`；Slang 内不写权重偏移常量，偏移由 Python 反射写入 `Params`。
- 只用 Falcor 8.0 锁定的 Slang 2024.1.34 已验证写法（固定数组、`typedef` 绑定 associated type、`[unroll]`）；SlangPy 携带更新的 slang，新语法先由双编译探针（`p1_v2_plan.md` P2.7）确认。
- 注释中文；每个 backend 文件头一句话说明"哪个文档定义了它"。

## C++（`apps/viewer/`）

- `namespace ncls`；校验用 `require(condition, "message")` 抛 `std::runtime_error`（`MethodBundle.cpp`）；JSON 用 `nlohmann::json`；路径用 `std::filesystem`，bundle 内 URI 必须 POSIX 相对且不越界。
- MSVC 编译加 `/utf-8`；源文件 UTF-8。
- 项目 shader 依赖显式列在 `apps/viewer/CMakeLists.txt`；新增 backend shader 必须加进去，否则不会被复制到运行目录。

## 完成前自检

- [ ] 新值对象 frozen + `__post_init__` 校验；错误消息说明期望
- [ ] 没有新的 `print` / `logging` / 静默 fallback
- [ ] 改了 JSON schema 或 ABI 就同步了 dataclass、生成文件、测试与 `docs/contracts/`
- [ ] Slang core 无 Falcor 依赖、无偏移常量、无数据相关循环
- [ ] 新 shader 已登记到 `CMakeLists.txt`；新 marker 已登记到 `pyproject.toml`
