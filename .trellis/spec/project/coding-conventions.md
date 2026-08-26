# 编码约定

Python 使用 future annotations、frozen dataclass、`__post_init__` fail-stop、`pathlib.Path` 与 canonical SHA-256；项目 Python 只在 `neural-shading` 环境运行。Slang 类型/函数以 `Ncls`/`ncls` 命名，固定循环有界。C++ 使用 `namespace ncls`、`nlohmann::json`、`std::filesystem`，package URI 必须安全相对。中文 UTF-8 文件在 PowerShell 读取时显式 `-Encoding UTF8`。
