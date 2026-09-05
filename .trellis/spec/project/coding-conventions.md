# 编码约定

Python 使用 future annotations、合适的 dataclass 和 `pathlib.Path`；输入、张量恢复和外部资源在对应边界验证一次，内部不反复重验。SHA-256 用于资源 identity 和来源记录，不把运行设置或实验观察变成全等门禁。项目 Python 只在 `neural-shading` 环境运行。Slang 类型/函数以 `Ncls`/`ncls` 命名，固定循环有界。C++ 使用 `namespace ncls`、`nlohmann::json`、`std::filesystem`，package URI 必须安全相对。中文 UTF-8 文件在 PowerShell 读取时显式 `-Encoding UTF8`。
