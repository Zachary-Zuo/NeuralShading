# ScatteringPackage@1 合同

`ncls.scattering-package@1` 是 reference 与 neural program 共用的部署目录。manifest 精确包含 program/source/scattering identity、typed program/material descriptors、validation、provenance、files 与 content hashes。

三个身份互不混用：`program_runtime_id` 覆盖程序、ABI、capability、module closure、defines 与 runtime blobs；`material_asset_id` 覆盖 source snapshot、program descriptor 与 material blobs/resources；`package_id` 覆盖前两者和 validation/provenance。

所有 URI 必须是 POSIX 相对路径且不得越界；逻辑 URI 唯一；hash 精确覆盖全部文件。loader 先做 schema、URI、存在性与 hash 校验，再创建 binding。包内 program 从 package 绝对路径加载，不要求预编入 viewer。
