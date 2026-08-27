# ScatteringPackage@1 合同

`ncls.scattering-package@1` 是 reference 与 neural program均可使用的部署目录。manifest 精确包含 program/source/scattering identity、typed program/material descriptors、validation、provenance、files 与 content hashes。viewer的特殊请求 `source-reference` 仍可直接调用 source family权威 transport；它不是磁盘 package，也不改变本合同。

三个身份互不混用：`program_runtime_id` 覆盖程序、ABI、capability、module closure、defines 与 runtime blobs；`material_asset_id` 覆盖 source snapshot、program descriptor 与 material blobs/resources；`package_id` 覆盖前两者和 validation/provenance。

所有 URI 必须是 POSIX 相对路径且不得越界；逻辑 URI 唯一；hash 精确覆盖全部文件。typed descriptor 当前覆盖 structured/byte buffer、`texture2d-rgba16float-dds@1` 与显式 sampler；shape、stride、alignment、usage 和 DDS header/extent/mip/format 必须一致。loader 先做 schema、URI、存在性与 hash 校验，再创建 binding；任何 mismatch 都拒绝，不 fallback。包内 program 从 package 绝对路径加载，不要求预编入 viewer。

导出器通过通用 `MethodDefinition.package_validation()` 冻结方法专属验证合同。NVIDIA 包写入独立 packed-FP16 oracle的固定 view/light与 `expected_response_cos`；viewer加载同一 package shader后必须在预先冻结的 FP16容差内通过，不能用“结果有限”代替数值 parity。验证内容参与 `package_id`，checkpoint step和 oracle identity必须可追溯。
