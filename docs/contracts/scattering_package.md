# ScatteringPackage@1 合同

`ncls.scattering-package@1` 是 reference 与 neural program均可使用的部署目录。manifest 精确包含 program/source/scattering identity、typed program/material descriptors、validation、provenance、files 与 content hashes。viewer的特殊请求 `source-reference` 仍可直接调用 source family权威 transport；它不是磁盘 package，也不改变本合同。

三个身份互不混用：`program_runtime_id` 覆盖程序、ABI、capability、module closure、defines 与 runtime blobs；`material_asset_id` 覆盖 source snapshot、program descriptor 与 material blobs/resources；`package_id` 覆盖前两者和 validation/provenance。

所有URI必须是POSIX相对路径且不得越界；逻辑URI唯一；hash精确覆盖全部文件。typed descriptor保留`kind/dtype/shape/stride/alignment/format/color_space/usage`，覆盖structured/byte buffer、typed texture与动态module source；shape、stride、alignment、usage和纹理header/extent/mip/format必须一致。loader先做schema、URI、存在性与hash校验，再创建binding；任何mismatch都拒绝，不fallback。包内program从package绝对路径加载，不要求预编入viewer。

导出器通过通用`MethodDefinition.package_validation()`冻结方法专属验证合同。NVIDIA包写入独立packed-FP16 oracle的固定view/light与`expected_f`；viewer加载同一package shader后必须在预先冻结的FP16容差内通过，不能用“结果有限”代替数值parity。验证内容参与`package_id`，checkpoint step和oracle identity必须可追溯。
