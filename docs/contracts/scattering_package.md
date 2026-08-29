# ScatteringPackage@2 合同

`ncls.scattering-package@2`把部署产物分成三个独立section：

- `program`：module closure、defines、共享typed blobs、无文件sampler state、ABI和capability，计算`program_id`；
- `asset`：source snapshot对应的compiled blobs、typed resources与无文件sampler state，计算`asset_id`；
- `instance`：原子绑定一个`program_id + asset_id`并保存instance参数，计算`instance_id`。

`package_id`覆盖三项identity、validation/provenance及其文件hash。instance binding必须与同一manifest的program/asset逐值一致；不能部分替换、猜测或fallback。URI均为POSIX相对路径且不得越界，逻辑名和物理URI唯一，content hash精确覆盖全部文件。typed descriptor保留`kind/dtype/shape/stride/alignment/format/color_space/usage`，纹理header、extent、mip和format必须与descriptor一致。sampler以`kind/usage/filter/address_mode`显式登记并参与program或asset identity，不得被writer忽略或由viewer猜测；所有runtime binding usage全局唯一。

Python loader先完成schema、identity、URI、存在性、hash和typed resource校验，再一次性创建`ProgramRuntime + AssetBinding + InstanceBinding`。viewer按`program_id`缓存程序runtime，asset与instance分别绑定；任一段失败都不改变现有slot。特殊请求`source-reference`仍直接调用权威source transport，不伪装成磁盘package。

导出器通过通用method artifact conformance检查required runtime artifacts和Slang entry points。NVIDIA package另外冻结packed-FP16 parity oracle；真实D3D12 package shader必须在预先冻结的容差内通过，不能以“结果有限”替代数值parity。
