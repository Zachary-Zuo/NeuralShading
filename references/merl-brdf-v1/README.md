# MERL 测量 BRDF reference package

这一族的 GT 是 MERL 发布的 100 个各向同性测量 BRDF 二进制表、它的 Rusinkiewicz half/difference 参数化、官方索引和 RGB scale。项目不会先拟合解析 PBR 参数再把拟合结果称为 GT。

原始 ZIP 固定为 Zenodo record `8101681` 的 `BRDFDatabase.zip`，以发布方 MD5 校验。adapter 支持原始表身份 round-trip、向量化查表、`response_cos` 查询和离线预览；离散表之外不声称存在原生连续材质参数。

数据集本身按 2023 发布包的 `README.md` 使用 `CC-BY-SA-4.0`。发布包附带的 `BRDFRead.cpp` 仍有较早的教育、研究和非营利用途声明，因此 manifest 将“数据许可证”和“示例代码声明”分开记录；项目 adapter 是依据公开参数化独立实现的代码，不复制该示例源码。
