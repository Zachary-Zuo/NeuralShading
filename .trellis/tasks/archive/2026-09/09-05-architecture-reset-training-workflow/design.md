# 项目架构重置：技术设计

> 状态：已交付，用户授权提交并归档，用户已批准实施，当前代码与本机验证完成。Linux 保留数值 validation 和统一图像 eval 接口，图像执行当前为空操作；已批准实施。

## 1. 用户入口

唯一日常训练入口：

```bash
python -m ncls train 0 --config configs/training/runs/example.yaml
python -m ncls train 0,1,2,3 --config configs/training/runs/example.yaml
```

在 neural-shading 环境运行。GPU 只指定一次；单卡与 Linux DDP 共用 config、engine、方法实现和输出管理。Windows 多卡保留当前不支持边界，本次不开发新的分布式 backend。

`ncls.__main__` 先进入仅依赖标准库和轻量配置读取的启动层。启动层定位项目与当前环境、准备 Falcor/CUDA 动态库、解释设备列表，然后用当前解释器启动实际进程；需要 Linux LD_LIBRARY_PATH 生效时在新进程导入原生库。多卡由同一层内部启动 torchrun，rank 进入私有 worker 后只看到自己的物理卡映射为 cuda:0。

训练 CLI 不再承担路径拼接、模型注册适配、reader 分流与平台 shell 选择。现有单纯转发的 multi.sh、旧训练 CLI 和重复设备解析删除；仍有实际环境用途的调用方切到同一 Python 启动模块，不保留旧入口做兼容。

## 2. 目录与输出

| 目录 | 职责 |
|---|---|
| src/ncls | 项目 Python 实现 |
| shaders/ncls、apps/viewer | 共享着色器和 Windows 应用 |
| configs | 正式可编辑配置；不保存运行后生成的配置副本 |
| references | source/reference 清单、版本和必要合同 |
| assets | 原始源材质、纹理和 viewer 场景输入 |
| external | 锁定第三方源码/SDK |
| build | 可重建的构建结果与编译缓存 |
| outputs | 按 config 聚合的新训练成果 |
| artifacts | 临时研究产物，以及用户要求原地保留的旧 viewer 图像 |
| docs、.trellis | 稳定文档与任务记录；原始诊断脚本/清单在当前任务 scratch |

新输出布局：

```text
outputs/<config-key>/<run-id>/
  config.yaml
  resolved.yaml
  run.json
  checkpoints/
  tensorboard/
  eval/
  exports/
  logs/
```

一个小型 RunPaths 对象生成所有路径并交给消费者。config-key 默认为入口文件名；run-id 自动生成，用户日常不需要填写。短 run-id 避免 Windows 嵌套 shader 路径过长。新启动创建新 run；明确 --resume 才回到原 run 并恢复 step。创建 run 的进程在 DDP 启动前决定目录，各 rank 不各自生成时间戳。

run.json 只记录配置来源、种子、命令、代码提交、设备、开始/结束状态等事实，不成为多层身份协议。resolved.yaml 便于人阅读最终参数；checkpoint 保存恢复所需的模型和训练配置。日志、图像和导出全部使用 run 路径；只写已实际使用的子目录。

旧成果保持原位置。新代码不读取旧 artifacts 来补配置、加载权重或发现默认 package。viewer 的新包发现/导出路径指向 outputs；原始 viewer 资产仍属于 assets。

## 3. 代码职责

```text
Python 启动层
  -> 配置解析 + RunPaths
  -> 公共训练 engine + online data session
       -> 当前方法直接提供 model / objective / data / lifecycle
       -> checkpoint
       -> TensorBoard
       -> 统一图像 eval hook（Windows 本机实现 / Linux 空实现）
  -> 当前 checkpoint 的评估/导出
       -> 公共 package 与 viewer
```

保留 core、source_materials、references、data 的真实领域边界。reference 继续保留五族原生语义和 GPU online query；不保存训练 batch。

方法仍只有一个显式注册入口，按 Nvidia/Metal 目录并置 model、objective、data 和 compiler。接口绑定真实实现，不再通过六个 `_Definition*` adapter 回调旧 MethodDefinition。移除旧 definition 与没有独立含义的 facet hash。公共数值 helper 仅在存在当前多个消费者时进入 common；不为每个函数新建 service/registry/factory。

训练状态、配置和 checkpoint 不再在 v1/v4/runner/snapshot 之间转换。方法自身普通权重优先使用标准 state_dict；确有方法专属状态时由同一当前接口显式提供，不能再复制一套文件 reader。

旧 full Metal 的当前复用 helper 先提取到共享位置，然后删除旧 method/model/runtime/独占 shader、配置和测试；不能让当前 budgeted 模型继续继承一个退役模型 adapter。新模型设计与训练质量不在本任务内。

## 4. 配置与检查

YAML 是可调值的唯一来源。保留有用的 base/method/data/recipe 组合，但去掉用户必须手写的内部 schema/version/实现 hash 标签和“只能等于当前 recipe”限制。解析完成后得到一次构造的配置对象，engine、hook、compiler 直接消费。

| 检查类型 | 处理 |
|---|---|
| reference_spp 恰好 1024、neural_spp 必须小于 reference | 删除；spp 独立从 YAML 传给对应 renderer |
| 公共配置只能采用某一 checkpoint selection、固定研究阶段名称/实验标签 | 删除人为门禁；实现只判断所需操作/数据是否存在 |
| 每层重新检查相同 dictionary 字段、freeze/thaw、重复整文件 hash/load | 收到真正输入边界，内部不重复 |
| 所有 YAML 与源码 hash 完全相同才能加载模型 | 删除；来源记录与执行正确性分开 |
| preview 必须训练至少一步、导出必须 formal/complete/完整梯度证据 | 改为报告；允许当前模型在可执行状态下预览/导出，包括初始化状态 |
| 配置拼写/类型、spp 正整数、缺失文件、当前模型张量结构 | 在对应输入/加载边界处理一次，错误直接说明原因 |
| GPU 物理映射、DDP collective 次序、跨语言资源 ABI 与大小 | 保留真实执行要求，避免静默运行错误资源 |

有意义的 source/resource 内容 identity 继续用于定位资产、缓存失效和实际绑定；不再把同一身份复制到每个中间对象并重复全等检查。训练稳定性、梯度覆盖和性能 profile 是可观察诊断，不是普通 train 必须先过的一组研究门禁。

## 5. 训练状态、配置来源与运行设置

一份当前 checkpoint 包含：模型结构配置/权重、optimizer/scheduler/必要 precision 状态、phase/global step、RNG 与 online query cursor、恢复所需的数据定义，以及轻量来源记录。阶段完成也不因“complete”标签强制丢弃用户可能用于继续训练的 optimizer 状态。

- 更换输出位置、日志 cadence、TensorBoard、图像 spp、相同卡数下的物理 GPU 编号，不因源码或整个 plan hash 改变而拒绝加载。
- 评估/导出只消费模型与所需资产，不先构建完整训练 engine 或要求历史运行环境相同。
- 精确续训检查实际要恢复的 optimizer/phase/data cursor 与模型结构，不对所有记录字段做全等比较。
- 不承诺改变卡数或训练 batch 后仍产生相同随机轨迹；实现按实际状态恢复能力给出具体行为，不能用一个全 plan hash 错误掩盖原因。不同拓扑的弹性状态迁移不属于本次范围。
- TensorBoard 续训从恢复 step 接续，并正确处理回退 checkpoint 之后已有的事件；不要只截断 JSONL、让曲线保留另一条历史。

只读一种新架构状态，不提供 v4/v1 importer、格式识别分流、转换工具或旧 checkpoint 测试 fixture。旧权重本任务不用，也不修改。

## 6. 平台与 eval

数值 validation 在两个平台保留，使用同一配置、engine 调度和指标汇总；Linux DDP 的数值统计仍由训练 rank 共同计算，rank 0 写出。图像 eval 同样保留一个公共调用点与接口，平台差异只在启动装配时选择实现，不在 engine/method/data 内分叉训练流程。

最小图像接口为 `evaluate(model, context) -> VisualResult | None`。context 仅传递当前 step、图像配置、source 与输出位置等执行需要的数据；它是进程内对象，不序列化成版本化请求。Windows 实现返回图像结果，Linux 空实现直接返回 None；公共 hook 仅在有结果时交给 TensorBoard。配置关闭图像评估时也可使用同一个空实现，不另建禁用流程。

Linux 空实现不导入 Windows renderer，不先准备 probe、编译 package、复制模型、保存 checkpoint、创建 eval 目录或访问 GPU 再返回；没有请求队列、后台服务、额外 collective 或训练 RNG 消耗，也不发布“图像评估成功”事件。公共 cadence 和接口保留，未来增加 Linux renderer 只替换装配出的实现。该接口具有明确的当前职责，不保留旧 spool/worker 协议或增加插件发现框架。

Windows 图像实现是本机一次普通操作：在配置 cadence 暂停该单卡训练的优化步骤，使用当前模型和独立 probe 状态生成对照图，由公共 hook 交给同一个 TensorBoard writer。probe 与渲染资源只在实际实现执行时准备。同步渲染的耗时单独记录；TensorBoard 的文件写入可沿用简单有界后台线程。无需保存完整 optimizer 快照、请求协议、spool、claim、worker 身份、迟到 collector 或独立 eval GPU。

默认 reference_spp=128，neural 默认 deferred；用户可通过 YAML 改变可执行的模式和 spp，不增加对某个数字的版本门禁。渲染器实际使用配置的采样数，最后一次 dispatch 截断到剩余预算；该规则表达执行正确性，不锁定数字。

Windows 训练图像与正式物理评测用途明确记录。PT-vs-deferred 的 difference 不能自动解释为纯模型误差；本任务不扩展另一套 renderer 或研究新的积分算法。独立 Windows eval 可读取新训练结果，无论其权重来自哪台训练机。

checkpoint、数值 validation、图像 eval 与日志按自己的 cadence 调度；不为出图强制写 checkpoint，也不为保存模型强制运行 validation。

## 7. 删除与文档收口

具体证据见 research/architecture-audit.md。删除目录/符号清单至少覆盖：

- legacy_checkpoint、v1/v4 往返转换、旧 evaluation/deployment 读取分叉。
- MethodDefinition 到 MethodPlugin 的转发适配和退役 full Metal 独占链。
- visual_eval 的跨机 contracts/spool/worker/collector 与专用于旧协议的 CLI、配置、测试；替换为统一进程内图像接口，保留两个平台共同调用点和 Linux 空实现。
- shell 多卡薄转发器、重复设备参数入口、旧 corpus/HDF5/CLI 说明与已无消费者工具。
- 冻结当前配置 hash、固定 spp、历史格式兼容成功等只保护旧实现的测试。
- 仅为旧训练 handoff、固定历史 checkpoint 和历史部署包路径服务的代码。

README、AGENTS/CLAUDE、repository_policy、architecture、learning、Linux 使用说明、TESTING 和相关 spec 同步更新。历史研究结论仍可留作有范围说明的记录，不将它们重新包装为当前方法要求，也不为了保留旧链接继续保留运行兼容。

## 8. 实施边界与回退

用户已批准并完成本次实现，本机验证与 Linux 交接见 research/validation.md。采用一个任务中的顺序改造，因为入口、状态和输出相互依赖；不建立额外任务树或长期并行新旧系统。

回退依靠 Git 中的代码版本。用户既有 artifacts、assets、external 不因回退被移动/删除；不存在旧成果迁移或新旧模型格式桥接。
