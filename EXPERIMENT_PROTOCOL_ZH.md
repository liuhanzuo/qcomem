# Q-CoMem Edge 实验前提与报告协议

本文定义哪些运行可以进入正式表格。目标是把模型/算法差异与笔记本的动态电源、温度
和内存状态分开。程序会自动记录可观测条件，但仍需由实验者固定房间环境和后台任务。

## 1. 正式运行的硬前提

| 项目 | 正式要求 | 原因 | 自动处理 |
|---|---|---|---|
| 电源 | 接入 AC，整个 run 不拔电 | 电池模式会改变 CPU/GPU 功率预算 | 默认不满足即退出 |
| 电源模式 | 关闭低功耗；固定为同一模式 | `powermode` 会改变吞吐、温度和频率 | 记录活动 profile；低功耗即退出 |
| 温度 | 冷态主表开始/结束均为 nominal；持续负载表单独报告热状态轨迹 | 热节流会使后半程系统性变慢 | 前后读取 thermal state 和 `pmset` |
| 后台负载 | 开始前系统 CPU 不高于 35% | 浏览器、索引和同步会污染时间 | 超限即判无效，并记录高负载进程 |
| swap 增长 | 单次 run 不超过 128 MiB | 换页会污染延迟且掩盖统一内存压力 | 前后比较，超限即判无效 |
| 模型 | 仓库与 immutable commit 固定 | `main` 更新会改变权重或配置 | 内置模型固定 commit；其他远程模型强制传 revision |
| 数据 | corpus、query、选择结果冻结 | 防止不同配置看到不同证据 | JSON 数据集和 frozen selector |

已有少量 swap 不会被程序直接判无效，因为 macOS 可能在内存压力解除后仍保留 swap。
但论文性能实验应在重启并静置后开始，报告初始 swap，并优先使用初始 swap 接近 0 的
run。只要本次增长超过阈值，该 run 仍会自动失效。

默认正式入口：

```bash
make comem-multidoc
make mlx-replay
```

如果前提不满足，程序只写出 `status=preflight_failed` 的 JSON，然后在下载/加载模型前
退出。`--power-policy record-only` 只允许继续做功能排错，不会绕过有效性标签。

`make mlx-replay` 固定执行 H20 选出的 Qwen3.5 hybrid state policy：depth 7、residual Q4、
full-attention cache Q4、linear recurrent state Q8。它必须把模型权重 revision、完整
`residual + lower cache` 字节、exact Q16 replay、packed replay、配置顺序和 warm-up 一起
记录，不能只报告 residual 文件大小。

## 2. 人工固定但程序无法完全保证的条件

正式采集前应完成以下操作：

1. 接上相同功率的电源适配器，关闭低功耗模式，记录所用电源模式。
2. 重启后在 `22±2°C` 室温静置至少 10 分钟；正式开始前间隔 60 秒检查两次，均须为
   `thermal_state=nominal`，且静置期间 swap 不继续增长。
3. 关闭 Spotlight 大规模索引、Time Machine、云盘同步、视频会议和浏览器重负载；
   固定显示器连接方式、屏幕亮度和外接设备，不在 run 中改变它们。
4. 先完成模型下载。冷下载、首次文件读取和 warm inference 必须分开报告。
5. 正式比较使用相同 model commit、tokenizer、数据集文件、split depth、chunk/overlap、
   group size、采样/teacher-forcing 设置和代码 commit。
6. 每个配置至少运行 5 个全新进程；使用冻结的循环/随机顺序交错 baseline 与 candidate，
   报告 median、IQR、CV、完整范围和配对 speedup，而不是只报告最快值或独立均值之比。
7. 性能测试与质量测试分开：teacher forcing 用于稳定比较分布，不等于真实生成质量；
   最终还需固定解码参数做端到端 QA 指标。

### 2.1 冷态/交互式主表

冷态主表回答“用户在散热恢复后的单次查询能有多快”。每个 context length 单独运行，
流程为：模型常驻加载、一次不计时 warm-up、同一轮内配对执行所有路径、下一轮改变路径
顺序。每个独立进程开始时必须 nominal；若结束为 fair/serious/critical、swap 增长超过
128 MiB 或后台负载超限，则整轮标记为 invalid。只重跑失效编号，不挑选更快的替代值。

主表至少给出 TTFT/prefill、decode tok/s、Write、Read、持久化字节、峰值统一内存，及
candidate/baseline 的逐轮配对比值。冷启动模型加载时间单列，不混进 warm-path 延迟。

### 2.2 持续负载/热稳态表

持续负载表回答“连续查询时是否降频”。它与冷态主表使用不同 session：从 nominal 开始，
丢弃一次 warm-up，固定 prompt、greedy 解码、输出 token 数和轮间间隔，连续执行 20 轮。
逐轮记录 throughput、TTFT、thermal state、swap、内存和可用的功耗代理，报告完整曲线、
进入 fair/hot 的轮次、稳定平台段 mean±std/CV。这里 thermal 变热是被测现象，不是删掉
后半程；但出现 swap 时仍需单独标注为 memory-pressure regime，不能解释成纯热降频。

这两个口径不得混在同一 speedup 数字中。2026 年持续端侧评测采用了“重启、22±2°C
静置 10 分钟、一次 warm-up、60 秒温度稳定检查、20 轮”的流程；早期 iPhone 实机研究
则固定满电、无后台、同 OS、greedy 解码、warm-up 与轮间间隔，并用 9 次测量取平均。
PowerInfer-2 报告 10 次平均，但没有同等详细地公开环境温度/thermal-state 控制，因此
我们的协议采用前两者更严格的可审计条件：

- https://arxiv.org/pdf/2603.23640
- https://arxiv.org/pdf/2312.12472
- https://arxiv.org/pdf/2406.06282

## 3. 多文档实验中的受控变量

主量化实验使用 `selection=frozen`。每个 query 的文档 ID 在数据集中预先给定，因此：

- dense baseline、BF16 CoMem 和各 bit 配置读取完全相同且顺序相同的文档；
- dense 与 BF16 CoMem 的差距记为 **interface gap**；
- BF16 CoMem 与量化 CoMem 的差距记为 **quantization gap**；
- frozen 文档与 relevant 文档的交集记为 evidence recall。

`selection=bm25` 是另一组 selector sensitivity 实验，不能与 frozen 主结果混在同一列。
未来加入学习型 selector 时也应先冻结其输出，再比较存储位宽。

每篇文档在一个 split depth 只做一次 lower-layer Write，随后复用于全部 query。各 bit
配置由同一 BF16 residual 量化得到。query residual 保持 BF16，当前实验只改变持久化
文档 residual 的位宽。

## 4. 当前质量指标及其边界

若 query 提供 `expected_answer`，程序一次性输入答案并对每个答案位置计算：

- BF16 reference 到 candidate 的 mean KL 与最大 position KL；
- 每个答案位置的 top-1 agreement；
- reference/candidate 对目标 token 的 NLL 及其差值；
- BF16 CoMem 相对 dense prompt 的同类 interface 指标。

这是 teacher-forced correctness calibration，适合低成本筛掉危险的 bit/depth，不代表
自由生成答案已经等价。后续论文评测还需加入 exact match、F1、引用正确率、长答案生成
和多轮 query；性能路径则需加入带 KV cache 的增量 decode，不能用当前整段 forward
时间冒充 decode tok/s。

## 5. 一个配置进入正式表格的条件

运行级条件：

- JSON 中 `status == "completed"`；
- `formal_result_eligible == true`；
- `environment_assessment.reasons` 为空；
- 所有重复 run 使用相同固定条件，且没有中途休眠或其他交互。

算法级默认校准条件（可在 CLI 修改）：

- corpus residual relative RMSE 不超过 0.05；
- 所有答案位置的最大 KL 不超过 0.02；
- 每个 query 的 top-1 agreement 达到配置阈值；
- 在满足约束的候选中选择实际 `corpus_stored_nbytes` 最小者，而不是名义 bit 最小者。

## 6. 报告最小字段

论文或实验日志至少报告：

- 芯片、GPU 核数、统一内存、macOS、MLX/MLX-LM 与代码 commit；
- 模型仓库、模型 commit、权重量化格式；
- 电源来源、活动 `powermode`、运行前后 thermal state、初始/新增 swap；
- corpus 文档数/token 数、query 数、选择方法与 top-k；
- split depth、chunk size、overlap、residual group size 和位宽策略；
- Write 总时间与摊销假设、每 query Read/解量化延迟、实际字节数；
- interface gap、quantization gap、任务质量，以及所有重复运行的离散程度。

`powermetrics` 能提供更细的能耗信息，但通常需要管理员权限，因此不放进自动默认流程。
正式能耗实验应单独使用 Instruments 或经明确授权的 `powermetrics`，并把采样开销和
采样频率写入报告。
