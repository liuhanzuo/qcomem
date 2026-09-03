# Qwen3.5 vLLM Q16 Paged Attention 正式门禁：负结果与归因边界

## 一句话结论

Job `237281` / Trial `1840009` 在 PG-19 train-only correctness gate 中按协议
fail-closed：8 个 rank 中，5 个完成门禁，3 个分别在 full-attention 第 31、31、19 层未通过
Transformers eager 与 vLLM Triton `unified_attention` 的逐元素数值兼容阈值。因此本次运行
**没有产生 validation 授权，没有读取 LongBench test-v2，也没有启动 validation 或性能
benchmark**。

这个负结果应准确表述为：**TF-eager compatibility gate 失败**。它不能直接推出
“document page reuse / CoMem block layout 错误”，因为本轮比较同时更换了 attention 数值后端；
它也不能证明 page reuse 已经正确。要单独判断布局，必须在同一个 vLLM kernel 上比较
request-owned full materialization 与 shared-document + private-tail fork，并另用 FP32 dense oracle
拆分两种后端各自的数值误差。

## 运行身份与冻结条件

- QS：Job `237281` / Trial `1840009`；
- 页面：<https://qs2.devops.xiaohongshu.com/model/production/job/trial/237281/1840009>；
- 队列 / 资源：queue `436`，8 张 H20；
- 终态：`failed_closed_before_authorization`；
- 执行时间：`0d00h17m45s`；
- 最后完成阶段：`02_static_dry_run_ok`；
- 失败阶段：`pg19-gate`；
- 模型：Qwen3.5-35B-A3B，40 层，16 query heads、2 KV heads、head dim 256；
- full-attention 层：`3, 7, 11, 15, 19, 23, 27, 31, 35, 39`；
- kernel：vLLM `0.26.0+cu129` 的 Triton `unified_attention`；
- 其余冻结环境：PyTorch `2.11.0+cu129`、Transformers `5.14.1`、Triton `3.6.0`、
  FlashInfer Python `0.6.14`；
- cache：Q16 NHD block pool，page size 128；门禁样本为 1024 个 document tokens 加
  32 个 query tokens，即 1056 个逻辑 KV tokens；GQA groups 为 8；
- 数据：只使用冻结的 PG-19 train calibration windows；LongBench validation 未开始，
  source >=68 / test-v2 未读取；
- preflight：22 项测试通过；代码、模型 manifest、模型 artifact 和 14 个 weight shards 的
  完整性检查均先于 GPU gate 通过。

关键冻结 SHA256：

| 对象 | SHA256 |
|---|---|
| code ledger | `29a5401b36a580f85c14f9ca87e89e06a471cf20db1abb3797a1414592042117` |
| model manifest | `72c9e06109702dbca958a6a528d6686b68d6b8e3376d116c0261b4c319e3da29` |
| model artifact ledger | `fa050ef64c76caaa353223541c6ad8b80be8a5f6f5c11430db2d7d4f2c4dfb5c` |
| model weight ledger | `a0352fd3fd47b4edcebf3269b5f8745490d3defb9eaedf2a4c4dc8ccae32ddf2` |
| PG-19 data | `ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c` |
| PG-19 manifest | `5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c` |
| PG-19 windows | `6010555f4a1e08220d1ece9a3590ebeb99005ba7ead9950f17659d60c7bffb5b` |
| static dry-run | `6d9bf6fe4f04720948b1475783a55830b6fc23fbbf0af41576da2ef48f6d3ce3` |

## Gate 结果

8 个 rank 各负责一个独立 PG-19 train window。每个 rank 先比较 10 个 full-attention 层的
isolated output；只有 isolated gate 全部通过，才计算该窗口的端到端 semantic row。

| rank | isolated gate | 停止位置 | max abs | mean abs | relative L2 | semantic row |
|---:|:---:|---:|---:|---:|---:|:---:|
| 0 | 通过 | — | — | — | — | 完成 |
| 1 | 通过 | — | — | — | — | 完成 |
| 2 | 通过 | — | — | — | — | 完成 |
| 3 | 通过 | — | — | — | — | 完成 |
| 4 | **失败** | layer 31 | 0.09375 | 0.00093356 | 0.00601637 | 未执行 |
| 5 | 通过 | — | — | — | — | 完成 |
| 6 | **失败** | layer 31 | 0.09375 | 0.00083133 | 0.00684625 | 未执行 |
| 7 | **失败** | layer 19 | 0.12500 | 0.00099538 | 0.00640090 | 未执行 |

逐层门槛是 `torch.allclose(rtol=0.02, atol=0.05)`，即对每个元素检查
`abs(candidate-reference) <= 0.05 + 0.02*abs(reference)`。因此不能只根据全 tensor 的
`max_abs` 判断 pass/fail：已通过的 row 中也出现过 `max_abs=0.125`，而失败的
`max_abs=0.09375` 可能落在 reference 幅值较小、允许误差更低的位置。

在 5 个完整 rank 的 50 个 isolated rows 上：

- relative L2：min `0.00138817`、mean `0.00527671`、max `0.00769012`；
- mean abs：min `0.00065111`、mean `0.00092900`、max `0.00141390`；
- max abs：min `0.015625`、max `0.125`；
- layer 31 的跨窗口 mean relative L2 最高，为 `0.00682148`。

这些量都有限且整体较小，但协议要求所有 layer × window 逐元素通过，所以 3 个失败不能被
均值掩盖，也没有事后放宽阈值。

## 已完成 semantic rows

失败 rank 在 isolated gate 处立即退出，因此只有 rank 0、1、2、3、5 产生 semantic row：

| rank | PG-19 source | greedy top-1 | full-vocab forward KL | max abs logit error |
|---:|---|:---:|---:|---:|
| 0 | `train/10049.txt` | 一致 | 0.00299178 | 0.421875 |
| 1 | `train/10045.txt` | 一致 | 0.00197685 | 0.7890625 |
| 2 | `train/10032.txt` | 一致 | 0.02970412 | 2.390625 |
| 3 | `train/10020.txt` | 一致 | 0.00219527 | 0.3720703 |
| 5 | `train/10021.txt` | 一致 | 0.00165545 | 0.1972656 |

5/5 greedy top-1 一致，但这不能替代完整 semantic gate。5 个 KL 的 example-equal mean 为
`0.00770469`，且每个已完成窗口都高于预注册的 global mean KL 阈值 `0.001`。所以即使忽略
3 个 isolated failures，这 5 个已观察窗口也不支持生成 authorization。由于另外 3 个窗口没有
semantic row，这不是完整 8-window KL 估计，不能拿它当最终质量均值。

## 这次负结果说明什么

### 可以确认

1. 当前 Q16 vLLM paged 路径不能满足预注册的 **Transformers eager 数值兼容契约**。
2. 失败具有 workload / layer 依赖性；layer 31 在已完成窗口中也是 relative L2 最敏感的层。
3. 门禁按预期 fail-closed；没有 authorization、validation shard、ABBA timing 或 memory
   benchmark artifact，因此没有可报告的 TTFT、TPOT、CUDA/NVML peak 或多请求容量收益。
4. 失败调用的审计几何一致：physical pool `(9,128,2,256)`、active table `(1,9)`、
   query 32、KV 1056、GQA 8、Q16；审计还记录 0 dense fallback、0 full-KV concat、
   0 full-document query-fork staging copy。

### 不能据此确认

1. **不能确认 page reuse layout 失败。** 本轮 oracle 是 HF eager dense attention，candidate 是
   vLLM Triton paged kernel；数值 kernel 与 cache ownership/layout 同时变化，归因没有被隔离。
2. **也不能确认 page reuse layout 正确。** 上述几何与 0-copy telemetry 只证明候选路径确实被
   调用，不等价于 logical K/V payload、page table 与 request-owned control 完全一致。
3. 不能把 5/5 top-1 一致表述为“几乎无损”；KL 门槛未过，且缺少 3 个 semantic rows。
4. 不能报告部署加速或显存节省；正式 validation/benchmark 根本没有运行。

## 数值根因：当前证据与置信边界

目前最强的解释是两种 attention backend 的 BF16 舍入与 reduction 顺序不同：

- HF eager 先用 dense BF16 QK matmul 形成整行 score，使用 FP32 softmax，再把概率 cast 回
  BF16 并进行 value GEMM；
- vLLM Triton `unified_attention` 使用 page/tile 化 `tl.dot`、FP32 online-softmax 的 running
  max/sum，并按不同顺序做概率 cast、value reduction 与最终归一化；
- 即使 post-RoPE Q/K/V、mask、scale、GQA 映射和逻辑 token 顺序相同，不同归约树也不会保证
  BF16 逐位一致；内容相关的 near-zero 元素会使同一全局误差量在不同窗口触发或不触发逐元素
  `allclose`。

这与“误差有限但随窗口和层变化、layer 31 较敏感”的观测一致，但仍是**强假设而非已完成的
FP32 oracle 归因**。在完成下面两个解耦对照前，不能排除 packing/table/layout bug：

1. 对完全相同的 post-RoPE Q/K/V，使用同一个 `unified_attention` kernel 比较 fresh
   request-owned materialized block pool 与 shared-document + private-tail fork；比较 canonicalized
   logical page table 和仅含 valid positions 的 logical K/V，不比较无语义的物理 block id、预留
   capacity 或未初始化 padding；
2. 用 FP32 dense attention oracle 分别测 HF eager 与 vLLM 输出误差。该 oracle 只做后端数值
   归因，不混入 CoMem layout correctness gate，也不据此事后改阈值。

## Artifact 与本地完整性复核

本地只读镜像：

`results/gpu-qwen35-vllm-paged-q16-formal-negative-20260814b/`

核心文件：

- `failure-summary.json`：负结果摘要；
- `failure-artifacts.sha256`：原始运行 artifacts 的 28 项账本；
- `pg19-gate-shards/pg19-gate-shard-{0..7}.json`：8 个 rank 的门禁结果；
- `logs/pg19-gate-rank-{0..7}.log`：rank 日志；
- `code.sha256`、`model-artifacts.sha256`、`model-weights.sha256`：冻结账本；
- `static-dry-run.json` 与 preflight / integrity logs。

在本机从该目录执行 `sha256sum -c failure-artifacts.sha256`，28/28 项全部为 `OK`。
账本自身与摘要的 SHA256 分别为：

- `failure-artifacts.sha256`：
  `4152263048062a34067a3131994e9a3729bdfdb9ce2cf258c02c86c7e48c39f1`；
- `failure-summary.json`：
  `1b77802f3d1110da8ad79b77a13719641d0d3ada378a3be5bf264089300a24d7`。

目录共 30 个文件：28 个被账本覆盖的原始 artifacts，加账本自身和事后只读整理的
`failure-summary.json`。后两者不自包含在 28 项账本内，以上分别给出独立 SHA256，避免循环
校验或模糊覆盖范围。
