# Qwen3.5 vLLM Q16 Paged Multi-Fork Resident 正式结果

## 一句话结论

Job `237580` / Trial `1840837` 完成了同一份约 4K PG-19 train-only 文档同时服务
`N={1,2,4,8,16,32}` 个常驻请求的 Q16 same-kernel 容量实验。fresh arm 为每个请求完整
物化独立文档 block pool；reuse arm 只保留一个只读文档 pool，并为每个请求预留私有
tail/append pages。8 个 rank 的全部 N 点上，两个 arm 的生成 token、每一步完整词表 logits、
最终逻辑 K/V 和 GDN state 都 exact；同一 query 在不同 N 中也保持 exact。

可重放的解析分账为：fresh `80 MiB + 90 MiB × N`，reuse
`80 MiB + 5 MiB × N`，即每新增一个常驻请求节省 `85 MiB`，其中避免了 `80 MiB`
物理文档 block copy。N=32 时，fresh/reuse 的 full-attention pool 分别为
`2960/240 MiB`，节省 `2720 MiB`。对应的 PyTorch allocator production absolute peak
allocated 中位数为 `74,623,183,360 B` 和 `71,765,915,136 B`，相差
`2,857,268,224 B`（`2724.903 MiB`，约占 fresh peak 的 `3.83%`）。

这不是并发吞吐或延迟正结果。N 个对象在整个 setup/generation 期间同时存活，但模型步在
单 CUDA stream 上按 round-major 顺序执行；本轮没有 ABBA timing 汇总，也没有 vLLM engine
scheduler、continuous batching、ragged batch、NVML peak、F1 或多文档实验。

## 本实验回答什么问题

此前的 [same-kernel fair v2](RESULTS_GPU_QWEN35_VLLM_PAGED_FAIR_V2_2026-08-14_ZH.md)
已经证明单请求 fresh full-copy 与 shared-document reuse 在同一个 vLLM
`unified_attention` kernel 下 bitwise exact，并观察到单请求 cache 增量显存下降，但
absolute 模型 peak 只差约 `83.64 MiB`。本轮只扩展一个变量：同一文档的常驻 request fork
数量。

两个 arm 的受控定义为：

- `vllm-q16-fresh-full-copy-control`：保留公共 source 以固定 common prefill/pack，再为 N 个
  请求分别分配独立 document + private pool，并复制完整物理文档 blocks；
- `vllm-q16-shared-document-reuse`：N 个请求共享同一不可变 document pool，只使用 source
  arena 为每个请求预留的 private pages 承担 partial-tail COW 和 append。

所有请求对象在 setup、8 轮生成和 allocator snapshot 期间都有强引用并同时存活。reuse
requests 共享同一个 source arena，但 private block reservations 互不重叠；fresh requests
的 arena storage 两两独立。各 N 严格使用同一冻结 query bank 的前 N 条，所以容量曲线是
嵌套可比的。

## 运行身份与冻结条件

- QS：Job `237580` / Trial `1840837`；
- 页面：<https://qs2.devops.xiaohongshu.com/model/production/job/trial/237580/1840837>；
- 终态：`Complete`，运行时间 `0d00h13m32s`；
- 资源：8 张 H20-3e，每张 `143,771 MiB`，driver `570.133.20`；
- 模型：Qwen3.5-35B-A3B，40 层，16 query heads、2 KV heads、head dim 256；
- full-attention 层：`3,7,11,15,19,23,27,31,35,39`；
- 其余 30 层为 linear GDN，两个 arm 使用相同 functional state 语义；
- 环境：PyTorch `2.11.0+cu129`、Transformers `5.14.1`、vLLM
  `0.26.0+cu129`、Triton `3.6.0`、FlashInfer Python `0.6.14`；
- kernel：`vllm.v1.attention.ops.triton_unified_attention.unified_attention`；
- cache：Q16 NHD block pool，page size 128，batch 1；
- N：`1,2,4,8,16,32`，每个 rank 都完整运行全部 N，而不是把 N 拆到不同 GPU；
- generation：每请求 32-token query + 8 个 greedy tokens；
- 文档：4095 tokens，`4095 mod 128 = 127`，是故意选择的近满 partial-tail stress；
- 数据：每 rank 一个不同 PG-19 train book/window；每个 book 冻结 32 个互不重叠、内容
  两两不同的 32-token 原文 query，stride 64；没有 synthetic marker；
- 数据治理：`longbench_consumed=false`、`source_6_9_consumed=false`、
  `source_68_99_consumed=false`、`test_v2_consumed=false`。

关键 SHA256：

| 对象 | SHA256 |
|---|---|
| code ledger，135 项 | `44c3a86a2cd7db7afcb7ee0cb29af91625ec3fcf5374c509146d92a451824ff9` |
| runtime protocol manifest | `975bc6a12f43447024b889889d4156ca71c2f89b68de6157ac609b4a9687e9c0` |
| protocol config | `eb18a5da70c8eeb6da2ab054f188b74772c07005fe31f88c81287dda09aaa6e4` |
| QS YAML | `931a9ef0bff7356fb3a4542b13599ec5b0db71de6fb917f8b0c4ca855933e5d0` |
| model manifest | `72c9e06109702dbca958a6a528d6686b68d6b8e3376d116c0261b4c319e3da29` |
| model artifact ledger | `fa050ef64c76caaa353223541c6ad8b80be8a5f6f5c11430db2d7d4f2c4dfb5c` |
| model weight ledger | `a0352fd3fd47b4edcebf3269b5f8745490d3defb9eaedf2a4c4dc8ccae32ddf2` |
| PG-19 train data | `ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c` |
| PG-19 manifest | `5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c` |
| PG-19 4095-token windows | `27ad6c687e5cab28f361bbd89dd1844788aecbecc6f2d25dbd0c60b7705a55f8` |
| static dry-run | `cacd703cf670931e8525e4a122e10618af7d865c7284cd77c7292992f99184b2` |
| final summary | `cbb5d86112e0242b50369c68ede25978a4a5ce51d0517c69d7925fcf671c0401` |
| 35 项 scientific ledger | `14fd1c0cf7a670d8d62daf1af227255973a4d7e42c39fd331ab2188e44faf363` |

## 发布治理

正式 Pod 的阶段顺序为：

`00_start -> 01_static_preflight_ok -> 02_resident_shards_ok -> 99_done`

没有 `FAILED` 或 `FAILED_PHASE`。在 8 个 GPU shard 启动前：

- runtime code ledger 135/135、6 个模型 artifacts 和 14 个 weight shards 全部通过；
- 真实 Transformers 5.14.1 focused suite 为 `33/33`、0 skip；
- static dry-run 记录 `gpu_initialized=false`、8 个冻结 query banks、
  `test_v2_consumed=false`；
- PG-19 data/manifest/windows、模型、代码和 protocol manifest 全部匹配预注册 SHA。

8 个 rank 都成功原子写出约 29.5 MB raw shard 后，aggregate 从 raw trajectory、call ledger、
storage、allocator、query provenance 和 cleanup baseline 重新计算结果，不信任 shard 中的
顶层 parity/fit 布尔值。summary 还绑定了 8 个 shard 的 SHA 与字节数。scientific ledger
生成并 35/35 校验成功后，launcher 才写 `99_done`。

## Correctness 结果

最终 summary 给出：

- `same_kernel_full_logit_token_logical_kv_gdn_exact_fraction = 1.0`；
- `cross_n_prefix_isolation_exact = true`；
- 8/8 rank、所有 N、所有请求和全部 8 个生成步骤均通过；
- fresh/reuse 的生成 token、完整词表 logits、最终 logical K/V、每请求 GDN state exact；
- persistent GDN state 与 source document pool 不变；
- 10/10 full-attention 层均调用同一 `unified_attention` descriptor；
- dense fallback 和 full-K/V concatenate 均为 0。

cross-N gate 很重要：同一 rank 的 request i 在所有包含它的 N 上，都必须与最小包含它的
N 具有相同 token、full-logit、K/V 和 GDN digest。它排除了“fresh/reuse 同时被共享状态污染，
所以两边看起来一致”的一类假阳性。

这里的 exact 是两个 same-kernel ownership arm 的等价性，不是相对 dense/HF eager 的质量
证明，也不是 LongBench 下游能力。Trial `1840009` 的 TF-eager compatibility negative 仍然
有效，不能被本轮改写为“HF eager 与 vLLM logits exact”。

## 解析容量结果

10 个 full-attention 层合计的固定几何是：

- 每个物理 page block：`2 × 128 × 2 × 256 × 2 B × 10 = 2,621,440 B`；
- 4095-token 文档 allocation：32 pages，`83,886,080 B = 80 MiB`；
- 有效文档 payload：`83,865,600 B`；padding：`20,480 B`；
- 每请求 private reservation：2 pages，`5,242,880 B = 5 MiB`；
- 每个 fresh 独立 request pool：document 80 MiB + private 5 MiB = `85 MiB`；
- 127-token tail 的首步 staging copy：`2,600,960 B`/请求，两边共同发生；
- 生成后实际 appended tokens 为 `32 + 7 = 39`，tail + append 为 166 tokens，active
  private payload `3,399,680 B`/请求，仍占 2 个预留 pages。

公共 source arena 也为 N 个请求预留 private pages，因此受控总量为：

- fresh：`80 MiB + 5 MiB × N + 85 MiB × N = 80 + 90N MiB`；
- reuse：`80 MiB + 5 MiB × N`；
- saving：`85 MiB × N`；
- avoided physical document copy：`80 MiB × N`。

8 个 rank 的重放 fit 完全一致，斜率与截距的 `R²=1.0`：

| N | fresh pool | reuse pool | 受控节省 | 避免物理 copy |
|---:|---:|---:|---:|---:|
| 1 | 170 MiB | 85 MiB | 85 MiB | 80 MiB |
| 2 | 260 MiB | 90 MiB | 170 MiB | 160 MiB |
| 4 | 440 MiB | 100 MiB | 340 MiB | 320 MiB |
| 8 | 800 MiB | 120 MiB | 680 MiB | 640 MiB |
| 16 | 1520 MiB | 160 MiB | 1360 MiB | 1280 MiB |
| 32 | 2960 MiB | 240 MiB | 2720 MiB | 2560 MiB |

这张表只计算经过 raw layer ledger 重放的 full-attention Q16 pools。combined unique tensor
inventory 被保留为非授权诊断，不进入线性 fit 或主 claim；active private payload 是 reservation
的子集，也没有被重复相加。

## PyTorch allocator 结果

下表是 8 个 rank 的字段中位数。delta 的基线是 common document prefill 和 Q16 pack 完成后、
request setup 之前；absolute 值包含模型权重与进程内其他 PyTorch allocations。

| N | fresh setup+generation current delta | reuse current delta | 差值 | fresh/reuse absolute production peak allocated |
|---:|---:|---:|---:|---:|
| 1 | 147.135 MiB | 61.883 MiB | 85.252 MiB | 64.896 / 64.813 GiB |
| 2 | 298.522 MiB | 126.766 MiB | 171.757 MiB | 65.048 / 64.881 GiB |
| 4 | 591.286 MiB | 249.520 MiB | 341.767 MiB | 65.345 / 65.011 GiB |
| 8 | 1177.831 MiB | 495.045 MiB | 682.786 MiB | 65.936 / 65.270 GiB |
| 16 | 2355.406 MiB | 990.081 MiB | 1365.325 MiB | 67.126 / 65.792 GiB |
| 32 | 4705.065 MiB | 1980.162 MiB | 2724.903 MiB | 69.498 / 66.837 GiB |

N=32 的 production absolute peak allocated 精确值为：

- fresh：`74,623,183,360 B`；
- reuse：`71,765,915,136 B`；
- 差值：`2,857,268,224 B = 2724.903 MiB = 2.661 GiB`，约为 fresh 的 `3.83%`。

该差值与解析 `2720 MiB` saving 很接近，额外约 4.9 MiB 来自 request tables/metadata 等
实际 allocation。它支持“共享一个文档时，常驻请求容量随 N 显著改善”，但不能写成总模型
显存缩小 12.3 倍：`2960/240` 是 full-attention pool 比值，不是整个 35B 进程比值。

所有 allocator 数字都来自 PyTorch allocated/reserved 计数器，不是 NVML、设备总占用或
系统级峰值。reserved 受 caching allocator 行为影响；本轮没有把它包装成可移植容量公式。

## Timing 边界

raw shard 保留 common prefill、Q16 pack、request setup、generation wall 和每步 seconds，
并验证它们是 finite/nonnegative。这些时间包含 validation instrumentation，且每 rank/N/arm
只有一次正式观察；arm order 也不是 ABBA。aggregate 因此明确不汇总 timing，不输出 ratio，
不声称 TTFT、TPOT、kernel latency、吞吐或速度提升。

单流 round-major 的服务顺序是：先对全部 N 个请求执行 round 0 的 32-token query，再按
request index 完成该 round；随后 7 个 decode rounds 依次处理所有请求。它证明 N 个 cache
对象可同时驻留并保持隔离，不等同于 N 个 kernel 并发、continuous batching 或 vLLM scheduler
吞吐实验。

## 可以与不可以声称的结论

可以声称：

1. Q16、batch 1、同一约 4K PG-19 train 文档、10 个 full-attention 层、N<=32 条件下，
   fresh full-copy 与 shared-document reuse 的 token/full-logit/KV/GDN 全部 exact；
2. 同一 request 的结果不受 N 变化影响，cross-N prefix isolation exact；
3. full-attention pool 从 fresh `80+90N MiB` 变为 reuse `80+5N MiB`，每请求节省
   `85 MiB`，避免 `80 MiB` 物理文档 copy；
4. N=32 时 PyTorch absolute peak allocated 中位数降低约 `2.661 GiB`，但这只是
   allocator 口径下的本实验值。

不可以声称：

1. vLLM engine 多请求调度、真实并行吞吐、continuous batching 或 ragged batch 已完成；
2. total GPU/NVML memory 缩小 `12.3×`，或把 `2720 MiB` pool saving 直接称为整个
   35B 进程/NVML 显存节省；
3. aligned 4096 文档也具有相同 private-page/tail-COW 成本；本轮只测了
   `4095 mod 128 = 127` 的 worst-case partial-tail stress；
4. 多文档、Q8/Q4、量化 KV、scheduler 回收/复用、F1/EM 或下游能力已验证；
5. 延迟、TTFT、TPOT、isolated kernel 或吞吐有任何加速；
6. HF eager 与 vLLM 的 logits bitwise 等价。

## Artifact 与本地完整性复核

本地非-pycache 原始镜像：

[`results/gpu-qwen35-vllm-paged-multifork-resident-20260814a/`](results/gpu-qwen35-vllm-paged-multifork-resident-20260814a/)

目录共 38 个非-pycache 文件：35 个由原始 `scientific-artifacts.sha256` 覆盖的科学
artifacts，加账本自身、账本生成后的 integrity log 和最终 `99_done` 标记。运行时外置的
`pycache/` 不是 scientific artifact，未进入本地归档。原始账本保留远端绝对路径且未改写；
本机用下列 path-only 映射重放，35/35 全部为 `OK`：

```bash
RUN=results/gpu-qwen35-vllm-paged-multifork-resident-20260814a
REMOTE=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/qwen35-vllm-paged-multifork-resident-20260814a/
(cd "$RUN" && sed "s#  $REMOTE#  #" scientific-artifacts.sha256 | shasum -a 256 -c -)
```

核心文件：

- `multifork-resident-summary.json`：8-rank 聚合 summary；
- `resident-shards/multifork-resident-shard-{0..7}.json`：8 个完整 raw shards；
- `static-dry-run.json`、代码/模型 ledgers、GPU 快照与全部阶段/完整性日志；
- `scientific-artifacts.sha256`：35 项原始 scientific ledger。

本报告没有修改 raw JSON、log 或原始 ledger。
