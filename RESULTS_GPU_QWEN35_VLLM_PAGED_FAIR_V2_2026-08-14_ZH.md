# Qwen3.5 vLLM Q16 Paged Attention 同 Kernel 公平对照正式结果

## 一句话结论

Job `237468` / Trial `1840486` 完成了 Q16、batch 1、单请求条件下的正式
same-kernel 对照：fresh arm 为每个请求完整物化文档 block pool，reuse arm 复用只读
文档 pool 并只保留请求私有 tail/append。两边调用同一个 vLLM 0.26 Triton
`unified_attention` callable。

8 个 validation workload 上，两个 arm 的全部生成 token 与每一步 full-vocabulary logits
均 bitwise exact。reuse 没有带来延迟加速：paired median cached-document request TTFT
ratio 为 `1.009029`，即约慢 `0.90%`；TPOT ratio 为 `1.000620`，基本持平。它明确降低了
缓存增量显存：paired median incremental CUDA peak allocated ratio 为 `0.496010`，约低
`50.4%`；每请求避免的物理文档 block copy 中位数为 `83,886,080 B`，即 `80 MiB`。

这个结果不能写成“总模型显存减半”。fresh/reuse 的 absolute allocator peak 中位数分别为
`69,682,303,232 B` 和 `69,594,598,656 B`，只差 `87,704,576 B`，即约
`83.64 MiB` / `0.126%`。本实验也没有 LongBench F1/EM、多 query、ragged batch、
NVML peak 或 isolated kernel latency 结果。

## 为什么需要这个对照

Trial `1840009` 同时比较了 Transformers eager attention 与 vLLM Triton
`unified_attention`。3/8 PG-19 rank 没有通过逐元素兼容阈值，另外 5 个完整 semantic
row 的 KL 也没有通过授权门槛。该结果只能叫 **HF-eager compatibility negative**：
attention 数值后端与 cache ownership/layout 同时改变，不能把差异归因给 page reuse。
详见
[vLLM paged Q16 负结果](RESULTS_GPU_QWEN35_VLLM_PAGED_Q16_FORMAL_NEGATIVE_2026-08-14_ZH.md)。

本轮 fair v2 把主问题收窄为同一个 kernel 内的两个 ownership arm：

- `vllm-q16-fresh-full-copy-control`：每个请求分配独立物理 pool，并复制完整文档物理
  blocks；
- `vllm-q16-shared-document-reuse`：共享不可变文档 blocks，只使用预留 private blocks
  承担 partial-tail COW 与 continuation append。

两个 arm 必须具有相同的 post-RoPE Q、有效逻辑 K/V、position ids、causal contract、
scale、GQA、sequence lengths、生成轨迹与完整词表 logits；10 个 full-attention 层均须命中
同一个 `unified_attention` callable，dense fallback 和 full-K/V concatenate 必须为 0。
30 个 linear GDN 层不是本轮优化变量：两边从同一 persistent tensor base 开始，并执行同一
functional rebind/update。

## 运行身份与冻结条件

- QS：Job `237468` / Trial `1840486`；
- 页面：<https://qs2.devops.xiaohongshu.com/model/production/job/trial/237468/1840486>；
- 终态：`Complete`，运行时间 `0d00h12m03s`；
- 资源：8 张 H20-3e，每张 `143,771 MiB`，driver `570.133.20`；
- 模型：Qwen3.5-35B-A3B，40 层，16 query heads、2 KV heads、head dim 256；
- full-attention 层：`3,7,11,15,19,23,27,31,35,39`；
- 环境：PyTorch `2.11.0+cu129`、Transformers `5.14.1`、vLLM
  `0.26.0+cu129`、Triton `3.6.0`、FlashInfer Python `0.6.14`；
- cache：Q16 NHD block pool，page size 128；
- 主协议：batch 1、单请求、等长语义、8 个 continuation tokens；
- PG-19 授权：8 个 train-only windows，每个 `1025` document tokens + `32` query
  tokens；
- validation：QASPER / 2WikiMQA 的 source index `6--9`，每数据集 4 条；
- source `68--99` 与 LongBench `test-v2` 均未读取。

关键 SHA256：

| 对象 | SHA256 |
|---|---|
| code ledger，128 项 | `e5ea5c8ac95c988520ca952148296b919e947983c4dd0756384a726fd2a0bcd1` |
| runtime protocol manifest | `39529179030e7b855bb8b88bf6d15904206cc87c4aa561687c8d9647d431bafd` |
| QS YAML | `c4b55bec9081f503e7ac62bcfc18a25d0bc0ddb1cc0463a296d423ab446dd4ad` |
| model manifest | `72c9e06109702dbca958a6a528d6686b68d6b8e3376d116c0261b4c319e3da29` |
| model artifact ledger | `fa050ef64c76caaa353223541c6ad8b80be8a5f6f5c11430db2d7d4f2c4dfb5c` |
| model weight ledger | `a0352fd3fd47b4edcebf3269b5f8745490d3defb9eaedf2a4c4dc8ccae32ddf2` |
| PG-19 train data | `ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c` |
| PG-19 manifest | `5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c` |
| PG-19 windows | `5cac8b9b60c0b5e345e393001c6ed982d703c65ebb8808a528a3d52140e9d4b7` |
| validation data | `1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe` |
| static dry-run | `fa815ec54dce10374be7616d8d0c6a66f6f1696aeed70eef7856cb2a02d489d6` |
| PG-19 authorization | `f516a0964577f109d0bde872fb9343199af7bd03b356a99f4da1ac50fb400842` |
| final summary | `2a8af1734dbf60e4250dca00922c8736eb5116c5a6cc658b0c502803fccdb31b` |
| 58 项 scientific ledger | `2f8023f2ac7b615efabcb4414e64f4f12d9a03279dd69f4701c6915f8f3a4e24` |

## 发布治理与阶段顺序

正式 c 版在真实 Pod 中依次产生：

`00_start -> 01_static_preflight_ok -> 02_pg19_shards_ok ->
03_pg19_authorized -> 04_validation_hash_authorized ->
05_validation_shards_ok -> 99_done`

没有 `FAILED` 或 `FAILED_PHASE`。preflight 在 GPU 门禁前完成：

- runtime code ledger 128/128 重放通过；
- 14 个模型 weight shards、模型 artifacts 与数据/manifest digest 均通过；
- 真实 Transformers 5.14.1 focused suite 为 `43/43`、0 skip；
- static dry-run 记录 `gpu_initialized=false`、`validation_consumed=false`、
  `validation_hashed=false`、`source_68_99_consumed=false`、`test_v2_consumed=false`。

8/8 PG-19 train-only shards 全部通过后，聚合器才生成 SHA-addressed authorization。
authorization 明确记录 same-kernel layout 与 full-vocab logit gate 均通过，并且 HF
compatibility 不参与授权。只有 `03_pg19_authorized` 完成后，launcher 才首次存在性检查、
哈希并读取 validation；validation digest 为预注册的 `155373...`。

### 保留的 locale preflight 失败

Job `237414` / Trial `1840344` 是发布治理负结果，不是算法或 GPU benchmark 结果。
该 Pod 在 `00_start` 后生成的 128 行 code ledger 内容和路径集合均正确，但 launcher 继承
`en_US` 排序，而冻结侧使用 C locale，导致 ledger SHA 从预期
`3ef6c71c...` 漂移为 `c0b0cedd2f2d7cd38f0bb78fcdcb907aa2cb8d0118b8c0e886a5618ebb65be50`。
它没有进入 static、PG-19、validation 或 GPU kernel 阶段。

c 版把 launcher 全局 locale 与 code/scientific 两条排序管线都显式锁为
`LC_ALL=C`，并新增 C / `en_US` 实际排序回归和 preflight failure-marker 回归。
真实 c Pod 的 code ledger 精确恢复为预注册的 `e5ea5c8a...`。

## Correctness 结果

PG-19 authorization 与 8 个 validation shards 都要求：

- fresh/reuse canonical logical K/V bitwise exact；
- 每次调用为同一个 kernel callable identity；
- 10/10 full-attention 层每步命中；
- 生成 token 完全一致；
- 每一步完整词表 logits bitwise exact；
- source document pool 不可变；
- dense fallback、full-K/V concat 均为 0。

最终 8/8 validation workload 全部通过 token 与 full-logit exact gate。因此在**同一
vLLM kernel、Q16、单请求**边界内，document reuse 没有引入数值或生成差异。

HF eager 只作 non-authorizing backend compatibility 诊断：8/8 workload 的 8-token
greedy 轨迹与 vLLM 相同，但 0/8 workload 的逐步 full-vocab logit SHA 相同。这与
Trial `1840009` 的负结果一致：跨 backend 的 BF16 reduction/rounding 差异不能冒充
CoMem layout 误差。

本实验没有计算 LongBench F1、EM 或 answer-type 指标。8-token exact 证明的是两个
same-kernel cache ownership arm 一致，不等于完整下游质量评测。

## 性能结果

主性能采用 fresh-state ABBA：每个 arm 一次 warmup、四个 measurement trials；奇偶 rank
反转顺序。表中所有 ratio 均为 `reuse / fresh full-copy control`，小于 1 才代表 reuse
更快或更省。

| 指标 | paired median ratio | 解释 |
|---|---:|---|
| cached-document request TTFT | `1.009029` | reuse 约慢 `0.90%`，没有 TTFT 加速 |
| continuation 第一个完整模型步 | `1.013289` | reuse 约慢 `1.33%` |
| TPOT | `1.000620` | 基本持平，约慢 `0.06%` |
| incremental CUDA peak allocated | `0.496010` | reuse 约低 `50.4%` |
| incremental CUDA peak reserved | `1.000000` | allocator reserved 没有下降 |
| request-setup peak allocated | `0.00005827` | fresh 的完整 pool allocation/copy 基本被消除 |
| request-setup peak reserved | `undefined` | 两边分母都是 0，协议拒绝伪造 ratio |

`cached-document request TTFT` 只包含 request setup 与 continuation 首步，不包含公共
dense document prefill 或 dense-to-NHD Q16 pack。公共 prefill / pack 中位时间分别为
`0.706056 s` / `0.006633 s`。continuation 首步包含整个 backbone、lm_head、argmax 与同步，
不是 isolated `unified_attention` kernel latency；因此不能写成 kernel speedup。

逐 workload 主结果：

| rank | workload | doc tokens | query tokens | TTFT ratio | TPOT ratio | incremental peak ratio | same-kernel full logits exact | HF token / logit SHA match |
|---:|---|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | `qasper-6` | 3997 | 64 | `1.014984` | `1.004152` | `0.497270` | 是 | 是 / 否 |
| 1 | `qasper-7` | 3999 | 64 | `0.995592` | `0.998855` | `0.498443` | 是 | 是 / 否 |
| 2 | `qasper-8` | 3448 | 64 | `0.996782` | `0.994239` | `0.522796` | 是 | 是 / 否 |
| 3 | `qasper-9` | 4000 | 64 | `1.004650` | `1.001881` | `0.498431` | 是 | 是 / 否 |
| 4 | `2wikimqa-6` | 4035 | 61 | `1.013408` | `0.993549` | `0.494750` | 是 | 是 / 否 |
| 5 | `2wikimqa-7` | 4048 | 48 | `0.982854` | `1.004246` | `0.487963` | 是 | 是 / 否 |
| 6 | `2wikimqa-8` | 4050 | 46 | `1.024543` | `0.999360` | `0.487614` | 是 | 是 / 否 |
| 7 | `2wikimqa-9` | 4046 | 50 | `1.018783` | `1.005090` | `0.488321` | 是 | 是 / 否 |

8 个 workload 的 TTFT ratio 横跨 1，且中位数非常接近 1。本轮支持“内存下降且延迟近似
中性”，不支持“几倍加速”或统计显著 TTFT 改善。

## 显存与 storage 分账

### Per-request allocator 增量

| 阶段 | fresh | reuse | 差值 |
|---|---:|---:|---:|
| request setup peak allocated delta | `87,880,192 B` | `5,120 B` | `87,875,072 B`（`83.80 MiB`） |
| setup + generation peak allocated delta | `172,874,240 B` | `85,571,072 B` | `87,303,168 B`（`83.26 MiB`） |
| setup + generation peak reserved delta | `8,388,608 B` | `8,388,608 B` | 0 |

paired ratio 的 `0.496010` 是每 workload 的增量 peak ratio 中位数；它不是上表两个全局
中位数再相除。CUDA allocator reserved 没有下降，说明 PyTorch caching allocator 会保留
相同 reservation，不能用 allocated 的下降替代 reserved 结论。

### Full-attention storage

| 指标 | fresh | reuse | reuse 减少量 |
|---|---:|---:|---:|
| 每请求物理文档 block copy（含 padding） | `83,886,080 B` | 0 | `80 MiB` |
| combined unique，continuation 前 | `240,521,020 B` | `152,701,500 B` | `87,819,520 B`（`83.751 MiB`） |
| combined unique，decode 后 | `305,401,660 B` | `217,582,140 B` | `87,819,520 B`（`83.751 MiB`） |

跨 workload 对各字段分别取中位数后，共同 source 文档的有效 payload、物理
document allocation、padding 分别为 `82,278,400 B`、`83,886,080 B`、
`1,136,640 B`；source arena 的 private reservation 字段中位数为 `5,242,880 B`。
这些是不同字段各自的跨 workload 中位数，不能把它们相加减后当成某一个 workload
的 storage equation；每个 raw trial、每层的精确分账公式均由 aggregator 硬门禁。
该 reservation 也不能包装成“纯文档 payload”。
两边首个 continuation step 都在共同 append path 中产生 `1,484,800 B` partial-tail
staging copy；它不是 reuse setup 成本，也不是本轮节省量。

### 为什么不能说总模型显存减半

包含模型权重与其余 persistent state 后，setup + generation 的 absolute peak allocated
中位数是：

- fresh：`69,682,303,232 B`；
- reuse：`69,594,598,656 B`；
- 差值：`87,704,576 B`，即约 `83.64 MiB` / `0.126%`。

因此正确表述是“**单请求 cache 增量峰值约减半，并避免约 80 MiB 文档 block copy**”。
要观察总模型显存或文档容量的更大比例变化，需要多请求/多文档同时共享一个 source pool，
这不在本轮实验中。

## 可以与不可以声称的结论

可以声称：

1. Q16、batch 1、单请求、10 个 full-attention 层条件下，同一个 vLLM
   `unified_attention` kernel 的 fresh full-copy 与 shared-document reuse 在 8/8
   validation workload 上生成 token 与逐步 full-vocab logits bitwise exact。
2. reuse 避免中位 `80 MiB` 物理文档 block copy，combined unique storage 减少
   `87,819,520 B`。
3. per-request incremental CUDA peak allocated paired median 约降低 `50.4%`，但
   allocator reserved 不降。
4. cached-document TTFT/TPOT 近似中性；本轮没有速度正结果。

不可以声称：

1. 总模型显存减半、几倍容量提升或多 query serving 已完成；
2. Q8/Q4、ragged batch、多请求调度或 Apple/MLX backend 已验证；
3. isolated attention kernel 加速、NVML peak 降低或能耗改善；
4. LongBench F1/EM 或下游能力无损；
5. HF eager 与 vLLM logits bitwise 等价。

## Artifact 与本地完整性复核

本地原始镜像：

[`results/gpu-qwen35-vllm-paged-fair-v2-20260814c/`](results/gpu-qwen35-vllm-paged-fair-v2-20260814c/)

目录共 61 个非-pycache 文件：58 个由原始 `scientific-artifacts.sha256` 覆盖的科学
artifacts，加账本自身、账本生成后的 integrity log 和最终 `99_done` 标记。原始账本保留
远端绝对路径且未改写；在本机用下列 path-only 映射重放，58/58 全部为 `OK`：

```bash
RUN=results/gpu-qwen35-vllm-paged-fair-v2-20260814c
REMOTE=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/qwen35-vllm-paged-fair-v2-20260814c/
(cd "$RUN" && sed "s#  $REMOTE#  #" scientific-artifacts.sha256 | sha256sum -c -)
```

核心文件：

- `fair-v2-summary.json`：正式 8-workload summary；
- `pg19-fair-v2-authorization.json`：PG-19 train-only authorization；
- `pg19-gate-shards/pg19-fair-v2-shard-{0..7}.json`：8 个授权 shards；
- `validation-shards/fair-v2-shard-{0..7}.json`：8 个 validation shards；
- `static-dry-run.json`、`code.sha256`、模型 ledgers 与全部阶段/完整性日志；
- `scientific-artifacts.sha256`：58 项原始 scientific ledger。

本报告没有修改 raw JSON、log 或原始 ledger。
