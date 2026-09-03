# vLLM Q16 ragged batch 可行性审计（2026-08-14）

## 结论

冻结的 vLLM 0.26 Triton `unified_attention` API **可以表达**同一 Q16 物理 block pool 上、多个 sequence 具有不同 `q_len` 与 `kv_len` 的一次 ragged 调用。当前新增适配器已经在 CPU/mock kernel 下证明：flattened query、`cu_seqlens_q`、`seqused_k`、二维 `block_table` 的基数和顺序一致；逻辑 K/V、tail-causal position/mask 与 16Q/2KV 的 GQA 结果都能对上 dense oracle；路径中没有完整 K/V 拼接或物化。

这个结论只到 **adapter feasibility**。本轮没有提交 GPU、没有加载模型、没有接入 vLLM scheduler，也没有读取任何 LongBench、validation 或 test-v2 数据。因此它不能支持真实 kernel、全模型正确性、并发调度、吞吐或延迟结论。

## 冻结 API 的静态审计

审计对象是既有部署环境：

- vLLM：`0.26.0+cu129`
- `triton_unified_attention.py` SHA-256：`992a2bc892e2e2b43fbd3c8163816ccf7e97ced56cb542bd827adb0ddb2df9fa`
- `triton_attention_helpers.py` SHA-256：`8c730611e7b3c5fb7579ec7846d56a2ab7e348ce06b39136da22072ecc363c95`
- entrypoint：`vllm.v1.attention.ops.triton_unified_attention.unified_attention`

源码中的 16 个必需参数依次为：

```text
q, k, v, out,
cu_seqlens_q, max_seqlen_q,
seqused_k, max_seqlen_k,
softmax_scale, causal, window_size,
block_table, softcap,
q_descale, k_descale, v_descale
```

这不是只看函数名做出的推断。冻结源码还明确给出了 ragged 语义：

- `num_seqs = len(seqused_k)`；
- kernel 通过 `cu_seqlens_q` 的累计前缀和，把 flattened query block 反查到 sequence 与 sequence 内偏移；
- 每个 sequence 的 `query_len = cu_seqlens_q[i+1] - cu_seqlens_q[i]`；
- `context_len = seqused_k[i] - query_len`；
- causal query 绝对位置是 `context_len + query_pos`；
- 每一行用 `block_table[i]` 将逻辑 KV block 映射到共同的物理 K/V pool。

所以在普通、无 padding 的 causal tail 场景下，`seqused_k[i] - q_len[i]` 正好是该请求 query tail 的起始位置。适配器不重复做 RoPE；它接收 post-RoPE query，并用显式 `position_ids` 检查这个 tail 位置契约。

`audit_frozen_vllm_ragged_api()` 使用包 metadata、AST 与文件 SHA 做静态核验，不 import vLLM 的 CUDA/platform 初始化模块。任何版本、源文件或签名漂移都会得到 `matches_frozen_api=false`。

在远端冻结环境中实际运行该审计得到 `matches_frozen_api=true`；这只是源码/API 身份核验，没有调用 GPU kernel。

## 适配器数据面

实现文件：`gpu/qcomem_vllm_ragged_batch.py`。

当前作用域有意收窄为 Qwen3.5 Q16 几何：

| 项目 | 冻结值 |
|---|---:|
| query heads | 16 |
| KV heads | 2 |
| GQA groups | 8 |
| head dim | 256 |
| page size | 128 |
| K/V dtype | `float16` 或 `bfloat16` |
| K/V layout | contiguous NHD `[blocks, 128, 2, 256]` |

一次调用的构造是：

```text
request 0: q[0:q0],      kv_len=k0, table row 0
request 1: q[q0:q0+q1], kv_len=k1, table row 1
...

q                = [sum(q_i), 16, 256]
cu_seqlens_q     = [0, q0, q0+q1, ...] int32
seqused_k         = [k0, k1, ...] int32
block_table       = [N, max(ceil(k_i/128))] int32
max_seqlen_q      = max(q_i)
max_seqlen_k      = max(k_i)
```

所有 request 必须是不同的 `Q16PagedSequence`，而且必须共享同一个 `Q16PagedArena`。每个 request 的 query K/V 必须已 append 到自己的私有 reservation；适配器只复制 query 行与 block-table metadata。短 table 的右侧用物理 block 0 填充，`seqused_k` 保证 kernel 不会读取这些 padding entry。

## fail-closed 条件

dispatch 前会拒绝：

- 非 16Q/2KV/head256/page128 或非 Q16 dtype；
- query、K/V、position、mask、table 的 dtype/device 不一致；
- 非 contiguous NHD K/V pool、K/V storage alias；
- 多个 request 来自不同 arena，或同一个可写 sequence 重复出现；
- request-private 物理 block 跨请求 alias；
- table 含负数、越界 id、sequence 内重复 block，或不可变 document prefix 被改写；
- query 还没有对应地 append 到 K/V；
- `position_ids` 不是精确的 contiguous causal tail；
- padding、自定义 additive bias、额外 mask 类型、prefix-LM、sliding-window 或其他非 canonical mask；
- 非正或非有限的 softmax scale；
- CPU 上未注入 mock kernel 却试图调用真实 Triton entrypoint。

严格检查当前会读取 position/table/mask 的值，其中部分检查在 CUDA 上会产生 host synchronization。它适合 correctness gate，不应直接被宣传成无开销的生产 scheduler fast path。后续若接入 scheduler，需要由 scheduler 生成并持有可信的 CPU metadata/ownership ledger，再另外证明其与 device table 一致。

## CPU/mock 证据

focused test：`gpu/test_qcomem_vllm_ragged_batch.py`。

```bash
PYTHONPATH=gpu python3 -m unittest -v \
  gpu/test_qcomem_vllm_ragged_batch.py \
  gpu/test_qcomem_vllm_paged_kernel.py
```

本地结果：15/15 通过，其中 ragged 新增测试 8/8 通过；远端冻结 Python/torch 环境的 CPU/mock focused 复跑也是 8/8 通过。新增覆盖包括：

- `q_len=(1,3,5)`、`kv_len=(130,132,134)` 的真实 ragged cardinality；
- flattened query 顺序与 `cu_seqlens_q=(0,1,4,9)`；
- 每行 block table 还原的逻辑 document+private-tail K/V 与 dense 预期逐元素相同；
- 16Q/2KV GQA、tail causal mask 与 dense oracle 输出一致；
- `torch.cat` 被显式设为报错时，适配路径仍成功，audit 为 `full_kv_concatenations=0`、`full_kv_materializations=0`；
- bool mask、additive canonical mask 与 `None` no-padding contract；
- 错误 position/mask、几何、dtype/device、arena、private ownership、table、未 append query 与 scale 均在 kernel dispatch 前被拒绝；
- 既有 single-request paged-kernel 测试 7/7 同时通过，新增文件未修改其语义。

## 仍需 H20 gate

在把 feasibility 升级为真实 kernel 结论前，至少需要一次冻结协议的 H20 gate：

1. 在 job 内运行 `audit_frozen_vllm_ragged_api()`，要求版本、两个源码 SHA、完整参数顺序和 required-count 全部命中。
2. 使用同一个 Q16 pool 构造至少 3 个 request，选择不同 `q_len` 与 `kv_len`，让真实 `unified_attention` 完成一次 BF16 ragged 调用。
3. 对每个 sequence 分别用 dense/eager oracle 比较完整 attention output（不能只比较 top-1 token）；记录 max/mean error、NaN/Inf 与逐 sequence hash。
4. 证明一个 dispatch 的 kernel callable/entrypoint 身份、输入 table/length ledger、一次 kernel hit，以及调用前后 document block 不变、private block 不 alias。
5. 覆盖 partial-tail 与跨 page append；至少包含 `kv_len % 128` 不同的行。
6. 把 kernel-level adapter 接入 10 个 full-attention layers 后，另做全模型 logits/token parity gate。GDN request state、RoPE/position 与 cache update 顺序都必须单独核验。
7. scheduler batching、真正的同时 resident/concurrent execution、吞吐、TTFT/TPOT、allocator/NVML 与容量曲线必须使用另一套部署协议；不能由本适配器测试外推。

在以上 gate 通过前，准确表述只能是：**冻结 vLLM 0.26 API 与 CPU/mock 结果支持 Q16 shared-pool ragged 调用的适配器可行性；真实 H20 kernel 与端到端部署效果尚未验证。**
