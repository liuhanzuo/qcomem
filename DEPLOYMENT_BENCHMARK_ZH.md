# Q-CoMem KV Cache 与部署显存实验

这条实验线回答两个问题：Q-CoMem 的增量生成是否正确、是否真的只计算新 token；以及在
真实部署生命周期中，它能减少多少持久化文档状态和 GPU 显存，代价是多少。

它默认使用公开 LongBench 的 Qasper、2WikiMQA validation 子集，或完全合成的
4k/8k/16k/32k workload，不消费为 mixed-bit 最终评测预留的 test-v2。

## 1. KV cache 路径与正确性门槛

底层实现位于 `gpu/qcomem_torch.py`，生产式会话和测量封装位于
`gpu/qcomem_deployment.py`。

| 配置 | 文档 Write | query prefill | 后续 token |
|---|---|---|---|
| `dense-recompute` | 无持久状态 | 重算 document + query | 每步重算全部历史 |
| `full-prefix-q16` | 保存全模型 document cache | 从 document cache 继续 | 全模型 cache 增量一步 |
| `qcomem-*` | 保存 split residual + lower KV/recurrent state | lower cache 继续 query；suffix 先用 document residual 建 cache，再用 query residual 续写 | lower 与 suffix cache 都只追加一个位置 |

Q-CoMem 的 suffix cache 很关键。若每生成一个 token 都重新把整个 boundary residual 送入
suffix，persistent store 虽小，在线计算仍然是二次的。当前会话先用 document residual 单独建立
suffix cache，再从 document offset 用 query residual 续写；之后只向 lower 和 suffix cache
各追加一个新位置。不能将前两步合成一个 chunk：Qwen3.5 的 GatedDeltaNet/conv cache
会保留 chunk 边界语义，合并后可在第二个 token 就改变 recurrent state 和 logits。

每个 GPU 在计时前执行硬 gate：

```text
dense full-history recompute oracle
        == exact full-prefix incremental decode
        == Q16 residual + Q16 lower-state incremental replay
```

gate 对每个生成位置比较 token ID，任一位置不一致就停止 shard。logits 的 bitwise equality、
给定 `atol` 的 equality 和最大绝对误差也写入 JSON。默认只把逐 token equality 作为硬条件，
因为不同 attention kernel 可能产生不改变 token 的微小浮点差异；需要更严格检查时添加
`--require-exact-logits`。

当 `FORK_STRATEGY=paged-cow-staging` 时还有第二道、不能关闭的直接配对 gate：从**同一份**
Q16 persistent state 分别执行 eager deep-clone 与 COW 的完整 autoregressive trace。两边每步
token 和完整 logits tensor 都必须 `torch.equal`，JSON 逐步保存 max-abs 与 relative-L2；同时
在 eager 后、COW 后分别用全 tensor snapshot 检查 persistent source 未被修改。COW fallback
即使恰好生成相同 token 也判失败，因为它没有实际覆盖待验证路径。

## 2. 默认比较配置

| 配置 | residual | lower full-attention | lower linear/recurrent |
|---|---:|---:|---:|
| `qcomem-d7-r16-a16-l16` | Q16 | Q16 | Q16 |
| `qcomem-d7-r8-a8-l8` | Q8 | Q8 | Q8 |
| `qcomem-d7-r4-a4-l4` | Q4 | Q4 | Q4 |
| `qcomem-d7-r4-a4-l8` | Q4 | Q4 | Q8，冻结的 state-type policy |
| `qcomem-d7-mixed` | policy JSON 决定 | 逐层 policy | 逐层 policy |

`qcomem-d7-mixed` 兼容 `aggregate_layer_sensitivity.py` 输出的
`residual_bits + cache_layer_bits`。也可显式传：

```text
qcomem-d7-r4-layers=8,8,4,4,8,8,8
```

逐层位宽数必须与实际 lower cache layer 数严格相等，不会自动截断或补齐。每个 repeat 内
配置顺序都会用预注册 seed 随机打乱并写入结果。

## 3. 显存口径

以下字段含义不同，不能随意相加后统称“显存”：

| 口径 | 主要 JSON 字段 | 含义 |
|---|---|---|
| 模型权重 | `environment.model_parameter_nbytes` | parameter 唯一 storage 的逻辑字节 |
| 模型 buffer | `environment.model_buffer_nbytes` | buffer 唯一 storage 的逻辑字节 |
| CUDA 模型基线 | `model_cuda_allocated_baseline_bytes` | 模型加载后 allocator 已分配字节 |
| persistent/document | `persistent_document_nbytes` | corpus 中每篇文档长期保存的 packed 状态 |
| selected/fork active | `selected_fork_active_state_*` | query 选中并 fork/dequantize 后的 dense 活跃状态 |
| decode KV | `decode_kv_*` | query/generated lower 增量，加完整 suffix decode cache |
| CUDA allocated | `cuda_peak_allocated_bytes`, `steady_state_cuda_allocated_bytes` | PyTorch tensor 的当前/峰值分配 |
| CUDA reserved | `cuda_peak_reserved_bytes`, `steady_state_cuda_reserved_bytes` | caching allocator 保留空间 |
| NVML process | `nvml_sampled_peak_process_bytes` | NVML 检查点看到的进程显存，包含非 PyTorch allocation |

NVML 是离散采样值，不伪装成连续硬件峰值；CUDA peak 来自 PyTorch allocator counter。
Q-CoMem 在 suffix prefill 后释放 active fork 中已消费的 dense document residual，因此同时
报告 active peak 与 steady state。persistent packed store 仍存在，两者不是同一份状态。

## 4. 时延和文档容量

- `write_build_seconds`：构建一篇可复用 store；量化配置包含真实 pack 时间。
- `ttft_seconds`：已有 persistent store 时，从 fork/dequantize、query prefill 到首 token
  logits；dense-recompute 包含 document + query 全重算。
- `tpot_seconds`：首 token 之后每个新 token 的同步 GPU 时间。
- `throughput_tokens_per_second`：`generated_tokens / (TTFT + sum(TPOT))`。
- `instrumented_wall_seconds`：包含 allocator/NVML 采样开销，只用于审计。

容量估算使用：

```text
store-only = floor((device_total - model_allocated - safety_headroom) / bytes_per_doc)
one-active = floor((device_total - model_allocated - safety_headroom
                    - measured_active_request_overhead) / bytes_per_doc)
```

默认 headroom 为 4 GiB。它是当前单进程、单并发的估算，不包含碎片、多并发和 CUDA graph
pool；最终论文还应加入实际逐篇装载直到 OOM 的 capacity sweep。

## 5. LongBench validation

数据必须由 `prepare_longbench_subset.py` 固定 source repo、immutable revision 和 source
index。runner 会记录文件 SHA256、revision、index 和 prompt protocol。缺失 revision、混入
其他数据集会直接退出。

为避免提前查看 mixed-bit test-v2，默认禁止 `_source_index >= 68`。只有显式传入
`--allow-test-v2` 才能绕过；部署主实验不应使用这个选项。

单卡功能命令：

```bash
CUDA_VISIBLE_DEVICES=0 "$ENV_DIR/bin/python" gpu/run_deployment_bench.py \
  --model "$MODEL_DIR" --workload longbench --data "$VALIDATION_JSONL" \
  --limit-per-dataset 1 --world-size 1 --rank 0 \
  --max-input-tokens 2048 --max-new-tokens 4 \
  --warmups 1 --repeats 1 --run-dir "$RUN_DIR/smoke"
```

8×H20 验证：

```bash
CODE_DIR=/path/to/qcomem_gpu \
MODEL_DIR=/path/to/Qwen3.5-35B-A3B \
DATA_FILE=/path/to/longbench_validation.jsonl \
RUN_DIR=/path/to/runs/qcomem/deployment-validation \
ENV_DIR=/path/to/vllm-cu129-v1 \
WORKLOAD=longbench LIMIT_PER_DATASET=4 MAX_NEW_TOKENS=32 WARMUPS=1 REPEATS=3 \
MIXED_POLICY_FILE=/path/to/layer_policy.json \
bash gpu/launch_deployment_8gpu.sh
```

4 条 Qasper + 4 条 2WikiMQA 刚好每张 GPU 一个 workload；每张卡内部跑所有配置，避免
将不同策略固定到不同 GPU 后混入卡间差异。已准备但尚未提交的 QS 文件为
`qs/qcomem-deployment-bench.yaml`。

## 6. 合成容量实验

```bash
CODE_DIR=/path/to/qcomem_gpu MODEL_DIR=/path/to/Qwen3.5-35B-A3B \
RUN_DIR=/path/to/runs/qcomem/deployment-capacity ENV_DIR=/path/to/env \
WORKLOAD=synthetic WARMUPS=1 REPEATS=3 \
MIXED_POLICY_FILE=/path/to/layer_policy.json \
bash gpu/launch_deployment_8gpu.sh
```

默认 4k/8k/16k/32k 各重复两份，共 8 个 workload。它测容量斜率、TTFT/TPOT 和 OOM
边界，不回答自然任务质量；LongBench F1 与合成容量必须分表。

## 7. 输出和验收

```text
deployment-shard-0.json ... deployment-shard-7.json
deployment-summary.json
gpus-before.csv
gpus-after.csv
deployment-tests.log
logs/rank-*.log
stages/00_start ... stages/99_done
```

最低验收条件：

1. 8 个 shard 都为 `status=completed`；
2. `all_exactness_gates_passed=true`；
3. 每个 config/workload 有 3 次重复，`randomized_orders` 完整；
4. source revision 和 SHA256 唯一，`test_v2_consumed=false`；
5. NVML 不可用时明确为 null，不能拿 CUDA allocated 冒充 NVML；
6. Q16、Q4/Q8、mixed 的 F1 与显存/时延同时报告。

## 8. 当前风险

- fork 是真实 deep clone，不是 copy-on-write；它反映现有实现，后续可用只读 prefix page
  sharing 降低 active peak。
- persistent store 目前由 GPU tensor 保持。SSD/CPU corpus capacity 还需要异步 IO 和
  pinned-memory 路径，不能用当前结果冒充 IO 完成版。
- suffix decode cache 覆盖整篇文档，所以 persistent store 压缩十几倍不代表活跃 query 的
  peak 也压缩十几倍；这正是两类字段分开的原因。
- 容量公式必须由真实 OOM sweep 复核。
- H20 用于 GPU infra/自然任务验证；Apple 端还需用相同逻辑字段在 MLX unified memory
  复测，不能直接类比 CUDA reserved 或 NVML。

## 9. 可审计的 COW staging 原型

`gpu/qcomem_paged.py` 提供可选的 `paged-cow-staging` fork 策略。它解决的是“每个 query
一开始就 deep-clone/dequantize 全部 lower state”的问题，但不是 vLLM/PagedAttention
意义上的真正分页 attention kernel。

```bash
FORK_STRATEGY=paged-cow-staging \
CODE_DIR=/path/to/qcomem_gpu MODEL_DIR=/path/to/model \
DATA_FILE=/path/to/longbench_validation.jsonl RUN_DIR=/path/to/run \
ENV_DIR=/path/to/env WORKLOAD=longbench \
bash gpu/launch_deployment_8gpu.sh
```

状态生命周期如下：

| 状态 | fork 行为 | 原因 |
|---|---|---|
| document boundary residual | 只读共享 | suffix 入口先 `torch.cat`，不会把原 tensor 交给原地 kernel |
| full-attention document K/V | fork 时只读共享 | 原型安装受控的 read/`torch.cat`/rebind update，不调用未知的原地 update |
| linear `conv_states` | fork 时复制 | Transformers 使用 `copy_`，Qwen decode kernel也会原地推进 |
| linear `recurrent_states` | fork 时复制 | Transformers 使用 `copy_`，必须 query-private |
| query/generated K/V | query-private | 从首次 query prefill 开始增长 |

对于 Q4/Q8 persistent state，原型只在准备阶段 dequantize 一次 dense staging template，后续
query 共享它；输出会同时报告：

- `persistent_document_nbytes`：packed durable store；
- `persistent_materialized_staging_nbytes`：额外 dense template；
- `persistent_total_resident_nbytes`：两者实际同时驻留的总量；
- `cow_initial_shared/private_nbytes`：刚 fork、尚未 query prefill；
- `cow_after_query_shared/private_nbytes`：query prefill 后；
- `cow_final_shared/private_nbytes`：生成结束。

容量估算使用 `persistent_total_resident_nbytes`，不会只按较小的 packed bytes 得到虚假的
resident-document 数量。

安全策略是 fail closed：只支持可识别的 non-sliding dynamic `keys/values` 和明确的
`conv_states/recurrent_states`。出现 sliding cache 或未知 tensor 字段时自动记录
`deep-clone-fallback` 和原因，不会把 fallback 宣称为 COW。每次 query 结束还检查共享 tensor
的 storage pointer、可用的 PyTorch version counter 和固定采样值；任何变化立即抛错。

当前标准 Transformers attention kernel 需要连续完整 K/V，因此首次 query 写入时
`torch.cat` 仍会物化一份 `document + query` K/V。这个原型能够：

1. 避免 fork 瞬间的长 KV eager clone；
2. 避免 packed lower state 每个 query 重复全量 dequantize；
3. 准确暴露 staging 与首次写物化的内存峰值。

它不能消除 query 活跃期间的完整 K/V 物化。要实现真正的 shared document pages + query
delta，需要替换 attention kernel，使它直接读取 page table 或 prefix pages；在这一步完成前，
论文中应称为 **COW staging prototype**，不能称为完整 PagedAttention infra。

### 9.1 Q16 fork 修正与旧结果边界

2026-08-12 审计发现，旧版 `PackedTensor.dequantize()` 在 Q16 分支只执行
`reshape(...).to(same_dtype)`；PyTorch 可直接返回原 storage 的 view。Qwen3.5 的
`conv_states/recurrent_states` 会原地更新，因此旧版所谓 Q16 deep-clone 可能污染 persistent
cache。现在 Q16 cache leaf 会显式 `clone()`，fork A、fork B 与 packed source 三份 storage
必须互不相同；boundary residual 仍按上述只读 contract 共享。

因此，任何旧的“同一 Q16 store 连续服务多个 query”结果都应标记为受影响并重跑。现有
168-row deployment trial 每一行都重新 build store、只发一个 request，不能把它解释为
multi-query correctness 证据；它的首次 query 数值仍可保留，但 persistent reuse 结论必须以
新的 eager-vs-COW direct paired gate 为前提。Q4/Q8 cache dequantize 本来就产生新 tensor，
所以该 storage-alias bug 不影响 frozen Q4/Q8 或逐层 mixed 的 cache fork。

### 9.2 H20 direct gate 结果与边界

修复后的 QS Trial `1830738` 已运行，详见
[COW direct 报告](RESULTS_GPU_COW_DIRECT_GATE_2026-08-12_ZH.md)。必须同时保留两项状态：

- Trial 总状态 `Failed`；外层 dense single-chunk vs incremental token gate 为 5/8 通过、
  3/8 失败；
- 同一 Q16 persistent source 的 eager-vs-COW direct sub-gate 为 8/8 通过：每步完整 logits
  `torch.equal`，max-abs/relative-L2 都为 0，source immutable、actual COW 且无 fallback。

本次只使用 256-token document、32-token query 和最多 4 个 emitted logits。由于总 gate 没有
通过，未运行 short benchmark；不能从 direct correctness 推出 4k active peak、TTFT 或吞吐
改善。现有 artifact 能证明 full-prefix 与 Q16 COW 的 token 序列 8/8 相同，但旧 schema 未保存
二者逐步 logits pairwise，不能称它们 8/8 bitwise exact。runner 已补三方 pairwise 字段供未来
小门禁使用，本轮不重复申请 8 卡。
