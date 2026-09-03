# QComem paged attention 公平基础设施协议 v2

## 为什么需要 v2

Trial 1840009 把 Hugging Face eager attention 与 vLLM 0.26
`unified_attention` 放在同一个 correctness gate 中。两者的 BF16/FP32
归约顺序不同，因此即使缓存的逻辑 K/V 正确，logit 仍可能出现后端数值差异。
该负结果保留为历史证据；v2 不放宽阈值，也不把后端差异解释为 QComem
误差。

v2 的主问题只有一个：在完全相同的 vLLM kernel、输入和模型路径下，
完整复制文档 block pool 与复用只读文档 block pool 有什么差异。

## 两个主比较 arm

- `vllm-q16-fresh-full-copy-control`：为每个请求分配独立物理 pool，并在
  request setup 内复制全部文档物理 block。复制字节包含最后一页 padding；
  padding 由 `seqused_k` 屏蔽，不算有效文档 payload。
- `vllm-q16-shared-document-reuse`：复用不可变文档 block，只使用预留的
  private blocks 完成 partial-tail COW 和 query/decode append。

两边必须共享同一个 Python callable 对象及同一
`unified_attention` module/qualname/signature；每次调用还硬验相同的
post-RoPE Q、scale、GQA、causal、`seq_lens`、block-table 逻辑内容和当前
append delta。10 个 full-attention 层均须命中 fused kernel，dense fallback
和 full-K/V concatenate 都必须为 0。

物理 block ID、pool 总容量和无效 padding 不是 correctness invariant。
授权门禁把 active block table canonicalize 后，只对有效逻辑 K/V 做 bitwise
比较，并要求两个 arm 的 kernel 输出、完整词表 logits 和生成轨迹 bitwise
一致。PG19 特意使用 1025-token 文档、32-token query 和 page size 128，覆盖
非页对齐 tail 及跨页 append。

模型的 30 个 linear GDN 层不是优化变量。两边从同一 persistent tensor
storage 开始，随后使用相同 functional rebind/update；门禁要求查询后两边
functional state bitwise 一致且 persistent base 不变。因此主结论仅覆盖
10 个 full-attention 层的缓存 ownership。

## 数据治理与授权顺序

1. 静态 preflight 锁死 Q16、8 ranks、page 128、PG19 参数、LongBench
   source 6--9、每数据集 4 条、max input/query/new 为 4096/64/8、source
   revision，以及代码、模型和数据 SHA。
2. 首个 GPU 阶段只读取冻结的 PG19 train 数据。8 个 rank 各完成一个唯一
   train window；只有 same-kernel layout 与 full-logit gate 可以授权。
3. 聚合器必须把当前 code/model/data/runtime-protocol identity 与 8 个 shard
   做 exact match，生成 SHA-addressed authorization。
4. authorization 生成并复核后，launcher 才允许对 LongBench validation
   做第一次 existence/hash/open。validation loader 重新读取后的 digest 也须
   与 authorization 一致，以关闭 hash-to-open 漂移。
5. validation 只使用 QASPER 与 2WikiMQA 的 source 6--9。source 68--99 和
   `test-v2` 在本协议中 fail-closed，不能读取，也不能用于选择或授权。

runtime protocol manifest 是独立 JSON 文件，包含全部 formal 参数及
code/model/data digest；QS YAML 只钉 manifest 的外部 SHA，避免让一个文件
试图包含自身 SHA。外层 QS YAML 仍须在提交前做本地/远端 SHA、只读、dry-run、
唯一 job name 和 RUN_DIR absent 审计，但不伪称运行时能自验证提交 YAML。

所有依赖文件名顺序的账本都把排序规则冻结为 bytewise C locale：launcher
全局导出 `LC_ALL=C`，code ledger 与 scientific-artifact ledger 的管线仍各自
显式使用 `LC_ALL=C sort -z`。冻结、远端重放和 Pod 运行时必须使用同一规则，
不能继承提交机的 `en_US`/`C.UTF-8` 排序差异。任何 trap 安装后的 preflight
digest 或资源门禁失败，都必须以非零返回触发统一失败处理并留下 `FAILED`、
`FAILED_PHASE` 与阶段标记，不能在 helper 内直接 `exit` 绕过取证。

## 性能与内存报告口径

主性能使用 fresh-state ABBA：每个 arm 先 warmup 一次，随后每 arm 四个
独立 trial；奇偶 rank 使用相反顺序。每次 trial 后显式 unregister backend、
释放临时对象、`gc.collect()`、`empty_cache()`、同步，并要求 allocated 与
reserved baseline 精确恢复。

必须分开报告：

- 公共 dense document prefill；
- 公共 dense-to-NHD Q16 pack；
- per-request setup（fresh arm 在这里计入 pool allocation 和完整物理 block
  copy；reuse arm 只计 request fork 与 block table）；
- continuation 第一个完整模型步；
- `cached_document_request_ttft = request setup + continuation first model step`；
- setup + 全部 continuation generation 的 allocator 峰值。

allocator 的 absolute current/peak allocated 与 reserved、以及相对 setup 前
baseline 的 delta，分别在公共 prefill、公共 pack、request setup、
setup+首个 continuation step、setup+全部 generation 五个阶段落账；因此
partial-tail COW 触发后的首步显存状态可独立审计。

`cached_document_request_ttft` 明确不含公共文档 prefill/Q16 pack，因此不是
raw-prompt 标准 TTFT。continuation first step 包含整个 backbone、lm_head、
argmax 和同步，并非 isolated attention kernel latency；v2 不报告 kernel
speedup。HF eager 只作绝对延迟/输出兼容性诊断，FP32 dense oracle 只用于
数值归因；二者都不进入主 speedup 分母或授权条件。

partial-tail COW 由 `Q16PagedSequence.append()` 在首个 continuation model
step 中触发，两个 arm 都走同一 append 路径；它不属于 request setup，但被
`cached_document_request_ttft` 覆盖。artifact 与 summary 必须把两边的
`partial_tail_staging_copy_nbytes` 分列并要求相等。

allocator 同时记录 current/peak allocated 与 reserved。若 fresh 分母为 0，
ratio 必须为 `null/undefined`，不能用 1-byte 人工分母制造“下降”。

full-attention storage 每层使用：

`block_bytes = 2 * page_size * kv_heads * head_dim * element_size`

并分列有效文档 payload、source document allocation、source padding、source
arena 预分配 private reservation、fresh duplicate document allocation、fresh
private reservation、active tail/append payload、active physical pages、reserved
unused、source/fresh/request block tables 和 CPU reservation metadata。source
arena 的预留 private capacity 不能包装成“纯文档常驻内存”；active payload
和 active pages 已包含在 reservation 内，不能再次加到 total。combined unique
storage 只作为 before-continuation/after-decode 诊断，不作为纯文档节省口径。

## 可以与不可以声称的结论

可以报告：Q16、batch=1、单请求、等长输入、10 个 full-attention 层、同一
vLLM kernel 的 full-copy control 对比 document reuse；以及对应的物理复制、
缓存常驻、allocator 峰值和 cached-document request TTFT。

不可以外推：Q8/Q4、ragged batch、多 query 并发、跨请求调度、isolated kernel
加速、Apple/MLX 后端、或 HF eager 与 vLLM 之间的 bitwise 等价。多 query
serving 和端侧能耗需要独立实验。
