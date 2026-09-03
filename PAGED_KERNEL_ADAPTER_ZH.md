# Qwen3.5 Q16 真正分页注意力适配方案

## 冻结环境审计

目标 H20 环境中已安装：PyTorch `2.11.0+cu129`、Transformers `5.14.1`、
vLLM `0.26.0+cu129`、FlashInfer `0.6.14`、Triton `3.6.0`。目标模型的
full-attention 几何是 16 个 Q head、2 个 KV head、GQA 分组 8、head dim
256；40 层中有 10 个 full-attention 层，没有 sliding window。

首选入口是 vLLM 的
`vllm.v1.attention.ops.triton_unified_attention.unified_attention`。它直接接收
固定大小 K/V block pool、block table、每个请求的 KV 长度和 query
indptr，支持 causal GQA、head dim 256、单 token decode 以及多 token append/
prefill。FlashInfer 的 `BatchPrefillWithPagedKVCacheWrapper` 与
`BatchDecodeWithPagedKVCacheWrapper` 也可用，但需要管理 plan 和 128 MiB 等级
workspace；对当前 Transformers 同 caller 接口而言，vLLM Triton 入口更薄。

## 内存布局与接口

`gpu/qcomem_vllm_paged_kernel.py` 新增了独立 Q16 adapter：

- K/V 都使用 `[physical_block, page_token, kv_head, head_dim]`；
- block table 使用 `[batch, logical_block]` 的 `int32` 物理块编号；
- Q 从 Transformers 的 `[B, Hq, Q, D]` 只重排为 `[B*Q, Hq, D]`；
- `cu_seqlens_q=[0,Q,2Q,...]`，`seq_lens` 记录 document+query 的真实长度；
- RoPE 已由 Qwen3.5 在 `cache.update` 之前完成，所以 kernel 内不再做位置编码；
- 普通无 padding 的 tail-causal mask 由 kernel 的 `causal=True` 表达，任意
  additive bias 或 padding mask 默认 fail closed。

文档建立时只做一次 dense-to-block-pool copy。请求 fork 只复制很小的 block
table。若文档末尾不是整页，第一个 query append 仅把最后一个不完整页复制到
请求私有块，再写 query K/V；不会复制或 `torch.cat` 完整文档 KV。arena 在建立
时按最大并发 fork 数和每请求最大 append token 数预留私有块，容量不足会在写入
前报错。

## Q8/Q4 分阶段原因

vLLM 0.26 的 Triton kernel 已原生支持 `INT8_PER_TOKEN_HEAD` 和 packed
`INT4_PER_TOKEN_HEAD`，并有对应的 reshape-and-cache kernel。但是它们与当前
QCoMem `PackedTensor` 格式不同：当前 Q8/Q4 是按 group 的 affine scale+bias；
vLLM Q8 是 per-token-head 动态 scale，Q4 还包含 Hadamard rotation 和非对称
zero-point。因此不能把当前字节直接交给 kernel。

分阶段建议：

1. 先用新 adapter 完成 Q16 H20 fused capability/parity/内存/时延门禁；
2. Q8/Q4 新增一次性的 document quantize-to-vLLM-block-layout 与 append quantize
   路径，分别调用 vLLM 的 per-token-head cache writer；
3. 重新做量化误差、下游语义以及跨层 mixed-bit 校准，不能沿用当前 Q8/Q4 数值
   结果冒充同一种量化格式。

## 仍需 H20 验证的门禁

当前只完成冻结包源码/API 审计与 CPU mock kernel 测试，没有提交 GPU 作业。正式
接入现有 runner 前必须在 H20 同一 caller 上验证：10 个 full-attention 层全部命中
fused backend、dense eager 与 Q16 token/hidden parity、block pool/document base 不被
多请求修改、请求路径 full-document staging copy 为 0，并分别报告 32-token 首次
query 与单 token decode 的 TTFT/TPOT。通过这些门禁后才能把它称作生产形态 kernel，
当前 Python two-pass 数据不能作为生产 TTFT 结论。
