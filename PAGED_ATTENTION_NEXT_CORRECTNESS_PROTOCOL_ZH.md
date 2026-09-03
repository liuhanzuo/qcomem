# 页式 attention 下一版 correctness 协议（未执行、未提交）

## 目的与数据隔离

本协议是在 Trial `1834110` 得到真实负结果后制定的下一版预注册协议，不回改已失败的
final-logit 等价门禁，也不使用该 Trial 的 source 6--9 数值来选择新阈值。

- correctness/calibration 数据只用 **PG-19 train split** 的冻结窗口；
- LongBench validation source 6--9 只用于随后独立的 deployment benchmark / downstream
  quality，不参与 kernel 修复、阈值选择或 early stopping；
- source >=68 / test-v2 始终禁止读取；
- PG-19 窗口 ID、原始文件 SHA256、builder SHA256、tokenizer/model manifest、抽样 seed、
  document/query 边界与最终 JSONL SHA256 必须在首次 GPU 运行前冻结；
- correctness 窗口与任何 LoRA/SFT optimizer 使用的样本按 source document hash 隔离，避免训练泄漏。

阈值为事前定义：final greedy top-1 agreement 必须为 `1.0`，example-equal mean KL 必须
`<=1e-3`。它们不是根据 LongBench validation 或 Trial `1834110` 的误差拟合得到。

## Gate A：逐层 isolated kernel parity

目的：把“页式 kernel 本身的误差”和“上游隐藏状态误差继续传播”分开。对每个配置导出的
full-attention 层（当前模型为 10 层，但实现不得硬编码层号），在 stock eager 同一次前向上用
pre-hook 捕获该层已经计算好的：

- `query_states`（已 q-norm 与 RoPE）；
- `key_states` / `value_states`（包含 document cache 和当前 query 后的逻辑 KV）；
- 完整 additive/bool `attention_mask`；
- `scaling`、GQA head metadata、dtype/device 和 page table。

随后不再运行 decoder 后层，而是把**同一份捕获输入**分别交给 HF eager oracle 和候选
production paged kernel。因此第 7/11/... 层的比较不会包含前一 full-attention 层候选输出造成的
上游漂移。每层、每窗口至少记录：

- attention output max-abs、mean-abs、relative-L2；
- NaN/Inf、mask/page offset、GQA head mapping；
- eager/candidate dtype 与 accumulator mode；
- 所有 page hit、dense fallback count、最大解包页大小；
- candidate 与 eager 的 output shape/layout/contiguity。

Gate A 必须 100% layer × window coverage、0 dense fallback、0 mask/page/GQA mismatch。具体
local numeric tolerance 由 production kernel 的声明契约固定在 run manifest；不得在看到 PG-19
结果后修改。若目标 kernel 声明 bitwise/deterministic parity，则必须按该更强声明门禁。

## Gate B：端到端 semantic parity

在同一冻结 PG-19 train-only 窗口集上，以完全相同 document/query caller 比较 stock eager 与
候选 production paged kernel：

1. 每个 query 最终位置保存完整 logits 的 FP32 log-softmax；
2. greedy top-1 agreement 必须为 `1.0`（所有样本完全一致）；
3. 对每个样本计算 `KL(p_eager || p_paged)`，再做 example-equal mean，必须 `<=1e-3`；
4. 另存 max/p50/p95 KL、logit relative-L2、top-5 overlap，但这些是诊断项，不可替代两项硬门禁；
5. 同时要求所有 full-attention 层拦截完整、0 dense fallback、cache length/page ownership exact。

KL 必须使用完整词表 FP32 logsumexp 计算，禁止只取 top-k 后把 tail 忽略；如果显存不足，可按
vocab block 做数值稳定的 streaming logsumexp / KL，但结果必须通过小词表 oracle 回归。

Gate A 与 Gate B 都通过后，才可在 LongBench validation source 6--9 上运行性能与 downstream
quality。validation 的结果不能反过来改本协议阈值。

## Production kernel 接口契约

下一实现建议由 Triton/CUDA/FlashInfer 类 production kernel 提供统一接口，而 Python reference
仅负责 page store、量化、page table 与 oracle：

```python
def paged_attention(
    query,                  # [B, Hq, Q, D], post-norm/post-RoPE
    page_table,             # logical page order and valid lengths
    key_pages, value_pages, # dense/Q8/Q4 payload descriptors
    attention_mask,         # exact caller mask; no implicit second causal mask
    *,
    scaling,
    num_key_value_groups,
    output_dtype,
    accumulator_mode,
    audit,
): ...
```

实现还必须提供 `capture_oracle_inputs(...)` 或等价 hook adapter，使 Gate A 可直接复用同一
q/k/v/mask；`audit` 至少返回 kernel name/version、Triton/CUDA commit、page hits、fallbacks、
dtype/accumulator、temporary bytes 和 deterministic flag。Q16 correctness 先通过后才测试 Q8/Q4；
量化质量误差与页式 kernel 算术误差要分成不同 gate/report。

## Fail-closed 调度

正式 8-GPU job 的 rank0 先执行 Gate A/B；其余 rank 不加载模型并等待带 SHA256 的授权 artifact。
超时、artifact 缺失、schema/hash 不匹配或任一硬门禁失败时，全 job 在 benchmark 前退出。只有
授权 artifact 验证通过，才启动 8-rank LongBench validation source 6--9 benchmark。失败 artifact
永久保留，不自动提交 replacement，不自动改阈值。
