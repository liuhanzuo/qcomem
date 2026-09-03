# Functional cache：最小可微 reference 与 Qwen3.5 接入面

## 当前结论

`qcomem_functional_cache.py` 提供了一个纯 PyTorch、小尺寸、CPU 可运行的
functional cache reference。它只验证一条关键的工程假设：只要 cache transition
不原地改写旧状态，document prefill 产生的计算图可以安全地跨过 document/query
边界，且分段执行能够与 merged causal forward 得到相同的 hidden、最终状态和梯度。

当前 reference 覆盖两类 Qwen3.5 hybrid cache 状态：

- full attention：K/V 通过 `torch.cat((old, new))` 生成新历史；
- GDN-like linear attention：causal-conv tail 和 recurrent matrix 都由表达式生成新
  tensor，输入 state 的 storage 与 version counter 不变。

测试命令：

```bash
PYTHONPATH=gpu python -m unittest test_qcomem_functional_cache -v
```

本地四项测试均通过：

1. document/query 分段（包括 document 内再次切段）与 merged forward 的所有
   hidden 和最终 cache tensor 对齐；
2. document 输入、query 输入和全部参数梯度与 merged backward 对齐；
3. query continuation 不改变 document state 的值或 `_version`，且新旧 state
   不共享 storage；
4. 空初始 state 合法，空 segment 序列会 fail closed。

这些结果不能解释为“真实 Qwen3.5 functional cache 已实现”。reference 中的
GDN 方程只复现 conv/recurrent state 的因果与可微接口，不是 Transformers 中
Qwen3.5 fused GatedDeltaNet 的数值复刻。真实 35B 接入仍为 fail-closed；已知
mutable cache 路径在 Trial 1830867 的 8/8 rank 上均因 autograd version mismatch
失败。

## Reference 的状态契约

每层调用遵循：

```text
(hidden_segment, old_layer_state) -> (new_hidden, new_layer_state)
```

必须满足：

- 不调用 `copy_`、索引赋值或原地 recurrent update；
- `new_layer_state` 的任何 tensor 都不能复用 `old_layer_state` 的 storage；
- document state 仍保留 autograd graph，不能用 `detach` 冒充完整两段训练；
- query 可以一次输入完整序列，因果 mask/scan 在序列内部继续推进 state；
- state 按 layer 显式传入并显式返回，不能依赖一个由所有层共享、内部变异的
  `DynamicCache`。

注意：完全 out-of-place 会在训练图中保留 document 与 query 的中间状态，首要目标
是语义与梯度正确，不是直接作为部署时最高性能实现。推理仍可使用 mutable cache，
但必须额外验证两条实现的 logits/hidden 等价。

## 接进 Qwen3.5 的明确 patch surface

### 1. CoMem adapter 层

文件：`gpu/qcomem_torch.py`

- `TorchSplitCausalLM._run_layers` 当前只返回 hidden，并让每层共同修改
  `past_key_values`。新增 `_run_layers_functional`（或把返回值升级为
  `(hidden, next_state)`），逐层读取 `state[layer_idx]` 并接收该层返回的新 state。
- `_layer_context` 当前让 causal-mask helper 从 `DynamicCache` 推断已缓存长度。
  functional 路径需显式传入 `past_length`/position offset，并保证 full-attention 与
  recurrent mask 的尺寸与原实现相同，不能为生成 mask 临时修改 cache。
- `write_lower_replay`、`continue_lower_replay` 与 suffix document/query 两段执行需有
  functional state 版本。训练路径保留有图 state；存储/部署路径可继续使用现有
  packed mutable state，两者不得共用含糊的 metadata 名称。

文件：`gpu/qcomem_lora.py`

- `quant_student_suffix_hidden` 新增真正的 `functional-cache` 分支：先以 document
  residual 得到 suffix state，再把同一有图 state 传给完整 query continuation；
  selected-logit loss 覆盖所有 query positions。
- 只有 tiny-Qwen 与目标模型的 forward/backward、state immutability、全目标模块
  gradient coverage 均通过后，才能把 `functional_cache_capability_gate()` 的真实
  Qwen gate 改为 true。
- checkpoint metadata 固定执行语义，例如
  `functional_document_prefill_then_full_query_continuation`；downstream load 必须
  hard-check depth、bit policy 和该执行语义。

### 2. Transformers Qwen3.5 layer 层

远端 8×H20 环境已经锁定为 Transformers `5.14.1`；实际实现文件是
`transformers/models/qwen3_5/modeling_qwen3_5.py`，需要接入的类是
`Qwen3_5Attention.forward` 和 `Qwen3_5GatedDeltaNet.forward`。配置里的
`model_type=qwen3_5_moe` 不能当作 Python module path。

- full attention：绕开 `DynamicCache.update`，接收本层 `(past_k, past_v)`，用
  out-of-place concat 生成 `(all_k, all_v)` 并随 layer output 返回；GQA head layout、
  RoPE position 与 mask 必须保持原实现。
- GatedDeltaNet conv：当前 cache 的 `conv_states` 更新替换为显式输入 tail、显式输出
  next tail。训练 functional 路径不得调用写入 cache buffer 的 fused causal-conv
  update API。
- GatedDeltaNet recurrent rule：显式输入 recurrent state，并返回 scan 的最终新
  state；需要调用支持 non-mutating initial state 的 kernel，或先写 PyTorch/Triton
  unfused reference。不能在 wrapper 外 clone 一次后继续让 query 内部多 token 原地
  更新，因为这仍可能触发 version mismatch。
- layer output：为训练路径返回 `(hidden, next_layer_state)`；常规推理 API 保持原状，
  避免无意降低线上 decode 性能。

### 3. 必须通过的集成 gates

- 随机 tiny Qwen3.5 hybrid config：merged 与 document/query 两段的逐位置 hidden、
  logits、最终 K/V、conv 与 recurrent state 对齐；
- 对同一标量 loss 比较 document/query input gradients 和所有 suffix 参数梯度；
- document state 在 query 前后 value、storage pointer、`_version` 完全不变；
- attention q/k/v/o 和每个目标 suffix layer 的 gradient present/finite/nonzero，且
  gradient elements 与 trainable shard elements 一致；
- 真实模型 1-step 在 anomaly detection 下通过，再做 8-rank gradient coverage；
- functional 训练路径与 mutable deployment 路径做 all-query-position semantic gate，
  不能只比较最后一个 query token。

## 工程量与主要风险

在已有 tiny reference 基础上，研究级、未融合的 Qwen3.5 接入预估为
**8–14 个工程日**：

- full-attention functional K/V adapter 与 mask parity：1–2 日；
- GDN conv/recurrent 的非原地实现及对固定 Transformers 版本的数值 parity：3–5 日；
- layer/state plumbing、CoMem 训练接口与 metadata hard gate：2–3 日；
- tiny-Qwen、单卡真实模型、DDP/FSDP gradient/semantic gates：2–4 日。

最大风险是 GatedDeltaNet 的 fused scan/causal-conv API 可能只提供 mutable cache
接口。若需要自定义 autograd 或新 Triton kernel，在上述正确性原型之后还需约
**1–2 周**；若目标是可长期维护且接近原 fused 性能的生产实现，总周期更现实地是
**3–6 周**。因此下一步应先完成固定 Transformers commit 上的 tiny-Qwen CPU/CUDA
parity，不应直接把 35B smoke 当作开发调试手段。
