# 35B 全参数 Supervised SFT H20 Smoke（2026-08-12）

> 后续状态：正式 384-step 训练、heldout 选择和三个 DCP 已完成，见
> [正式 full-SFT 结果](RESULTS_GPU_DENSE_SFT_FORMAL_2026-08-12_ZH.md)。本文保留为一步
> execution / numerical-update 门禁记录。

## 结论

Qwen3.5-35B-A3B 的真实全参数一步更新已经在 8×H20-141GB 上通过数值门禁。修正后的
Trial `1831289` 使用 FSDP1 `FULL_SHARD`：模型先以 BF16 加载，wrap 后只把每个 rank 的
持久 optimizer shard 提升为 FP32；forward 仍为 BF16，gradient reduce、Adam parameters
与 moments 均为 FP32。它完成 answer+EOS global-token-weighted CE、backward、全局裁剪和
AdamW step，并证明：

- 34,660,610,688 个参数全部有 gradient tensor，missing/nonfinite 均为 0；
- 40 层、embedding、final norm、lm_head 的 FP32 update 都非零；
- FP32 master 中 `32,377,707,746` 个元素改变，占 93.41%；
- 转成下一次 BF16 forward 可见值后仍有 `768,886,188` 个元素改变，占 2.22%。

所以现在可以说“35B dense full-model supervised SFT 的一步参数更新真实发生且 8×H20
可执行”。仍然不能说质量已经改善：这里只用了 8 条 train-only 样本、一步、metadata-only，
没有保存可评测 checkpoint，也没有运行独立 validation。

首轮 Trial `1831074` 仍保留为重要的审计反例：它虽然完成 BF16 backward 与 AdamW 调用，
但 BF16 参数/BF16 moments 配 `lr=1e-6` 缺少 delta 证据。它只能算 execution capability
通过，不能单独证明有效更新。第二轮正是为关闭这个漏洞而运行。

它也不是 Q-CoMem cached suffix SFT。dense smoke 明确 `use_cache=False`；真正的两段
document-prefill/query-continuation 训练仍被 mutable-cache autograd 问题阻断。

## 首轮 BF16 execution smoke：保留的反例

- Job / Trial：`234718 / 1831074`，终态 `Complete`；
- Web UI：<https://qs2.devops.xiaohongshu.com/model/production/job/trial/234718/1831074>；
- 硬件：8×NVIDIA H20-3e，每卡 `143771 MiB`；
- 模型：Qwen3.5-35B-A3B，冻结目录标识 `59d61f3`；
- 文本模型参数：`34,660,610,688`，全部 `requires_grad=true`；
- 分布式：8-rank FSDP1 `FULL_SHARD`、`use_orig_params=true`；
- mixed precision：parameter/reduce/buffer 均为 BF16，目标位置 logits 转 FP32 计算 CE；
- 激活重算：40 个 MLP/MoE block 使用 non-reentrant checkpoint，attention 不重算；
- 数据：Qasper train 4 条 + 2WikiMQA train 4 条，共 8 条；
- 监督：只监督 `selected_answer + EOS`，prompt labels 全为 `-100`；
- 序列上限：1024 token；7/8 样本发生 context head+tail 截断，答案和 EOS 均保留；
- prepared JSONL SHA-256：
  `b4df78bacc6d08f2d2e615b4c803f795d5a188c1d230f3b84a18d382561a91b9`；
- manifest SHA-256：
  `56bb22bc215bd3af5475b3cdde474b8d3706066dab1ccf98154bf46469cb5e97`；
- LongBench test-v2：未读取，metadata 记录 `test_or_validation_used=false`；
- 本地产物：`results/gpu-dense-supervised-sft-smoke-20260812a/`。

运行时同时核验了训练代码六文件 ledger 和模型/tokenizer 六文件 ledger；二者均
`all_artifacts_exist_and_match=true`。Tokenizer 为 `Qwen2Tokenizer`，EOS id `248046`，
chat-template SHA-256 为
`a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`。

## 一步训练结果

| 指标 | 结果 |
|---|---:|
| 全局样本数 | 8 |
| 全局 answer+EOS target tokens | 99 |
| 8 rank 的样本 CE 简单平均 | 2.069428 |
| 裁剪前 global grad norm | 99.5 |
| grad norm 门禁 | finite、`>0`，通过 |
| max grad norm | 1.0 |
| learning rate | `1e-6` |
| metadata 内一步区间 | 8.46 s |
| launcher `00_start` 到 `99_done` | 2 min 36 s |
| QS 作业执行时间 | 5 min 45 s |

`fsdp.clip_grad_norm_(1.0)` 返回的是 BF16 口径下的裁剪前全局 norm；对应缩放系数约
`0.01005`，但本轮没有单独测量裁剪后 norm，不能声称它数值上精确等于 1.0。随后代码执行
`optimizer.step()`。step 后每个 rank 都观察到 Adam state：约 `16.14 GiB/rank`，主体为
两个 BF16 moment tensor。这些状态以及 `last_step=1` 证明执行流越过了 backward 和 step，
但不是权重 delta 的替代证据。

这里的 CE 是“每 rank 先对本样本 target token 求均值，再对 8 个 rank 等权平均”，不是把
99 个 target token 放在一起做 token-weighted mean；后者按保存的逐样本结果重算为
`2.308673`。8 条 smoke 样本太少，且 7 条 context
被截断；这个数值只用于 finite/可训练门禁，不用于比较模型质量。

## 显存账本

| 阶段/对象 | 每 rank 观测值 |
|---|---:|
| FSDP wrap 前完整 BF16 文本模型 allocation | 64.56 GiB |
| FSDP wrap 后参数 shard allocation | 8.07 GiB |
| trainable shard numel | 4,332,576,336 |
| step 后 Adam state | 16.14 GiB |
| reset 后训练阶段 max allocated | 40.42 GiB |
| reset 后训练阶段 max reserved | 89.68--90.14 GiB |

`max_reserved` 是 CUDA allocator 保留量，不等于同时存活的 tensor，也不应写成模型真实
active memory。当前 loader 会在 FSDP wrap 前让每个 rank 短暂持有完整 BF16 文本模型；
这是可优化的启动峰值，但本次没有 OOM。run 只写 metadata，不保存约 52 GiB 的全局 BF16
model checkpoint，也不保存 optimizer checkpoint。

运行时还明确提示 flash-linear-attention 与 causal-conv1d 快路径不可用，GatedDeltaNet/
causal-conv 回退到 Transformers 的 torch 实现。因此本轮时间和显存是 capability smoke 口径，
不能当作安装 fused kernels 后的生产训练吞吐。

## 修正后的 FP32-shard 数值更新 smoke

### 可复现信息

- Job / Trial：`234809 / 1831289`，终态 `Complete`；
- Web UI：<https://qs2.devops.xiaohongshu.com/model/production/job/trial/234809/1831289>；
- QS 执行时间：7 min 32 s；launcher `00_start` 到 `99_done` 为 6 min 03 s，其中包含
  14 个约 72GB weight shards 的一次完整 SHA-256 校验；
- 训练 metadata 区间：11.28 s；不是纯 kernel time；
- PyTorch `2.11.0+cu129`、Transformers `5.14.1`、CUDA `12.9`、NCCL `2.28.9`；
- 数据与 tokenizer 和首轮完全相同，LongBench test-v2 仍未读取；
- code/runtime/model-artifact/14-weight-shard 四份 ledger 均通过；
- metadata SHA-256：
  `6028be694b5df208aa769999abbddaf88457fa35f22b1f9132c9198ec9779344`；
- metrics SHA-256：
  `3e273704b95542fe95d896a3e22ad3f616db9723cb25ed6a42bdfbb3e749b210`；
- 本地产物：`results/gpu-dense-supervised-sft-fp32-delta-smoke-20260812a/`。

两进程 tiny FSDP preflight 在加载 35B 前先验证了相同 runtime 的原生 dtype 路径：persistent
parameter FP32、forward BF16、gradient FP32、Adam moments FP32 且 parameter delta 非零。

### 目标和梯度

第二轮不再按 8 个 QA 等权反传，而是把每 rank 的 local mean CE 乘以
`local_target_tokens * world_size / global_target_tokens`，补偿 FSDP 的 rank-average，得到
真正的 99-token global-token-weighted objective。

| 指标 | 结果 |
|---|---:|
| global-token-weighted answer+EOS CE | 2.308673 |
| 仅作参照的 8-sample 等权 CE | 2.069428 |
| 裁剪前 global grad norm | 49.847645 |
| 裁剪阈值 | 1.0 |
| gradient tensor coverage | 34,660,610,688 / 34,660,610,688 |
| finite、nonzero gradient elements | 32,635,971,413（94.16%） |
| missing / nonfinite elements | 0 / 0 |
| 裁剪后 gradient L2 | 0.99999995 |

40 个 transformer layer 加 embedding、final norm、lm_head 共 43 个组都存在 finite、nonzero
gradient。这里按 fused MoE parameter 统计；未路由 expert 的局部元素可以为零，但没有整层丢失。

### Optimizer 与真实更新

8 个 rank 各自持有 `4,332,576,336` 个 FP32 optimizer parameter elements。每个非空 local
parameter 的 Adam step 都精确为 1；每 rank 的两个 FP32 moments 共 `8,665,152,672`
elements，全部 finite，且 state coverage 完整。

| Delta 口径 | Changed elements | 比例 | L2 | Max abs |
|---|---:|---:|---:|---:|
| FP32 persistent optimizer shards | 32,377,707,746 | 93.41% | 0.148730 | 1.01328e-6 |
| cast 为 BF16 后对下一次 forward 可见 | 768,886,188 | 2.22% | 0.036635 | 1.90735e-6 |

所有 40 层的 FP32 delta 都非零；embedding、final norm、lm_head 也都非零。final norm 的
单步 FP32 delta 尚未跨过 BF16 ULP，因此其 BF16-visible changed count 为 0；这恰好说明
为什么必须保留持久 FP32 shard 来累积 sub-ULP 更新，而不能每步只写回 BF16 后丢掉低位。

### 显存

| 阶段/对象 | 每 rank 观测值 |
|---|---:|
| FSDP wrap 前完整 BF16 文本模型 allocation | 64.56 GiB |
| wrap 后 FP32 persistent parameter shard | 16.14 GiB |
| step 后 FP32 Adam moments | 32.28 GiB |
| delta gate 的 FP32 before snapshot（逻辑预算） | 16.14 GiB |
| 训练阶段 max allocated | 83.14 GiB |
| 训练阶段 max reserved | 109.27--123.24 GiB |

相对首轮 BF16 smoke，FP32 parameter/gradient/moments 和 snapshot 把 max allocated 从约
40.42 GiB 提高到 83.14 GiB，但仍低于 H20 的 140.4 GiB 可见容量且没有 OOM。reserved
仍只是 allocator 保留量。正式多步训练不需要永久保留一步 before snapshot；可以仅在
预注册 probe step 做 delta gate，从而释放约 16.14 GiB/rank。

## 与 LoRA 的关系

目前三条结果必须分开解释：

1. Interface LoRA：200-step 后 validation 相对 frozen chunk-d7 提升 `+0.04384` mean F1，
   但 overall CI 跨 0，且仍未追平 dense；属于部分恢复。
2. Frozen-static quant LoRA：200-step 的末 20-step KL 比首 20-step 更高，趋势 gate 失败。
3. Dense full SFT：FP32-shard 版本已经通过 1-step 真实 parameter-delta 门禁；但尚未运行
   多步训练、保存 checkpoint 或做 validation，也没有直接针对 Q4/Q8 cache 误差。

Dense full SFT 可以作为“任务适配上界”或独立 baseline；它本身不会自动证明模型学会容忍
Q-CoMem 的 Q4/Q8 状态量化。要回答压缩恢复，最终仍需比较同一训练后模型的 dense 与
Q-CoMem 路径，或先实现可微、无原地 mutation 的 functional cache 再做 suffix full SFT/QAT。

## 下一步门禁

FP32 parameter-delta、全层 gradient coverage、global-token weighting、完整依赖/权重 ledger 和
runtime version 门禁均已通过。现在多步 full SFT 的剩余前置条件是：

1. 构建更大的、与已消费 validation/test hash 不重叠的 balanced train-only 数据；
2. 使用 FSDP/DCP 分片保存和恢复 FP32 model/optimizer state，禁止 rank-0 full gather；
3. 只在首步/预注册 probe step 保留 before snapshot，正常 step 释放这 16.14 GiB/rank；
4. 固定 scheduler、shuffle、gradient accumulation、checkpoint interval 与停止条件；
5. 用独立 train-heldout CE 监控过拟合，再在冻结 checkpoint 后运行 LongBench validation；
6. 同一个 full-SFT checkpoint 同时评测 dense 与 frozen-static Q-CoMem，才能判断任务适配与
   Q4/Q8 压缩容忍度是否同时改善。

正式质量矩阵至少需要：原始 dense、dense full-SFT、full-SFT 后的 frozen-static Q-CoMem，
以及已有 Interface LoRA；报告 paired F1、bootstrap CI、灾难性退化率和持久/训练资源成本。
在此之前不能声称 full SFT 优于 LoRA，也不能声称 full SFT 恢复了 Q4/Q8 精度。
