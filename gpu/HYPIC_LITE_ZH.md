# HYPIC-lite：Qwen3.5 hybrid suffix 的 TTFT–bytes 对照原型

这是一条 **HYPIC-inspired 参考原型**，不是完整 HYPIC 复现，也不是新的 exact replay
路径。现有 `qcomem_torch.py` 和部署 exact replay 不做任何修改。原型回答三个可以分开验证的
问题：

1. 直接相加各段的 GatedDeltaNet zero-start end state（naive）会怎样；
2. 使用真实 GatedDeltaRule 内部 kernel 提取并组合
   `S_out = T_C @ S_in + S_C|0` 会怎样；
3. 边界重算 `w=0/8` 能恢复多少质量，以及为减少 suffix TTFT 实际增加多少持久化字节。

## 能力边界

目标 Qwen3.5 的 depth-7 suffix 有 33 层：24 个 linear-attention 层和 9 个
full-attention 层，而且两类层交错。Transformers 的公开 `DynamicCache` 对 linear 层只暴露
conv tail 和 recurrent end state，不暴露累计 transition；full-attention 层则需要完整文档
KV。因此：

- `linear transition-only` 不能让这个 hybrid suffix 跳过文档 token forward；
- `transition + seam-only KV` 仍不能提供 full-attention 的全历史 causal lookback；
- 当前唯一能实际跑到 query logits 的 HYPIC-lite 配置还要持久化各段独立算出的 full-attention
  body KV，并把它们拼接。这会遗漏跨段 hidden-state 影响，所以明确标为 approximate；
- `w=8` 在线重算每个内部边界的 8 个 token，只修复 seam，不能使独立 body KV 变成等价结果。

当前 Transformers/FLA 版本可以从内部 `chunk_gated_delta_rule` 提取真实 dense transition：
正常 zero-state 调用得到 `S_C|0`，再用 `S0=I`、zero value/write stream 调用同一 kernel 得到
`T_C`。公共 cache API 做不到这一点。生产实现至少需要 fused kernel 在一次 prefill 中输出
`(T_C, S_C|0)`、保存 causal-conv 的 `k-1` tail，并提供 transition compose 和量化 cache/KV
kernel；当前 identity-state 第二次调用只是 reference path。

## depth-7 / 4096-token 字节账

下面按真实配置计算：full-attention 是 2 KV heads × 256 head dim；linear-attention 是
16 key heads、32 value heads、128 key/value dim、conv kernel 4。`T_C` 必须按 **32 个 value
heads** 而不是 16 个 key heads计数，因为 key 会 repeat 到 value heads，而 `g/beta` 又是
per-value-head；所以每个 value head 都有不同的 dense transition。

| 场景 | full suffix KV | `S_C|0` | `T_C` | conv tail | seam KV budget | suffix 合计 |
|---|---:|---:|---:|---:|---:|---:|
| 1 segment, runtime state FP32 | 75,497,472 B | 50,331,648 B | 25,165,824 B | 1,572,864 B | 0 B | 152,567,808 B |
| 1 segment, all-BF16 payload | 75,497,472 B | 25,165,824 B | 25,165,824 B | 1,572,864 B | 0 B | 127,401,984 B |
| 4 segments, w=8, runtime state FP32 | 75,055,104 B | 201,326,592 B | 100,663,296 B | 6,291,456 B | 442,368 B | 383,336,448 B |
| 4 segments, w=8, all-BF16 payload | 75,055,104 B | 100,663,296 B | 100,663,296 B | 6,291,456 B | 442,368 B | 282,673,152 B |

`suffix 合计` 不包含约 9.71 MiB 的实测 mixed Q-CoMem lower state。作为尺度参照，已有
full-prefix 实测约 139.94 MiB：一段的 BF16 payload 加 lower state 约 131.2 MiB，只剩约
1.07×；Transformers 当前 FP32 recurrent state 下甚至更大。四段时 `S/T` 按段复制，内存
明显劣于 full prefix。这是该对照的重要负结果，不能只报告 TTFT。

原型还输出 Q8/Q4 的“全 suffix tensor 统一压缩”payload-only 下界：w=8 时一段分别为
63,700,992 / 31,850,496 B，四段分别为 141,336,576 / 70,668,288 B。它们是新的
compressed-HYPIC 组合估算，不是 HYPIC 原结果，也尚不可执行；未计 affine-group metadata，
且缺少量化 transition compose/KV kernel，不能作为已实现显存结果。

## TTFT 省掉什么

当前 Q-CoMem 请求要在线重建文档的整个 suffix：4096 × 33 = 135,168 个
suffix token-layer forwards。四段 `w=8` 的 HYPIC-lite 在线只重算 3 × 8 × 33 = 792 个
seam token-layer forwards，账面减少 99.41%；`w=0` 不重算文档 token。仍然保留：lower
query forward、suffix query forward、24 层逐段 state composition、9 层逐段 KV splice，
以及 cache fork/反量化。真实 wall-clock speedup 必须以 H20 输出为准。

## 正确性门槛和运行

GPU runner 在正式计时前硬性要求：

- 现有 dense/full-prefix/Q16 Q-CoMem token exactness gate 通过；
- 单 segment（没有跨段近似）HYPIC-lite 与当前 Q-CoMem first-token top-1 相同，且
  max-abs logit error ≤ 0.05；
- `T_C @ probe + S_C|0` 与同一个 GatedDeltaRule 从 probe 续算的 relative L2 ≤ 0.02。

多段本身是近似路径，不把“必须等价”伪装成 gate；每条结果同时记录对 same-packed Q-CoMem
和 exact full-prefix 的 top-1 agreement、logit relative L2/KL、真实 tensor persistent bytes、
private suffix cache bytes 和 token-layer work ledger。

8×H20 启动入口是 `launch_hypic_lite_8gpu.sh`，QS 模板是
`qs/qcomem-hypic-lite.yaml`。默认只读 validation source index 6–35，排除 calibration 4–5，
并拒绝 source index ≥68 的冻结 test-v2；same-packed Q-CoMem 使用预注册 frozen-static
策略 `[8,8,8,4,8,8,8]`。当前只准备了 launcher/YAML，未提交外部任务。
