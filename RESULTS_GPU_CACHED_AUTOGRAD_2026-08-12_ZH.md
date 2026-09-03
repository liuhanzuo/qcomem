# Q-CoMem cached-two-stage 反向图负结果（2026-08-12）

## 结论

真实 Qwen3.5-35B-A3B、8×H20 的 1-step smoke 证明：当前 Transformers
DynamicCache/GatedDeltaNet mutable cache 下，`cached-two-stage` quant student 可以完成
forward，但**不能完成 backward**。8/8 rank 都在 `loss.backward()` 报同一个 inplace
autograd version mismatch。因此：

- 不能声称 `cached-two-stage` 已经可训练；
- 不能把历史 `merged-uncached` 训练结果说成两段 cache 部署语义的训练结果；
- 本次不是 OOM，后续 NCCL abort 是某个 rank 先发生 autograd 错误后的清理，不是根因；
- 没有完成 optimizer step，也没有产生 gradient-coverage 或 checkpoint。

## 可复现信息

- Job / Trial：`234637 / 1830867`，终态 `Failed`；
- Web UI：<https://qs2.devops.xiaohongshu.com/model/production/job/trial/234637/1830867>；
- 模型：Qwen3.5-35B-A3B；硬件：8×H20-141GB；
- data SHA-256：`ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c`；
- frozen-static student：depth 7、residual Q4、attention Q4、linear Q8、
  `cache_layer_bits=[8,8,8,4,8,8,8]`；
- LoRA：suffix attention 36 modules、6,193,152 trainable parameters；
- 本地产物：`results/gpu-lora-quant-cached-two-stage-smoke-20260812c/`；
- train log SHA-256：`3183ecbe2e13be1f0de8f4e3738463742a6b2044b040cc09f5c4ac098f67a49d`；
- metadata SHA-256：`d377e8d5ff417b4543e5b436e86142d4f18a8106d20197bb8a31397e4515b597`；
- `test_v2_used=false`，没有读取 LongBench test-v2。

launcher 在启动前完成 13 项 LoRA 定向测试并写入 `code.sha256`。运行时 8 个 rank 均成功
加载模型并进入真实 student forward/backward；失败不是静态检查或模型加载失败。

## 失败签名

8/8 rank 的核心错误一致：

```text
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation: [torch.cuda.FloatTensor [1, 32, 128, 128]],
which is output 0 of torch::autograd::CopyBackwards, is at version 2;
expected version 1 instead.
```

这说明 document suffix prefill 写入 mutable cache 后，query continuation 继续原地更新同一类
cache tensor；backward 需要的旧版本已被覆盖。日志中的 NCCL `Abort` 发生在 rank autograd
异常之后，只是 `torchrun` 终止其余进程的结果。每卡约 66.5 GiB allocation，没有 CUDA OOM。

## 未通过的 gate

- `last_step`：不存在；
- optimizer update：未执行；
- `last_gradient_coverage`：不存在，因为 backward 在 coverage 审计前失败；
- 36-module LoRA A/B finite/nonzero coverage：未测得；
- checkpoint：未生成；
- deployment-semantic trainability：失败。

因此 forward 成功不能替代 trainability gate。

## 资源止损

suffix-full FSDP Trial `1830869` 使用相同 mutable `cached-two-stage` 语义，而且训练参数为
27,751,037,952。它在 `Uncommit`、无 Pod、无 run directory 时被停止，终态为 `Terminated`；
没有为已知必然失败的反向图消耗 8×H20 训练资源，也没有提交替代任务。

## 后续设计边界

后续 `detached-document-cache` 路径已经由 Trial `1832364` 做了真实冷启动 1-step capability
gate。该版本将 document suffix prefill 放在 `no_grad` 下，并对 cache tensor 执行
`detach().clone()`；真实 forward 成功，但 8/8 rank 仍在 query continuation 的
`loss.backward()` 失败：同形状 `CopyBackwards` tensor 的 version 为 1、expected 0。没有
step、gradient coverage、cache immutability、checkpoint 或 semantic shard。完整证据见
[RESULTS_GPU_DETACHED_CACHE_AUTOGRAD_2026-08-12_ZH.md](RESULTS_GPU_DETACHED_CACHE_AUTOGRAD_2026-08-12_ZH.md)
和 `results/gpu-lora-quant-detached-capability-20260812a/`。

两轮错误签名共同说明，仅在 wrapper 边界 clone cache 不足以得到可微的完整 query：document/query
跨边界 mutation 已被隔离，但 query 内部 mutable recurrent/cache update 仍会覆盖 autograd 保存的
状态。因此现在只保留两个语义不同的后续方向：

1. `functional-cache` 主线：attention K/V 使用 out-of-place concat，GatedDeltaNet conv/recurrent
   state 显式返回新 state，并先通过 tiny-Qwen parity 与同一真实 1-step gate。当前生产 Qwen3.5
   integration 尚未实现，不能提交 200-step 训练。
2. frozen readout adapter 对照：冻结整个 cached Q-CoMem forward，只训练最终 hidden/readout 小
   adapter。它可以测试纯 readout adaptation，但不能称为内部状态量化恢复。

在 functional path 完成真实 backward、全 module/layer finite+nonzero query gradient、cache
immutability、optimizer update 与 deployment semantic gate 前，不准备 200-step 或正式训练。
