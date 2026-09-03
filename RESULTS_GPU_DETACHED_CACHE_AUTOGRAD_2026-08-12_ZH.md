# Detached-document-cache LoRA 能力门禁负结果（2026-08-12）

## 结论

真实 Qwen3.5-35B-A3B、8×H20 的冷启动 1-step capability gate 证明：把 document suffix
prefill 放入 `no_grad`，再对 document cache 执行 `detach().clone()`，仍不足以让完整 128-token
query continuation 可微。Forward 已完成并进入 `loss.backward()`，但 8/8 rank 都发生相同的
mutable-state autograd version mismatch。

因此：

- `detached-document-cache` 没有通过 trainability gate；
- 没有 optimizer step、LoRA update 或 checkpoint；
- gradient coverage 与 cache immutability 虽已实现为 hard gate，但执行顺序在 backward 之后，
  本次没有到达并持久化这些 gate；
- 后置的 8×128 all-query detached-vs-deployment semantic gate没有运行；
- 不准备 Interface warm-start、learning rate `2e-5` 的 200-step 配置，也不自动重提任务。

## 任务与复现信息

- Job / Trial：`235202 / 1832364`；终态 `Failed`；
- 页面：<https://qs2.devops.xiaohongshu.com/model/production/job/trial/235202/1832364>；
- 硬件：单节点 8×H20-141G；QS node health 为 Healthy；
- 数据：PG-19 train-only smoke64，SHA-256
  `ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c`；
- 配置：512 document + 128 query、depth 7、residual Q4、attention Q4、linear Q8、
  `cache_layer_bits=[8,8,8,4,8,8,8]`；
- LoRA：36 个 suffix attention projections，6,193,152 个可训练参数；
- immutable code snapshot：
  `/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_gpu_detached_lora_capability_20260812a`；
- remote run directory：
  `/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/lora-quant-detached-capability-20260812a`；
- 本地产物：`results/gpu-lora-quant-detached-capability-20260812a/`；
- code manifest SHA-256：
  `2ec7a9c583cefcdfe2cc9bb42ec0c7064bc5b135b502de9e160cf472dbb8b4de`；
- train log SHA-256：
  `059da95e3a3d4755f68c291573117e313df71a430cb5f58305693fc926e907c9`；
- metadata SHA-256：
  `1f13507f1ed06e87a962f89a5e90f31aa8fd13081ff24be28351e0c1dbfca6c3`；
- `artifact.sha256` 固定全部下载产物，manifest SHA-256 为
  `16469100f13a3b4a653dbcc4550683bee25ad8213c3cbf87f39aa186a44ad0ea`，
  `sha256sum -c` 全部通过；`test_v2_used=false`。

提交前本地与远端同环境共 19 项 focused tests 通过，包括 tiny cache 的 storage/version
immutability、全 query semantic 聚合与 checkpoint/update 审计。它们只验证 gate 本身；真实模型
是否可微必须由本 Trial 决定。

## 失败签名

8/8 rank 都在 `loss.backward()` 报告：

```text
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation: [torch.cuda.FloatTensor [1, 32, 128, 128]],
which is output 0 of torch::autograd::CopyBackwards, is at version 1;
expected version 0 instead.
```

历史 mutable `cached-two-stage` Trial `1830867` 的同形状错误为 version 2、expected 1；本轮先
detach+clone document cache 后仍出现 version 1、expected 0。结合错误发生在完整 query
continuation backward，可合理定位为：**query continuation 内部的 mutable recurrent/cache update
本身仍覆盖了 autograd 保存的状态**。这是从两次实验得到的工程推断，不是 kernel 级 traceback。

本次不是 OOM。节点健康，所有 rank 都成功加载模型、安装 36 个 LoRA 模块并完成 student forward；
NCCL/torchrun 的后续退出是 rank autograd 异常后的清理。

## 未达到的 gate

metadata 只包含初始化配置，明确显示：

- `last_step`：不存在；
- `last_gradient_coverage`：不存在；
- `last_detached_capability`：不存在；
- optimizer state：空；
- checkpoint 数量：0；
- detached semantic shard 数量：0。

所以不能因为 forward 成功或 tiny immutability test 通过，就声称真实 cache immutability、query
gradient 或部署语义已经通过。

## 下一条可行路线

不再继续尝试 wrapper 外层的 mutable-cache clone 变体。科学上与原目标一致的下一步是为固定
Transformers 5.14.1 的 Qwen3.5 suffix 实现 **query-side functional cache**：

1. full-attention K/V 用 out-of-place concat 并显式返回新 state；
2. GatedDeltaNet conv/recurrent state 使用不原地覆盖 initial state 的 transition；
3. 每层采用 `(hidden, old_state) -> (new_hidden, new_state)`，完整 query graph 保留；
4. 先做 tiny-Qwen forward/state/gradient parity，再运行相同的真实 1-step capability gate。

成本更低但方法不同的对照，是冻结整个 cached Q-CoMem forward，只在最终 frozen hidden/readout 上训练
小 adapter。它绕开 cache backward，能够回答“纯 readout adaptation 是否有用”，但不能称为 CoMem
内部状态量化恢复，也不能替代 functional-cache 主线。
