# MacLLM-Bench

面向 Apple Silicon 的可复现本地大语言模型推理实验。项目的第一个里程碑是使用
MLX-LM 测试一个 3B、4-bit 模型；在建立可信的基线后，再逐步加入 llama.cpp、
更多量化格式和长上下文负载。

## 测试机器基线

- MacBook Pro，Apple M4 Pro
- 24 GB 统一内存
- arm64 架构
- macOS 26.4.1

## 运行项目

进入项目目录后，执行：

```bash
make run
```

该命令会在需要时为当前用户安装 `uv`，安装由 `uv` 管理的 Python 3.12，创建
`.venv` 虚拟环境，安装 MLX-LM，下载配置中指定的模型，记录当前机器信息，然后
运行基准测试。整个过程不需要 Homebrew 或 `sudo`。首次运行需要下载模型权重，
因此耗时会更长。

完成环境安装后，也可以分别执行以下命令：

```bash
make system-info
make smoke
source .venv/bin/activate
```

运行单进程的 512、2048、8192 token 长上下文测试：

```bash
make context
```

运行显式控制 Apple GPU 的 `mlx.core` 手写版本：

```bash
make manual-context
```

运行 Q-CoMem 的分层 residual 量化 smoke benchmark：

```bash
make comem-smoke
```

运行多文档、一次 Write 多次 Read 的正式入口：

```bash
make comem-multidoc
```

该命令默认先检查电源、低功耗模式、温度、后台 CPU 和 swap；条件不满足时会在模型
加载前退出。若只是验证实现，可运行 `make comem-multidoc-diagnostic`，但输出中的
`formal_result_eligible` 仍会是 `false`，不能作为正式性能数据。

## 第一个实验

Smoke 测试使用 `mlx-community/Llama-3.2-3B-Instruct-4bit` 模型，先进行一次
预热，然后正式运行三次；每次最多生成 64 个 token：

```bash
make smoke
```

实验输出保存在 `results/` 目录：

- `system_info.json`：硬件、操作系统、Python、内存和电源设置
- `runs.jsonl`：每次运行对应一条记录
- `summary.json`：各项指标的中位数、平均值、最小值和最大值
- `raw/`：完整的 MLX-LM 原始输出，用于核对结果和调试

当前的基础解析器会记录 MLX-LM 输出的输入处理吞吐量（prefill）、生成吞吐量
（decode）、峰值内存和端到端运行时间。基础 Smoke 测试不会通过总时间推算首个
token 延迟（TTFT），因为这种推算并不准确。真正的 TTFT 由长上下文测试中的
Python 流式生成循环单独测量。

## 长上下文实验

`make context` 会在同一进程中加载一次模型，然后分别测试 512、2048 和 8192
token 的输入。每个长度默认运行三次，每次最多生成 128 个 token。

该实验会记录：

- 模型加载时间
- 首个 token 延迟（TTFT，Time To First Token）
- 输入处理速度（prefill tokens/s）
- 逐 token 生成速度（decode tokens/s）
- 峰值内存
- 实际输入和输出 token 数

将模型加载与不同上下文长度的测试放在同一进程中，可以避免重复加载模型对结果
造成干扰，更适合观察上下文长度对推理性能和内存占用的影响。

## 手写 MLX Core 推理

`make manual-context` 使用与长上下文实验相同的模型、输入长度、运行次数和输出
长度，但不调用 MLX-LM 的 `stream_generate()` 或 `generate_step()`。该版本显式：

- 检查 MLX Metal backend 是否可用
- 将默认计算设备设置为 Apple GPU
- 创建并使用独立的 GPU stream
- 使用 `mlx.core` array 管理每层 KV Cache
- 分块执行 prefill，并通过 `mx.eval()` 物化缓存
- 使用 `mx.argmax()` 完成确定性采样
- 逐 token 执行 decode，并明确设置 CPU/GPU 同步点
- 记录 MLX 峰值内存、活跃内存和 KV Cache 大小

MLX-LM 在这一版本中只负责加载模型定义、量化权重和 tokenizer。实验输出写入
`results/manual_mlx_context_benchmark.json`，可以与
`results/context_benchmark.json` 对照。

## Q-CoMem：不同深度使用不同位宽

`make comem-smoke` 会把文档分块独立执行到多个候选 split depth，捕获每个深度的
中间 residual，然后分别以 2/4/8/16 bit 保存。低于 16 bit 的数据由
`mlx.core.quantize()` 真正打包为整数张量，并保存对应的 affine scale 和 bias；
不是把低精度数值继续放在 FP16 数组中模拟压缩。

对每个 `(depth, bits)`，实验会测量：

- residual RMSE 和相对 RMSE
- 相对于同深度 BF16 suffix 的 logit KL divergence
- 首选 token 是否一致
- packed residual、scale 和 bias 的实际字节数
- Write、quantize、dequantize 和 suffix Read 时间
- 相对于完整 dense prompt 的 chunk-local interface gap

校准器会在 KL、relative RMSE 和 top-1 一致性约束下，为每个深度选择占用最小的
位宽 `b*(j)`。默认结果写到 `results/q_comem_depth_benchmark.json`，选中的
residual store 写到 `results/q_comem_store/`。默认命令只使用一个固定 prompt，
因此输出的是用于检查实现的 smoke policy；正式部署前必须在完整任务校准集上聚合
这些指标，不能直接把 smoke policy 当成论文结论。

例如，可以显式测试层 4、7、9，并用手工策略覆盖自动选择：

```bash
.venv/bin/python -m macllm_bench.comem_bench \
  --depths 4 7 9 \
  --bits 2 4 8 16 \
  --depth-bits 4:4 7:4 9:8
```

本节的本地 MLX 实现是第一阶段 training-free 原型：它已经覆盖 split
Write/Read、真实 bit packing、持久化、depth-aware policy 和 CPU BM25 对照。更完整的
suffix LoRA 与生成阶段跨层 cache 实现在后文的 H20/PyTorch 实验线中。完整研究设计见
[Q-CoMem-Edge proposal](PROPOSAL_QCOMEM_EDGE_ZH.md)。

## Q-CoMem 多文档实验

`make comem-multidoc` 使用
[`configs/comem_multidoc_demo.json`](configs/comem_multidoc_demo.json) 中的 6 篇文档和
6 个查询。每个候选 split depth 的执行顺序是：

1. 每篇文档独立通过 lower layers，生成一次 BF16 residual。
2. 每篇 residual 分别打包成 2/4/8/16 bit；同一份 store 被所有查询复用。
3. 主实验按数据集冻结的文档 ID 选择证据，避免把检索误差混入量化误差。
4. query 独立通过 lower layers，随后与选中文档 residual 拼接并执行 suffix Read。
5. 对参考答案的全部 token 做 teacher forcing，比较量化结果与同深度 BF16 Read。

结果写入 `results/q_comem_multidoc_benchmark.json`，包含：

- corpus 总字节数、每个 query 平均读取字节数和实际压缩率；
- 文档 Write、query Write、dequantize 和 suffix Read 时间；
- 答案 token 的 mean/max KL、top-1 agreement、NLL 变化；
- BF16 CoMem 与完整 dense prompt 的 interface gap；
- selector evidence recall；
- 运行前后的电源、温度、内存、swap、负载及有效性判定。

自动策略不再由单个 prompt 决定，而是在所有 query 上聚合后，为每个深度选择满足
约束的最小位宽。也可以把 selector 切换成确定性的 CPU BM25，单独测检索敏感性：

```bash
.venv/bin/python -m macllm_bench.comem_multidoc_bench \
  --selection bm25 --top-k 2
```

内置数据集只用于打通实验协议和发现实现问题，并不是论文评测集。正式结论需要换成
规模更大的冻结 corpus/query split，并进行多个全新进程的重复实验。详细前提、失效
条件和报告模板见 [实验协议](EXPERIMENT_PROTOCOL_ZH.md)；接电后采集数据时直接按
[多文档实验执行手册](TEST_RUNBOOK_ZH.md) 操作。

2026-08-11 的第一轮 AC 正式结果、5 次重复统计、BM25 对照与已发现的 interface gap
记录在 [多文档实验结果报告](RESULTS_MULTIDOC_2026-08-11_ZH.md)。

## GPU 下游任务验证

GPU 版本使用 PyTorch 手工拆分 Qwen3.5-35B-A3B 的 lower/suffix layers，在 H20 上执行
真实 Q16/Q8/Q4 residual 打包、自由生成和 LongBench F1。8-sample pilot 的代码、配置、
autograd 内存问题修复及当前 64-sample validation 状态见
[GPU 下游实验报告](RESULTS_GPU_DOWNSTREAM_2026-08-11_ZH.md)。`gpu/` 保存执行与聚合代码，
`qs/` 保存可审计的 QS 任务配置；这些任务面向远端 CUDA GPU，不是 Mac 本地 `make`
入口。

当前版本还实现了 Qwen3.5 hybrid lower replay：除 split residual 外，同时复用普通注意力
KV 和 GatedDeltaNet convolution/recurrent state。目标 35B 模型在 depth 7/10/13 的
8-sample 诊断中均与 dense 逐 token 完全一致，说明早期接口损失可以 training-free 消除。
真实 packed-state 实验进一步发现，full-attention cache 可以降到 Q4，而 linear recurrent
state 需要保留 Q8；早期 64 条 validation 上，depth-7 配置将完整持久状态缩小
14.10x，mean F1 相对 dense 为 +0.0012，并通过预注册 overall/per-dataset mean margins。
当时冻结的另 64 条现已是 consumed legacy test：压缩 14.12x、mean F1 delta -0.0087，也通过相同 mean
margins；paired 95% CI 为 [-0.0237, +0.0012]，因此报告为“观察到近无损”，不宣称严格
统计无损，也不再把它当作新 policy/adapter 的 untouched test。真正 blind 的 test-v2
68–99 仍未读取。
实现现已支持 residual 与每个 lower cache layer 独立选择 Q2/Q4/Q8/Q16，并使用实测
字节数和校准误差在给定内存预算下自动求解 mixed-bit 策略；不再局限于“所有 attention
统一 Q4、所有 linear 统一 Q8”。`gpu/run_cached_smoke.py` 先验证 cached dense、exact
prefix、cached replay、fixed-order multi-document 和 per-layer Q16 与重计算 oracle 的
逐 token 一致性；通过后 `gpu/launch_layer_sensitivity_8gpu.sh` 才启动 8 卡逐组件校准。
完整存储必须按 `residual + lower state` 计算；研究定位、HYPIC/CacheBlend/KVTuner 等
直接基线和 Apple 端侧 Pareto 实验见
[Q-CoMem-Edge proposal](PROPOSAL_QCOMEM_EDGE_ZH.md)。
修正 suffix chunk boundary 后的 60 条 H20 联合验证见
[mixed-bit validation 结果](RESULTS_GPU_MIXED_VALIDATION_2026-08-12_ZH.md)：当前可信 knee
是 frozen static 的 14.10× 观察性近无损点；4-prompt 校准得到的 mixed policy 尚未胜过它。

### 2026-08-13–14：新版训练与 infra 结论

新版训练固定从 post-trained `Qwen3.5-35B-A3B` 开始，而不是从 Base 重新做对齐。
1024 条数据同时覆盖 4K grounded long instruction、一般指令 replay 和 frozen-teacher
preservation。Dense Full SFT control 完成 128 step 后，固定 60 条 validation 上的
dense/frozen-static F1 相对 Base 点估计为 `+0.01352/+0.01384`，但置信区间均跨 0；
frozen-static 仍仅占 `9.66 MiB`/文档，相对 Q16 state 压缩 `3.59×`。详见
[训练报告](RESULTS_GPU_DENSE_LONG_PRESERVATION_SFT_CONTROL_2026-08-13_ZH.md)和
[full-state 评测](RESULTS_GPU_DENSE_LONG_PRESERVATION_SFT_FULL_STATE_2026-08-13_ZH.md)。

部署边界一致的 native functional-cache LoRA 也已首次正式跑通：GatedDeltaNet 的
conv/recurrent cache 写入由原地 `copy_` 改成 tensor rebind，解决了此前 backward 的
version mismatch；128 step、36 个 LoRA 模块的梯度/更新和 991-position semantic gate
全部通过。它把内部 heldout KL 降低约 `23.58%`，但固定下游相对未启用 adapter 的
frozen-static F1 为 `-0.01543`、CI 跨 0，因此当前只证明训练链路成立，不能声称 LoRA
恢复了精度。详见 [native LoRA 报告](RESULTS_GPU_NATIVE_FUNCTIONAL_LORA_2026-08-13_ZH.md)。
为什么 KL 改善却没有转成 F1、现有 step0/64/128 的归因缺口，以及下一轮
answer-supervised/task-balanced 预注册方案，见
[训练结果诊断](RESULTS_POSTTRAIN_DIAGNOSIS_2026-08-13_ZH.md)。

checkpoint 归因实验进一步把 adapter-disabled、Interface warm-start step 0、native
训练 step 64/128 放在同一 full-state frozen-static caller 上配对比较，Overall F1 分别为
`0.54360/0.53392/0.52138/0.52830`。三个启用 LoRA 的 checkpoint 相对 disabled 的
95% CI 均跨 0：偏移在 step 0 已经出现，后续纯 query-token KL 训练也没有稳定恢复生成式
回答。因此下一版改为 answer+EOS supervision、任务均衡和 answer-position teacher
preservation，而不是继续调这条 loss。详见
[LoRA checkpoint 归因报告](RESULTS_GPU_NATIVE_LORA_CHECKPOINT_ATTRIBUTION_2026-08-13_ZH.md)。
这两个 2026-08-13 run 仍保留为旧 36-module、answer-free query-KL 路线的历史负基线，
不被新结果覆盖。

2026-08-14 的 answer-supervised native-cache LoRA B（Job `237290` / Trial `1840023`）
按独立 official-train heldout 冻结 step 128，再在同一 8×H20 job 中完整评测
adapter-disabled 与 step 0/64/128。它覆盖 36 个 full-attention 和 120 个 GDN 关键
projection，共 156 modules / 26,689,536 个 FP32 参数；MLP 未覆盖，不得称为
“全 suffix 所有线性层”。它使 heldout loss 从 `0.740831` 降到 `0.369624`；
step 128 frozen-static F1 为 `0.548575`，相对 adapter-disabled 为 `+0.005559`，
95% CI `[-0.019821,+0.030127]`。已知的 step-0 yes→no 在仅 2 个 reference-yes 定向
样本上消失，但总体 CI 跨 0，只能说失败模式被定向修复、总体点估计转正，
不能声称统计显著提升。

成本必须分账：frozen-static 持久文档状态是 `9.6609 MiB`/文档，相对 Q16
`34.6831 MiB` 小 `3.5901×`；另有每个模型进程共享的 `101.8125 MiB` FP32
adapter。只统计 persistent state + shared adapter 时，break-even 为 `4.0689`，
从第 5 个常驻文档起 Q4+adapter 总增量驻留低于 Q16；这不包含模型权重、
active workspace 或 allocator reserve。step 128/disabled TTFT 中位比是 `1.06451×`，
但只是单次固定顺序诊断，不是 ABBA 或严格性能 claim。初始化也是 36 个旧
full-attention warm start + 120 个 GDN cold start，所以实验检验的是“answer
supervision + 扩 surface 能否修复旧系统”，不是纯 cold-start LoRA。本轮已消费
validation 6–35，不得用它重选 checkpoint；test-v2 68–99 未读取。完整协议、
证据和边界见
[answer-supervised LoRA B 正式报告](RESULTS_GPU_ANSWER_SUPERVISED_NATIVE_LORA_B_2026-08-14_ZH.md)。
训练的 whole-answer multi-token continuation 也不等同于真实 token-by-token decode：
26 条 heldout / 206 positions 虽然 top-1 无分叉，mean KL 仍为 `0.000507959`，
因此不声称 chunk-boundary 数值等价。

真正 paged KV 的 Python online-softmax reference 仍未通过 logits correctness gate。
two-pass BF16 修正版虽在浅层更接近 Transformers eager，但误差在 10 个 full-attention
层间累积，最终 max-abs `1.4375`、relative L2 `0.07662`；正式 benchmark 已被阻断，
没有新的 TTFT/active-memory 正结果。详见
[paged reference 负结果](RESULTS_GPU_QWEN35_PAGED_REFERENCE_NEGATIVE_2026-08-13_ZH.md)和
[下一版 correctness 协议](PAGED_ATTENTION_NEXT_CORRECTNESS_PROTOCOL_ZH.md)。下一实现将转向
融合 Triton/vLLM/FlashInfer kernel，而不是调宽正确性阈值。

vLLM `unified_attention` 版本在 Trial `1840009` 也按协议 fail-closed：8 个 PG-19
train-only rank 中 5 个完成 semantic row，3 个在 full-attention layer 31/31/19 未通过
Transformers eager 与 vLLM Triton kernel 的数值兼容门。这是 **TF-eager compatibility
negative**，既不证明 page-reuse layout 错误，也不证明它正确；LongBench
validation/test-v2 和性能 benchmark 都未运行，因而没有 TTFT、TPOT 或显存收益。
详见 [vLLM paged Q16 负结果](RESULTS_GPU_QWEN35_VLLM_PAGED_Q16_FORMAL_NEGATIVE_2026-08-14_ZH.md)。

同 kernel fair v2 已在 Job `237468` / Trial `1840486` 正式完成：同一个
`unified_attention` callable 下，fresh full-copy 与 shared-document/private-tail reuse
在 8/8 validation workload 的全部生成 token 和逐步 full-vocab logits 上 bitwise exact。
reuse/fresh cached-document TTFT paired median ratio 为 `1.009029`，没有加速；缓存增量
CUDA peak allocated ratio 为 `0.496010`，约低 `50.4%`，并避免中位 `80 MiB` 物理
文档 block copy。combined unique storage 减少 `87,819,520 B`，但包含模型后的 absolute
peak 只差约 `83.64 MiB`，不能写成总模型显存减半。范围严格限于 Q16、batch 1、
单请求、10 个 full-attention 层；没有 F1、多 query、ragged batch、NVML peak 或
isolated kernel speedup。详见
[same-kernel fair v2 正式报告](RESULTS_GPU_QWEN35_VLLM_PAGED_FAIR_V2_2026-08-14_ZH.md)。
此前 Trial `1840344` 仅因未固定 locale 导致 preflight code-ledger 排序漂移，未进入
static/PG-19/validation；c 版已用 `LC_ALL=C` 在真实 Pod 中闭合该发布治理问题。

同一 Q16 kernel 的 multi-fork resident 容量曲线也已在 Job `237580` / Trial `1840837`
完成。8 个 rank 都在同一份 4095-token PG-19 train-only 文档上完整运行
`N={1,2,4,8,16,32}`：fresh 为 N 个请求分别物化完整文档 pool，reuse 为一个只读文档
pool 加 N 份 private tail/append。所有 N 上的 token、逐步 full-vocab logits、最终 K/V/GDN
和 cross-N request isolation 均 exact。可重放分账为 fresh `80+90N MiB`、reuse
`80+5N MiB`；N=32 时节省 `2720 MiB` full-attention pool，PyTorch absolute peak
allocated 中位数相差约 `2.661 GiB`。N 个对象虽然同时驻留，模型步仍在单 CUDA stream
上 round-major 顺序执行；因此这不是 concurrent serving、throughput 或 TTFT 正结果，也
不覆盖 aligned 4096、ragged、多文档、NVML 或 F1。详见
[multi-fork resident 正式报告](RESULTS_GPU_QWEN35_VLLM_PAGED_MULTIFORK_RESIDENT_2026-08-14_ZH.md)。

在此基础上，仓库新增两条不混淆口径的验证线：

- [部署/KV 基准](DEPLOYMENT_BENCHMARK_ZH.md) 同时报告 GPU 驻留的持久文档状态、每请求
  active fork、decode KV、模型权重、CUDA allocator 峰值、NVML 离散采样、TTFT/TPOT 和
  文档容量估计；对照包含 dense recompute、标准 full-prefix cache、Q16/Q8/Q4 和逐层
  mixed-bit。168-row 主表使用 eager deep clone；后续另以 audited COW staging short 测试，
  两种实现都不冒充 paged-KV 的生产上限。
  修正 suffix chunk boundary 后的 8×H20 Q16 hard gate 见
  [部署 exactness 结果](RESULTS_GPU_DEPLOYMENT_EXACTNESS_2026-08-12_ZH.md)；完整 168-row
  显存/容量/TTFT/TPOT 结果见
  [部署基准结果](RESULTS_GPU_DEPLOYMENT_2026-08-12_ZH.md)。后续审计发现旧 Q16 fork 可能与
  persistent source 共用 storage；旧 COW 检查也只是 token-to-dense，不是同一 source 的
  eager-vs-COW direct paired gate。修复后
  [4k COW short](RESULTS_GPU_COW_4K_SHORT_2026-08-12_ZH.md) 已在 QS `1832356` 完成：
  相同 caller boundary 的 incremental full-prefix/Q16 eager/Q16 COW 三方 gate 为 8/8，
  source immutable、eager/COW logits bitwise equal、无 fallback。frozen-static 的 durable
  payload 为 `14.024×` 压缩，但 dense staging-inclusive total resident 只有 `3.056×`；
  CUDA/NVML peak 反而多约 `548/615 MiB`，TTFT 为 `4.122×`。所以它验证了 COW correctness，
  没有验证 active-memory 或 TTFT 优势，也不是真正的 PagedAttention。
- [LoRA 训练线](LORA_TRAINING_RUNBOOK_ZH.md) 将原始 CoMem 的 Interface LoRA 与状态量化
  恢复 LoRA 分开。前者学习 residual-only/chunk-local 接口，后者用 Q16 replay teacher
  蒸馏真实 packed Q4/Q8/mixed state；模型权重不量化，因此后者不是 QLoRA。Qwen3.5
  MoE 默认只训练 suffix attention `q/k/v/o_proj`，并有一亿参数 hard gate。真实 8×H20
  一步训练链路与权重更新核验见
  [LoRA 结果](RESULTS_GPU_LORA_SMOKE_2026-08-12_ZH.md)。随后 Interface LoRA 完成 200-step
  训练和固定 checkpoint 的 60 条 validation：相对未训练 chunk-d7 mean F1 提高 `+0.04384`，
  但 overall 95% CI 跨 0，且相对 dense 仍低 `-0.05197`，所以只报告“部分恢复”。
  Frozen-static quant LoRA 也完成 200 step，但末 20 step loss 比首 20 step 高 `6.72%`，
  没有通过预注册趋势 gate；它是负结果，不能作为量化精度恢复证据。进一步尝试部署形态的
  document-prefill/query-continuation cache 反向图时，8/8 H20 rank 都发生 mutable-cache inplace
  autograd version mismatch：forward 可运行，但 backward 失败、没有 optimizer step。详见
  [cached-two-stage 负结果](RESULTS_GPU_CACHED_AUTOGRAD_2026-08-12_ZH.md)，因此目前不能说
  cached-two-stage 已可训练。
- [全参数 supervised SFT capability smoke](RESULTS_GPU_DENSE_SFT_SMOKE_2026-08-12_ZH.md)
  分两步完成审计：首轮 BF16 Trial `1831074` 只证明 backward/optimizer execution；修正后的
  FP32-shard Trial `1831289` 使用 BF16 forward、FP32 gradient reduce/Adam，已经证明全部
  34,660,610,688 参数有 gradient coverage，40 层均有非零 FP32 delta，且约 7.69 亿个参数值
  对下一次 BF16 forward 可见地改变。因而 8×H20 上的真实一步 full-model update 已通过；
  但它仍是 8 条 train-only 样本的一步 smoke，没有 checkpoint/validation，不能声称质量改善，
  也不是 cached suffix SFT。
- [全参数 supervised SFT 正式训练](RESULTS_GPU_DENSE_SFT_FORMAL_2026-08-12_ZH.md)
  已完成 384 steps / 3 epochs（QS `1831595`）。独立 train-split heldout 的 token-weighted CE
  在 step 128 从 `2.2415` 降至 `1.6664`（`-25.66%`），随后 step 256/384 回升，故冻结最佳
  FP32 model-only DCP 为 step 128。
- [SFT × 完整 lower-state Q-CoMem 统一下游评测](RESULTS_GPU_SFT_FULL_STATE_DOWNSTREAM_2026-08-12_ZH.md)
  已在同一 8×H20 作业内完成 base/SFT 的 dense、完整 Q16、完整 Q8 和 frozen mixed-bit
  配对评测（QS `1832184`）。dense SFT 相对 base dense 的 F1 点估计为 `-0.02308`，95% CI
  跨 0，因此没有观察到下游恢复。SFT Q8 相对 SFT dense 仅 `-0.00697` 并通过 mean margin；
  frozen 相对 SFT Q16 仅 `-0.00139`、state 再缩小 `3.59×`，说明量化增量近似无损，
  主要损失来自 dense→split/replay interface。Q4/Q8 仍只指 persistent document state，
  模型权重保持 BF16。

下一阶段的 LoRA、COW/paged state 和 HYPIC-inspired TTFT 比较在查看新结果前已固定
数据边界和验收条件，见 [下一阶段实验预注册](NEXT_STAGE_EXPERIMENTS_ZH.md)。其中
HYPIC-inspired transition/seam 原型、depth-7/4k 的分项字节账及不能声称的能力边界见
[HYPIC-lite 说明](gpu/HYPIC_LITE_ZH.md)；它是参考对照，不是完整 HYPIC 复现。

mixed-bit validation 固定使用 LongBench source index 6--35，部署 validation 使用 6--9；
校准样本 4--5 明确排除，冻结的 test-v2（68--99）在策略和 adapter 冻结前不会读取。
LoRA 功能 smoke 使用 Google DeepMind 官方 PG-19 bucket 的 train-only 64-book 子集，逐对象
校验 GCS MD5；该子集只回答真实 forward/backward 能否工作，不替代完整 PG-19 正式训练。

## Apple MLX Hybrid Replay

当前 Apple 路径不再只是量化 residual。`macllm_bench.mlx_replay` 在 `mlx.core` 上直接维护
Qwen3.5 lower layers 的两类状态：full-attention `KVCache`，以及 GatedDeltaNet
`ArraysCache` 中的 convolution/recurrent state。冻结配置与 H20 独立 test 一致：

```text
depth = 7
residual = Q4
full-attention cache = Q4
linear recurrent state = Q8
group size = 64
```

实现包含真实 bit packing、query-local cache fork、exact Q16 replay、greedy generation、
完整 persistent bytes、state-type RMSE、逐层 mixed-bit 自动预算以及 safetensors 保存/恢复。
多文档 exact replay 当前支持冻结顺序的连续写入；任意重排/删除仍需要 segment transition
composition，不能把 fixed-order bundle 表述成 HYPIC 式任意组合。默认正式入口使用固定
revision 的 `mlx-community/Qwen3.5-9B-4bit`：

```bash
make mlx-replay
```

命令默认要求 AC，检查不通过会在模型加载前退出；未接电时不要用 diagnostic 数值填论文
表格。接电后的完整步骤见 [实验执行手册](TEST_RUNBOOK_ZH.md)。

## 实验规范

1. 发布实验结果前，固定模型仓库和具体版本。
2. 记录真实的输入和输出 token 数，不使用字符数代替。
3. 区分冷启动、预热运行和正式测量。
4. 所有对比实验使用固定的采样参数。
5. 报告中位数和数据波动范围，不能只报告最快的一次。
6. 记录电源模式、后台负载、系统 swap 和温度状态。
7. 不要把实现方式不同的 4-bit 格式视为完全等价的量化方案。

## 学习重点

阅读和运行本项目时，建议先理解以下概念：

- **统一内存**：Apple Silicon 的 CPU 和 GPU 共享同一个内存池。
- **Prefill**：模型一次性处理输入 token，并构建注意力 KV Cache 的阶段。
- **Decode**：模型读取权重和已有 KV Cache，逐个生成新 token 的阶段。
- **TTFT**：提交输入后，到生成第一个 token 所经过的时间。
- **KV Cache**：保存历史 token 的 Key 和 Value；上下文增长时，它会持续占用内存。
- **量化**：使用较低位宽保存权重，以减少模型体积和内存访问量；位宽更低并不保证速度按比例提高。

## 后续计划

- 进一步拆分 tokenizer 时间和真实 TTFT。
- 完善 512、2048、8192 token 的固定输入样本。
- 加入 7B/8B 模型和可控的内存压力测试。
- 加入 llama.cpp 构建、GGUF 元数据采集和 `llama-bench` 结果解析。
- 加入进程级内存与 swap 采样，以及持续负载下的温度测试。
- 在 runtime 基准测试足够可信后，再加入 Core ML。
