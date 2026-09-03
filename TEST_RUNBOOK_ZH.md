# Q-CoMem 多文档实验执行手册

这份手册用于在 M4 Pro MacBook Pro 上采集可进入分析表格的数据。算法设计和指标定义见
[proposal](PROPOSAL_QCOMEM_EDGE_ZH.md)，机器状态的判定标准见
[实验前提协议](EXPERIMENT_PROTOCOL_ZH.md)。

当前内置实验是 **6 篇文档、6 个 query 的协议验证集**。它可以验证多文档复用、真实
bit packing、逐深度校准和环境记录是否工作，但规模不足以支持论文结论。

## 1. 明天运行前的检查单

在开始前完成以下事项：

- [ ] 使用同一个电源适配器接入 AC，整个实验过程不要拔电。
- [ ] 在系统设置中关闭低功耗模式；固定同一种电源模式。
- [ ] 如果刚运行过重负载或发生过 swap，正式采集前重启，并在 `22±2°C` 静置至少 10 分钟。
- [ ] 关闭浏览器重负载、Spotlight 大规模索引、Time Machine、云盘同步、视频会议和系统更新。
- [ ] 不在运行中切换显示器、屏幕亮度或外接设备。
- [ ] 模型已经下载完成，避免把网络下载时间混入实验。
- [ ] 最好重启后再采集论文性能数据，使初始 swap 尽量接近 0。

进入项目并人工查看状态：

```bash
cd /Users/liuhanzuo/MacLLM-Bench
pmset -g batt
pmset -g custom
pmset -g therm
sysctl vm.swapusage
```

期望看到：

- `Now drawing from 'AC Power'`；
- 当前 AC profile 的 `powermode` 不是 `0`；
- `pmset -g therm` 没有 thermal/performance warning；
- swap 没有在静置时持续增长。

程序还会再次自动检查，所以人工检查只是为了避免等到命令退出后才发现条件不合格。
正式开始前应相隔 60 秒重复读取一次；两次 thermal state 都必须为 nominal。完整论文实验
分为“冷态/交互式主表”和“20 轮持续负载曲线”，两者不得混用。具体判定与文献依据见
[实验前提协议](EXPERIMENT_PROTOCOL_ZH.md#21-冷态交互式主表)。

## 2. 先做快速代码检查

下面只运行单元测试，不加载 3B 模型：

```bash
make test
```

预期所有测试均为 `OK`。若单元测试失败，不要继续采集正式结果。

## 3. 先下载冻结的 Qwen3.5 模型（不计入性能实验）

Apple hybrid replay 默认使用 9B 模型验证算法和系统路径。模型下载不应混入正式计时；可以
在接电前完成：

```bash
cd /Users/liuhanzuo/MacLLM-Bench
.venv/bin/hf download mlx-community/Qwen3.5-9B-4bit \
  --revision 8b2b98c00a6b4d291155e4890773ca8f769aee53
```

这一步只下载文件，不产生实验结论。确认下载完成后，再按第 4 节接电运行。

## 4. Apple 完整诊断矩阵：冻结的 Hybrid Q4/Q4/Q8 Replay

接入 AC、通过第 1 节检查后运行。该命令用于一次性验证三个长度、存储、正确性与内存
边界；只有全程 nominal 且无新增 swap 时，延迟才可进入主表：

```bash
make mlx-replay
```

默认设置为：

- 模型：`mlx-community/Qwen3.5-9B-4bit`；
- immutable revision：`8b2b98c00a6b4d291155e4890773ca8f769aee53`；
- split depth：7；residual Q4；full-attention cache Q4；linear state Q8；
- context：512、2048、4096 tokens；每个长度 3 次；greedy 生成 8 tokens；
- 四条路径：dense、exact prefix、exact Q16 replay、packed Q4/Q4/Q8 replay；
- 配置计时顺序按 run 循环轮换，每个长度先做各路径 1-token warm-up；
- 输出：`results/qcomem_mlx_hybrid_replay.json`；
- store：`results/qcomem_mlx_hybrid_store/tokens-*.safetensors`。

运行结束后先检查：

```bash
.venv/bin/python - <<'PY'
import json
p = json.load(open("results/qcomem_mlx_hybrid_replay.json"))
print(json.dumps({
    "status": p["status"],
    "formal_result_eligible": p["formal_result_eligible"],
    "reasons": p["environment_assessment"]["reasons"],
    "config": p["frozen_replay_config"],
    "summary": p["summary"],
}, indent=2, ensure_ascii=False))
PY
```

只保留 `status=completed`、`formal_result_eligible=true`、`reasons=[]` 的 run。重点核对：

- `persistent_compression_vs_prefix` 是否随长度保持约一个数量级；
- `replay_q16.all_tokens_match_dense`，它是 exact replay 正确性门槛；
- `replay_q4_q4_q8.all_tokens_match_dense`，它是生成敏感度而非唯一质量指标；
- residual/attention/linear 三类 `relative_rmse`；
- swap 是否增长、运行结束是否出现 thermal warning。

若完整矩阵因后半程升温或 swap 被判 invalid，保留 JSON 作为审计/功能结果，不要连续
重跑，也不要提高 swap 阈值。性能主表应改为每个 context length 的独立冷态 session，
在各 session 间恢复 nominal，并跨重复轮换四条路径顺序；持续负载结果另跑 20 轮并报告
逐轮热状态曲线。

单个冷态 session 的命令模板如下；每次只跑一个长度和一轮。将 `SESSION` 依次设为
`0..4`，每轮之间人工等待 thermal state 恢复 nominal，输出文件不得互相覆盖：

```bash
CONTEXT=512
SESSION=0
.venv/bin/python -m macllm_bench.mlx_replay_bench \
  --context-lengths "$CONTEXT" --runs 1 --order-offset "$SESSION" \
  --max-new-tokens 8 --no-save-store \
  --output "results/mlx-cold-c${CONTEXT}-s${SESSION}.json"
```

分别对 512、2048、4096 执行 5 个有效 session。`--order-offset` 会让 dense、exact
prefix、Q16 replay、Q4/Q4/Q8 replay 在不同 session 中处于不同计时位置，降低升温和
缓存顺序对某一方法的系统性偏置。

35B-A3B 只作为第二阶段容量压力实验。先确认 9B run 有效、统一内存和 swap 有充分余量，
再显式使用：

```bash
.venv/bin/python -m macllm_bench.mlx_replay_bench \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --model-revision 1e20fd8d42056f870933bf98ca6211024744f7ec \
  --context-lengths 512 2048 4096 \
  --runs 3 --max-new-tokens 8 \
  --output results/qcomem_mlx_hybrid_replay_35b.json
```

如果 preflight 可用内存不足、发生明显 swap 或 memory pressure，不要强行把 35B 结果解释为
算法失败；先缩短 context 或只报告“该 model-plus-context working set 在本机不可行”。

## 5. 旧版 residual-only 多文档主实验

先运行一次完整矩阵并持久化自动选中的 residual store：

```bash
make comem-multidoc
```

默认设置为：

- 模型：`mlx-community/Llama-3.2-3B-Instruct-4bit`；
- 模型 revision：`7f0dc925e0d0afb0322d96f9255cfddf2ba5636e`；
- split depth：`5, 7, 9`（28 层模型的约 `1/6, 1/4, 1/3`）；
- residual bits：`2, 4, 8, 16`；
- group size：64；chunk size：64；left overlap：16；
- selector：数据集中冻结的两个文档；
- 输出：`results/q_comem_multidoc_benchmark.json`；
- store：`results/q_comem_multidoc_store/`。

若电源或系统条件不合格，命令会在加载模型前退出，JSON 中写入
`status=preflight_failed`。不要用 `record-only` 绕过后再把结果当成正式数据。

## 6. 立即核对旧版多文档运行是否有效

运行结束后执行：

```bash
.venv/bin/python -c 'import json; p=json.load(open("results/q_comem_multidoc_benchmark.json")); print(json.dumps({"status":p["status"], "formal_result_eligible":p["formal_result_eligible"], "reasons":p["environment_assessment"]["reasons"], "policy":p.get("selected_policy"), "power_before":p["environment_before"]["power"]["source"], "power_after":p.get("environment_after",{}).get("power",{}).get("source"), "swap_growth_bytes":p["environment_assessment"].get("observed_swap_growth_bytes")}, indent=2))'
```

这一轮只有同时满足以下三项才保留：

```text
status == completed
formal_result_eligible == true
environment_assessment.reasons == []
```

任何一项不满足，都把该 run 标成 invalid，并根据 `reasons` 处理后重新运行。不要删除
无效 JSON；把它移动到单独的 `results/invalid/` 目录，保留审计记录。

## 7. 旧版多文档正式重复运行

单次计时不能用于论文表格。第一轮有效后，静置 1 分钟，再使用 5 个全新 Python
进程重复。下面的命令给每个 run 单独保存 JSON，且不重复写 store：

```bash
cd /Users/liuhanzuo/MacLLM-Bench
run_tag=$(date +%Y%m%d-%H%M%S)
run_dir="results/formal-${run_tag}"
mkdir -p "$run_dir"

for run_index in 1 2 3 4 5; do
  .venv/bin/python -m macllm_bench.comem_multidoc_bench \
    --power-policy require-ac \
    --selection frozen \
    --no-save-store \
    --output "$run_dir/frozen-${run_index}.json"
done
```

注意：这里的 5 次是独立进程，但仍应逐个检查每个 JSON 的有效性。若其中一次热状态、
后台 CPU 或 swap 增长超限，只重跑该编号，不要用最快的一次替代它。

最终报告 median、最小/最大和 p5/p95，不要只报告最快值。内置 benchmark 已把算法的
各阶段拆开，但目前每个进程内部的阶段计时只有一次；论文性能版后续还应增加随机化配置
顺序和更多独立进程。

## 8. 第二轮：BM25 selector sensitivity

冻结选择实验有效后，再单独运行 BM25。它回答“检索选择改变后结果是否稳定”，不能与
冻结选择的量化主实验混在同一列：

```bash
.venv/bin/python -m macllm_bench.comem_multidoc_bench \
  --power-policy require-ac \
  --selection bm25 \
  --top-k 2 \
  --no-save-store \
  --output results/q_comem_multidoc_bm25.json
```

重点核对 `selection.mean_evidence_recall` 和 `selection.selected_by_query`。当前 BM25 是
确定性的 CPU baseline，尚未把 selector latency 算进在线 Read 时间，因此这一轮主要
用于质量和证据敏感性，不可声称是端到端 selector 加速结果。

## 9. 应从 JSON 提取什么

每次运行至少保存和汇总以下字段：

| 目的 | JSON 字段 |
|---|---|
| run 是否有效 | `formal_result_eligible`, `environment_assessment` |
| 电源与温度 | `environment_before/after.power`, `thermal_state` |
| 内存压力 | `swap_bytes`, `observed_swap_growth_bytes`, `mlx_peak_memory_bytes` |
| 自动位宽 | `selected_policy`, `policy_sensitivity` |
| corpus 容量 | `aggregate_results[].corpus_stored_nbytes` |
| 压缩率 | `aggregate_results[].corpus_compression_ratio` |
| 在线读取量 | `aggregate_results[].mean_selected_stored_nbytes` |
| 量化质量 | `mean_kl`, `max_position_kl`, `mean_top1_agreement_rate` |
| 答案分布 | `mean_answer_nll_delta`, query 级 teacher-forced 指标 |
| split 误差 | `depth_results[].mean_interface_kl_vs_dense` |
| 时间 | Write、query Write、dequantize、suffix Read 分段字段 |

`selected_policy` 使用预注册的严格阈值。`policy_sensitivity` 只展示 top-1 阈值变化时
位宽是否改变，不能在看到任务答案后任意挑一个最好看的阈值作为主结论。

## 10. 看到异常时怎么处理

- `not_connected_to_ac_power`：接电、确认系统识别 AC 后重跑。
- `low_power_mode_enabled`：关闭低功耗模式，重新检查活动 AC profile。
- `background_cpu_above_limit`：关闭占用进程，静置后开启新 run。
- `thermal_state_*` 或 `pmset_warning`：停止，等待机器降温，不要连续重跑。
- `swap_growth_above_limit`：关闭内存重负载；正式采集建议重启。
- `record_only_diagnostic_mode`：这只是一轮排错数据，不能改标签。
- Q4/Q2 未被选中：先看 residual RMSE、max-position KL 和每 query agreement；不要为了
  得到更低 bit 而事后放宽阈值。

## 11. 当前实验不能证明什么

即使 6×6 demo 全部成功，也还不能证明：

- 大 corpus 的 SSD/RAM 分层收益；
- LongBench、LoCoMo、RULER 等自然任务质量保持；
- 自由生成答案和带 KV cache 的 decode 等价；
- 能耗降低；
- Q4 在所有模型和 split depth 都可用。

demo 的目标是冻结一套不会混淆 selector、split interface 和量化误差的执行方式。下一步
再把同一 JSON schema 扩成真实多文档数据集，并加入 RAM/SSD、重复 query 数和端到端
生成实验。

## 12. H20：cached exactness gate 与逐层 mixed-bit 校准

这一轮不是 Mac 性能测试，不要求笔记本保持接电；它使用 QS 的 8×H20 package。先把
下列文件同步到远端 `qcomem_gpu/`，不要只替换 runner 而遗漏底层接口：

```text
gpu/qcomem_torch.py
gpu/run_cached_smoke.py
gpu/run_layer_sensitivity.py
gpu/aggregate_layer_sensitivity.py
gpu/launch_layer_sensitivity_8gpu.sh
```

远端先做静态检查：

```bash
ENV_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/envs/vllm-cu129-v1
CODE_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_gpu
"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/qcomem_torch.py" \
  "$CODE_DIR/run_cached_smoke.py" \
  "$CODE_DIR/run_layer_sensitivity.py" \
  "$CODE_DIR/aggregate_layer_sensitivity.py"
bash -n "$CODE_DIR/launch_layer_sensitivity_8gpu.sh"
```

提交 [QS 配置](qs/qcomem-layer-sensitivity.yaml)。launch 脚本先只在 GPU0 上运行 512-token、
3-token generation smoke，以下五项必须全部与重计算 oracle 一致，否则作业立即退出，不
启动校准：

```text
cached dense
exact full prefix
cached hybrid replay
fixed-order three-document replay
per-layer Q16 replay
```

gate 通过后，8 张卡分别负责 residual 与 lower cache layer 0–6，在 validation 子集测
Q2/Q4/Q8/Q16 的实际 component bytes、first-token KL、relative logit MSE 和 top-1 match。
聚合输出位于：

```text
.../runs/qcomem/layer-sensitivity-20260811a/cached_smoke.json
.../runs/qcomem/layer-sensitivity-20260811a/layer_policy.json
```

`layer_policy.json` 的策略只是在 calibration 上选出的候选，不能直接写成下游无损。选择
同内存策略或 `minus_25_percent` 后，应冻结完整 `residual_bits + cache_layer_bits`，再在
untouched test 上只运行一次自由生成/F1。`extreme_q2_floor` 是 rate-distortion 端点，不
默认进入主表。
