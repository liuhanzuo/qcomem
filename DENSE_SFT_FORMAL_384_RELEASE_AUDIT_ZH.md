# 384-step dense full-SFT 正式发布协议（2026-08-12）

## 结论与用途

本文冻结第一次正式 dense full-model SFT 的实验选择，并作为 QS 提交前的 fail-closed
清单。它不是 smoke 协议。只有训练器、launcher、配置和代码账本全部落定且下列检查全部通过，
才允许创建唯一一个 QS Trial。

本轮只训练 dense 模型。它用于回答“全量 SFT 能否改善同一 train-only 诊断集上的监督 CE”，
不能单独证明 Q4/Q8 CoMem 精度恢复；后者还需要对冻结后的同一 checkpoint 做 dense 与各量化
CoMem 推理矩阵。CE heldout 也不是最终下游测试集。

## 1. 冻结的正式训练配置

| 项目 | 冻结值 |
|---|---|
| 模型 | `Qwen3.5-35B-A3B-59d61f3`，text 参数 `34,660,610,688` |
| 资源 | 单节点 `8 × H20-141G`，FSDP1 `FULL_SHARD` |
| 持久参数 / 前向 / 梯度归约 | FP32 shard / BF16 / FP32 |
| optimizer | AdamW，FP32 moments，`foreach=false` |
| 总 step | `384`（训练集 3 个等效 epoch） |
| 每 step 全局样本 | `8`，每卡 `1`，Qasper `4` + 2Wiki `4` |
| gradient accumulation | `1` |
| 训练样本暴露量 | 每数据集各 `1,536` 次，共 `3,072`；恰好 `3` 个等效 epoch |
| learning rate | peak `1e-6` |
| schedule | linear warmup `20` steps，随后 cosine decay 到 0 |
| weight decay | `0.0` |
| global grad clip | `1.0` |
| seed | `31` |
| 最大 sequence | `1,024` tokens |
| loss | answer + EOS only，按全局 target-token 加权 |
| heldout CE | baseline step `0`；正式候选 step `128 / 256 / 384` |
| early stop | 禁止；本轮一定执行到 step 384，除非出现运行错误 |
| checkpoint | step `128 / 256 / 384` 各保存 model-only DCP；不保存 optimizer |
| checkpoint 选择 | 三个候选中 overall token-weighted heldout CE 最低者；并列取更早 step |

不得在看到 step 0 或中途 CE 后修改上述任何超参数。若因错误重提，必须使用新的 run ID，
并在结果报告中同时保留失败 Trial。

## 2. 冻结数据与泄漏边界

远端目录：

```text
/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/
qcomem-supervised-sft-scale-20260812a/
```

| artifact | 冻结 SHA256 | 计数 |
|---|---|---:|
| `supervised-train-512plus512.jsonl` | `b6b1a88226b3060b6ba6b600793d90470820511ae38096b4db99af8b65f05257` | 1,024 |
| `supervised-heldout-ce-64plus64.jsonl` | `069c6649e73a0bdbe7b300a1a32f6b89fa5ad23d43fcfa03c85f101d5c7ac10e` | 128 |
| `supervised-scale-512-64.manifest.json` | `e527eeac4f110005057bcc3936093c6b6ce60252591cd57373785c4995f2ff15` | — |

正式 manifest 已记录并通过：

- train 为 Qasper 512 + 2Wiki 512，CE heldout 为 64 + 64；
- 两者 source ID、四种内容指纹和全局指纹 component 的交集全部为 0；
- 所有行的 top-level 与 provenance `source_split` 都是 `train`；
- `validation_or_test_rows_used=false`、`raw_test_v2_read=false`；
- test-v2 状态固定为 `deferred_not_read`。

训练与 CE evaluator 的参数接口不得接受 LongBench test-v2 路径。正式训练完成、checkpoint 和
Q4/Q8 策略全部冻结之前，继续禁止读取 test-v2 source index 68--99。训练期间使用的 128 条
heldout 只能称为 train-split CE diagnostics。

## 3. 资源、显存与存储预算

采用已完成的 corrected one-step FP32-shard 运行作为容量证据：每卡峰值 allocated 约
`83.14 GiB`，峰值 reserved 约 `109.27--123.24 GiB`，在 H20-141G 上完成了 forward、
backward、clip 和 AdamW update。正式运行维持相同 sequence cap、每卡 micro-batch 与
gradient accumulation，因此预期单步内存阶数不变；但仍必须记录每卡峰值，并将任何 OOM 或
reserved 持续增长视为正式失败，不能把部分 step 当作正式结果。

理论持久状态预算：

- 全局 FP32 参数约 `129.12 GiB`，每卡 shard 约 `16.14 GiB`；
- 全局两份 FP32 Adam moments 约 `258.24 GiB`，每卡约 `32.28 GiB`；
- 每个 model-only DCP 的有效 tensor payload 约 `129.12 GiB`；三个候选约 `387.36 GiB`。
  考虑 DCP metadata、临时文件和文件系统余量，提交前为 run dir 预留至少 `480 GiB`；
- 本轮禁止 optimizer checkpoint。单个包含参数和两份 Adam moments 的完整恢复点会接近
  `387.36 GiB`，三个会超过 `1.1 TiB`，不符合本轮资源边界。

2026-08-12 提交准备时，JuiceFS 显示约 `18 PiB` 可用，因此容量足够；这一状态必须在真正
create 前重新查询。现有 runs/qcomem 占用约 `719 MiB`，不包含即将产生的 final DCP。

## 4. QS 资源选择与唯一性

资源套餐固定为：

```text
cloudId=6
clusterId=53
resourcePackageId=183     # 8Gpu/170C/1800Gi，H20-141G
workerNum=1
restartNum=0
```

队列不属于实验变量。create 前用 `qs -o json resources options` 查询上述套餐的实时 remaining，
从已有非 borrowed 且至少剩余 8 GPU 的授权队列中选择余量最大的一个。2026-08-12 审计快照中
`Reasoning_Rollout (queueId=436)` 尚余 112 GPU，优先使用它；若提交时状态变化，可以只改
queueId，必须把资源快照和最终 queueId 写入 run metadata。

冻结的 run 名称与目录：

```text
liuhanzuo-qcomem-dense-supervised-sft-formal-384-20260812a

/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/
dense-supervised-sft-formal-384-20260812a
```

launcher 必须拒绝非空 `RUN_DIR`。create 前还要从 QS 搜索同名 Job/Trial；目录存在或任务名
已存在时禁止复用，改成新的不可变 run ID。

## 5. 提交前 fail-closed 清单

以下任一项失败都不得 create：

1. `bash -n`、`py_compile` 与相关单测未全部通过；
2. trainer 没有硬校验 8 ranks、34,660,610,688 text 参数、40 layers、FP32 persistent shards、
   BF16 forward 与 FP32 gradients/moments；
3. 正式配置不是 384 steps、warmup 20、cosine、lr 1e-6、clip 1、wd 0；
4. 全局每 step 不是冻结的 Qasper 4 + 2Wiki 4；
5. heldout 不是严格 `torch.no_grad()`，或任何 heldout tensor 进入 backward；
6. baseline heldout 不是 step 0，或候选 heldout 时点不是恰好 128/256/384；
7. step 128/256/384 的 model-only DCP 没有在所有 rank 完成后原子发布 `_SUCCESS`、包含
   文件清单、directory SHA ledger、逻辑 payload 与实际占用字节，或 best 选择没有按最低
   overall token-weighted CE（并列取更早 step）；
8. launcher 会读取 test-v2、validation、dev，或者数据 SHA/manifest 审计不一致；
9. 代码、配置、模型 artifacts、14 个 weight shards、数据和运行依赖没有不可变 SHA ledger；
10. run dir 非空、QS 同名任务已存在、共享文件系统余量不足 480 GiB；
11. 实时资源不是单节点 8 × H20-141G，或选择了 borrowed/余量不足的队列；
12. QS `--dry-run` 不能正确解析最终 YAML，或 dry-run 展示的 command/资源与本协议不一致。

## 6. 正式结果的最低验收条件

训练完成不能只看 Trial 状态。至少需要同时满足：

- step 1 的 full-gradient、FP32 Adam 和 nonzero FP32 parameter-delta 门禁通过；其后 383 个
  step 都必须存在有限 train CE、grad norm 与冻结 LR schedule 记录；
- `train-metrics.jsonl` 恰有 384 个连续、互不重复的 step，且全部 loss/grad/LR 有限；
- step 0/128/256/384 的四次 heldout evaluation 都存在，每次 128 条、Qasper/2Wiki 各
  64 条，无 padding duplication；
- step 128/256/384 的 model-only DCP 完整，所有 rank 写入结束，`_SUCCESS`、目录 ledger、
  逻辑 payload 和实际占用字节复验通过；
- `best` 指针或 manifest 只指向三个完整候选中 overall token-weighted CE 最低者；
- metadata 明确记录 `raw_test_v2_read=false`、`final_downstream_evaluation=false`；
- 保存 QS trial/job ID、GPU UUID、driver/CUDA/PyTorch/NCCL、wall time、峰值显存、数据与代码
  SHA；
- 报告 step 0→384 的 paired CE/NLL 变化及分数据集结果，但不把它表述为最终下游能力。

只有以上门禁都通过，这个 checkpoint 才进入下一阶段：同一冻结模型分别做 dense、Q16、Q8、
Q4 与 mixed-bit CoMem 推理，再在完全冻结选择后打开未消费的最终评测分片。

## 7. 实际发布审计记录

2026-08-12 17:40（Asia/Shanghai）完成了正式发布审计：

- `bash -n` 和 Python `py_compile` 通过；
- `test_supervised_sft_longrun` + `test_sft_quality_validation` 共 11 项测试通过；
- 远端同一 PyTorch 环境的 2-rank FP32 FSDP DCP save/load exact round-trip 通过；
- 远端正式 train、heldout 与 split manifest 的 SHA 与第 2 节一致；
- 正式 code ledger SHA256 为
  `d7911f441a996f0ab9d8f2062f8dcbaea33d3664bcedb4b358c3a53ecf29c42e`；
- model-artifact ledger SHA256 为
  `fa050ef64c76caaa353223541c6ad8b80be8a5f6f5c11430db2d7d4f2c4dfb5c`；
- 14-shard model-weight ledger SHA256 为
  `a0352fd3fd47b4edcebf3269b5f8745490d3defb9eaedf2a4c4dc8ccae32ddf2`；
- QS YAML 为 `qs/qcomem-dense-supervised-sft-formal-384-20260812a.yaml`，SHA256
  `cf142b3d2f83daf80c3d02a489861d2cf7318606bca73259adb0a8910107a4ea`；
- `qs --dry-run` 展示的资源为 queue 436 / cloud 6 / cluster 53 / package 183，command
  与冻结 YAML 一致，没有发出 dry-run POST；
- dry-run 后的资源快照：`Reasoning_Rollout` 非 borrowed，H20-141G 余量 112 GPU；
- create 前目标 run dir 不存在，JuiceFS 可用约 18 PiB。

随后唯一正式任务已创建，不能再次 create：

```text
Job 234857
Trial 1831544
status at audit: Uncommit
https://qs2.devops.xiaohongshu.com/model/production/job/trial/234857/1831544
```

`qs training get 1831544` 返回的 command、三个 ledger SHA、数据 SHA、8-H20 资源和
`restart_num=0` 均与冻结 YAML 一致。后续只监控这一 Trial；若它失败，保留失败记录并使用
新的 run ID 提交，不得复用当前非空目录。
