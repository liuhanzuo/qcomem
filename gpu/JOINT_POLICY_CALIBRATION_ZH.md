# Q-CoMem 扩大 PG-19 Joint Mixed-Bit 校准协议

## 目的与不可越过的数据边界

旧逐层策略只在 4 条 LongBench prompt 上分别扰动一个组件，再把各组件 first-token KL
相加。它适合功能验证，但不足以支持“自动 mixed-bit 优于 frozen static”。本协议只使用
Google DeepMind PG-19 的官方 `train/` 对象，扩大样本覆盖，并在候选生成后执行真实的
**联合量化**，避免把组件误差可加当作结论。

数据边界固定如下：

- 输入必须是 `deepmind-gutenberg/train/*.txt`，JSONL 与独立 manifest 都要通过预期
  SHA-256，逐对象 GCS MD5、bucket、prefix 和对象集合交叉核验；
- 当前 development calibration 使用 provenance-locked 的 64-book PG-19 train 子集，
  按固定 hash 从中选择 32 本，每本一个窗口；它不是完整 PG-19 正式语料；
- 禁止路径名包含 LongBench/Qasper/2WikiMQA，禁止 QA/evaluation row schema，并按已知
  SHA-256 拒绝 LongBench validation 和 frozen test-v2；
- **不读取 LongBench source 6--35，也不读取 68--99。** 旧正式 validation 的预测、F1、
  答案和既有 policy 标签都不能参与自动策略选择；
- 自动策略使用新名字 `pg19-joint-*`，不得覆写或冒充已经在 LongBench 上评过的
  `replay-d7-same-memory-mixed` 等标签。

因此本轮的 next-token NLL 只是自然文本 continuation proxy，不是下游 QA 分数。即使自动
候选在 PG-19 上优于 frozen static，也只能称 calibration candidate，不能继承旧 validation
结论。

## 固定窗口与目标

- 32 本 PG-19 train books；选择 seed `20260812`；
- tokenizer 后的冻结窗口 SHA-256 为
  `5d295ac27424ed71b3036bcdb0c3ee2bc30e6bf92460a53770ab92d3d1a0a3a6`；
- 每本最多查看开头四个候选窗口，再由 source hash 固定一个窗口；
- document `1024` token，query continuation `128` token，候选窗口 stride `512`；
- 在 query 的 8 个均匀位置测量完整 vocabulary logits，每个位置都有下一个真实 PG-19
  token 作为 target；
- depth 固定为 7，group size 64，候选 bit 为 Q2/Q4/Q8/Q16。

对 Q16 teacher 和联合量化 candidate，累计：

- `KL(teacher || candidate)`；
- 真实 PG-19 next-token NLL delta 以及只保留有害方向的 positive NLL delta；
- teacher/candidate top-1 agreement；
- relative logit L2 与 max-abs logit error；
- 含 scale/bias/packing metadata 的真实 persistent bytes。

预先固定的 calibration scalar 为：

```text
mean forward KL
+ mean positive PG-19 next-token NLL delta
+ 0.1 * (1 - top-1 agreement)
```

这个 scalar 只用于 PG-19 calibration 排序。所有分项都必须保留，不能只报告 scalar。

## 两阶段策略搜索

### 阶段 1：扩大 component profile

8 张 H20 各负责 residual 或一个 lower cache layer。每个组件分别切换
Q2/Q4/Q8/Q16，其余组件保持 Q16。相比旧版，这里从 4 个 QA first-token prompt 扩大到
32 本书、每本 8 个 query positions，并加入自然 continuation NLL。

阶段 1 的 additive score 只用于从 `4^8 = 65,536` 个组合中提名候选：

- frozen-static 同内存预算下前 6 个；
- frozen-static 75% 预算下前 6 个；
- 去重后与 Q16、frozen-static、uniform-Q8 controls 一起冻结到
  `joint-policy-candidates.json`。

### 阶段 2：真实 joint quantization

候选文件落盘后不再修改。8 张卡分摊候选，但每个候选都在同一组 32 个窗口上执行完整
`residual + 7 lower cache layers` 的联合 pack/dequant，并采用部署一致的 suffix 边界：先用
document residual 建 suffix cache，再一次性 continuation query residual。最终选择基于阶段 2
的真实联合指标，而不是阶段 1 的可加近似。

始终保留三个 control：

- Q16 `[16; 16,16,16,16,16,16,16]`；
- frozen static `[4; 8,8,8,4,8,8,8]`；
- uniform Q8 `[8; 8,8,8,8,8,8,8]`。

Q16 在每个阶段都必须满足 mean forward KL `<=1e-6`、top-1 agreement `=100%`、max-abs
logit error `<=1e-5`，否则整项实验失败。same-memory 候选的实测 bytes 不得超过 frozen
static；minus-25 候选不得超过 frozen static 实测 bytes 的 75%。

## 结果状态与后续冻结

最终 `joint_policy.json` 只会生成：

- `q16_control`；
- `frozen_static_control`；
- `pg19_joint_same_memory_candidate`；
- `pg19_joint_minus_25_percent_candidate`。

后二者始终带 `calibration_candidate_only` 状态。它们不能用已经揭晓的 source 6--35 F1
挑选或重命名。若候选值得继续，需在整个算法/LoRA/runtime 选择全部冻结后，制定新的独立
下游协议；frozen test-v2 68--99 仍只允许最终一次使用。

## H20 资源审计

- 单个任务，8×H20-141G，queue 400，与 LoRA queue 408 分离；
- 两阶段都只有 inference，无 backward、optimizer、checkpoint 或第二份模型；
- 每卡一份约 64.6 GiB BF16 模型；结合此前同模型 replay 实测，预计峰值仍在约 68 GiB
  以下，明显低于 141 GiB；
- 阶段 1 每卡 32 windows × 4 component bits；阶段 2 每卡约 1--2 个联合 policy；
- 产物只有 JSON/log/ledger，预期远小于 1 GiB；
- launcher 拒绝非空 run dir，避免同名重复任务；提交前必须 `qs --dry-run`，且只提交一次。

入口：`gpu/launch_joint_policy_calibration_8gpu.sh`。静态门禁：

```bash
PYTHONPATH=gpu python3 -m unittest gpu/test_joint_policy_calibration.py -v
python3 -m py_compile gpu/qcomem_joint_policy.py \
  gpu/run_joint_policy_profile.py gpu/aggregate_joint_policy_candidates.py \
  gpu/run_joint_policy_eval.py gpu/aggregate_joint_policy_eval.py
bash -n gpu/launch_joint_policy_calibration_8gpu.sh
```
