# Q-CoMem Mixed-Bit H20 Validation（2026-08-12）

## 实验边界

- QS trial：`1830116`，状态 `Complete`，执行时间 10 分 14 秒；
- 模型：Qwen3.5-35B-A3B；
- 硬件：8×H20-141GB；
- 数据：LongBench Qasper/2WikiMQA 各 30 条，共 60 条；
- source index：6--35；校准样本 4--5 排除；
- test-v2 68--99：未读取；
- 最大输入/生成：4096/128 token；
- split depth：7；
- Q16 cached smoke：在正式 worker 启动前通过。

本实验是策略冻结后的 validation，不是最终 test-v2。预注册的均值门槛为 overall F1 delta ≥ -0.02、每数据集 delta ≥ -0.03；通过均值门槛不等价于统计上严格证明无损，仍需同时报告 paired bootstrap CI。

## 主要结果

| 配置 | 持久状态 MiB | 相对 full-prefix 压缩 | 相对 Q16 replay 压缩 | mean F1 | Δ dense | Δ Q16 replay | vs Q16 95% CI | token exact vs Q16 | 灾难性退化率 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| dense | — | — | — | 0.54289 | 0 | -0.00335 | [-0.04606, 0.02665] | 93.33% | 0% |
| full-prefix Q16 | 136.235 | 1.00× | 0.255× | 0.54682 | +0.00393 | +0.00058 | [0, 0.00175] | 96.67% | 0% |
| split replay Q16 | 34.683 | 3.93× | 1.00× | 0.54624 | +0.00335 | 0 | [0, 0] | 100% | 0% |
| frozen static | 9.661 | 14.10× | 3.59× | 0.54237 | -0.00052 | -0.00387 | [-0.02037, 0.01076] | 85.00% | 0% |
| same-memory mixed | 9.395 | 14.50× | 3.69× | 0.53365 | -0.00924 | -0.01259 | [-0.04279, 0.00977] | 86.67% | 1.67% |
| minus-25% mixed | 7.536 | 18.08× | 4.60× | 0.49186 | -0.05103 | -0.05438 | [-0.11561, -0.00171] | 68.33% | 8.33% |

“灾难性退化”定义为单样本相对 dense 的 F1 delta ≤ -0.5。token exact 是完整生成 token 序列相对 Q16 replay 完全一致；frozen static 的平均 token-position agreement 为 89.46%，same-memory mixed 为 90.70%，minus-25% 为 75.77%。

## 数据集分解

| 配置 | 2WikiMQA Δ dense | Qasper Δ dense | 是否通过预注册均值门槛 |
|---|---:|---:|---|
| split replay Q16 | +0.02000 | -0.01330 | 是 |
| frozen static | +0.01444 | -0.01549 | 是 |
| same-memory mixed | -0.00381 | -0.01467 | 是 |
| minus-25% mixed | -0.05048 | -0.05158 | 否 |

## 如何解释

### 1. 已找到可信的近无损 knee

frozen static 相对 full-prefix 将持久文档状态压缩 14.10×，mean F1 相对 dense 只差 -0.00052、相对 Q16 replay 差 -0.00387，CI 跨过 0，且没有灾难性退化。严谨表述应是“在当前 60 条 validation 上观察到近无损的 14.1× Pareto knee”，不能写成统计上严格无损。

### 2. split 本身就是独立卖点

不做低比特量化时，split replay Q16 已将 full-prefix 的 136.2 MiB 降至 34.7 MiB，即 3.93×。这是只持久化 boundary residual 与前 7 层 lower state、而不是全 40 层 prefix cache 的收益。低比特进一步把 Q16 split 压到约 9.66 MiB。

### 3. 当前 mixed-bit policy 没有胜过 static

same-memory mixed 只比 frozen static 少 2.75% 持久字节，却带来更大的 F1 损失和 1/60 的灾难性退化。这不否定逐层不同 bit 的研究方向，但说明由 4 个 calibration prompts 得到的误差代理不足以可靠排序下游重要性。下一版 policy 应增加校准覆盖，并把 downstream/logit sensitivity、层类型和误差传播共同纳入优化，而不是只按局部 state RMSE 分配 bit。

### 4. 极限压缩目前不是“无损”卖点

minus-25% 策略达到 18.08×，但相对 Q16 的 F1 CI 上界仍小于 0，并有 8.33% 灾难性退化。它适合作为量化条件 LoRA 的恢复目标或“可部署容量优先”档，不应与 14× knee 一起宣传为近无损。

## 下一步

1. 论文主线先使用 frozen static 作为 near-lossless operating point；
2. 对 frozen static、same-memory mixed、minus-25% 分别训练 bit-specific LoRA，验证能否恢复到 Q16；
3. 扩大 PG-19 train 与 validation calibration，再重新求一次 mixed policy；
4. adapter/bit policy 冻结后才运行一次 LongBench test-v2；
5. 结合 deployment benchmark 报告“可驻留文档数、TTFT/TPOT、adapter 常驻开销”，判断 18× 档是否真的能让原本放不下的工作集变得可部署。
