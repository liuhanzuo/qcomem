# Correlated Majority-Vote Early Stopping：多 rollout 何时可以提前停——细致中文解析

> 公开审阅快照：Owner A11；状态 `WAITING_HUMAN`。本页保留论文问题、方法、理论、实验与局限的细致讲解；内部运行轨迹、逐轮状态附录、服务信息和本机路径未公开。
>
> [冻结 PDF](../manuscripts/a11-correlated-majority-vote/paper.pdf) · [冻结 LaTeX 全文](../manuscripts/a11-correlated-majority-vote/paper.tex)

Self-consistency 对同一问题采样 \(N\) 个答案，再取多数票。许多问题在前几个 rollout 后赢家已稳定，继续采样浪费 token；但 rollout 的正确性和答案模式常相关，不能用独立硬币公式判断“多数票不会翻转”。本文从 calibration problems 学习每题最终 pass-count 的 mixture prior，计算给定当前前缀状态后 full-\(N\) majority 翻转的精确 posterior probability，并在该条件风险低于阈值时停止。

## 1. 状态、最终多数与 flip event

为简化说明，把每个 rollout 相对当前候选答案编码为二元票。第 \(k\) 步已观察前缀状态 \(x_k\)，包括当前票数或 margin；完整预算为 \(N\)。最终多数答案记 \(M_N\)，当前前缀多数记 \(M_k\)。停止错误事件是

\[
F_k=\{M_N\neq M_k\}.
\]

目标不是预测 gold correctness，而是控制“如果现在停，最终多数是否会翻转”。这使证书可用无 gold rollout 状态计算。

## 2. mixture prior \(H\) 表示什么

对每个 calibration problem，完整 \(N\) 次中某候选获得的最终票数为 \(K\in\{0,\ldots,N\}\)。跨问题的经验分布记 \(H(K)\)。它允许不同题有不同难度和相关结构：不假设每个 rollout 来自同一固定 Bernoulli \(p\)，而是先抽一个 problem-level count \(K\)，再条件于该 count 观察无放回前缀。相关性被压缩到 count mixture。

## 3. 条件前缀分布为何是 hypergeometric

条件于完整序列中有 \(K\) 个“成功票”，若 rollout 顺序可交换，前 \(k\) 步看到 \(r\) 个成功的概率是

\[
P(R_k=r\mid K)
=\frac{\binom Kr\binom{N-K}{k-r}}
{\binom Nk}.
\]

这是 hypergeometric 分布：从含 \(K\) 个成功的有限总体无放回抽 \(k\) 个。它不要求 rollout 独立，但要求在给定最终 count 后顺序交换。ordered drift 会破坏该假设，论文另做压力测试。

## 4. BAYES-H posterior flip probability

给定前缀 \((k,r)\)，Bayes 公式更新 count posterior：

\[
P(K\mid k,r,H)
\propto H(K)
\frac{\binom Kr\binom{N-K}{k-r}}{\binom Nk}.
\]

对每个可能 \(K\)，最终 majority 是否与当前前缀 majority 相反是确定事件。于是

\[
q_H(k,r)
=\sum_KP(K\mid k,r,H)
\mathbf1\{M_N(K)\neq M_k(r)}.
\]

BAYES-H 在 \(q_H(k,r)\lealpha\) 时停止，否则继续到下一 rollout。

## 5. “adaptive is free”定理

令 \(\tau\) 是只依赖前缀的 stopping time，且仅在 \(q_H(x_\tau)\le\alpha\) 的状态停止。条件风险定义正是

\[
q_H(x_\tau)=P(F_\tau\mid\mathcal F_\tau).
\]

塔式法则给出

\[
P(F_\tau)
=E[P(F_\tau\mid\mathcal F_\tau)]
=E[q_H(x_\tau)]
\le\alpha,
\]

其中 forced stop at \(N\) 的条件风险也按定义计入。不需要对 \(k=1,\ldots,N\) union bound，因为只评估实际停止状态的条件风险。

## 6. 为什么这不等于无限制 optional stopping

定理要求 \(q_H\) 是真实生成 law 下的条件概率，\(\tau\) 对前缀可测，且停止规则只在认证状态触发。如果 \(H\) 从同一 TEST 调整、顺序不可交换或状态遗漏影响未来的变量，\(q_H\) 就不再是真条件风险。adaptive 免费的是“在正确 posterior table 上选停止时间”，不是任意调阈值、挑规则或跨分布复用。

## 7. 未知 mixture 的 population layer

\(H\) 从 \(m\) 个 CAL problems 估计。对每个候选规则 \(j\)，可精确重放每个 CAL problem count 得到 flip risk \(g_i^{(j)}\in[0,1]\)。经验 Bernstein 上界为

\[
\bar g_j+
\sqrt{\frac{2s_j^2log(4/\delta_j)}m}
+\frac{7\log(4/\delta_j)}{3(m-1)}.
\]

对预声明的 \(J=64\) 规则分配 \(\delta_j\)，同时事件覆盖搜索族。只有上界低于目标，规则才能进入 TEST。

## 8. fixed-k baselines 为什么重要

Hoeffding-fixed-k 不建 mixture，只用分布无关界选择一个统一前缀长度；安全但节省少。empirical-Bernstein fixed-k 用 CAL 估计每个 k 的 population flip，允许 mixture-aware 但不按题自适应。window heuristic 根据最近几票稳定性停止，便宜但无有效错误控制。BAYES-H 必须同时胜过最安全固定预算和最强 mixture-aware fixed budget，才能证明 per-problem posterior 的新增价值。

## 9. ordered drift 压力测试

front-loading 把某类票集中在前面，linear trend 让成功概率随 rollout index 改变，adversarial block swap 交换前后块。它们都破坏给定 K 后的可交换顺序。实验比较 stopper 的实际 flip 与节省随 drift 强度变化；BAYES-H 相对优雅退化，但原始 \(\alpha\) 证书不再自动成立。解决办法是增加 certificate margin 或用明确 drift budget 收紧阈值。

## 10. total-variation 鲁棒证书

固定由 \(\hat H\) 生成的规则，令 \(g(K)\) 为在最终 count K 下的精确 flip probability。对 total-variation ball

\[
\mathcal B_R={q:\lVert q-\hat H\rVert_1\le2R}
\]

最坏风险

\[
V(R)=\max_{q\in\mathcal B_R}\sum_Kq_Kg(K)
\]

是线性规划。最优策略从低 \(g\) atoms 抽质量，加到高 \(g\) atoms，直到搬移 \(R\)。排序后 \(O(N\log N)\) 可精确计算，比粗界 \(E_{\hat H}g+R\) 更紧。

## 11. critical radius 与守恒曲线

定义

\[
R^*(\alpha)=\sup\{R:V(R)\le\alpha\}.
\]

它表示 frozen rule 可证明承受多大 count-prior 漂移。规则越激进，base risk 和 \(\max_Kg(K)\) 越大，critical radius 越小。后期定理进一步说明 full-radius crossing set 对 flip budget 是前缀：一旦 base risk 越过 \(\alpha\)，更松预算不能恢复全半径鲁棒；但 \(R^*\) 在小于 1 区间可非单调。

## 12. 主实验数据与 split

主载体 OpenMathReasoning 有 11,607 个数学问题，每题 \(N=32\) 个 rollout，严格按 problem 划 FIT/CAL/TEST，TEST 单次读。比较 BAYES-H、Hoeffding fixed-k、EB fixed-k 和 heuristics。主指标是相对 full-32 的 token/rollout 节省、前缀多数与 full majority 的 flip rate，以及 paired savings gap。第二 shard 独立复现并做 prior transfer，观测 total variation 约 0.042。

## 13. 主结果与跨载体边界

BAYES-H 在 \(\alpha=0.025/0.05\) 一类工作点节省约 81%/73% rollout，同时满足冻结证书；分布无关 fixed-k 约节省 16%，强 mixture-aware fixed-k 可节省约 47% 或在另一工作点更高，但 paired 比较仍显示 adaptive 增益。第二 shard 和 prior transfer 的指标移动小于约 0.4 个百分点。OpenR1-Math 每题只有 \(M=2\) 时证书几乎无提前空间，RLVE \(N=8\) 形成中间工作点，说明 horizon 是可达性的结构变量。

## 14. 为什么 prior mismatch 不能只看均值

两个 \(H\) 可以有相同平均 pass count，却把概率质量放在 majority boundary 附近或远离边界。flip profile \(g(K)\) 在边界附近最大，所以风险取决于 \(H\) 与 \(g\) 的对齐。TV LP 直接沿 \(g\) 最坏搬质量，避免用均值差或 KL 单一标量遗漏危险 atoms。它也给出清晰的部署诊断：估计的新 prior 超过 \(R^*\) 时回退 full-N 或重新校准。

## 15. 直觉 takeaway 与最终边界

BAYES-H 把“当前领先多少票”与“这类题最终通常有多少票”结合，计算当前多数被 full budget 推翻的真实条件概率。塔式法则让正确 posterior 下的自适应停止无需逐时刻 Bonferroni，CAL layer 则为未知 task mixture 付一次有限样本成本。终审必须核对 exchangeability、problem-level split、64 候选族、forced-stop 计入方式和 drift 结果；不能把与 full-majority 一致误写成与 gold 一致。

## 16. 一个 N=5 的 posterior 手算

设 CAL prior 只在 \(K=2\) 与 \(K=4\) 各有 0.5 质量。观察前 \(k=2\) 票都成功，即 \(r=2\)。likelihood 分别为

\[
P(r=2\mid K=2)=\binom22\binom30/\binom52=.1,
\]

\[
P(r=2\mid K=4)=\binom42\binom10/\binom52=.6.
\]

posterior 质量归一化后约为 1/7 与 6/7。当前多数为成功；K=2 最终会失败，K=4 会成功，所以 flip risk 约 1/7，而不是按独立 \(p=.5\) 计算的尾概率。

## 17. full-majority surrogate 的科学含义

早停目标是复现花满 \(N\) rollout 的答案，而 full majority 自身可能错误。若 full majority accuracy 为 \(a\)，stopper flip rate 为 \(f\)，早停相对 gold accuracy 最多只能用 coupling 给粗差 \(|Acc_{stop}-a|\le f\)。低 flip 保证计算近似，不创造超过 teacher majority 的正确性。若论文讨论 accuracy gain，必须来自另一个机制或真实 gold 结算，不能由 flip theorem 直接推出。

## 18. rollout 顺序的生产合同

exchangeability 要求 rollout 的 seed、temperature、prompt 和并发调度不系统地随 index 变化。若先跑低温、后跑高温，或 retry 后切模型，前缀 law 就有趋势。最简单实现是预先采样全部 seed 列表并随机排列，再按该顺序流式请求；缓存 full-N 仅用于离线评测。线上若 provider drift，应使用 TV margin 或回退固定 k。

## 19. token saving 与 wall-clock saving 的区别

少 rollout 通常省生成 token，但并发系统的 wall time 可能由最慢批次、队列和网络开销决定。若 32 个 rollout 一次并发发出，观察到第 6 个答案时其余请求已消费资源，早停无法回收全部成本。可部署实现需要分批发射或支持 cancel，并报告实际 completed/cancelled tokens、请求数和 latency。论文的 rollout saving 是核心可比单位，不能自动等同电费或延迟百分比。

## 20. 人工核验 posterior table 的最小例子

对小 \(N=5\) 穷举全部 \(2^5\) 二元序列，按最终 count K 汇总，再对每个前缀状态比较枚举得到的 flip fraction 与 BAYES-H table。随后随机抽 stopping path，确认算法只在 table≤\(alpha\) 时停、\(N\) 强制停止正确记风险。这个小例可以一次验证 hypergeometric、tie、posterior normalization 和 stopping semantics，而无需运行完整模型。
