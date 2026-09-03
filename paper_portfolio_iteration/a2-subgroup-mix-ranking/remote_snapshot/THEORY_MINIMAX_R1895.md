# Fixed-Budget Minimax Subgroup Allocation for Safe Ranking — Theory (r1895)

项目：A2_SAFE_MODEL_RANKING_SUBGROUP_MIX。本文件把 MGR 指令 r1895 的中心对象
`min_{n_g, sum_g n_g = R} max_j sum_g w_g r_g(n_g)` 形式化，给出 KKT / 水填充 /
可核验解析条件，精确划定 uniform 何时是最优(minimax)、何时不是(构造性对称反例与
非对称反例)，并把一个**凸 minimax allocation** 作为同预算强基线，用于四 carrier 前台比较。

状态：定理推导完成；数值核验见 `results/SUBGMIX_MINIMAX_R1895.json`（前台 EXIT=0）。
本文只给解析条件；严格建立在真实读回的 outer/errs 之上。

---

## 1. 设定与对象

**风险近似（方差主导）**：对候选模型 $i$ 与组 $g$，$r_g(n_g)$ 表示"花 $n_g$ 个 CAL 标签
估计后组 $g$ 风险可达到的精度水平"。在配对差证书中（Thm m25），每个 $\mathrm{UCB}_{i^*j,g}$
是关于配对差 $\bar d_n$ 的经验-Bernstein 上界，其**不确定半宽** $\asymp
\sigma_{ij,g}/\sqrt{n_g}$（Maurer–Pontil 的 $\sqrt{2vX L/n}$ 项 + 高阶项），其中
$\sigma_{ij,g}$ 是组内第 $i,j$ 模型错误指示的差的标准差。因此对任意组合权重
$w\in\Delta^{G-1}$，最少需要的组 $g$ 标签数满足一个**可加成本**关系：要同时保证所有组
的加权不确定 ≤ 目标，等价于解读下面的凸规划。

**形式化对象（MGR 指令原文，逐字固定）**：
$$
\mathrm{minimize}_{n\in\mathbb N^G,\ \sum_g n_g=R}\ \ 
\max_{j}\ \sum_{g} w_{g}\, r_{g}(n_g) . \tag{$\mathcal P$}
$$

- 这里 $r_g(n_g)$ 是组 $g$ 的径向"不确定"量。为使问题可解且 von-Neumann minimax 可交换，
  我们令 $r_g(n_g)$ 为组 $g$ 配对差 UCB 的一组寻址所需半宽的**单调下降凸**函数。
  在经验-Bernstein 中自然选择 $r_g(n)=\sigma_g\,\Phi(1/\sqrt{n})$，即
  $r_g(n)=\beta_g/\sqrt{n}+\gamma_g$，$\beta_g\propto\sigma_{ij,g}$（组内差标准差），
  $\gamma_g$ 是可忽略项。这是**凸**（在 $x=1/\sqrt{n}$ 意义下：
  $1/\sqrt{n}$ 在 $[0,\infty)$ 凸，$\beta_g\ge0$ 保持凸）。
- $w_g$ 是一个未来 mixture 的组权重；$\max_j \sum_g w_g r_g(n_g)$ 中 $j$ 索引的最坏候选。

**注意与跑步机的关系**：这是在 $r1894$ 观测"uniform 在对称网格 10/10 胜 neyman"基础上的
**形式化**，且要把现象与条件性剥离：uniform 的极大极小最优只在对称族成立，非对称族必须
用凸 minimax 分配。

---

## 2. 一般凸代理的最小化：KKT / 水填充

### 2.1 连续松弛与拉格朗日
把 $n_g$ 松弛为实数 $n_g>0$，$\beta_g>0$（若某组 $\beta_g=0$ 则该项恒零，对该组最优
alloc 为零，见 counterexample）。目标
$$
F(n)=\max_j \sum_g w_g\,\big(\beta^j_g/\sqrt{n_g}+\gamma^j_g\big).
$$
令 $\Phi_j(n)=\sum_g w_g\beta^j_g n_g^{-1/2}$（忽略常数 $\gamma$，它们对 $n$ 不变）。
凸函数最大值是凸函数，故 $\max_j\Phi_j(n)$ 是 $n$ 的凸函数。目标即
$$
\min_{n:\,\mathbf 1^\top n=R} \ \max_j \Phi_j(n).
$$
引入 $\lambda\ge0$ 对预算，$t$ 对 $\max$ 层（epigraph）：
$$
\min_{n,t}\ t \quad \text{s.t.}\ \ \Phi_j(n)\le t\ \ \forall j,\ \ \mathbf 1^\top n=R.
$$

### 2.2 可核验的一阶 KKT 水填充条件
假设在解处激活了非空的最坏候选集 $J^*(n)=\{j:\Phi_j(n)=t^*\}$（I 类，纯粹回归；
若没有候选在解处激活——即 $\max_j\Phi_j(n)$ 由某个孤点最坏实现——则问题退化为单一
$\Phi_{j^*}$ 的全局极小，见 §4）。在正则性（LICQ：激活约束梯度线性无关；此处各
$\nabla\Phi_j$ 分量 $\propto -w_g\beta^j_g n_g^{-3/2}$，方向各由 $w_g\beta^j_g$ 缩放，只要
权重向量不全成比例，满足非退化）下，存在乘子 $\lambda,\mu_j\ge0,\sum_{j\in J^*}\mu_j=1$，使
$$
\forall g:\ 
\sum_{j\in J^*}\mu_j\, w_g\,\beta^j_g\,\tfrac{1}{2}n_g^{-3/2}
\;=\; \lambda
\ \ \text{(每组的边际收益相等，即水填充)} , \tag{KKT-1}
$$
$$
\sum_g n_g=R,\qquad \Phi_j(n)=t^*\ \ \forall j\in J^*,\ \ \Phi_j(n)\le t^*\ \forall j. \tag{KKT-2}
$$

**可核验解析条件**令 $\bar\beta_g^{\rm eff}(n)
=\Big[\sum_{j\in J^*}\mu_j\,\beta^j_g\Big]$ 为激活候选上的 $\mu$-加权组差标准差。则
$$
n_g\ \propto\ \big(w_g\,\bar\beta_g^{\rm eff}\big)^{2/3} . \tag{WATER}
$$
即：**最优 minimax 分配按 $(w_g \beta_g)^{2/3}$ 分配样本**（$2/3$ 幂次来自
半宽 ∝ $\beta_g/\sqrt{n_g}$ 的边际 $-\tfrac12\beta_g n_g^{-3/2}$，在与预算列等式后取
$2/3$ 幂）。这是直接的**可核验解析条件**：给定组差标准差 $\beta_{ij,g}$ 与组合权重 $w$，
该比例可从数据估计并在 O(G) 内算出；若它 ≠ uniform 比例，则 uniform 不是该族的最优分配。

> 注：2/3 幂来自"方差是 1/n 衰减"（错误指示差经中心极限/经验-Bernstein）。若换用别的
> 凸代理（例如 $\beta_g \log(1/n_g)$ 或 $\beta_g n_g^{-1}$），幂次会变为 $1/2$ 或 $1$。
> 本文所有结果只用 $1/\sqrt{n}$（error-indicator 的自然尺度），故 $2/3$。

### 2.3 对偶（r1897 修正前，已作废 INVALID-DIAGNOSTIC）

> ⚠️ **r1897 修正**：本节 `d^*=\lambda^* R` 是**错误维数**（$\lambda$ 已含 $R^{-3/2}$，
> 故 $\lambda R$ 的齐次性是 $R^{-1/2}$，与半径目标一致，但值本身应为 $2\lambda R$ 而非
> $\lambda R$，见 §2.3bis）。旧 minimax 还把**所有候选等权**当活跃集，实算只在对称平凡例
> 上成立。本节与 r1895 两个 JSON 一律按 INVALID-DIAGNOSTIC 保留，数字不作主张。

（旧文：令对偶变量为预算乘子与 minimax 层乘子…… $d^*=\lambda^* R$……）

---

### 2.3bis 修正后的强对偶 / 值·预算齐次性（r1897, MGR 593c907d2ccd）

**引擎系数矩阵**。把孩子 $(j,w)$（候选模型 × 部署 mixture）合称一个**引擎** $k$，其
组 $g$ 宽度系数 $C_{kg}=w_g\,\beta_{jg}$。于是 $\Phi_k(n)=\sum_g C_{kg}\,n_g^{-1/2}$，
原问题（对 worst mixture **和** worst candidate 联合）是
$$
\min_{n:\,\mathbf1^\top n=R}\ \max_k \Phi_k(n). \tag{$\mathcal P'$}
$$
$\max_k\Phi_k(n)$ 是 $n$ 的凸函数，故 $\mathcal P'$ 是凸问题。

**Sion 强对偶（minimax 交换）**。因 $\max_k\Phi_k(n)=\max_{\mu\in\Delta^{K}}\sum_k\mu_k\Phi_k(n)$
凸函数的最大值是凸的，且对 $(n,\mu)$ 是 convex-in-$n$ / linear-in-$\mu$（因此 concave-in
-$\mu$），Sion 极小极大定理适用：
$$
d^* \;=\; \min_n \max_\mu \sum_{k,g}\mu_k C_{kg}n_g^{-1/2}
\;=\; \max_\mu \min_n \sum_{k,g}\mu_k C_{kg}n_g^{-1/2} \;=\; P^*.
$$

**固定 $\mu$ 的内层闭式**。$a_g(\mu)=\sum_k\mu_kC_{kg}$。对固定 $\mu$，内层是等权总生意式
$\min_{n,\mathbf1^\top n=R}\sum_g a_g(\mu)n_g^{-1/2}$，其唯一水填充解为
$$
n_g(\mu) \;=\; \frac{R\,a_g(\mu)^{2/3}}{S(\mu)},\quad
S(\mu)=\sum_g a_g(\mu)^{2/3}, \qquad
V(\mu)=\frac{S(\mu)^{3/2}}{\sqrt{R}} . \tag{WATER-R}
$$
对偶值即 $\max_{\mu\in\Delta^{K}}V(\mu)$。**预算乘子**（对 $\mathbf1^\top n=R$）
$$
\lambda \;=\; \tfrac{1}{2}S(\mu)^{3/2}R^{-3/2}
\ \ \Longrightarrow\ \  \boxed{\,V(\mu)=2\lambda R\,},
$$
即 **值 = $2\lambda R$，且值 ∝ $R^{-1/2}$**（半径目标的齐次性），而不是旧的
$d^*=\lambda R$。

**$S(\mu)$ 凹 ⇒ 一族凸规划**。每个 $a_g(\mu)^{2/3}$ 是线性型 $u\mapsto a_g(u)$ 的
$2/3$ 幂（$0<2/3<1$，凹），加和仍凹。故
$$
d^* \;=\; R^{-1/2}\Big[\max_{\mu\in\Delta^{K}} S(\mu)\Big]^{3/2},
$$
`max_{\mu} S(\mu)` 是单纯形上的**凹最大化**（等价凸规划），可用投影/梯度（Duchi
simplex projection）或任一凸求解器精确求解。**这就是 MGR 要求的"联合求 $\mu$ 与 $n$ 的
凸解/可证等价算法"**：$\mu^*=\arg\max_\mu S(\mu)$，$n^*=n(\mu^*)$。

**互补松弛 ⇒ 活跃集**。在 $n^*$ 处，活跃引擎是使 $\Phi_k(n^*)=h(n^*)$ 者（= 对偶承重的
$k$），由互补松弛精确给出
$$
k\in\text{active}\ \Longleftrightarrow\ \mu^*_k>0 .
$$
**绝不是"所有候选都活跃"**：凡被支配的引擎（某 $k$ 的 $C_k$ 逐组 ≤ 某凸组合）即
$\mu^*_k=0$，不活跃。r1895 把全部候选等权当活跃集是启发式，仅在对称平凡例上无害；真实
四 carrier 上活跃引擎只有 K 的 6–25%（见 `SUBGMIX_MINIMAX_ABLATION_R1897.json`）。

**可核验数值 fixture**（`results/MM_FIXTURE_R1897.json`，前台 EXIT=0）：
- 四类 $(M,G)$ 与 $\beta$ 情境，连续松弛对偶间隙 $|P^*_{\rm cont}-d^*|/d^*<10^{-6}$（强对偶实证）；
- 恒等式 $V=2\lambda R$ 到 $10^{-9}$；
- $R\in\{400,800,1200,2400,4800\}$ 上 $V\sqrt{R}$ 平坦到 $<10^{-16}$（$R^{-1/2}$ 齐次性）；
- 互补松弛：活跃集=承重引擎、非活跃 $\Phi<\max\Phi$、活跃内等值（到整数舍入误差）；
- 整数化代价：非对称 case P* 连续 vs 整数分配相对间隙 $10^{-6}$–$10^{-3}$（floor+n_min 的质数
  费，诚实披露，非对偶错误）。

**容量墙的修正读法**。旧文把"$\lambda R$ 超 $\tau$"当 infeasible；修正后目标=值
$V=2\lambda R=S^{3/2}/\sqrt{R}$。digits/news 的 $\beta$ 大 ⇒ $S$ 大 ⇒ $V$ 超 $\tau$ 的
可证预算仍然存在（结论不变，仍是 `n<680` 容量墙），但判定量是 $S^{3/2}/\sqrt{R}$ 而非
$\lambda R$。

---

## 3. uniform 何时是最优 minimax？(可核验对称性条件)

在 §2 的水填充解上与 uniform（$n_g\propto1$）比较。uniform 是 $\mathcal P$ 的极小解
当且仅当当 $(KU)$ 设置代入 WATER 时退化为常数。具体地：

**Proposition (uniform 最优的对称性判据)。** 设最优分配为水填充
$n_g^{**}\propto (w_g\bar\beta_g^{\text{eff}})^{2/3}$。uniform 是 $\mathcal P$ 的最优解
（在 $\sum n_g=R$ 下）当且仅当水填充对每个激活组常数，即 $$w_g\,\bar\beta_g^{\rm eff}$$
在 $g$ 上恒等。特别地，**以下任一不对称即会破坏 uniform 最优性**：
- (i) **方差不对称**：$\bar\beta_g^{\rm eff}$ 非常数（一些组差方向显著高于其他组）；
- (ii) **权重不对称**：$w$ 不是均匀（未来 mixture 偏向某组）；
- (iii) **候选族对称性**：对激活候选 $J^*$，$\beta_g^j$ 视 $j$ 在组间非等间隔。

**证明 sketch。** 若 $w_g\bar\beta_g^{\rm eff}=c$ 恒定，则 WATER 给 $n_g\propto c^{2/3}$ 常数，
即 uniform。反过来，若 uniform 最优，水填充（唯一，因 $\Phi_j$ 严格凸）必为常数，故
$w_g\bar\beta_g^{\rm eff}$ 恒定。∎

**阈值/可核验**：只需检验样本标准差 $\hat\beta_{ij,g}$ 的组间 CV（变异系数）与
$w$ 的熵。若两者都小，uniform 是可核验最优；否则 convex-minimax（水填充）严格领先。

**r1899 门 operating characteristic 实证核验（控制变量）**：用受控合成族（G=4、同一
paired-MPB 证书、R=1200、20 seed、τ 扫描）独立变化 deployed set ∈ {spanning, non-spanning} ×
CV(β̂) ∈ {低, 高}，见 `results/SUBGMIX_GATE_OC_R1899.json`。判据证实"驱动源=deployed set 是否
spanning，非 CV(β̂)"：非 spanning 时 minimax committed-rate 曲线在**两个 CV 水平**都支配 uniform
（signed area +0.25，minimax 在 τ=0.08 即达 1.0 而 uniform 停在 0.25–0.93）；spanning 低 CV 持平
（area +0.00，uniform=maximin 成立处）；**spanning 高 CV 时 minimax 反而更差**（area −0.08，
worst −0.49@τ=0.12，因 minimax 集中预算到高 β 组、饿死 spanning 集低方差顶点）——这正是
"else uniform"分支的安全半边。⇒ 门"非 spanning → convex-minimax，否则 uniform" safe+effective；
CV(β̂) 只定位承重高方差组、非第二驱动。20 seed 合成 OC 比先前 3 seed 真实表（r1896）
在统计上更稳。

---

## 4. 确定性反例（非对称族，uniform 严格次优）

**反例 A（方差不对称，均匀权重，破坏 uniform）。** 取 $G=2$，均匀 $w=(1/2,1/2)$，
$\beta=(0,1)$：组 1 的配对差方差为 0（两模型错误指示恒一致，无需样本），组 2 方差为 1。
(KKT-1) 里组 1 的 $\bar\beta=0$，故其边际收益恒 0，最优把**全部** $R$ 样本给组 2：
$n^{**}=(0,R)$，目标 $=\tfrac12\cdot 1/\sqrt R$。uniform $n=(R/2,R/2)$ 目标
$=\tfrac12(0+1/\sqrt{R/2})=1/\sqrt{2R}$，前者严格更小（比值 $\sqrt2$）。uniform
在此**严格次优**。这就是 MGR 要的"至少一个非对称 w 或方差的确定性反例"。

**反例 B（权重不对称，均匀方差）。** $G=2$，$w=(0.9,0.1)$，$\beta=(1,1)$。
WATER：$n_g\propto w_g^{2/3}$，故 $n^{**}=(yR,(1-y)R)$ 其中 $y=0.9^{2/3}/(0.9^{2/3}+0.1^{2/3})$
$\approx 0.966$。均匀 $n=(0.5,0.5)$ 分配给低权重组的样本被浪费（$0.1/\sqrt{\cdot}$ 项被 0.9
掩蔽），目标高于最优。uniform 再次严格次优。

**为何 MGR 说"uniform=maximin 一般最优/恒支配 only 按对称候选网格现象"。** 在既有对称
网格（uniform/skew/interp 全含均匀列、错误率低、$\beta_{ij,g}$ 组间 CV 小）上，水填充
退化到 uniform 附近，故 uniform 观测性胜出；但这是**该族对称性的产物**，不是普适定律。
非对称族（A/B）中 uniform 的不是 minimax 最优，凸 minimax（水填充）严格更好。

---

## 5. 凸 minimax allocation 作为同预算强基线（可执行算法, r1897 修正）

r1897 用联合求解取代旧 §5 只对单个 $w$ 的水填充。算法（`code/minimax_core_r1897.py`）：

1. 组引擎系数矩阵 $C_{kg}=w_g\,\beta_{jg}$（对全部 $(j,w)\in$ 模型×部署网格，只读 FIT 的
   $\beta$，不读 CAL，静态 MPB 证书健全）。
2. 凸解：单纯形上凹最大化 $\max_{\mu\in\Delta^{K}}S(\mu)$（投影梯度/Duchi 单纯形投影，
   多起点重启）到收敛。
3. 分配 $n_g=R\,a_g(\mu^*)^{2/3}/S(\mu^*)$，整数化（floor + 预算余量 round-robin +
   $\ge n_{\min}$ 下限）；活跃集=$\{\mu^*_k>0\}$ 由互补松弛给出。
4. 同预算 R、同评估网格、同 M2.5 paired-MPB 证书，与 uniform/neyman/widthgreedy/sens/
   前台比较 committed_rate / 证书宽度 / 各组 n / 成本 / violations / 弱域。

结果见 `results/SUBGMIX_MINIMAX_R1897.json`（修正后）、
`results/SUBGMIX_MINIMAX_ABLATION_R1897.json`（all-active 启发式消融：活跃引擎仅 K 的
6–25%，修正 d* 为 worst-case 混合，故 ≥ uniform-mixing 的旧启发式值）。旧
`results/SUBGMIX_MINIMAX_R1895.json` 为 INVALID-DIAGNOSTIC，不作主张。

---

## 6. 诚实边界（同既有项目纪律延续）

- 本文只核验"水填充分配对 $\mathcal P$（为何 minimax 目标）是最优"，不声称任何"真实
  regret 更低"。committed_rate 是证书支配的度量；分配改变各组 n → 改变 UCB 宽度 →
  改变一张证书能 committed 的蛋糕份额，但绝不保证真实 regret 单调。
- 配对证书为 **simplex-同时**（非逐 w 逐点）：UCB[(i,j)][g] 独立于 w，只在有序对×组上做一次
  Bonferroni（δ/(M(M−1)G)），同一事件 E 同时对**整个连续单纯形**上所有 w 与数据依赖选择 i*(w)
  给 UB(w)=max_j Σ_g w_g UCB。r1898 用每 (carrier,seed,预算) 1000 个离网格 Dirichlet w 扫描
  核验：committed 单元格覆盖率仍 1.0（mnist π=0.5 离网格 2816/3000 commit、fashion 0.8 5/5、0.95
  70/70），容量墙 carrier（digits/news）仍 0 commit。**旧"网格联合待做"是过度保守的自我降级**，
  正确口径=证书已对连续单纯形联合，无需 per-grid 再 Bonferroni。3 seed（5 seed 复验为 TBD，非阻塞）。
- digits/news $n<680$ 容量墙保留：$\mathcal P$ 在容量墙内不可行（target 超 τ）时，
  无论分配多好都 0 commit；这是可证容量边界不是分配缺陷。
- WATER 的活跃集 $J^*$（引擎集）由互补松弛 $\mu^*_k>0$ 给出，r1897 不再假设"所有引擎活跃"；
  实测真活跃引擎仅 K 的 6–25%（`SUBGMIX_MINIMAX_ABLATION_R1897.json`）。旧 r1895 的
  "全候选作活跃集"是 INVALID-DIAGNOSTIC 启发式。
- 值-齐次性统一：修正后值 $V=S^{3/2}/\sqrt R=2\lambda R$，容量墙判定用 $V$ 超 $\tau$（
  digits/news 仍不可行，结论不变）。

**关联网**：[[a2-r1894-certified-ub-is-max-of-weighted-means]]、[[a2-r1886-m3-budget-allocation]]、
[[a2-r1816-union-dir-bandmass-dominance]]、[[a2-r1813-union-direction-correction]]。