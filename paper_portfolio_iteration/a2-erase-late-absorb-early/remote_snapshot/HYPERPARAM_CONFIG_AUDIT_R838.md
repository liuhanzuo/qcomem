# 第二十四维自主审计：训练配置/超参数声明 vs 一手脚本/JSON（HYPERPARAM_CONFIG_AUDIT_R838）

**日期**：2026-08-10（第838回合）。**范围**：A2 线投稿包（workspace 镜像 `paper/`，与 run 根字节一致）。**性质**：纯只读 + 内存态故障注入（字符串副本），冻结字节零接触。

## 动机（为何是第二十四维）

前 23 维覆盖：数值锚点（r790/815/816/822/825/826）、渲染/引用/交叉引用（r817/827/828/830）、时间序/量化词/选择报告（r820/829/831）、近邻/跨文档/编号方程/单位计数/声明合规（r832/834/835/836/837）、README 自声明（r833）。**正交缺口**：论文声称的**训练配置本身**（η 网格、T/split、batch size、momentum/wd、架构、数据源、块比例、种子数、运行时、曝光范围）是否与实际执行脚本+落盘 JSON 一致。hostile reviewer 复现视角的第一入口就是照附录配置重跑——配置声称若与脚本默认值不符，全部结果不可复现。consistency_r790 核结果数字不核配置；r833 核 README 环境/文件计数不核训练超参。本维补该缺口。

## 协议

脚本 `hyperparam_audit_r838.py`（exit 0=全过）。L0 内存态故障注入（改字符串副本，证实检测器能抓声称-脚本失配），再对冻结包跑 L1-L5 主审计。

## 结果（28 pass / 0 fail，5 层 + 自校验；1 新 flag R1）

**L0 自校验 2/2**：注入 `eta_hi 0.05→0.5`（pilot19 脚本副本）被抓；注入 `100\%→50\%`（tex 副本）被抓。

**L1 图像臂五元组 4/4**：tex 附录声称 `240 epochs, split 120, η_hi=0.05, η_lo=0.005, bs 256, momentum 0`。实测 7 脚本（pilot17_mnist_fashion/pilot18/pilot19/diag20/eqra×3）train() 默认参数五元组逐字段精确一致；全部 8 处 `torch.optim.SGD` 构造均 `momentum=0.0, weight_decay=0.0`。off-task 臂 pilot17_cifar10n.py 独立五元组 bs=128（η/T/split 同体系）——**该脚本不属于附录五元组枚举集**（枚举=pilot17/18/19, diag20, eqra_*），无冲突（见"判定"节）。

**L2 架构/数据集 5/5**：tex 称 SmallCNN/small CNN——实测 pilot19 版 3conv+3通道（CIFAR 32×32×3）、pilot17 版 2conv+1通道（MNIST 28×28），脚本类名均 SmallCNN；数据源 URL（MNIST/Fashion-MNIST S3+GitHub）与 pilot19 JSON `data_provenance`（CIFAR-10_human.pt, Wei et al. 2022, cached 2026-08-07 + official CIFAR-10 tarball）一致。

**L3 种子/块比例/T 5/5**：三臂 JSON `seeds=[0..5]`、`T=240`、`quick=false` 落盘；pilot17 nD=nA=60000 ↔ tex `100\% of A`；pilot18 dfrac=0.03/nD=1800 ↔ tex `3\% of A`；pilot19 dfrac=0.10/nA=15000/nD=1500 ↔ 正文 `10%≈10%` 口径。

**L4 曝光/运行时 9/9**：`E_full=0.8*120+0.08*120=105.6` 在 verify_t1_invariance.py 脚本层精确重算 ↔ tex `E=105.6`（L214/218/371）；「200× budget range」由 verify_t1_invariance2 的 frac 扫描（0.005→1.0）坐实：实测区间 [0.528,105.6]、ratio=199.2、全扫描 damage range max=0.000125 ≤ tex 声称 0.0004。正文无 0.8/0.08 与图像体系混淆（合成 η 仅存在脚本层，tex 从不声称数值）。

**L4 flag R1（第15例，展示层/口径粒度）**：tex 附录声称 `~20 min per full 6-seed sweep on one H100`。一手 JSON `runtime_min` 实测：pilot19=25.4、eqra_salvage=12.2、eqra_loss=8.3、eqra_loss_p3=19.3（均与 ~20 min 相容，25% 偏差内）；但 **pilot17_realconflict=66.1、pilot18_smallconflict=51.8**（60k MNIST 全量臂，超声称 2.6–3.3×）。README §3 对应行仅写「GPU，全量 6 种子」无数值声称，矛盾仅限 tex 附录一句。**零研究主张影响**（运行时不入任何定理/结果；复现者只是多等 40 分钟）；属诚实性措辞——「~20 min」对最昂贵两臂低估 3×。备答辩预案：解冻后改为「~10–25 min per sweep（CIFAR-10N/eqra 臂）；60–70 min（60k-MNIST 全量臂 pilot17/18）」或删数值留「minutes-scale on one H100」。挂 MGR 授权窗口与 S1/C1/M1/D1-D3 同批收口。

**L5 双 η 体系区分 3/3**：合成体系 0.8/0.08 仅在脚本层（pilot_phase_datavalue/pilot2/pilot11）；附录五元组（0.05/0.005）实测归属「Real-image arms (one GPU)」段落内部；合成段落（L953）不带任何 η 数值——两套体系在文档层无串位，复现者不会拿错配置。

## 审计器自身修正（3 例，非稿件缺陷；累计 16 例）

1. **注释剥离器吃掉 `\%`**：初版按裸 `%` 截断行，把 LaTeX 转义百分号当注释起点，导致 `100\% of $A$` 类声称在 tex 侧不可见（L3 两例假 FAIL）。修为逐字符扫描、仅未转义 `%` 起注释。
2. **SGD 调用正则 `[^)]*` 遇嵌套括号截断**：`SGD(model.parameters(), lr=..., momentum=0.0, ...)` 中 `model.parameters()` 的内层 `)` 使匹配提前终止，误判 momentum/wd 缺失（8 例假 FAIL）。修为 DOTALL+行尾锚定。
3. **段落锚正则 `\}\.` 与实际文本 `.\}` 字序不符**（`per run).}`）：L5 假 FAIL。先读原始字节再定正则，不凭记忆写锚点。

## 判定

训练配置/超参数维度**全部声称与一手脚本/JSON 一致**（28/28），唯一发现是运行时口径 flag R1（附录「~20 min/sweep」对两个 60k-MNIST 臂低估 2.6–3.3×，零研究主张影响，备答辩措辞已备）。off-task 臂 bs=128 与附录 bs=256 的表观差异经枚举集归属分析确认**非冲突**（附录五元组显式枚举适用脚本，pilot17_cifar10n 不在其列；tex 对该臂仅引用负结果定性结论不带配置）。hostile reviewer 照附录配置重跑 7 个枚举脚本将得到与论文一致的设置。

## 教训(ah)固化

训练配置是独立于结果数值的复现攻击面——结果数字全对≠配置声称对（hostile 照附录重跑时第一层接触的就是配置）。LaTeX 审计的注释剥离必须区分转义 `\%`（字面量）与裸 `%`（注释），否则把合法声称从审计视野中抹掉（与 r836「90含9」假阳性同族：token 匹配须理解语言层转义）。运行时声称要按臂的实际数据规模分档核：单句「~X min per sweep」对异质臂（15k CIFAR vs 60k MNIST）天然有 3× 离散度，声称方应给区间或按最慢臂锚定。枚举式配置声称（"脚本 a/b/c: 配置 X"）的适用范围须逐脚本核对——集外脚本的偏差参数不构成矛盾，但必须在审计中显式判定归属而非忽略。
