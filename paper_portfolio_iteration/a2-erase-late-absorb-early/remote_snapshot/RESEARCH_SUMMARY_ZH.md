# 中文研究总结 — EraseLateAbsorbEarly（LR 阶段依赖的数据吸收/擦除不同构理论）

研究员 A2，方向注册：数据价值/数据选择/训练动态理论。主稿 `paper/paper.tex`（编译 0 错 0 未定义引用，16 页 = 正文 9 页满 [计数口径细粒度 8.88：编号 Limitations 起 p8、Conclusion 终 8.70、References 起 p10 行86/99] + 豁免声明约 2.88 页 [Reproducibility/Ethics/AI-use 三个 \section* 合计] + 参考文献约 1 页 + 附录约 4 页 [p11 起]；r850 AI-use 任务级披露扩段 +1 页，r863 细则同步，pdfinfo 实测 16 页锚定不变）。

## 一句话贡献

数据价值不是样本的端点属性，而是训练路径的泛函：有害块在高 LR 窗口被"吸收"进表示，
在低 LR 尾部可被"免费擦除"——吸收与擦除不同构；该非对称性由标量收缩预算
B = Σ_steps η（= K·E_win = N_win·η）控制；理论导出"延迟检测 + 预算规则"方法，
并在真实人类标注噪声（CIFAR-10N）上发现可逆性边界，最终由理论指导的早期 loss 隔离
（EQRA-loss）把滞后（hysteresis）regime 转回可逆 regime，取得预注册正收益。

## 理论贡献（三定理 + 一个引理，凸 logistic cell）

- **T1（schedule/K 不变性）**：always-on 损伤只是累积暴露 E(s) 的函数，与 LR schedule
  形状、确定性更新计数 K 无关（per-seed 逐位一致）。
- **T2（免费擦除）+ Lemma C1**：drop 后损伤以 exp(−λ_eff·E_tail) 回归吸引子，机制钉在
  强凸吸引子回归（不是 ridge 下界、不是 logistic 曲率形状；diag5 决定性证伪两个流行归因）。
- **T3（first-cause/recency 反转 = 收缩预算 crossing）**：反转由标量预算 B 单独控制，
  优化器身份与梯度噪声非因果（pilot 12/13/14 三重证伪流行归因）；无参数预测
  1−e^(−λ_path·B)，λ_path=0.2023 由 A3 独立复算验证；F2 修复后定理按 per-step 暴露
  预算陈述，proof 补无放回 shuffle 鞅差条件（Azuma–Hoeffding）。
- **统一（§3.4）**：transient vs equilibrium 视角统一三定理。

## 理论导出的方法（§4）

1. **延迟检测规则**：受控腐化 regime 下，检测/删除可推迟到 LR 衰减前，恢复质量无损
   （R2 恢复 1.012/1.069 ≥ 0.7 预注册通过）。
2. **预算规则**：B 预测 late injection 是否存活（pilot 15 双向 budget switch）。
3. **EQRA-loss（salvage 正端点）**：预算律说损伤在高 LR 窗口被吸收 → warmup 期把最高
   loss 10%（=真实噪声占比，预注册非调参）样本隔离，LR 衰减时 re-admit。

## 实验结论（全部预注册，6 种子，阈值未改）

- 凸 cell（pilots 1–16）：T1/T2/T3 预注册预测全过；三个流行机制归因被证伪（SGD 噪声、
  noise floor、epoch 内 batch 结构）。
- 真实图像臂（pilots 17/18/19）：受控腐化（MNIST shuffle/cyclic）R1/R2 过；
  **真实人类标注分歧（CIFAR-10N）R2 预注册失败（recovery 0.240 < 0.7）——诚实报告，
  不包装**：可逆性边界依赖于数据模态/模型/腐化类型。
- diag20 机制定位：hysteresis = 持久表示记忆 + 持续负冲突 + 晚期步代价高；
  fixed-drop 网格 s=90..210 最大恢复 0.406，无 detox 死线。
- **Gate C 正端点**：EQRA-loss +0.0243 vs 最强 baseline（fixed-drop），5/6 种子正，
  距 never 上界仅 −0.0083（P1/P2 双过）。同框架下 cos 信号失败（P1/P2 双负）——
  检测信号是决定性变量，正负结果都如实写入论文。
- **P3 精确性边界（诚实负）**：无害清洁块上 EQRA-loss −0.026（6/6 种子）——q_frac 须
  匹配真实噪声率才精确；路由规则：噪声率可估且不可忽略 → EQRA-loss，否则 fixed-drop。

## 局限（论文 §7 如实声明）

凸理论定量不点态迁移到真实像素（R3 report-only）；真实人类噪声不可免费擦除；
EQRA-loss 对难例有假阳性代价（q_frac 失配时）；MLP+GD 擦除时间尺度条件依赖；
单卡轻量实验规模有限。

## AI 使用声明（ICLR 2027 要求，r850 闭环）

任务级披露已写入 `ai_use_statement.tex` 并 `\input` 进主稿 PDF（§AI usage statement；
三个豁免 statement——Reproducibility / Ethics / AI-use——合计约 2.88 页，
sha256 `89101b36…`）：AI 辅助 6 项（选题/理论/实验代码/数据绘图/手稿/文献）、纯人工 3 项
（任务规范与资源边界/最终责任/方向与门控决策）、共享 3 项（A3 r787 与 A6 r786 证明审计、
A2+A6 引用互查、MGR 协调）。science 内容零改动，仅披露细化（MGR 指令 `8f935df34e73`）。
正文计数口径（r863 更正）：到 References 起点 = 8.88 页 ≤ 9 硬门，编号 Limitations 起 p8
在正文 span 内（A1 r743 官方豁免口径 trivial-pass；详见 PAGE_COUNT_SCOPE_AUDIT_R863.md）。

## 复现命令

- 凸 cell（CPU 秒级）：`lr_phase_datavalue_r1/pilot1..16, diag4..9, verify_*.py`，
  6 种子锁定，端点预注册于各 PREREGISTER 文件。
- 真实图像臂（单 H100，每全量 6 种子约 20 分钟）：`pilot17/18/19, diag20,
  eqra_salvage.py, eqra_loss_salvage.py, eqra_loss_p3_precision.py`；
  SmallCNN + WSD（240ep, split 120, η_hi 0.05, η_lo 0.005, bs 256）。
- 判读文件：GATEC_VERDICT_R785.md / DIAG20_VERDICT.md / EQRA_COS_VERDICT.md /
  EQRA_LOSS_VERDICT.md / EQRA_LOSS_P3_VERDICT.md（阈值全程未改）。
- 原始输出：每个 run 对应 `*_out.json` 与脚本同目录。

## 未解决风险

- T1 尚缺第二名独立审计（A3 审 T2/T3 已通过关闭；A1 复审待 MGR 调度）。
- q_frac 自适应（GMM/mixture-model 估计噪声率）记为未来工作，未实现。
- ~~主稿迁移至 run 根目录 paper/~~ 已关闭（r792 迁移完成，MGR 授权 `5c5bd6630e1f`）；
  Figure 1 AutoFigure 重绘已备预案（r848 `edit_a2_fig1.py` 刷新至当前论文三面板，
  py_compile 过未调 API），仍待 MGR 裁决是否执行。
