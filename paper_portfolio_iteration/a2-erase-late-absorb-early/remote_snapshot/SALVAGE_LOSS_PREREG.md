# SALVAGE-LOSS 预注册：per-sample loss quarantine（EQRA 的 fallback）

触发：EQRA-cos 首轮负信号（cos 在 warmup 期 epoch<60 为正，损伤已进入吸引子时才转负，
隔离太晚）。机制诊断（diag20 H-b REFUTED）显示 cos 不是好早期信号。fallback=per-sample
loss：真实人类噪声样本（aggre≠clean）在 warmup 高 LR 期 loss 显著高于清洁样本（模型
尚未拟合噪声标签）——这是经典 noise-detection 信号（Arazo/MentorNet/Co-teaching 谱系）。
本文件在 EQRA-loss 启动前写定，阈值/端点同 SALVAGE_PREREG，禁止事后改。

## 方法（最小实现）
EQRA-loss：warmup 期（t<split）每 epoch 末按当前模型算每个训练样本 loss，把 loss 最高
的 q_frac=10%（=D 占比，已知；真实场景用 GMM/阈值自适应）标记为 quarantine，下一 epoch
降权 w_q=0；LR 衰减时（t>=split）若样本 loss 降至清洁分布内则 re-admit。
与 EQRA-cos 唯一区别=检测信号（per-sample loss vs block 梯度 cos），隔离/再接纳框架同。

## 预注册端点（与 SALVAGE_PREREG 完全一致）
- P1：EQRA-loss acc ≥ fixed-drop@120 + 0.02；P2：≥ never − 0.01；P3（受控臂不误伤）。
- 停止线：EQRA-loss < drop@120 + 0.005 → NEGATIVE-ASSET，论文停在可逆性边界+机制。
- 公平性：6 seeds 同 pilot19 划分，同 240 epoch 同 CNN 同 WSD，配对差，均值±std。
- q_frac=10% 启动前定（=D 真实占比，非看结果调）。

## 诚实声明
这是"用经典 noise-detection 信号做早期隔离"——若过 P1，贡献是"收缩预算律预测何块
会被吸收+早期 loss 隔离阻止吸收"的理论-方法闭环；novelty 在理论驱动而非检测信号本身
（loss-based detection 是已知技术，论文如实引用 Arazo/MentorNet/Co-teaching）。
