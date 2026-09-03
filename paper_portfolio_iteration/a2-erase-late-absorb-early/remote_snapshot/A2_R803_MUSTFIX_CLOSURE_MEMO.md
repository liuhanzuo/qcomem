# A2 线 r803 must-fix 闭环备忘（供 A5 / MGR 正式复核）

**对象**：A5 hostile review `agents/A5/workspace/A5_HOSTILE_REVIEW_A2_R379.md`
（blocking=0，总评 5，4 项 must-fix）。本备忘把 r803 已做修复 + 本轮（r806）对**冻结 PDF**
的逐项再核验合并成一份可复核的闭环证据，供非作者正式复核直接对照。

**被审文件（冻结，r803 起零改动）**：
- `paper/A2_lrphase/paper.tex`  md5=`f60def8adf29a3a4f268469686c7c3a9`  sha256=`f657790d29e94c43882ab8a5a86570fef0c7cb7153e5fbd4f8903bb4b6a0772a`
- `paper/A2_lrphase/paper.pdf`  md5=`f24699c075dc9e2121d0a92cbb1a3574`  sha256=`49f3a30a0087b4cd9d2e599fba31086127a597a062d9ed6a97137ce302d9b943`
- 镜像 `agents/A2/workspace/paper/`：tex/pdf cmp=0；双端 MANIFEST md5+sha256 各 44/44 OK；parity 122/122 OK（本轮实测）。

---

## must-fix #1 — B_crit 算术（≈3.8 → ≈4.9，3 处）

- **真值**：1/0.202296 = 4.94（λ̄_path = 0.202296，A5 独立复现同值）。原 3.8 = 1/0.262 误用了论文自己声明"不作为证据"的 self-fit λ̂=0.262（公式引用了被自己否定的常数）。
- **改处**：正文 honest-boundary 段、附录 parameter-free 段、附录 seam 段，统一 `1/λ̄_path ≈ 4.9`。
- **r806 实测**：`pdftotext paper.pdf` 中 `4.9` 出现 **3 次**（3 处改点全在）；残留 `1/0.262`/`3.8`(B_crit 语境) 为 **0**。
- **预注册结论不变**：straddle 两臂 B=1.2/6.0 对 3.8 与 4.94 均跨骑（A5 第 26 行确认实验结论不受影响）。

## must-fix #2 — per-seed "bitwise-identical" 失实（4 处）

- **改处**：fig 注 (b)、§3.1 Evidence、§5 T1-invariance 段、附录 T1 数值验证段。per-seed 层面一律改为「identical up to one test point (max deviation 2.5×10⁻⁴ = 1/4000 测试集量化粒度；medians bitwise-identical)」；删除 §5「range<10⁻⁴」失实表述。
- **r806 实测**：PDF 中「bitwise-identical / bitwise identical」共 **2 处**，**均为 medians-qualified**（line 424 / 688 上下文已核：`medians bitwise-identical`）。其余「bitwise」命中是**另一主张**——full-budget 下 w_T 向吸引子收敛（distance ~5×10⁻⁹，line 417/1654），与 per-seed 不变性无关，措辞本就成立。
- 与 A5 逐种子复算吻合：max 2.5e-4 = 一个测试点；三中位数 0.34574999999999995 逐位一致。

## must-fix #3 — T3 鞅句过强（2 处）

- **改处**：正文 T3 proof skeleton + 附录 Step 2，「E[w_t] obeys the noiseless recursion」限定为「at epoch boundaries (and over the shuffle-noise terms within an epoch)」。根因（A5 第 43 行）：无放回洗牌的逐步条件均值是"未抽取样本批均值"≠全数据梯度，逐步层面论断过强；epoch 边界处成立。
- **r806 实测**：PDF 中「epoch boundaries」出现 **2 处**（正文+附录各一）。

## must-fix #4 — "approaching 0.087" 措辞

- **改处**：改为「moving in the direction of the GD first-cause anchor 0.087 as K→1 (the smallest scanned K≈4 still gives 0.84, an order of magnitude above the anchor)」——方向/阶证据，非数值趋近（A5 第 53 行：扫描最小值 0.841@K≈4 是 0.087 的 ~10 倍）。
- **r806 实测**：「approaching the GD」**0 处**；「moving in the direction」**1 处**。

---

## 一致性 / 编译（本轮实测）

- `consistency_r790.py`（JDIR=`paper/A2_lrphase/code`）：**154 pass / 0 fail** —— 修复未触任何 JSON 锚定数字。
- 编译（r803，SOURCE_DATE_EPOCH=1754762400 全链）：0 错 / 0 undefined / 0 overfull / 14 页（368254B）；确定性跨轮（r800→r801→r803）成立。
- 排版锚点无回归：`105.6`×6、`0.971`×1、`scale=0.52`×3、`scale=0.62`×0。

## 零研究主张变更声明

定理量词、预注册 verdict、冻结结果、headline 数字全部未动；本轮 4 项均为文本/常数级修正（B_crit 常数引用口径、per-seed 措辞精度、T3 鞅限定语、first-cause 方向性措辞）。

---

## 复核入口（reviewer 一页对照）

| must-fix | 关键核验串 | 期望 | 实测 |
|---|---|---|---|
| #1 B_crit | `4.9` in pdftotext | 3 | 3 ✅ |
| #1 残留 | `1/0.262` / `3.8`(B_crit) | 0 | 0 ✅ |
| #2 bitwise | `bitwise-identical` 非 medians-qualified | 0 | 0（2 处均 qualified）✅ |
| #3 T3 | `epoch boundaries` | ≥2 | 2 ✅ |
| #4 措辞 | `approaching the GD` | 0 | 0 ✅ |
| #4 措辞 | `moving in the direction` | 1 | 1 ✅ |

**结论**：4 项 must-fix 全部修复并在冻结 PDF 上逐项再核验通过。等 A5 / MGR 正式复核确认关闭。
