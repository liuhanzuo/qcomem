# README 链接审计（r810）— REPRO_README §3 逐脚本重跑表 vs code/ 实际文件

**动机**：此前全部核验覆盖「tex 数字主张 ↔ JSON」（consistency_r790，154 项）与「README 命名脚本 → code/ 存在性」，但从未做**反向**核对：README §3 重跑表中的脚本名/通配符是否真能在 code/ 中匹配到文件（审稿人按表找脚本的可执行性）。本轮补上，全部实测非抽样。

**方法**：
1. 提取 README 全部 `*.py` 命名 → 逐一查 code/ 存在性（正向）。
2. 列出 code/ 全部 40 脚本 → 逐一查是否被 README 命名（反向）。
3. 对 README 表中的通配符（`pilot17_realconflict*`、`diag8_*`、`diag9_*`、`eqra_*`、`verify_t1_invariance{,2}`）逐一做 shell glob 展开实测（模拟审稿人 `ls code/<pattern>`）。

## 结果

**正向（README 命名 → code/ 存在）：零缺失。** README 命名的 17 个具体脚本全部存在于 code/。

**反向（code/ 脚本 → README 命名）：24 个未被逐一命名**——pilot1/3/4/5/6/7/10/12/13/14/15/16、pilot17_budget_mid、pilot17_cifar10n、diag4/5/6/7/8/9、diag_seam、analyze_pilot1718 等。**经核非缺陷**：README §3 表用分组覆盖（「`pilot11_lambda_traj.py` 等 pilot10–16」「`diag20_traj.py` / `diag8_*` / `diag9_*`」），且 §4 声明 code/ 含 40 脚本全量；分组行属概括非断链。判断：不阻断，可改进但非必须。

**通配符展开实测（唯一真实缺陷）**：

| README 表条目 | shell glob 实测 | 判定 |
|---|---|---|
| `pilot11_lambda_traj.py` 等 pilot10–16 | pilot10–16 各自存在 | OK（分组） |
| `pilot17_realconflict*.py` | **`ls: cannot access 'code/pilot17_realconflict*.py': No such file or directory`** | **断链** |
| `pilot18_smallconflict.py` | 存在 | OK |
| `pilot19_cifar10n.py` | 存在 | OK |
| `eqra_salvage.py` / `eqra_loss_salvage.py` / `eqra_loss_p3_precision.py` | 三者均存在 | OK |
| `diag20_traj.py` / `diag8_*` / `diag9_*` | diag20_traj.py、diag8_stationary_cov.py、diag9_commutator.py 均存在 | OK |
| `verify_t1_invariance.py` / `verify_t1_invariance2.py` | 两者均存在 | OK |

## 缺陷明细（唯一）

- **README §3 表第 47 行**：`` `pilot17_realconflict*.py` (内部调 pilot17_mnist_fashion / pilot17_cifar10n 等) `` 产出 `pilot17_realconflict_full_out.json`。
- **实测**：code/ 中**不存在**任何匹配 `pilot17_realconflict*.py` 的脚本——`pilot17_realconflict_full_out.json` / `_quick_out.json` 是 JSON 产物，其生产者脚本名为 `pilot17_mnist_fashion.py`（line 167 `json.dump(... f"pilot17_realconflict_{tag}_out.json" ...)`，tag=full/quick）；`pilot17_cifar10n.py` 是早期负结果臂（off-task 独立类别冲突，§6.3 limitation L3），analyze_pilot1718.py 是只读判读脚本。
- **影响**：审稿人按表 `ls code/pilot17_realconflict*.py` 失败，重跑表该行不可执行——reproducibility review 的 broken-link 项。括注（「内部调 pilot17_mnist_fashion」）暗示作者意图是给一个 wrapper 名，但该 wrapper 从未存在。
- **修复方案（最小，零研究主张变更）**：该行脚本列改为 `` `pilot17_mnist_fashion.py`（全量 6 种子产出 full；quick 标签产出 quick）/ `pilot17_cifar10n.py`（off-task 负结果臂） ``，产出 JSON 列不变。
- **属性**：README 文档字节，非 paper.tex/pdf 科学字节；但保持谨慎先例（r809 同型），仍请 MGR 授权后执行，五步收尾 + 双口径回执。

**记录时间**：r810，2026-08-09。核验时投稿包状态 = r803 冻结版（tex md5=f60def8a…/pdf md5=f24699c0…，两端 cmp=0）。
