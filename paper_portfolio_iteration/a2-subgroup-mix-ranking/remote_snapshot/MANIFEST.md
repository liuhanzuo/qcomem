# MANIFEST — A2_subgroup_mix_ranking_author_candidate_r1915 (successor of r1911)

- 定稿性质：M5 author-candidate 的**单调后继候选**，自含只读发布包（真实资产拷贝，无 hardlink/symlink）。
  承 MGR 指令 **f147b3bf33e9**：从唯一 mutable 主稿直接装配一个单调后继候选，把 r1912 首图与 M10/M11
  段纳入，并依卡要求新增 fresh-5-seed M12 真采样审计（MGR 后续指令顺序为前端审查提供）。
- 装配来源：唯一 mutable 主稿 `subgroup_mix_ranking/paper/paper.tex`（r1915 M12 收口状态，含 r1912 图 +
  r1914 M11 + r1915 M12）。
- 前任候选（全部 exact bytes 只读保留，未触碰）：
  - `paper/A2_subgroup_mix_ranking_author_candidate_r1905`（最早 frozen M5）
  - `paper/A2_subgroup_mix_ranking_author_candidate_r1911`（上一前继）
- 状态：M5-C。正文主拱（9 页）与 r1905 逐字同；app:tau 在 M6+M7+M8+M9+M10+M11 基础上新增 **M12**
  （fresh-5-seed 真采样：M3/M3.5 预算总点上严格相对带 0 真实切换，绝对门守住部署且单调）。
- 本文件只锁**不可变发布包**（承重论文资产 + 证据 JSON + runner + verifier 中的定稿件）。mutable 代码/
  runner/config/防漂移脚本/**不哈希冻结**，可继续研究。

## 承重不可变文件（SHA-256，装配当日 2026-08-18 计算）
| 文件 | SHA-256 |
|---|---|
| paper/paper.tex | 258defa2a33a776fc377f920ef753dc5136e4882fc296c6e517e79e54f7e2d0c |
| paper/paper.pdf | fbcff897281a422fc05fc564b2203b70ebfad571215a28885eccc20325842c4d |
| paper/references.bib | 932bdc05acbe9dc29769251f4710c82b0b3cf1db1b21b98623b861d4e41499b1 |
| paper/fig_m10_frontier.pdf | 147c876bbc4823cae5df41ab3bf0ff20247eadacf9fadfdff9b92e9addaca5dd |
| code/subgmmix_m10_exactband_budget_r1911.py | 5bc0676bac60355427f2fcfa924a256d0b8da116b1524c317f8583a6140bee8c |
| code/subgmmix_m11_allocation_axis_closure_r1914.py | bd033bc27d6efd3b17a77c52d296cf3a4450e18711b9c1216806b197693a765f |
| code/subgmmix_m12_fresh5_budget_r1915.py | 0561bfe4f41ad9f20a1cb379c1a42b2fcfbde728253f068e7501f29c349958c1 |
| results/SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json | efe6885bba6242d2e5c9423e3fb5ae5e2c2bafd6b2b548269f86dd1bfdf31043 |
| results/SUBGMIX_M12_FRESH5_BUDGET_R1915.json | 6bcb5400e29d4b5ccd501367e4839788f91eff3bb80a4ce980c6ea10d7f4319d |
| results/M12_VERIFY_R1915.py | 460289ccb085d11d8aaf1efb117b49470e81574c68fde5f4af73207010b065ac |

## 候选根前台重放（装配当日真实执行，readback）
1. `python3 results/M5_VERIFY_R1902.py` → PASS 105, FAIL 0, EXIT=0
2. `python3 results/M7_VERIFY_R1907.py` → PASS 28, FAIL 0, EXIT=0
3. `python3 results/M8_VERIFY_R1909.py` → 17/17 PASS, EXIT=0
4. `python3 results/M9_VERIFY_R1910.py` → ALL PASS, EXIT=0
5. `python3 results/M10_VERIFY_R1911.py` → ALL PASS (125 infeasible rows), EXIT=0
6. `python3 code/subgmmix_m11_allocation_axis_closure_r1914.py` → (a)-(d) 7/7 断言, EXIT=0
7. `python3 results/M12_VERIFY_R1915.py` → PASS 245, FAIL 0, EXIT=0
8. 干净编译：pdflatex×3+bibtex → paper.pdf **14 页**，正文恰 9（Conclusion p9），References p10，
   fig_m10_frontier 随图 p14；0 LaTeX Error / 0 Overfull / 0 undefined。

## 页数与匿名性（真实 readback）
- `pdfinfo paper.pdf` → **14 页**；正文 Conclusion `newlabel{sec:concl}` 页=**9**（正文恰 9，ICLR 2027 限内）。
- 匿名：`\author{Anonymous}`，双盲无作者/机构/水印。

## r1915 新增内容（相对前驱 r1911）
- 论文：app:tau 在 M10/M11 段后新增 **M12** 段（fresh-5-seed 真采样审计：M3/M3.5 预算总点严格相对带
  0 真实切换、绝对门守住单调、digits/news 墙如实）。
- 证据：results/SUBGMIX_M12_FRESH5_BUDGET_R1915.json + results/M12_VERIFY_R1915.py。
- 已并入 r1912 首图 fig_m10_frontier 与 r1914 M11 全分配轴闭合段（源自 mutable 主稿）。

## 内容与许可
- 全部 code/runner/aggregation/verifier（r1881-r1915，标注回合）+ 全部 SUBGMIX_*.json 证据。
- 数据许可：Fashion-MNIST(MIT)、MNIST、KMNIST(CC-BY-SA-4.0)、sklearn digits(BSD-3)、20News(本地缓存)。
  原始数据未复制入候选；核心聚合 JSON 已自含，复现依赖 REPRO_README。