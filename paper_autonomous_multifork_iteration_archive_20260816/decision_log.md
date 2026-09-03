# 版本决策日志

## 2026-08-15
- 决定以 `RESULTS_GPU_QWEN35_VLLM_PAGED_MULTIFORK_RESIDENT_2026-08-14_ZH.md` 为主要证据源，先做一版可复现中文草稿。
- 决定不引入任何与实验边界不一致的并发吞吐或下游任务结论，保留明确“不可以声称”清单。
- 决定使用3位独立审稿人角色替代5位（资源限制），并在 state 中记录 `degraded_mode=true`。
- 决定保留 claim 口径为:
  - same-kernel exactness（C1）
  - cross-N prefix isolation（C2）
  - 解析容量线性节省公式（C3/C4）
  - 分配器口径差值（C5）
- 下一步将启动 snapshot + 盲审，按 review schema 生成 issue ledger 与修订。

## 2026-08-15（自迭代第1轮）
- 按 `RESULTS_GPU_QWEN35_VLLM_PAGED_MULTIFORK_RESIDENT_2026-08-14_ZH.md` 做一次约束增强修订：
  - 明确“单流 round-major”、无 ABBA、无 scheduler、无 timing speedup 这条执行边界；
  - 在 reproducibility/治理部分补充验证闭环的完整要素（query bank/query provenance/raw 重放与窗口 hash）；
  - 修正文案“未支持外延”与可复现性描述中的边界歧义；
  - 修正 `windows` 拼写和可追踪性注释。
- 使用 `xelatex -interaction=nonstopmode -halt-on-error manuscript.tex` 编译成功，生成 4 页 PDF。

## 2026-08-15（继续迭代 + 编译）
- 用 `review/round_03` 的最新评分快照（median=6.8）与问题清单补齐 `score_trajectory.json` 与 `build/build_record.json`。
- 将 manuscript 重编 3 次（xelatex + bibtex + xelatex×2）并复制为 `main.pdf`，最终页数为 6 页。
- 同步提交快照中的 `paper_state.json`、`issue_ledger.json` 与 `build_record.json`，保持主线与 round_03 快照元数据一致。
