# 复现 README — EraseLateAbsorbEarly (A2 线, r796)

本目录是 A2 线投稿主稿。本文档说明如何复现论文中全部数字主张。

## 1. 编译论文

```bash
cd paper/A2_lrphase        # 本目录
export SOURCE_DATE_EPOCH=1754762400
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

依赖仅为本目录内随附的 style 文件（`iclr2027_conference.sty/.bst`、`natbib.sty`、`fancyhdr.sty`、`math_commands.tex`）与标准 LaTeX 发行版 + pgfplots。
期望产物：`paper.pdf` 14 页（正文 9 页 + 参考文献 + 附录 A–C），0 error / 0 undefined reference。
r796 实测：`grep "^!" paper.log` = 0，undefined citation/reference = 0。

## 2. 数字链机器核验（最省力的复现入口）

论文正文与附录中的 **154 项数字主张**全部锚定到实验输出 JSON，由随附 checker 逐条比对：

```bash
cd paper/A2_lrphase
CONSISTENCY_JDIR=<json目录> python3 consistency_r790.py
# 期望输出: === r790+r793 consistency: 154 pass, 0 fail ===
```

`<json目录>` 需含本包 `code/` 下的全部 `*_out.json`（即把 `code/` 作为 JDIR 直接传入亦可：
`CONSISTENCY_JDIR=$PWD/code python3 consistency_r790.py`——checker 只读 JSON，不要求代码可运行）。

r796 实测（JDIR 指向作者 workspace 原始实验目录）：154 pass / 0 fail。

## 3. 从头重跑实验

环境：Python 3 + NumPy（凸格/合成实验）；PyTorch ≥2.3 + CUDA GPU 一张（pilot17/18/19 的 CNN 实验）。
作者环境实测：torch 2.3.0a0, numpy 1.24.4。

| 脚本（`code/` 内） | 产出 JSON | 对应论文主张 | 规模 |
|---|---|---|---|
| `pilot2_conflict.py` | pilot2_conflict_out.json | 吸收-擦除对象存在（凸格） | CPU 秒级 |
| `pilot8_lambda_eff.py` | pilot8_lambda_eff_out.json | 附录 B T2 λ_eff 测量 | CPU 秒级 |
| `pilot9_param_resid.py` | pilot9_param_resid_out.json | T2 参数层真回归 | CPU 秒级 |
| `verify_t1_invariance.py` / `verify_t1_invariance2.py` | verify_t1_invariance{,2}_out.json | 附录 B T1 不变性 + 预算饥饿边界（H3 falsifier） | CPU 秒级 |
| `verify_c1_direct.py` | verify_c1_direct_out.json | C1 直接核验（min margin 0.0392 等） | CPU 秒级 |
| `verify_prop2_closedform.py` | verify_prop2_closedform_out.json | T2 闭式核对 | CPU 秒级 |
| `verify_t3_ou.py` | verify_t3_ou_out.json | 附录 B T3 OU 涨落 | CPU 秒级 |
| `pilot11_lambda_traj.py` 等 pilot10–16 | 对应 *_out.json | §6.1 机制诊断（含 pilot11 V1/V3 预注册失败如实报告） | CPU 分钟级 |
| `pilot17_mnist_fashion.py`（full/quick 两臂，realconflict 生产者）+ `pilot17_cifar10n.py`（off-task 负结果臂，§6.3 L3） | pilot17_realconflict_full_out.json, pilot17_budget_mid_out.json | §6.2 预算-穿越律 B_crit 单调表（tab:budget） | GPU，全量 6 种子 |
| `pilot18_smallconflict.py` | pilot18_smallconflict_full_out.json | §6.2 受控冲突臂 | GPU |
| `pilot19_cifar10n.py` | pilot19_cifar10n_full_out.json | §6.2 真实人类噪声臂（R1/R2/R3，预注册 r783） | GPU |
| `eqra_salvage.py` / `eqra_loss_salvage.py` / `eqra_loss_p3_precision.py` | eqra_*_out.json | §6.3 salvage：EQRA-cos 负结果 / EQRA-loss 正结果 / P3 精确性边界 | GPU |
| `diag20_traj.py` / `diag8_*` / `diag9_*` | diag*_out.json | 附录诊断（交换子=0、OU 协方差、轨迹） | CPU/GPU 分钟级 |
| `arxiv_drift_r802.py` | arxiv_drift_r802_out.json | 近邻新颖性的投稿日漂移终检（7 查询 + 15 篇近邻 ID 探针，锚 2026-08-06；需联网） | CPU 分钟级 |

种子与阈值均在各脚本内预注册（先于运行固定），失败端点（pilot11 V1/V3、pilot17_budget_mid M_fit、EQRA-cos 双负）的 verdict 原样保留在 JSON 中，未删改。

### 外部数据（仅 pilot17/19 的真实噪声臂需要）

- CIFAR-10N 人类标注：`CIFAR-10_human.pt`，Wei et al. 2022（cifar-10n 官网，CC-BY），下载后放 `data/CIFAR-10_human.pt`（脚本内 `DATA` 常量，可用环境变量或改一行指向）。
- CIFAR-10 图像：官方 tarball（`data/cifar10`）。
- MNIST/Fashion-MNIST：torchvision 自动下载。
- 全部凸格/合成实验（pilot2–16、verify_*）无需任何外部数据。

## 4. 文件清单

- `paper.tex` / `ref.bib` — 主稿与参考文献（bib 全一手核验，无虚构条目）。
- `consistency_r790.py` — 154 项数字锚定 checker（§2）。
- `code/` — 40 个实验/验证脚本（r802 增投稿日漂移终检 `arxiv_drift_r802.py`）+ 35 个冻结结果 JSON（即论文全部数字与近邻终检的来源；r796 全部 `py_compile` 通过，r802 新增脚本同测）。
- `RESEARCH_SUMMARY_ZH.md` — 中文研究总结（贡献/假设/实验/局限/复现/风险）。
- `SUBMISSION_CHECKLIST.md` — 投稿前核对表（逐项状态与证据）。
- `HOSTILE_SELFREVIEW_R789.md` — 作者 hostile 自审（18 项；不替代正式独立 hostile review）。
- `ai_use_statement.tex` — AI 使用声明（已接入正文末，见 paper.tex 的 `\input{ai_use_statement}`）。
- `MANIFEST_md5.txt` — 提交文件 md5 清单（在本目录内 `md5sum -c MANIFEST_md5.txt` 核验）。
- `build_*.log` — 历次编译日志（r793a–e 为当前稿的构建轨迹）。

## 5. 已知非确定性说明

- `paper.pdf` 的 md5 默认跨编译不稳定（pdflatex 写入时间戳/Producer 元数据）。r800 实测修复：编译前 `export SOURCE_DATE_EPOCH=1754762400`（见 §1）后两次独立全链编译 `md5sum paper.pdf` 逐位一致（bd6b5bfa…），PDF 派生物亦可纳入 md5 清单核验；不设该变量时一致性核验仍应对源文件（.tex/.bib/.sty）而非 PDF。
- CPU 凸格实验在固定种子下逐位可复现；GPU CNN 实验（pilot17–19）受 cuDNN 非确定性影响，论文中均以 6 种子均值+离散度报告，单种子逐位复现不作主张。
