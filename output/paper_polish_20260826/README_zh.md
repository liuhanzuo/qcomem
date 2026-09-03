# 14 篇论文润色交付说明（2026-08-26）

本目录交付 14 篇论文：`A1`、`A10`、`A14`、`A23`、`A31`、`A32`、`C1`、`C2`、`C3`、`P1`、`P3`、`P5`、`P7`、`S1`。

每篇论文的**选定最新版**由 `review_snapshot_manifest.tsv` 中的 `snapshot_sha256` 唯一标识，对应只读冻结件：

```text
frozen/<ID>/<snapshot_sha256>/paper.pdf
```

`work/<ID>/source/main.pdf` 是工作区编译产物，不等同于选定最新版，也不应覆盖冻结件。若源码再次修改，必须重新编译、完成质量检查、计算新 SHA，并显式建立新的冻结快照后，才能把它视为新版本。

## 目录结构与查找方式

```text
paper_polish_20260826/
├── source_manifest.tsv              # 本轮选中输入 PDF、输入 SHA/页数、来源状态
├── qa_manifest.tsv                  # 最终 PDF 页数、主文页界、视觉与构建状态
├── review_snapshot_manifest.tsv     # ID 到最终冻结 PDF SHA 的绑定
├── frozen/
│   └── <ID>/<snapshot_sha256>/paper.pdf
├── work/
│   └── <ID>/
│       ├── source/                  # 可编辑 LaTeX 工程；根入口见下文
│       ├── revision_log.md          # 本轮修订日志
│       ├── needs_verification.md    # 尚需作者/证据核验的事项
│       └── semantic_lock.md         # 数值、结论边界、术语等语义锁
└── reviews/
    ├── R1/<ID>.json
    ├── R2/<ID>.json
    └── R3/<ID>.json
```

- 最终 PDF：按 `review_snapshot_manifest.tsv` 的 SHA 进入 `frozen/<ID>/<SHA>/paper.pdf`。
- 可编辑 LaTeX：`work/<ID>/source/`。14 个工程实际根入口均为各自目录内的 `main.tex`；必须在该 `source/` 目录中编译，不能只复制入口文件，否则可能丢失图、样式、章节或附录依赖。
- 修订内容：`work/<ID>/revision_log.md`。
- 未决核验：`work/<ID>/needs_verification.md`。
- 语义锁：`work/<ID>/semantic_lock.md`。继续修改前先读取并保持其中的数字、方向、限定词、实验状态、引用键和结论边界。

## 本轮选中输入 PDF

以下为 `source_manifest.tsv` 固定的输入版本。文件名中的 `(1)` 是所选文件名的一部分，不应自行替换为同名的无后缀版本。

| ID | 选中输入文件 | 页数 | 输入 PDF SHA-256 |
|---|---|---:|---|
| A1 | `A1-paper.pdf` | 23 | `7dcc065c0af6c2d5f6117fda524cb0aa1d5f39426d12cff8b4e107a4ce7803e3` |
| A10 | `A10-paper.pdf` | 10 | `8f7e4ebdc7c48e6ea427b858df4e96e5ff9cbf3d5273e3f32f83f35759eb348e` |
| A14 | `A14-paper (1).pdf` | 19 | `8d6123738eceb7fd64054672744d55b90881e0f4ad2bad80f95415478175066a` |
| A23 | `A23-paper.pdf` | 15 | `9397d82fc8982994a36709d3992c1e73606c2e5618c7ad45da95f078335b740e` |
| A31 | `A31-paper (1).pdf` | 19 | `5289ce4e03e32b243befd8ccea394529b9a81912ab2b0ec11fabbdfe86a17f90` |
| A32 | `A32-paper.pdf` | 12 | `b64f0a58ef38a6f82897aac6fc4c6278103fa67c1f6d1645df59c63bf62cffe3` |
| C1 | `C1-paper (1).pdf` | 28 | `ad12a865f6aa40b345319158357dcd696564c0825ac320ab9b07b9a75a1b8a57` |
| C2 | `C2-paper.pdf` | 30 | `7c8a617ddf4364056b71e15e97d7ee03421aba18edca01d94c53756a498fd6ef` |
| C3 | `C3-paper.pdf` | 14 | `6e4109840e80f6ee0e92142b717cee0ccf08393b13cd75d090a0684665f9ec01` |
| P1 | `P1-paper (1).pdf` | 25 | `bad75b40153309d97837b9a34bca37db62959570369c191e6f973ac5258fb303` |
| P3 | `P3-paper (1).pdf` | 10 | `06510f2d83e09a7647f7efcd1abfadc4350da3438b5d7a23452297ccf9d84c5b` |
| P5 | `P5-paper.pdf` | 9 | `9dfd6772ebab424f5da4d623fc44b8257ec7fa58951a6e3b877faac792850003` |
| P7 | `P7-paper.pdf` | 12 | `a6e9c44d17e1e89d1a2987c83883f59c8e2ce524762476f77624b791f11fe859` |
| S1 | `S1-paper (1).pdf` | 17 | `80a3b9dd0c1e49b863330f9f71cce333321d0b8bfe78d3375cc3c1ca81070a64` |

## 最终冻结 PDF：选定最新版、SHA 与页数

| ID | 最终页数 | 最终 PDF SHA-256（即选定快照） | 来源状态 |
|---|---:|---|---|
| A1 | 23 | `5bd024b4e45e0b174fcb88835505ca50a995ce03d2d71f68a3ca0ec19706cb40` | 精确 LaTeX |
| A10 | 10 | `3ec51ae6ffcf87653a3cf81af085bb834be3c6c87b953736487d0e0c9fe089b1` | 精确 LaTeX |
| A14 | 19 | `d97f36aebfe3377236c0acc84ee88257d4b704e0d298ad74e88ccb435793821f` | 精确 LaTeX 快照；附录为保守重建 |
| A23 | 14 | `b564bcaf9926acc12ba3c30bbf4ff09ab48b02dd88130b866bf3ef5eb432e625` | PDF-only 来源；同谱系源码重建 |
| A31 | 18 | `090599d59e42d3dbe36d9cf724ddc469f317f9ea9065ab90a58b321bda30aa6d` | 精确 LaTeX 快照 |
| A32 | 12 | `c557217eba583f4b47559e4fcbc1c75c71f688febb1f9c633cbdab4d45906277` | 精确 LaTeX 快照；资产需 repin |
| C1 | 28 | `319da71b563edd52d9f4d3e6303bcedcc188048744461a33c9d87c7bebb866dc` | 精确 LaTeX 快照；资产需 repin |
| C2 | 28 | `33380752971af33cdffd66c902f0b47bc87490a3e29c35c63920b2751cc5f13e` | 精确、自包含 LaTeX 快照 |
| C3 | 14 | `b1eed7ac44c179e68b6d31e3b7235d19ddb1094fc99de461246f36a1de2d5edb` | 精确、自包含 LaTeX 快照 |
| P1 | 25 | `074b0ec75463f223d717e02f07b0e792e9d8865bc2a08328ea8163f2ef081a88` | 精确 `main.tex`，依赖树已重建 |
| P3 | 8 | `c047b9974119a16a69aaf7981a3621e3dfcb0cf47672cabb075f6fad8cdf4b83` | PDF-only；历史 TeX 被覆盖后重建 |
| P5 | 10 | `61cb32cfc2143449d9d418c67ae71814cbca1a9fda188facac641abe0285f565` | 精确 LaTeX |
| P7 | 12 | `9a9cd2f2f253ea8e671235b98dec83387d7c07db00fb6491580825254b7602a2` | 精确 LaTeX |
| S1 | 16 | `a70d0593930038db805ed9ae99c770ea52d4f7efec5155f1a4689d3b61482ac7` | 精确冻结 LaTeX 快照 |

以上 SHA 已与冻结目录中的 `paper.pdf` 实际字节核对，页数已与 PDF 元数据及 `qa_manifest.tsv` 交叉核对。14 份最终 PDF 的视觉检查均为 `pass`。A1 的构建带非致命浮动体警告；A10、A14、A23 带 underfull 警告；其余构建状态为 `pass`。

## 重新编译

### 实际入口与已记录流程

| ID | 实际根入口 | 采用的重编译流程 |
|---|---|---|
| A1 | `work/A1/source/main.tex` | 保守的 pdfLaTeX + BibTeX 流程；允许材料未记录历史命令原文 |
| A10 | `work/A10/source/main.tex` | 保守的 pdfLaTeX + BibTeX 流程；允许材料未记录历史命令原文 |
| A14 | `work/A14/source/main.tex` | 保守的 pdfLaTeX + BibTeX 流程；同时保留 `appendix_refine5.tex` 与全部资产 |
| A23 | `work/A23/source/main.tex` | 保守的 pdfLaTeX + BibTeX 流程；这是 PDF-only 同谱系重建工程 |
| A31 | `work/A31/source/main.tex` | 保守的 pdfLaTeX + BibTeX 流程；允许材料未记录历史命令原文 |
| A32 | `work/A32/source/main.tex` | 已记录：pdfLaTeX、BibTeX、两次最终 pdfLaTeX，禁用 shell escape |
| C1 | `work/C1/source/main.tex` | 已记录：pdfLaTeX、BibTeX、两次最终 pdfLaTeX，禁用 shell escape；保留 `sections/` |
| C2 | `work/C2/source/main.tex` | 已记录：pdfLaTeX、BibTeX、两次最终 pdfLaTeX，禁用 shell escape |
| C3 | `work/C3/source/main.tex` | 已记录：pdfLaTeX、BibTeX、两次最终 pdfLaTeX，禁用 shell escape |
| P1 | `work/P1/source/main.tex` | 已记录：pdfLaTeX、BibTeX、两次最终 pdfLaTeX，禁用 shell escape |
| P3 | `work/P3/source/main.tex` | 已记录：通过 `latexmk` 使用 pdfLaTeX、禁用 shell escape，完整构建两遍 |
| P5 | `work/P5/source/main.tex` | 已记录：通过 `latexmk` 使用 pdfLaTeX、禁用 shell escape，完整构建两遍 |
| P7 | `work/P7/source/main.tex` | 已记录：通过 `latexmk` 使用 pdfLaTeX、禁用 shell escape，完整构建两遍；保留本地附录/表格输入 |
| S1 | `work/S1/source/main.tex` | 已记录：通过 `latexmk` 使用 pdfLaTeX、禁用 shell escape，完整构建两遍 |

不要从交付根目录直接编译。先进入对应工程：

```bash
cd /Users/liuhanzuo/MacLLM-Bench/output/paper_polish_20260826/work/<ID>/source
```

对 `A32`、`C1`、`C2`、`C3`、`P1`，按修订日志记录的顺序执行：

```bash
pdflatex -no-shell-escape main.tex
bibtex main
pdflatex -no-shell-escape main.tex
pdflatex -no-shell-escape main.tex
```

对 `P3`、`P5`、`P7`、`S1`，按记录通过 `latexmk` 驱动 pdfLaTeX，并完整构建两遍：

```bash
latexmk -pdf -pdflatex='pdflatex -no-shell-escape %O %S' main.tex
latexmk -pdf -pdflatex='pdflatex -no-shell-escape %O %S' main.tex
```

对 `A1`、`A10`、`A14`、`A23`、`A31`，可信交付材料确认了 `main.tex` 入口和成功构建，但没有记录历史命令原文。需要本地重编译时，可在工程副本中采用下面的**保守复现流程**；它是建议流程，不冒充历史命令：

```bash
pdflatex -no-shell-escape main.tex
bibtex main
pdflatex -no-shell-escape main.tex
pdflatex -no-shell-escape main.tex
```

编译后得到 `work/<ID>/source/main.pdf`。应重新检查页数、未解析引用/交叉引用、溢出、图表缺失与视觉版式；在完成重新冻结前，不得用它替换上述 SHA 对应的 `frozen/.../paper.pdf`。

## 必须保留的交付限制

### A1：九页主文限制是提交阻断项

A1 的参考文献从最终 PDF 第 13 页开始，主文约 12 页，比名义上的 ICLR 九页主文上限多约 3 页。当前 23 页冻结件通过了视觉与构建检查，但**不满足九页主文限制，不能按九页要求直接提交**。要压至九页，需要移动承载证据的表格和审计细节，属于高影响改稿；本轮未声称已解决。

### A23：PDF-only 同谱系重建

A23 所选输入 PDF 为 15 页，但对应的精确源码未被保存。最终 14 页 PDF 来自最接近的活动同谱系源码，不能视为对输入 PDF 每句话、每个浮动体位置或版式的精确源码复现。遇到两者的 claim-level 差异时，以所选输入 PDF 为内容权威；本轮没有猜测或静默补写缺失段落。

### P3：从 PDF 保守重建

P3 的历史精确 TeX 已不可用/被覆盖，最终 8 页工程依据所选 10 页渲染 PDF 重建。科学内容按 PDF 保守转录，但不能保证原始引用键、行号、浮动体位置或字节级源码一致；两幅图为输入 PDF 的忠实裁剪/提取，而非由缺失绘图源码重新生成。所选输入 PDF 仍是科学内容权威。

### A32、C1：资产需要 repin

A32 与 C1 的 LaTeX 文本来自精确快照，但来源状态明确标记为 `assets_repin_required`。当前 SHA 对应的冻结 PDF 可以作为本轮审阅对象；若要从源码形成新的可归档/可提交版本，须先把所有外部或易漂移资产重新固定到明确版本，确认引用解析到预期文件，再重新编译、视觉复核和冻结。未完成 repin 前，不应把一次新的本地构建称为可重复的替代快照。

### 其他需核验事项

每篇论文的未决证据、复现、文献或作者决策均在 `work/<ID>/needs_verification.md`。语言润色没有把未运行实验、未核验文献、缺失原始记录或未来工作改写为已完成证据。

## 审稿 JSON 隔离规则

三个审稿轮次在物理目录上隔离：

```text
reviews/R1/<ID>.json
reviews/R2/<ID>.json
reviews/R3/<ID>.json
```

每个 JSON 必须同时按论文 ID 和轮次解释，并通过 `review_snapshot_manifest.tsv` 绑定到该 ID 的唯一 `snapshot_sha256`。不同 ID、不同轮次或不同 PDF SHA 的记录不得合并、覆盖或交叉归因；如果 PDF SHA 改变，应建立新的快照绑定并重新审阅，而不是沿用旧 JSON。本说明不读取或汇总任何审稿分数；评分汇总由独立交付步骤生成。

## 完整性核对建议

交付或搬运后，至少核对以下四点：

1. `review_snapshot_manifest.tsv` 中的 14 个 SHA 均存在对应的 `frozen/<ID>/<SHA>/paper.pdf`。
2. 每个冻结 PDF 的实际 SHA 与上表完全一致，页数与 `qa_manifest.tsv` 一致。
3. 每个 `work/<ID>/` 都保留 `source/`、`revision_log.md`、`needs_verification.md` 和 `semantic_lock.md`。
4. 审稿 JSON 仍按 R1/R2/R3 和 ID 隔离，并指向同一份冻结快照；不要把工作区 `main.pdf` 当作审稿对象。
