# Q-CoMem supervised SFT 数据与目标构建说明

## 1. 本阶段的边界

本工具只准备监督式问答训练数据，不启动训练、不申请 GPU，也不自动下载任何数据。
正式训练输入只来自官方 `train` split：

- Qasper v0.3 train：888 篇论文、2,593 个原始问题；官方 LED reader 会跳过唯一一篇
  `full_text=[]` 的论文，因此可转换问题固定为 2,590，显式跳过 3 条；
- 2WikiMultiHopQA train：167,454 个问题。

LongBench 的 pilot、calibration、validation、legacy test 都已被实验消费，必须从训练集
排除。frozen test-v2 仍保持 untouched：当前流程不打开它，也不读取它的答案。由于现在
没有预先生成的 blind-hash manifest，清单会明确记录：

```text
heldout_protocol.test_v2_content_hash_check = deferred_not_read
```

训练方案、LoRA 超参和 bit policy 完全冻结后，才可以由独立审计流程提供一个只含 SHA256
的 test-v2 blind manifest。这个 manifest 不能包含 `_id` 原文、context、question 或 answer。

## 2. 官方来源与冻结值

正式 source spec 是
[`configs/supervised_qa_sources.json`](configs/supervised_qa_sources.json)，脚本会同时核对
archive SHA、解压后 train 文件 SHA、split、license 和样本数；任何一项不一致都会失败。

| 数据 | 官方来源 | 固定 revision | license | archive SHA256 | train 文件 SHA256 |
|---|---|---|---|---|---|
| Qasper v0.3 | [AllenAI Qasper](https://huggingface.co/datasets/allenai/qasper/tree/fdc9d8214fbab5dd782958601db4d678e6934a54) | `fdc9d8214fbab5dd782958601db4d678e6934a54` | CC-BY-4.0 | `a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a` | `9458bfe76074a8fa8d1685af02bcc73537aa6d338ad20591dfaff1946bc88bf4` |
| 2WikiMultiHopQA | [官方 GitHub](https://github.com/Alab-NII/2wikimultihop/tree/13800e5be57df1b4040b9b1588c6c811779e69e9) | `13800e5be57df1b4040b9b1588c6c811779e69e9` | Apache-2.0 | `e8e57c0aafc4a26d41131e320ebb5afb6f2aca86b8a6e6611b08f52033cb7d04` | `b3fddb4d5bb42cd797919cad67616545be51b24740e0a7dabdae7bf76b8f7bfa` |

Qasper 的下载 URL 也冻结在其官方 dataset loader 中；2Wiki 的归档 URL来自官方 README。
2Wiki 论文给出的训练构成是 `154,878 train-medium + 12,576 train-hard = 167,454`。
正式 build 只能使用带 `official_source_lock=true` 的正式 spec；本仓库正式 spec 已填入实测值，
脚本还会将 repo、URL、revision、SHA 和计数逐项与内置官方锁核对。

## 3. 安全下载（人工执行）

下面命令只下载官方 train archive。不要把 LongBench test-v2 放进同一个目录，也不要把
test-v2 路径传给数据准备脚本。

```bash
mkdir -p /mnt/.../qcomem-sft-source/qasper
mkdir -p /mnt/.../qcomem-sft-source/2wikimqa

curl -fL \
  https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz \
  -o /mnt/.../qcomem-sft-source/qasper/qasper-train-dev-v0.3.tgz

curl -fL \
  'https://www.dropbox.com/s/npidmtadreo6df2/data.zip?dl=1' \
  -o /mnt/.../qcomem-sft-source/2wikimqa/data.zip

sha256sum \
  /mnt/.../qcomem-sft-source/qasper/qasper-train-dev-v0.3.tgz \
  /mnt/.../qcomem-sft-source/2wikimqa/data.zip
```

确认 archive SHA 与上表完全一致后，再解压到隔离目录：

```bash
tar -xzf /mnt/.../qcomem-sft-source/qasper/qasper-train-dev-v0.3.tgz \
  -C /mnt/.../qcomem-sft-source/qasper
unzip /mnt/.../qcomem-sft-source/2wikimqa/data.zip data/train.json \
  -d /mnt/.../qcomem-sft-source/2wikimqa

sha256sum \
  /mnt/.../qcomem-sft-source/qasper/qasper-train-v0.3.json \
  /mnt/.../qcomem-sft-source/2wikimqa/data/train.json
```

若 Dropbox 下载得到 HTML、SHA 不匹配或目录结构改变，立即停止；不要修改 spec 来“适配”
未知文件，应先重新核对官方 README 和归档。

## 4. 构造已消费数据的 hash-only 去重账本

账本构建器只读取两个官方 LongBench JSONL 的物理前 68 行（source index 0–67），包括：

- pilot 0–3，共 8 条；
- calibration 4–5，共 4 条；
- validation 6–35，共 60 条；
- legacy test 36–67，共 64 条。

它使用 `itertools.islice(..., 68)`，不会解析第 69 行；因此即使 ZIP member 后面仍有
source index 68–99，也不会读取其 JSON。可直接流式读取固定 revision 的官方 archive，
无需解压完整 LongBench：

```bash
PYTHONPATH=gpu python3 gpu/build_sft_overlap_ledger.py \
  --dataset 'qasper=/mnt/.../longbench/data.zip::data/qasper.jsonl' \
  --dataset '2wikimqa=/mnt/.../longbench/data.zip::data/2wikimqa.jsonl' \
  --output /mnt/.../qcomem-sft/heldout-fingerprints.json
```

预期计数为 pilot 8、calibration 4、validation 60、legacy 64、frozen test-v2 0；最后一项
状态应为 `deferred_not_read`。账本只保留 dataset、source index 和四种 SHA256：

1. 规范化 `_id`；
2. 规范化 `(context, input)` 对；
3. context-only；
4. input-only。

规范化规则是 Unicode NFKC 加空白折叠。脚本会拒绝任何带原始 `_id`、context、input、
question 或 answers 的 held-out ledger。

## 5. 默认 smoke-manifest（不产出训练 JSONL）

默认模式完整扫描、校验两个官方 train 文件和四键去重，但只对每个数据集按官方 source
顺序选择前 4 条 eligible 样本写 hash-only smoke manifest；不会写可训练 JSONL，也不会加载
tokenizer。输出限制不会提前终止扫描：

```bash
PYTHONPATH=gpu python3 gpu/prepare_supervised_qa_train.py \
  --qasper-archive /mnt/.../qcomem-sft-source/qasper/qasper-train-dev-v0.3.tgz \
  --qasper-train /mnt/.../qcomem-sft-source/qasper/qasper-train-v0.3.json \
  --twowiki-archive /mnt/.../qcomem-sft-source/2wikimqa/data.zip \
  --twowiki-train /mnt/.../qcomem-sft-source/2wikimqa/data/train.json \
  --source-spec configs/supervised_qa_sources.json \
  --heldout-ledger /mnt/.../qcomem-sft/heldout-fingerprints.json \
  --manifest /mnt/.../qcomem-sft/smoke-manifest.json
```

默认 `--overlap-policy fail`：脚本继续完成 hash-only 审计、写出
`status=failed_overlap` manifest，但绝不会发布训练 JSONL。若 input-only 这样的严格规则命中
了合理的泛问题，先人工检查 hash 级 provenance；确认后显式使用 `--overlap-policy drop`。
这时 manifest 必须满足：

```text
status = passed
detected_overlap_count = sum(dataset_stats.*.dropped_examples)
output_overlap_count = 0
```

`overlap_report` 不包含原始 held-out 文本或答案。

2026-08-12 的正式 source 扫描结果是：23 条 unique Qasper train example 仅在
`input_sha256` 上命中已消费的 validation/legacy 问题，共 25 个 match entry（有两条各命中
两个 held-out reference），2Wiki 命中 0。LongBench Qasper 本身派生自官方 Qasper，因而这种
问题文本交叉是预期可能性；没有 context 或 context+input 重合。input-only 比完整样本去重更
保守，但只影响 0.89% 的 Qasper converted train，故保留。默认 fail
门禁正确停止；显式 drop 后 Qasper full eligible 为 2,567、2Wiki 为 167,454，published
output overlap 为 0。机器结果见
`results/supervised-sft-data-smoke-20260812/smoke-manifest-drop.json`。

## 6. 正式 JSONL 与 answer-token CE labels

build 模式需要目标模型使用的同一 tokenizer，并默认只从本地缓存加载。revision 必须是不可变
commit；只有明确加 `--allow-tokenizer-download` 才允许联网获取 tokenizer。

```bash
PYTHONPATH=gpu python3 gpu/prepare_supervised_qa_train.py \
  --mode build \
  --qasper-archive /mnt/.../qcomem-sft-source/qasper/qasper-train-dev-v0.3.tgz \
  --qasper-train /mnt/.../qcomem-sft-source/qasper/qasper-train-v0.3.json \
  --twowiki-archive /mnt/.../qcomem-sft-source/2wikimqa/data.zip \
  --twowiki-train /mnt/.../qcomem-sft-source/2wikimqa/data/train.json \
  --source-spec configs/supervised_qa_sources.json \
  --heldout-ledger /mnt/.../qcomem-sft/heldout-fingerprints.json \
  --tokenizer /mnt/.../Qwen3.5-4B \
  --tokenizer-revision REPLACE_WITH_MODEL_COMMIT \
  --max-sequence-tokens 1024 \
  --max-output-per-dataset 4 \
  --overlap-policy drop \
  --output-jsonl /mnt/.../qcomem-sft/supervised-train.jsonl \
  --manifest /mnt/.../qcomem-sft/supervised-train.manifest.json
```

已存在的输出默认不会覆盖；复跑相同路径需显式 `--overwrite`。发布前必须固定 JSONL SHA 和
manifest SHA，训练 launcher 应重新计算并 hard fail。

每行基础 schema 为：

```text
dataset, source_split='train', source_id, context, input,
answers, selected_answer, provenance
```

此外写入 `document_input_ids`、`query_input_ids`、`answer_input_ids`、`input_ids`、
`labels` 与 `token_counts`。document/query 直接调用生产评测代码
`run_downstream.prompt_parts`，包括 chat template、`enable_thinking=False`、中间截断策略。
labels 对全部 prompt token 写 `-100`，只对 `selected_answer + EOS` 写真实 token id；因此
loss 是真正的 answer-token causal CE，而不是继续做 PG-19 next-token proxy。实现直接复用
`supervised_sft.build_supervised_example`：先从 1,024-token sequence budget 保留完整 answer
和 EOS，再将剩余预算交给 `prompt_parts`，与 trainer 重建逻辑一致。全量 overlap scan 不做
tokenization，只有最后选中的 4+4 行会生成 token target。

Qasper 多标注规则固定为：NFKC/空白规范化后 casefold 分组，选出现次数最多的答案；并列时
选 canonical 字典序最小者，surface 再取字典序最小者。extractive 多 span 用官方 LED reader
相同的 `", "` 连接。全部候选答案、annotation id、worker id、answer type、分组计数和最终选择
策略都保存在 provenance，改变输入 annotation 顺序不会改变 target。

2Wiki context 固定为 LongBench 使用的格式：

```text
Passage 1:
{title}
{concatenated sentences}
Passage 2:
...
```

## 7. 发布前硬门禁

训练程序和 launcher 至少验证：

1. JSONL SHA 与外部预注册值、manifest `output_jsonl_sha256` 三者相同；
2. manifest SHA 与外部预注册值相同；
3. `schema_version=qcomem-supervised-qa-v1`、`status=passed`；
4. 每行 top-level 和 provenance 的 `source_split` 都严格为 `train`；
5. `output_overlap_count=0`；
6. fail policy 要求 `detected_overlap_count=0`；drop policy 要求 detected 等于 dropped 总数；
   有输出上限时每个数据集 `written_examples=selected_for_output_examples=4`，而
   `full_eligible_examples` 仍必须来自全量扫描；无上限时 written 才等于 full eligible；
7. `raw_test_v2_read_by_converter=false`；test-v2 status 只能是
   `deferred_not_read` 或未来独立产生的 `blind_hash_manifest`；
8. source revision、archive/file SHA、license、样本数、tokenizer revision 和 chat-template SHA
   均与预注册值一致。

## 8. 本地测试

```bash
PYTHONPATH=gpu /usr/bin/python3 -m unittest \
  gpu/test_prepare_supervised_qa_train.py -v
```

测试完全使用临时合成数据，不联网、不读取 test-v2、不运行训练，覆盖 prompt token exactness、
CE mask、Qasper 多答案稳定性、2Wiki flatten、四种 overlap、fail/drop 发布语义、source SHA /
license/split/count，以及“物理第 69 行为非法 JSON 也不会被账本构建器读取”的边界测试。

## 9. H20 环境正式 4+4 build（只准备数据，不提交训练）

推荐远端目录：

```text
SOURCE_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/qcomem-supervised-sft-source-20260812
OUTPUT_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/qcomem-supervised-sft-smoke-20260812
MODEL_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/models/Qwen3.5-35B-A3B-59d61f3
CODE_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_gpu
```

source 目录需要保存两个原始 archive、只解出的两个 train 文件、固定 revision LongBench
archive 和本轮 hash-only ledger。先逐项用本节 2 的 SHA hard check，再运行：

```bash
cd "$CODE_DIR"
PYTHONPATH=. "$ENV_DIR/bin/python" prepare_supervised_qa_train.py \
  --mode build \
  --qasper-archive "$SOURCE_DIR/qasper/qasper-train-dev-v0.3.tgz" \
  --qasper-train "$SOURCE_DIR/qasper/qasper-train-v0.3.json" \
  --twowiki-archive "$SOURCE_DIR/2wikimqa/data.zip" \
  --twowiki-train "$SOURCE_DIR/2wikimqa/data/train.json" \
  --source-spec "$CODE_DIR/../configs/supervised_qa_sources.json" \
  --heldout-ledger "$SOURCE_DIR/heldout-fingerprints.json" \
  --tokenizer "$MODEL_DIR" \
  --tokenizer-revision 59d61f3 \
  --max-sequence-tokens 1024 \
  --max-output-per-dataset 4 \
  --smoke-count-per-dataset 4 \
  --overlap-policy drop \
  --output-jsonl "$OUTPUT_DIR/supervised-train-smoke-4x2.jsonl" \
  --manifest "$OUTPUT_DIR/supervised-train-smoke-4x2.manifest.json"
```

这一步只加载 tokenizer，不加载 35B 模型、不申请 GPU、不训练。完成后用 `sha256sum` 固定
JSONL 和 manifest，并运行 `supervised_sft` 的 CPU preflight。只有 JSONL 恰为 Qasper 4 +
2Wiki 4、tokenizer/chat-template 与运行时模型一致、全部 manifest 门禁通过，才能填写 QS
template；仍需单独授权才提交训练任务。
