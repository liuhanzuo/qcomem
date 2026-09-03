# Supervised SFT 数据准备结果（2026-08-12）

## 结论

真正 answer-token CE SFT 的离线数据线已经完成并通过本地测试；本轮没有申请 GPU、没有
训练、没有生成正式全量训练 JSONL，也没有读取 frozen test-v2 内容或答案。

官方 train 全量 leakage scan 发现 23 条 Qasper 问题与已消费 LongBench 行发生严格的
input-only SHA 重合；2Wiki 没有重合。默认 fail policy 正确停止并不发布 JSONL；显式 drop
后得到净化候选池：Qasper 2,567 条、2Wiki 167,454 条，`output_overlap_count=0`。

## 冻结输入

| 数据 | revision | archive SHA256 | train SHA256 | 原始/转换问题数 |
|---|---|---|---|---:|
| Qasper | `fdc9d8214fbab5dd782958601db4d678e6934a54` | `a28fdf…bdb5a` | `9458bfe…8bf4` | 2,593 / 2,590 |
| 2WikiMultiHopQA | `13800e5be57df1b4040b9b1588c6c811779e69e9` | `e8e57c0…7d04` | `b3fddb4…7bfa` | 167,454 / 167,454 |
| LongBench heldout source | `5e628be450b7e67fb7ae6e201bd6d8f7056f7672` | archive `cb45b11…7f64` | 不读取/不重算 test-v2 | 仅解析每数据集 0–67 |

Qasper 原始 2,593 与转换 2,590 的差是论文 `1801.07804`：官方 v0.3 train 中其
`full_text=[]`，包含 3 个问题。转换器与官方 Qasper LED reader 一致跳过整篇，不用 abstract
静默替代，并把 raw/converted/skipped 三种计数分别写入 manifest。

## Held-out 边界

hash-only ledger：

```text
results/supervised-sft-data-smoke-20260812/heldout-fingerprints.json
SHA256 949090d0d6867f7ebe6d013086027fd3e0e4e5dcce23e20354966184658bb64f
```

账本从 LongBench ZIP member 以 `islice(..., 68)` 读取：pilot 8、calibration 4、validation
60、legacy test 64。frozen test-v2 entry 数是 0，状态为 `deferred_not_read`，账本标记
`raw_test_v2_read=false`、`raw_answers_used=false`。

去重同时检查规范化后的 `_id`、`(context,input)`、context-only 与 input-only SHA。这里的
`detected_overlap_count=23` 是 **unique train example 数**，不是 match entry 数；21 条各命中
一个 held-out reference，2 条各命中两个 reference，共产生 25 个 match entry。25 个 match
全部是同数据集 Qasper 的 `input_sha256`：validation 20、legacy test 5；`_id`、context+input、
context-only 和跨数据集 match 都是 0。报告只含 hash、dataset、split、source index，不含
held-out 原文或答案。

这说明不是文档本身重复，而是 LongBench Qasper held-out 本来就派生自官方 Qasper 数据；同一
论文可以有多个标注问题，规范化后的 question 文本会在官方 train 与 LongBench benchmark 中
交叉。input-only 是刻意偏保守的门禁：它可能比防止“完整样本泄漏”更严格，但只丢 23/2,590
（0.89%）条，代价很小，而且能避免模型直接见过评测问题。因此正式 smoke 保留 drop policy；
manifest 同时保存 unique-example 数、match-entry 数、按 split/fingerprint 的分解和每样本
match 数直方图，不会因一条样本命中多个 reference 而重复扣减。

## Smoke manifests

默认 fail 审计：

```text
results/supervised-sft-data-smoke-20260812/smoke-manifest.json
status=failed_overlap
detected_overlap_count=23
output_jsonl=null
manifest SHA256=5203de8d779a7c6073efe39ca171a11e3798f1c8087bfeabae320c744651b59e
```

显式 drop 后通过的 hash-only smoke manifest：

```text
results/supervised-sft-data-smoke-20260812/smoke-manifest-drop.json
status=passed
detected_overlap_count=23
output_overlap_count=0
Qasper full eligible=2567, selected=4
2Wiki full eligible=167454, selected=4
test_v2_content_hash_check=deferred_not_read
manifest SHA256=b7c1b9f082ba54654c74d32d15800c256175d06ed1653c317874aee26d3ce4f0
```

通过 manifest 已做两次全量复扫，第二次 SHA 与第一次逐字节一致。本轮 smoke-manifest 不加载
tokenizer，因此不发布训练 JSONL。正式 4+4 build 会固定
`max_sequence_tokens=1024` 和目标模型 tokenizer commit，只对最终 8 条 tokenization；全量
overlap audit 仍扫描 170,044 条可转换问题。

## Target 语义

- prompt 直接使用 `run_downstream.prompt_parts`；
- build 直接复用 trainer 的 `supervised_sft.build_supervised_example`；
- 先保留完整 `selected_answer + EOS`，再用剩余 sequence budget 截断 context；
- prompt labels 全为 `-100`，answer 与唯一 EOS 才参与 causal CE；
- Qasper 多答案按 canonical majority、再按字典序确定性选择，全部候选与 annotation
  provenance 保留；
- 2Wiki 使用唯一官方 answer；context flatten 与 LongBench passage 格式一致。

## 测试

全仓 GPU 单测：共 103 项，102 passed、1 skipped（本机未安装目标 Transformers build）。其中新增数据线
17 项覆盖 source SHA/license/split/count、Qasper 多答案、empty-full-text 口径、2Wiki
streaming、四类 overlap、fail/drop、ZIP 第 69 行不读取、4+4 输出限制、prompt/label exactness
与 answer-reserved sequence budget。

## 下一步

1. 在 H20 环境使用固定目标 tokenizer 构建 4+4 JSONL，记录 JSONL SHA 与 manifest SHA；
2. 先跑 CPU preflight 和单步 full-model supervised CE smoke，不直接跑长训练；
3. adapter、训练超参、bit policy 冻结前继续保持 test-v2 `deferred_not_read`；
4. 训练方案完全冻结后，才由独立 blind auditor 生成 test-v2 hash-only overlap 结果，不能读取
   answers 做样本选择。
