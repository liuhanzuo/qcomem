# 多步 full-SFT train-only 数据准备结果（2026-08-12）

## 结论

多步 full-SFT 的数据已经准备并通过独立审计，但**尚未训练，也没有提交 GPU**：

- 正式训练集：Qasper 512 + 2WikiMultiHopQA 512，共 1,024 条；
- train-split CE heldout：Qasper 64 + 2WikiMultiHopQA 64，共 128 条；
- train 与 CE heldout 在 source ID、四类精确指纹和全局指纹连通组上的交集均为 0；
- 1,152 条全部来自两个数据集的官方 `train` split，没有使用 dev、validation 或 test 行；
- LongBench frozen test-v2 没有被读取，状态继续是 `deferred_not_read`；
- 每条样本都重新审计了完整 answer + EOS、1,024-token 上限以及 answer-token-only CE mask；
- 固定配置的第二次独立拆分与第一次逐字节相同。

CE heldout 只允许用于训练过程诊断，以及在**预先冻结选择规则后**做 checkpoint/early-stop
选择；它不是最终下游测试集，不能据此宣称 CoMem 下游精度提升或“几乎无损”。一旦用它选择
checkpoint，它就是已消费的开发信号。

## 1. 上游全量扫描和 target-valid parent pool

继续复用已经固定的官方 train source、仅含 LongBench source index 0--67 的 hash-only
heldout ledger、Qwen3.5-35B-A3B commit `59d61f3` 的 tokenizer，以及
`prepare_supervised_qa_train.py`。转换器没有打开 test-v2 文件。

全量 leakage scan 的结果为：

| 数据集 | parsed train QA | 已消费 heldout 精确命中 | drop 后 full eligible | parent 输出 |
|---|---:|---:|---:|---:|
| Qasper | 2,590 | 23 | 2,567 | 576 |
| 2WikiMultiHopQA | 167,454 | 0 | 167,454 | 576 |

parent 使用预先固定的
`first_n_target_valid_eligible_in_official_source_order-v1`：先完成全量 leakage scan，再按
官方 source order 选择每数据集最早的 576 条 target-valid 行。完整 answer 与 EOS 先占预算，
剩余预算才用于 prompt；answer 不截断。为凑齐 Qasper 的 576 条，本次选择边界内跳过了 6 条
answer+EOS 超过冻结生成上限的行；这不是对全部 2,567 条 eligible Qasper 的全量 target-length
统计。2Wiki 在选择边界内跳过 0 条。

parent 结果：

```text
supervised-pool-576plus576.jsonl
SHA256 527d4d26068c1d8afb2ee5849ca380c5f44024bc0f27295f6d9049a3ab3d376b

supervised-pool-576plus576.manifest.json
SHA256 81cc7e1be46ce5f9f82a35ef357cd0bcd3d33eef438c39872c757fcb21e212e9
```

heldout ledger SHA256 为
`949090d0d6867f7ebe6d013086027fd3e0e4e5dcce23e20354966184658bb64f`；其中
frozen test-v2 entry 数仍为 0。

## 2. 为什么按指纹连通组拆分

简单地逐行随机拆分会让“同 context、同 question、不同 source ID”的近重复 QA 落到训练和
heldout 两边。这里先对 parent pool 的每行重新计算四类规范化 SHA256：

```text
id_sha256
context_input_sha256
context_sha256
input_sha256
```

只要两行在同一种指纹上相同，就连接起来；传递闭包形成一个不可拆分的全局 component。
component scope 跨 Qasper 和 2Wiki，避免跨数据集重复被遗漏。最终 1,152 行形成 705 个
component。

预注册配置为
`configs/supervised_sft_scale_split_512_64.json`，SHA256：

```text
2bd2f5cef3f46405ebdf403ed4d8670fd6e64770f619051b5187a0e34a875be8
```

每个 component 的排序键只由固定 salt、dataset 名和 source-ID SHA 产生，不读取 loss、模型
输出或 heldout 指标。排序后使用确定性的二维 subset-sum，选择第一个能够精确得到
`Qasper=64, 2Wiki=64` 的 component 集作为 CE heldout；其余为 train。两份输出都保持 parent
的官方 source order。相同配置和 parent SHA 的独立复跑逐字节一致。

## 3. 正式输出和固定 SHA

远端正式目录：

```text
/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/
qcomem-supervised-sft-scale-20260812a/
```

| artifact | 行数 | 数据集计数 | SHA256 |
|---|---:|---|---|
| `supervised-train-512plus512.jsonl` | 1,024 | 512 + 512 | `b6b1a88226b3060b6ba6b600793d90470820511ae38096b4db99af8b65f05257` |
| `supervised-heldout-ce-64plus64.jsonl` | 128 | 64 + 64 | `069c6649e73a0bdbe7b300a1a32f6b89fa5ad23d43fcfa03c85f101d5c7ac10e` |
| `supervised-scale-assignment.hash-only.jsonl` | 1,152 | parent 全覆盖 | `620a5401ee32eee95d2336c6ea22ea2a3a05902609efa819e5adc5652311cf33` |
| `supervised-scale-512-64.manifest.json` | — | split manifest | `e527eeac4f110005057bcc3936093c6b6ce60252591cd57373785c4995f2ff15` |

hash-only assignment ledger 只包含 parent row index、dataset、split、row/source/component SHA，
不包含 source ID 原文、context、question 或 answer。

token 统计：

| split | sequence token min / median / max | answer+EOS token min / median / max |
|---|---|---|
| train | 373 / 1,024 / 1,024 | 2 / 4 / 120 |
| CE heldout | 567 / 1,024 / 1,024 | 2 / 5 / 105 |

## 4. 独立审计

独立 auditor 没有复用 split 函数的选择逻辑，而是重新读取并验证最终 artifact：

1. 用外部固定 SHA 绑定 split manifest，再由 manifest 绑定 parent、train、heldout 和 ledger；
2. 证明 train 与 heldout 不相交，且二者并集恰好等于 1,152 行 parent pool；
3. 证明 ledger 逐行绑定 parent，输出保持 parent source order，ledger 没有 raw text 字段；
4. 对 1,152 行重新计算四类指纹，四种 train/heldout 交集全部为 0；
5. component 交集为 0，source-ID 交集为 0；
6. 对全部行重验 top-level/provenance `source_split=train`；
7. 重验 `document + query + answer == input_ids`、prompt label 全为 `-100`、answer label 与
   answer IDs 相同、末 token 是 EOS、sequence 不超过 1,024。

机器审计结果在
`results/supervised-sft-scale-20260812a/independent-audit.json`，状态为 `passed`。
本地 converter/split 单测共运行 22 项，全部通过；远端 split 专项 4 项也全部通过。

## 5. 使用边界和下一步

这批 artifact 已达到数据发布条件，但当前 `supervised_sft.validate_prepared_training_manifest`
仍是 4+4 smoke 专用：它硬编码每数据集 4 条和总计 8 条。因此，不能绕过校验把 512+512
文件直接交给现有 trainer，也不能把 scale manifest 伪装成 smoke manifest。

开始多步 full-SFT 前还需要：

1. 新增独立的 scale-mode runtime validator，按本报告的 manifest/SHA/partition 合同硬校验；
2. 在查看任何 CE-heldout 指标前冻结 epoch/step、LR、optimizer、checkpoint cadence 和
   checkpoint 选择规则；
3. train loader 只打开 `supervised-train-512plus512.jsonl`；CE evaluator 只读 heldout，绝不
   backward；
4. 对 LoRA 与 full-SFT 使用同一 train/heldout 划分，报告 train CE、heldout CE、显存、吞吐和
   wall time；
5. CE heldout 仅用于训练诊断/预注册 checkpoint 选择。最终自然任务能力结论仍要等模型和
   CoMem/bit policy 全部冻结后，使用独立的下游评测；当前 frozen test-v2 继续不读。

