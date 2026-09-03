# Q-CoMem LongBench 数据边界（冻结于 2026-08-12）

公开数据源：`zai-org/LongBench`

固定 revision：

```text
5e628be450b7e67fb7ae6e201bd6d8f7056f7672
```

只使用 `qasper` 与 `2wikimqa`。`_source_index` 按上述 revision 中对应 JSONL 的
原始行号定义。

| 用途 | 每数据集 source index | 总样本数 | 状态 |
|---|---:|---:|---|
| pilot | 0-3 | 8 | 已使用 |
| layer-bit calibration | 4-5 | 4 | 已使用 |
| mixed-bit validation | 6-35 | 60 | 在 validation 结果揭晓前冻结 |
| legacy test | 36-67 | 64 | 已用于固定 Q4/Q8 策略，不能作为新策略的 untouched test |
| mixed-bit test-v2 | 68-99 | 64 | 在新 validation 结果揭晓前冻结 |

test-v2 远端路径：

```text
/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/qcomem-longbench-test-v2/longbench_test_v2.jsonl
```

SHA256：

```text
fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f
```

数据由 `gpu/prepare_longbench_subset.py --start 68 --count 32` 生成；每行保留
`_source_repo`、`_source_revision` 与 `_source_index`。test-v2 在 mixed-bit validation
结果产生前创建，不用于校准、策略搜索、LoRA 训练或超参数选择。
