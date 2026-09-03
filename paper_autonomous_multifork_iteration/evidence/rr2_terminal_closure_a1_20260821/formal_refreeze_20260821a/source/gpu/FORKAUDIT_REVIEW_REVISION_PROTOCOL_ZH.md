# ForkAudit 审稿意见触发实验：独立发布与证据协议

状态：**静态治理层与正式 GPU producer 已实现并联合发布；当前尚未产生新的 GPU 实验结果。**

本协议对应 `run_qcomem_qwen35_forkaudit_review_revision.py`。目标不是把已有结果重新包装成肯定结论，而是在审稿意见指出所有权证据、独立数值 oracle 和运行时 mutant sensitivity 不充分之后，预注册一轮可审计的新实验。失败、逃逸 mutant、错误 gate 或数值超阈值都必须原样保留，并可形成负结果。

## 1. 当前不可越过的发布门

正式执行同时要求：

1. runner 中 `GPU_LOOP_IMPLEMENTED=True`，且实现经过独立代码审查；
2. launcher 中 `FORMAL_PIPELINE_RELEASED=true`，且与 producer 在同一冻结发布中变更。

当前两者均未放行。launcher 只能依次完成代码/模型/数据完整性检查、零跳过测试、path-independent preregistration manifest 和 runner 的 `--stage static`，然后写入 `BLOCKED_GPU_PRODUCER_NOT_IMPLEMENTED` 并以状态码 3 退出。它在该门之前不查询 GPU、不启动 shard、不聚合，也不包含任何集群资源创建命令。`99_done` 不得出现。

static 完成后、任何 candidate output 之前，launcher 只生成一次 128-bit lowercase-hex `run_id`。该值不能硬编码，也不能由各 rank 分别生成；它由一次随机 nonce 与 static artifact SHA、protocol manifest SHA 共同派生，并写入可机械重算的 `run-id-receipt.json`。正式路径把同一个 `--run-id`、同一份 `--run-id-receipt` 及其 `--expected-run-id-receipt-sha256` 显式传给全部 8 个 shard 和 aggregate。每个 shard 把该 receipt 原文及摘要嵌入 raw shard；detached receipt builder 在冻结 8 个 shard 前逐 rank 核对三者完全一致。aggregate 还必须独立重放 receipt 对 run ID、static SHA、protocol SHA 和 nonce derivation 的绑定；shard、phase、oracle 与 aggregate 必须证明全 rank 一致。

正式 shard 不接受旧的 `FORKAUDIT_*` ambient 配置。每个 rank 必须由 launcher 显式传入：`--rank`、`--run-id`、`--output`、artifact root、static artifact 及 SHA、RR2 main 及 raw SHA、PG19 data/manifest、prior-capacity manifest、model directory、code/model-artifact/model-weight ledgers、protocol manifest、共享 run-ID receipt 及 SHA，以及由 preflight `nvidia-smi` 八行清单按 rank 绑定的 expected GPU UUID。GPU inventory 在 candidate output 前只生成一次 canonical+LF assignment receipt；每行固定 rank、visible index、UUID、name、memory MiB、compute capability `[9,0]` 和 BF16 support，要求 8 个 index/UUID 均唯一。receipt 的完整 raw-byte SHA 与同一文件显式传给 8 shard 和 aggregate；launcher 必须使用 receipt 中该 rank 的稳定 GPU UUID（而非可能随宿主枚举变化的 numeric index）设置 `CUDA_VISIBLE_DEVICES`，rank 再将 process-local `cuda:0`、UUID、name、memory、compute capability 与同一行逐字段匹配，terminal 重新查询并要求 receipt byte exact。所有 shard output、witness、oracle tensor 和其他 detached sidecar 的唯一 artifact root 固定为 `$RUN_DIR/raw`；不允许把 `$RUN_DIR` 整体作为 root 混入 preregistration/receipts。aggregate 必须显式传入同一个 raw root、output、static artifact 及 SHA、detached receipt manifest 及 SHA、同一个 run ID 和同一 run-ID receipt 及 SHA。launcher/tests 与 release builder 从 runner 的冻结 parser 机械核对这两组 option set；任何缺项或 runner 未实际消费 aggregate receipt 都保持 fail closed。pre-aggregate raw ledger 与 terminal replay 均递归覆盖 raw root 下全部 regular sidecars，aggregate 还逐引用重放 bytes/SHA/shape/dtype binding。

正式运行只能使用已经分配给 launcher 的 8 张 GPU。分配或创建 GPU 作业属于 launcher 外部的独立审批步骤。本轮不创建独立 GPU smoke job/YAML，也不拆成多个同 rank 作业；release gate 闭合后只执行一次正式 8×H20 并行运行。正式 launcher 内部的无结果 preflight、最大 N warmup 与丢弃 warmup state 仍是同一次运行的必要阶段，不能作为可引用实验输出。

## 2. 冻结身份与路径无关 manifest

模型固定为：

- model ID：`Qwen/Qwen3.5-35B-A3B`
- revision：`59d61f3ce65a6d9863b86d2e96597125219dc754`（完整 40 字符 commit）

`build_qcomem_qwen35_forkaudit_review_manifest.py preregister` 生成四份 canonical JSON：

- `release-manifest.json`：治理约束、8-rank 分配和逻辑 artifact receipts；
- `frozen-identity.json`：runner `--stage static` 所需的冻结身份；
- `frozen-query-banks.json`：从 RR2 main input manifest 提取并重新 canonicalize 的 8 个、每个 32-query 的 bank；
- `oracle-selection-plan.json`：候选输出产生前固定的 8 个 oracle 样本。

RR2 的唯一权威输入是 `qcomem-forkaudit-rr2-pg19-input-manifest-v1` main manifest 的**完整原始字节**。launcher 必须同时冻结其 raw SHA，并要求 builder 由 main manifest 生成 query-bank/oracle 两个 sidecar；生成字节必须分别与外部预注册 sidecar 完全相同。不能只钉住 main 内部的 self-hash。runner static CLI 必须接收并逐字节验证：

- `--rr2-input-manifest` 与 `--expected-rr2-input-manifest-sha256`；
- `--prior-fp32-context-manifest` 与 `--expected-prior-fp32-context-manifest-sha256`；
- `--review-experiment-plan` 与 `--expected-review-experiment-plan-sha256`。

static artifact 保存三份 exact raw inputs 的 SHA、bytes 与可逆 base64，并重放 main 中的 banks/selection；因此后续 artifact 验证不能用语义相同但字节不同的 JSON 替换输入。
round-2 reviewer-response experiment plan 的 raw SHA 固定为 `e2be05e198d6c86276f229d4e862c3579c65e311d07c219e9e9c792a390cbfbb`；其中 RR2 必须仍为 `experiment_required`，不得在看到候选输出后替换计划。

main 内部 digest、自哈希与 raw SHA 只能证明“冻结后未改”，不能独立证明 token digest 源自 PG19。为此 launcher 在 release builder 之前先调用 RR2 input builder，使用 exact PG19 bytes、exact prior manifest 和 `local_files_only` tokenizer clean rebuild main/banks/oracle，并逐文件 bytewise compare。release builder 随后重新加载 tokenizer、第二次独立重建并再次逐字节比较；任何自洽改写 document/query/bank digest 的伪 manifest 都必须在 GPU 门之前失败。

生成文件不记录输入目录、输出目录、用户名、挂载点或 `file://` URI。代码、模型 artifact 和模型权重 ledger 必须使用相对逻辑名，并以 `LC_ALL=C` 的字节序排序。相同输入字节复制到不同根目录后应产生完全相同的 manifest 字节。

`CODE_DIR` 必须是真正不可变的递归代码快照，而不是只检查顶层若干文件。launcher 使用隔离模式 `python -I`，在任何 Python import/测试之前以及 terminal integrity 阶段各扫描一次完整树：拒绝 `CODE_DIR` 本身或树内任意 symbolic link，拒绝任意 writable 文件或目录，拒绝 `__pycache__`、`.pyc`、`.pyo` 和 FIFO/socket/device 等非 regular entry。代码闭包包含 curated `CODE_DIR` 内递归全部 regular files（不只 `*.py`），并显式要求 launcher、本文协议、review manifest builder、RR2 input builder 和 runner 存在；相对路径按 C byte order 排序。由此同名 `.so`/平台 extension、`.dylib`、`.dll`、配置或其他 import 输入也不可能在 ledger 外遮蔽 Python 源。两次扫描生成的 NUL 路径表和逐文件 raw SHA-256 ledger 必须逐字节相等；terminal 阶段还必须分别把保留的 preregistration ledger 和现场重建的 terminal ledger 重新核验到同一个外部 `EXPECTED_CODE_LEDGER_SHA256`，不能只证明二者彼此一致。因此即使攻击者在 preflight 后修改代码并协调重写路径表及两份 ledger，也不能逃逸；新增嵌套 import 文件、删除文件、内容篡改、后置 symlink 或 bytecode 污染同样不能由只重放新旧任一 ledger 的 `sha256sum -c` 漏过。

需要区分两类摘要：

- raw artifact receipt：对磁盘文件的完整原始字节（包括结尾换行）计算 SHA-256；
- canonical JSON digest：严格解析 JSON 后，以排序键、紧凑分隔符、拒绝 NaN/Infinity 的 canonical bytes 计算 SHA-256。

runner 的 `expected-static-sha256` 和 `expected-receipt-manifest-sha256` 使用第二种；shard 及其引用的 phase/oracle artifact 使用第一种。二者不可混用。

tokenizer provenance 必须与已经逐文件验证的 model artifact ledger 交叉绑定。允许的 tokenizer layout 恰好二选一：`tokenizer.json`，或完整的 `vocab.json + merges.txt`；两套同时存在、BPE 套件缺文件、SHA/bytes 不一致均拒绝。实际选择的 layout 及所有被 tokenizer/model 读取的配置文件都写入 RR2 main manifest 的 artifact set。正式 rank 不直接从原始 model directory 加载：release builder 先把 model-artifact ledger 与 14-weight ledger 的 exact union 原子物化到独立 private view。每个文件只能使用 FICLONE reflink，若文件系统不支持则逐字节 copy；hardlink、symlink 和源/目标同 inode 一律拒绝。view 中所有文件为 regular 0444、目录为 0555，copy manifest 记录每行 copy mode 与 source/view device+inode，完成 staging 后才整体 rename 并交给 lease guard/ranks。

private view 完成后，launcher 必须启动一个独立、贯穿全部 8 个 rank 模型加载期的 Linux `ModelLoadLease-v1` keeper。keeper 由一个在任何 Python/torch import 前先屏蔽 SIGIO 的隔离 bootstrap 启动；所有随后产生的 native worker 继承屏蔽状态，只有单一 Python main keeper thread 在安装 sticky handler 后解除屏蔽，从而使 terminal signal barrier 不依赖多线程的偶然投递。keeper 对 14 个权重文件各自保持固定 FD，在同一 FD 上先做完整 SHA-256，再申请 `F_SETLEASE(F_RDLCK)`；任一文件系统不支持 read lease、已有 writer、无法建立独立 view、固定 FD 的 stat/hash 与 ledger 不一致，均在任何 candidate output 前 fail closed，禁止降级为 `flock`、`lockf` 或仅比较 pathname stat。authority receipt 必须绑定 run ID、model artifact/weight ledger raw SHA、private-view manifest raw SHA及 14 行固定-FD hash/stat，并由 launcher 将同一文件和外部 raw SHA 显式传给 8 shard 与 aggregate。两者还必须接收同一个 `--private-model-view-manifest` 与 `--expected-private-model-view-manifest-raw-sha256`，逐行重算 copy mode、source/view inode 分离、live view stat 和 authority 权重覆盖；aggregate 还必须显式接收原始 model-artifact/model-weight ledgers，不能只相信 authority 中声称的 ledger digest。

每个 rank 在 `from_pretrained` 紧邻前后记录固定 14 权重的 stat envelope；keeper 对 lease break/SIGIO 采用 sticky failure。全部 rank 完成后 launcher 才发送一次绑定 authority SHA 的 close 指令；keeper 必须在固定 FD 上再次 full-hash、检查 stat 与 `F_GETLEASE` 仍为 read lease，再安全解除 lease/关闭 FD，并写 canonical+LF closure receipt。closure 的 `passed` 必须覆盖终态 rehash、break 状态、解除 lease 和 FD 关闭的最终结果，不能在 unlock 前提前冻结。aggregate 同时重放 authority 与 closure 的外部 raw SHA；terminal 再核两者与 private-view manifest。该机制的保证范围明确限定为 Linux 正常 VFS 下、无特权同身份进程；不声称防御 root/CAP_SYS_ADMIN、raw block device、恶意内核/文件系统服务端或 keeper 本身被攻破。

## 3. 数据边界

唯一允许的数据为官方 PG19 的 `train/*.txt`：

- 8 个 rank 各绑定一个不同 train book；
- manifest 必须声明 bucket `deepmind-gutenberg`、prefix `train/`，并明确 `test_or_validation_objects_used=false`；
- PG19 JSONL 的 SHA 必须与 manifest 内的 `jsonl_sha256` 一致；
- oracle selection 中每个 `source_object` 必须存在于该 train manifest；
- LongBench、validation、test、test-v2 均不得消费，也不得作为回退数据源。

数据 gate 在 preregistration 阶段执行，而不是在看到模型输出之后执行。

RR2 window/query-bank 预注册固定 `seed=20260817`、document 4095 tokens、query 32 tokens、bank 32、stride 64、resident prefixes `N={1,8,32}`。window algorithm digest 固定为 `39bc36bb2eb04d51122e66caaebfa72367c02b43b073f072a2da240ed068c166`；它与 main manifest raw SHA 是不同语义。builder 必须从 PG19 raw bytes 和冻结 tokenizer 独立重建 document/query token IDs 及其 SHA，而不是信任 sidecar。

上一轮 capacity protocol manifest 的 raw SHA 固定为 `975bc6a12f43447024b889889d4156ca71c2f89b68de6157ac609b4a9687e9c0`，旧 window digest 为 `27ad6c687e5cab28f361bbd89dd1844788aecbecc6f2d25dbd0c60b7705a55f8`。RR2 的 8 个 `(source_object, document_start_token, document_length)` 必须逐项证明不在旧 cohort，且新旧 window digest 必须不同。

## 4. 2×2 所有权实验

两个正交轴为：

1. KV：`full-copy-per-request` 与 `shared-document-paged-private-reservation`；
2. GDN：`materialize-request-base-functional-rebind` 与 `borrow-immutable-base-functional-rebind`。

居民请求数固定为 `N ∈ {1, 8, 32}`，每个请求生成 8 步。四个 cell 必须比较 token、每步全词表 logit 摘要、逻辑 KV 摘要和最终 GDN 摘要；同一请求前缀还必须跨 N 完全一致。KV policy 不能代理 GDN policy。

每个 factorial cell 拆为两个独立执行单元：

- `formal_memory`：不得创建 request guard，不得执行 witness hash，才有资格贡献主内存端点；
- `ownership_witness`：必须重新构建 persistent cache 和 request group，创建 guard 并捕获 setup、首次 transition、generation 三阶段；不得贡献主内存端点。

两者 `cell_id` 必须不同。这样 ownership instrumentation 自身的 tensor/guard 生命周期不会污染主内存测量。

GDN 时间语义为：首次 transition 前，borrow 模式允许请求只读别名 base；完成的请求在 transition 后必须把全部 60 个 GDN tensor 重新绑定到私有、与 base 及 peers 均不相交的 storage；未完成请求必须保留 setup binding 和内容。

## 5. 独立 FP32 oracle

oracle 样本在任何 candidate output 产生前固定。每个 rank 固定：book、window、layer、request、generation round 和 `sample_id`。选择摘要须由外部环境变量再次钉住。

raw oracle artifact 必须绑定：

- rank、book、source object、window、layer、request、step；
- Q/K/V、candidate output、position IDs 和 mask 的 SHA-256；
- sample-selection digest 与 preregistration digest；
- 独立 dense FP32 GQA reference 的输入和输出；
- oracle-only IEEE FP32/禁用 TF32 的有效状态及恢复状态。

聚合器从 raw tensor 重算 reference 和相对 L2，不信任 producer 的 `passed` 字段。预注册阈值为 relative L2 `≤ 0.005`。

该阈值还绑定一个只作 contextual validation、不得调参的历史 FP32 manifest，其 exact raw SHA 为 `fa64f663bb74a190a0a5c0898fda2a55528171c77a91af2b1321c24a5f310a1d`。release builder 必须盲重放全部 80 行的 exact schema/type/value（8 books × 10 layers，document=1025、query=32、comparison、scale 与 SHA），从 rows 重算 ordered-coordinate SHA、最大 relative L2 及其 coordinate、两倍边界、`0.005/max` ratio 和 document-token 集合。`document_length_disjoint=true` 只是待核对字段，不能作为证明；必须机械证明历史 `{1025}` 与全部 RR2 selection 的 `{4095}` 不相交。

## 6. 运行时 mutant sensitivity

rank 分配固定如下：

| Rank | PG19 book index | Mutants |
|---:|---:|---|
| 0 | 0 | M1, M9 |
| 1 | 1 | M2 |
| 2 | 2 | M3 |
| 3 | 3 | M4 |
| 4 | 4 | M5 |
| 5 | 5 | M6 |
| 6 | 6 | M7 |
| 7 | 7 | M8 |

每一个 clean/mutant case 都必须重新构建 document cache 与 request cache，case 完成后丢弃，禁止跨 mutant 复用。特别是 M3 可能改变 source tail padding，恢复 Python hook 不等于恢复 cache bytes，因此 fresh cache 是实验有效性条件。

每个 mutant 的 target witness 必须记录与 case 绑定的 pre、mutated、restored descriptors 和 SHA：`pre != mutated` 且 `restored == pre`。只有 detector 边界内抛出预注册 gate、完整执行 restoration，并通过独立恢复验证，才算 expected detection。enter/exit gate spoof、wrong gate、escape 和 unrelated crash 都不能计为成功。

## 7. 外部 raw receipts 与 blind aggregate

每个 rank 的 shard JSON 是外部 receipt 的第一层。shard 内的相对引用继续绑定 GDN phase、KV witness、oracle raw tensor 和 mutant target witness。launcher 在 shard 全部退出后：

1. 以 `LC_ALL=C` 构建所有 raw 文件的字节级 `sha256sum` ledger；
2. 验证该 ledger；
3. 逐 rank 验证 8 个 raw shard 均嵌入同一个预输出 run ID、run-ID receipt 和 canonical SHA，再用固定相对路径 `raw/shards/forkaudit-shard-{0..7}.json` 构建 detached receipt manifest；
4. 把其 canonical JSON digest 写入独立 terminal receipt；
5. 将该 digest、同一个 run ID、run-ID receipt 及其预输出 SHA 显式传给 blind aggregate；
6. aggregate 后再次验证 raw ledger 和 terminal receipt 未改变。

聚合器必须重放 timeline、重算 oracle、重建 mutant outcome dataclass 并重算四 cell exactness。producer 的结论布尔值不具有证据效力。

## 8. 测试、超时与终态顺序

正式前置测试必须包含真实 Transformers 5.14.1 的 Qwen 调用栈测试 `test_real_tf514_qwen_call_consumes_and_advances_position_ids`。focused suite 中任何 skip 都是发布失败；mock-only PASS 不能替代真实调用栈 PASS。

所有长阶段使用 `timeout --signal=TERM --kill-after=...`。launcher 捕获 `ERR`、`INT`、`TERM`，终止全部子进程，并写 `FAILED`、`FAILED_PHASE` 和带阶段名的失败标记。Python bytecode 只能写入 `RUN_DIR/pycache`，而 `RUN_DIR` 必须位于冻结 `CODE_DIR` 外；正式代码树在 preflight 和 terminal 两次审计中都必须保持 0 writable entry、0 symlink、0 bytecode cache/file。

允许写 `stages/99_done` 的唯一顺序是：

`8 shards → raw byte ledger → detached receipt → blind aggregate → aggregate audit → raw terminal recheck → scientific artifact ledger recheck → 99_done`

静态通过、单 rank 通过、仅 aggregate schema replay 通过或 producer 尚未发布都不能生成 `99_done`。

## 9. 当前证据边界

本文件及配套脚本证明发布协议能够在任一正式门缺失时 fail closed，并能以路径无关方式冻结 preregistration。任何 remote probe 都不能当作正式 artifact：必须从真实 PG19/tokenizer/model ledger 输入重建并外部钉住 main manifest raw SHA、两个 sidecar SHA 和 release manifest SHA。它们本身不是新 GPU 实验结果；只有真实环境零 skip、8-rank 正式运行和 terminal integrity 全部完成后，论文才可以引用新结果。
