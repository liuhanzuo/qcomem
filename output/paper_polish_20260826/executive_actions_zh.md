# 14 篇论文三评执行摘要

## 使用边界

本摘要只综合以下已完成材料：`reviews/R1..R3/<ID>.json` 共 42 份、`score_summary.tsv`、`baseline_scores.tsv` 与 `qa_manifest.tsv`。本文不新增审稿意见，不进行第四次评分，也不生成 meta-score。所有“当前判断”均只是对三份原评审及其离散分数的压缩复述；“升档”仅指原评审明确要求的证据门槛，不是新的分数预测。

新旧分数只能作描述性对照，**不能解释为润色造成的因果变化**。基线记录来自不同版本、不同评审上下文和不同来源，且 A1、A10、A23、A32 没有可用的旧独立评分工件；因此任何 `0` 或 `−2` 只表示两个记录中位数之差，不代表修改有效或无效。

## 分数与中位数核对

| 论文 | 当前 R1 / R2 / R3 | 当前中位数 | 基线记录 / 中位数 | 描述性差值 |
|---|---:|---:|---:|---:|
| A1 | 4 / 4 / 2 | 4 | 无可用工件 | — |
| A10 | 2 / 2 / 2 | 2 | 无可用工件 | — |
| A14 | 4 / 6 / 6 | 6 | 6 / 6 / 4；中位数 6 | 0 |
| A23 | 2 / 2 / 2 | 2 | 无可用工件 | — |
| A31 | 4 / 4 / 4 | 4 | 4 / 4 / 4；中位数 4 | 0 |
| A32 | 4 / 6 / 6 | 6 | 无可用工件 | — |
| C1 | 4 / 6 / 4 | 4 | 4 / 6 / 6；中位数 6 | −2 |
| C2 | 4 / 4 / 4 | 4 | 4 / 6 / 6；中位数 6 | −2 |
| C3 | 2 / 2 / 4 | 2 | 2 / 4 / 2；中位数 2 | 0 |
| P1 | 2 / 2 / 4 | 2 | 2 / 2 / 2；中位数 2 | 0 |
| P3 | 4 / 4 / 4 | 4 | 2 / 4 / 4；中位数 4 | 0 |
| P5 | 4 / 4 / 4 | 4 | 4 / 4 / 4；中位数 4 | 0 |
| P7 | 4 / 4 / 4 | 4 | 4 / 4 / 4；中位数 4 | 0 |
| S1 | 2 / 2 / 2 | 2 | 2 / 4 / 4；中位数 4 | −2 |

机器核对结果：42 份 JSON 的 `overall_score` 与汇总表逐项一致；14 篇在四类输入中均完整、唯一覆盖；当前中位数、基线中位数及差值均未发现不一致。

## 逐篇执行摘要

### A1｜答案解析敏感性审计

- **当前判断：** 三评为 4/4/2，中位数 4；最可信的是 canonical 指标掩盖了两个床截然不同的 fallback-rescue 组成，但当前只能支撑范围很窄的历史工件审计，不能支撑排序稳健性或跨床能力比较。
- **决策性阻断项：** ① J2 的 `0/23` 来自观察数据后的有利 gate，而原始结果是 `3/28`，且计划九臂中缺一臂；② 两床的题目、提示/解码与格式支持不同，strict/canonical 标签本身又缺少独立金标准；③ 生成配置、随机种子和逐项输出不完整，限制重建与外推。
- **无需新实验的首要动作：** 把 `3/28` 设为唯一主要描述，`0/23` 与其他 gate 全部降为探索性；把 J1 定位为“不可比较性诊断”，把 J3 保留为未识别；删除匿名稿中的既往评审分数和评分元叙事，进一步收窄标题与贡献，并优先处理下文单列的超页问题。
- **真正需要新证据的动作：** 恢复第九臂与完整生成 provenance；建立盲法人工金标和唯一冻结 parser；在相同题目、提示和解码配置的新床上预注册 J2 双向 gate，并扩展到其他模型、任务和 judge。
- **可升档的证据：** 未见数据上的预注册 gate 仍支持同一方向；主要结论在人工金标和独立 parser 下不变；完整九臂可逐项重建；跨模型/任务的 matched-output parser 干预复现 rescue-share 与结论变化。

### A10｜初始化负对照与退化报告

- **当前判断：** 三评一致为 2/2/2，中位数 2；固定短预算下 AR 初始化优于随机初始化的 loss 方向清楚，但这一结果高度预期，单提示上的两臂生成又都退化，尚不足以形成主会级科学命题或经验证的测量方法。
- **决策性阻断项：** ① 没有独立质量/任务锚，PPL、长度、词频和 entropy 的差异不能解释为质量或通用诊断价值；② 四个 data-order 重复主要属于同一固定权重对，三次权重 draw 只是 pilot，无法区分初始化总体、学习速度与最终性能；③ drift 重算门失败、MAUVE 缺失，主随机初始化的 checkpoint/RNG 链也不能从零重建。
- **无需新实验的首要动作：** 改写为配置级工程负对照或 artifact note；把四次运行表述为 fixed-pair consistency，把三次权重 draw 标为 pilot；删除总体初始化、收敛速度、生成质量和机制暗示；对失败诊断明确标记为不可用，并公开现有 checkpoint、RNG 状态和配置（若可恢复）。
- **真正需要新证据的动作：** 加入 step-0、密集学习曲线和更长或等 loss/等阶段比较；扩大为 weight-seed × data-seed 交叉设计；在多提示、checkpoint、模型/任务上加入盲人评或任务标签；修复 drift/MAUVE 后独立复跑。
- **可升档的证据：** 分层重复能稳定估计初始化总体效应；长程结果证明差异不是短程起点优势；内部信号在 held-out 质量锚上有可复现的增量效度；核心 checkpoint 与全部诊断可端到端重建。

### A14｜多选评分与 reveal 协议

- **当前判断：** 三评为 4/6/6，中位数 6；预注册 matched-L 的同题 +13 pp 是当前最强证据，足以说明该冻结床上的完整协议不可互换，但尚不能识别具体组件、一般计算效率或跨设置规律。
- **决策性阻断项：** ① s_conf 与对照同时改变 feedback、order/path、selection 与 estimator，无法把差异归因给某一组件；② 2L 与扩展预算是在主结果后加入，证据层级和多重性必须与原始 primary 分开，NFE/forward 次数也不是实际成本；③ 单一随机日程以及答案位置与 distractor construction 的交互，限制稳定性和外部效度。
- **无需新实验的首要动作：** 只把 matched-L 作为确认性主结果；将 2L、高预算和 fresh-pool 分析统一标为探索性/后续敏感度；把“compute-fair”收窄为 NFE/forward-count matching；把结论限定为床内“完整协议不可互换”，不宣称 confidence、order 或 feedback 的单独优势，并精简证据层级。
- **真正需要新证据的动作：** 在独立题池预注册完整预算 family；运行多个显式 protocol seeds；平衡选项位置和 distractor construction；做 feedback × order × path × estimator 的正交消融；测量 FLOPs、延迟、吞吐、显存，并在额外模型/任务上复现。
- **可升档的证据：** 独立、预注册、实际成本匹配的比较仍保留主要差异；跨 seed 的 gap 明显大于算法随机性；平衡位置/构造后方向稳定；factorial 设计给出可复现的组件级解释，并跨模型/任务成立。

### A23｜解码内置信 readout 与风险

- **当前判断：** 三评一致为 2/2/2，中位数 2；冻结流上 commit–all 的 AUROC 差与 risk@coverage 的近零差异是有用的探索性测量提醒，但 correctness 标签和生成 provenance 的缺口直接动摇全部主结果。
- **决策性阻断项：** ① 重建标签与另一 extractor 在人工样本上约有 30% 分歧，远大于 AUROC `+0.0278` 和约 1 pp 的风险尺度；② 事件流与 decoder 配置不能完整恢复，commit/first 又直接依赖 remasking 轨迹；③ AUROC 与 risk 使用的聚合层级/保留集合并非完全相同，coverage 与等价界还是事后选择，尚未执行真正的 held-out risk gate。
- **无需新实验的首要动作：** 把论文限定为“单一冻结流、重建标签下的离线 readout 敏感性”；删除可部署 risk-control 暗示；明确 AUROC 与 risk 的 estimand/聚合差异，把事后界值和所有分析选择标为假设生成；完整披露下文单列的 PDF-only 重建边界。
- **真正需要新证据的动作：** 对全部或概率抽样的 974 条输出做与 readout 盲隔离的双人语义金标；恢复或重建完整事件流与 decoder manifest；在新数据上预注册统一 score、coverage、risk 界、多重性和 held-out calibration/test，并加入多 seed、额外模型/任务。
- **可升档的证据：** 金标重算后方向和决策不变；新流可从配置、种子和哈希端到端再生；相同 decode-level 排序与 retained set 下仍观察到排序/风险分离；预注册 held-out gate 达到明确的 coverage–risk 目标。

### A31｜schedule geometry 与 recorder 周期

- **当前判断：** 三评一致为 4/4/4，中位数 4；五个已测 arm 中 `P*=L` 的描述性结果稳定，最适合作为“记录窗口会写入 schedule geometry”的 instrumentation warning，而不是模型内在周期或单一机制的证据。
- **决策性阻断项：** ① recorder、partition 和 per-block step allocation 共同变化，不能分离工件来源；② 选择性 detector 缺少保持空间结构的有效 null 与 family-wise 误报校准，development L32 和边界较少的 L24/L48 也不足以作独立确认；③ 单一私有 checkpoint、单任务和不可公开的核心资产限制外部复现。
- **无需新实验的首要动作：** 全文固定为 recorder-level 描述，撤回模型机制和 `depth/content` 因果措辞；把旧 gate 与未充分校准的 arm 降为开发/探索性；将独立床的内容分析移到附录；以唯一 canonical 结果表和事件版本清单消除解读歧义，并发布现有 events/scripts（若许可允许）。
- **真正需要新证据的动作：** 在同一轨迹并行 active-block 与 full-canvas/passive recorder；做 partition × allocation 正交实验；用独立 clean null 校准完整候选选择过程并重新运行 L24/L32；在公开 checkpoint、更多 sampler/任务/seeds 上前瞻复现。
- **可升档的证据：** 双 recorder 与 factorial 设计明确定位信号来源；完整选择算法通过预设的 family-wise I 类错误门；多个 offsets/boundary loci 与独立 L24/L32 复现稳定；公开模型和额外任务仍出现同类 schedule-following pattern。

### A32｜长度与校准差的因果审计

- **当前判断：** 三评为 4/6/6，中位数 6；同题、固定预算下的 `+37-token` 无内容填充没有复现观察到的正 SCE 差，是可信的窄反证，但不等于“自然长度没有因果作用”；第二床仍因 judge 校准不足而未决。
- **决策性阻断项：** ① filler-insertion-under-fixed-budget 同时改变位置、格式、注意力分配与单位 token 计算，不能外推为一般长度机制；② MATH500 的 accepted/rejected 两层尚无完整概率抽样人工金标，现有敏感性带不具可解释的 95% 覆盖；③ 单次解码若非确定性则未覆盖生成方差，方法新意也必须落在随机干预和可排除效应，而非代数恒等式。
- **无需新实验的首要动作：** 将标题、摘要和结论严格限定为该填充操作、单模型与固定 NFE；把第二床未校正正区间降为 sensitivity/POWER_LIMITED，不作为跨床复制；区分同题固定对比与 Random 臂的剂量随机化；用一张 claim→estimand→状态表替代多套谱系读数。
- **真正需要新证据的动作：** 完成 accepted/rejected 双层概率抽样、盲法双人人工标注及联合不确定性传播；证明解码确定性或加入多 seed；增加多剂量、位置/格式控制及内容承载但任务等价的长度干预，并在第二模型/解码设置上复现。
- **可升档的证据：** 人工金标使第二床校正区间达到预设决策标准；多 seed 下窄反证稳定；联合 contrast-of-contrasts 正式区分观察关联与干预效应；更自然的语义保持长度操作和额外模型仍支持同一边界结论。

### C1｜chance 与 input-blind floor

- **当前判断：** 三评为 4/6/4，中位数 4；同标准重算清楚显示 nominal chance 与最佳常量 floor 会改变未经校正的报告叙事，但当前更像有价值的校准/报告审计，尚非已确认的新统计原理或跨模型规律。
- **决策性阻断项：** ① best-constant winner 在同一题集上选择，却未在每次重采样内重选；共享题目、嵌套 arm 与多 cell 依赖也没有由总体/层级推断处理；② v2 阈值、family 和 materiality 规则在看过现有 cells 后形成，同一数据兼作开发与展示；③ OLMo healing 集中现象不能分离家族、损伤强度与训练暴露，且与最近基线相比的方法增量尚不清楚。
- **无需新实验的首要动作：** 把贡献限定为已评估 null family 下的报告校准；区分固定题集运营 floor 与总体参数；将 v2 和未经校正的 cell counts 明确降为探索性，删除 healing 机制暗示；主文只保留一张最终结果矩阵，并精确区分复用组件与协议整合贡献。
- **真正需要新证据的动作：** 恢复逐题 gold/prediction 后做每次重选赢家、保持 cell 依赖的联合 bootstrap/层级分析；在完全未见的模型×数据集 cells 上冻结验证 v2；若保留 healing 解释，运行至少两个家族的匹配 prune-only 与 prune-then-heal 对照。
- **可升档的证据：** selection-aware 总体推断在预设多重性控制下仍支持核心方向；v2 在真正 held-out cells 上稳定改善决策；跨家族匹配实验排除 OLMo-only/训练暴露解释；closest-work 对照显示协议组合产生可验证的新能力。

### C2｜cached-depth read-out

- **当前判断：** 三评一致为 4/4/4，中位数 4；同检索内容下 j=0 重前向优于部署式 j=12 cache 的方向性负结果较稳健，但只能归于整条 operational path，不能归于“depth itself”，也尚无端到端系统价值。
- **决策性阻断项：** ① A/E 的 lower-band context set 不同，所谓 depth-only 因果效应被混杂；② 当前主 paired risk-difference 区间没有传播 discordance 总量不确定性，4k 的 `+27 pp` 又是选择后的最大显著幅度；③ 单模型、qa1、oracle selector，缺少完整生命周期测量和同床最近系统基线。
- **无需新实验的首要动作：** 把 C8-D 改写为部署路径的联合差异，删除 `pure depth`/`independently`；以已有有效的 matched-pair/Newcombe 区间替代或重算主区间，把 4k 数字降为 selected-cell 描述；将 `0.3 GB` 和约 26 次查询的 break-even 严格限定为 read/prefill 账本，并把论文定位为窄的负面测量。
- **真正需要新证据的动作：** 增加 lower-band context、位置和检索结果完全相同、只改变 materialization/read depth 的 arm；在独立冻结 cell/reader 上复现；加入真实检索、多模型/任务；实现写入、索引、prefill、decode、并发、吞吐与峰值内存的全生命周期对照及最近基线。
- **可升档的证据：** 无混杂单因素 arm 仍复现质量下降；修正区间与独立 cell 保持方向和有意义幅度；跨模型/真实任务不依赖被选 4k cell；完整 benchmark 给出可信的质量—资源 Pareto 结论或有价值的系统级负结果。

### C3｜撤回项目的 claim ledger / scanner

- **当前判断：** 三评为 2/2/4，中位数 2；最可靠的是 grad-mode guard 让单一 32k full-context 操作点不再 OOM，以及对撤回边界的诚实记录；这既没有恢复 KV-serving 结论，也没有验证一个通用语义审计方法。
- **决策性阻断项：** ① claim universe 由作者自定义，字符串/局部窗口 scanner 只能证明给定 ledger 的一致性，不能证明语义完整性或发现共同遗漏；② 没有外部、多项目、独立标注的 precision/recall 或 reviewer utility，跨项目 hit count 只证明规则运行过；③ 最终表 caption 与行状态仍存在可目视的内部矛盾，且 serving 侧只有单点 guard 修复，没有 selector、adapter、质量和端到端系统证据。
- **无需新实验的首要动作：** 明确选择论文身份：artifact/negative-results note，或报告方法论文；把 `complete` 永久限定为 author-declared ledger，把 scanner 降为 lint；立即修正两处 caption，并让 final-render gate 检查 caption-to-row 状态；若不恢复系统实验，删除 KV-serving 贡献暗示并压缩项目内术语与撤回史。
- **真正需要新证据的动作：** 对外部、多领域档案建立独立语义金标与对抗释义集，并与 checklist/关键词/assurance 基线比较；若保留 serving 主题，完成 selector、adapter-matched、16k/32k、guard on/off 及质量—延迟—吞吐—内存的受控复跑。
- **可升档的证据：** 冻结 scanner 在隐藏 gold corpus 上取得可信的 precision/recall、覆盖和人工节省，且优于简单基线；或恢复一个可完整评价的 serving 问题，并由受控 ablation、独立重复和完整生命周期结果支撑。

### P1｜AUDIT-GATE 可重放性框架

- **当前判断：** 三评为 2/2/4，中位数 2；当前 runner 的若干机械分支和合成 mutation 确实可执行，但只能证明规则在作者构造输入上工作，不能证明自然 archive 上的 operability 分类有效。
- **决策性阻断项：** ① 当前完整规则是在观察失败后修订的，branch coverage 与 fixtures 同时承担开发和评价，缺少真正 held-out 的自然 archive；② operator 可以事后补写 cause 使漂移通过，且静态文件/字符串存在不等于 endpoint 可动态执行；③ 相对一般 schema、attestation、assurance/checklist 的新增语义能力与实际效用未建立。
- **无需新实验的首要动作：** 将当前版本明确称为 retrospectively developed、transparently versioned v2，而非完整预注册验证；把 `operable/replayable` 收窄为 `statically registered/referenced`；未经事前签名的 drift cause 一律降为 POSTPONE/人工仲裁；主文只保留最终规则、项目无关示例和唯一 canonical verdict 表，并修正可复制命令错误。
- **真正需要新证据的动作：** 冻结最终规则后在作者外、多项目自然 archives 上做盲标校准；加入动态沙箱 reachability；用带时间戳、签名的 append-only cause registry；与简单 schema/hash/checklist 基线比较，并让第三方 clean-room 执行。
- **可升档的证据：** 独立自然 archive 上的 false-open/false-kill 与漏检达到预设界限；新 held-out cell 不再参与规则形成；动态执行能发现静态检查漏掉的真实故障；第三方能够安装、运行并证明相对基线的新增价值。

### P3｜path-ordered data value

- **当前判断：** 三评一致为 4/4/4，中位数 4；同一样本早/晚移动与 routing-only 压力测试有效说明横截面 B2 不能单独识别路径效应，但当前负向读数仍无法与曝光量和 carrier-specific estimator bias 分开。
- **决策性阻断项：** ① 早进入样本获得更多训练曝光，现有设计不能识别“纯顺序”效应；② ridge 负控制与 sine 主载体几何不同且自身不居中，正向偏差不能排除 sine 上的负向伪影；③ 主要证据仍限于合成/简单载体，closest-work 边界和逐 seed 工件复现不足。
- **无需新实验的首要动作：** 将结论收窄为 entry-time/exposure-dependent marginal value，不宣称纯 order/path effect；把 ridge 结果称为经验诊断而非排除证书；分别定义 τ 与 g 的尺度和门槛，若无冻结 g 门则只报告区间与方向；补齐现有配置、逐 seed 表和版本哈希，并明确下文 PDF-only 重建限制。
- **真正需要新证据的动作：** 运行等曝光、仅改变相对顺序的配对实验；构造尽量保持 sine 几何/优化流程、但可证明 set-ordered 的匹配负控制，或给出可验证的偏差符号界；在自然数据或非凸模型上前瞻复现并发布完整匿名工件。
- **可升档的证据：** 等曝光实验仍给出预设幅度的负效应；匹配负控制的偏差界严格排除主结果为伪影；新 seeds 与自然载体对稳健尺度、删一 seed 和多重性处理稳定；干净环境可端到端重放所有主表。

### P5｜末期冲击与 checkpoint 选择

- **当前判断：** 三评一致为 4/4/4，中位数 4；时序干预和 64/64 shocked、0/64 healthy 的冻结结算内部清楚，但 gate 与 best-val/early-stop 完全同值，相对受损 last 的收益没有证明新增 checkpoint policy 或独立告警价值。
- **决策性阻断项：** ① 当前构造中 gate 没有找到既有验证选择器找不到的 checkpoint；② healthy 轨迹近乎单调，shock 机制单一且完全分离，无法估计真实训练波动下的阈值运输、误报与未知冲击召回；③ seed 1 是否在冻结前可见的时间线仍需证明，基线选择器配置与最近变化点/漂移方法对照也不完整。
- **无需新实验的首要动作：** 将论文定位为检测协议/负结果，而非新 checkpoint policy；明确 gate 与 best-val/early-stop 共延，禁止把 `last` 对比写成增量价值；给出不可变预注册时间线和 seed 1 访问状态，若无法证明则把当前结算标为部分前瞻；补齐基线公式、信息集、存储成本和 Monte Carlo 分辨率。
- **真正需要新证据的动作：** 执行能让 gate 与 best-val/early-stop 分离的 E2；在自然非单调 healthy 轨迹、不同 shock 强度/时点/类型和更多 FIT seeds 上做 E6 类结算；若 seed 1 受污染则用全新 seeds；加入成本匹配的变化点/漂移检测基线。
- **可升档的证据：** 冻结 E2 中 gate 提供独立 checkpoint/决策价值；跨未见 shock 与现实 healthy 轨迹仍满足预设 FAR/召回；阈值对 FIT seeds 和 shock 条件稳定；相对标准监控方法在错误率、时延或成本上有明确增量。

### P7｜activation-tail turnover 与量化损伤

- **当前判断：** 三评一致为 4/4/4，中位数 4；最可信的结果是 4B 上 turnover 与 block depth 高度纠缠、1.7B 上关系弱且未决，因此它目前是一条“任何层级信号必须先与 depth 比较”的设计警示，而不是经验证的运行时触发器。
- **决策性阻断项：** ① 探测层被深层过采样，block 也不是独立抽样单位；对单 checkpoint 的层 bootstrap/Fisher 推断不能外推，turnover 是否超越 depth 未建立；② masked/unmasked 与旧/新分析数字混入正文，影响区间、LOO 和跨载体结论；③ 只完成 2/8 注册 cells，且没有实际 fallback policy、真实 prompt shift 或系统收益。
- **无需新实验的首要动作：** 从唯一 canonical masked artifact 自动生成全文数字，删除所有 stale unmasked 值并附版本映射；把层重采样降为有限集合敏感性，不赋予总体覆盖含义；把 1.7B 统一写成 weak/unresolved；将模板代理域准确命名，并把贡献限制为 instrumentation lesson。
- **真正需要新证据的动作：** 至少在一个 checkpoint 做全层或事前深度均衡测量，并在多个独立 checkpoint/domain/quantizer 上复现；加入 depth、权重/激活幅度、Hessian/量化误差等竞争基线；完成冻结的 cell family，并实际运行 fallback policy、damage reduction 与成本结算。
- **可升档的证据：** turnover 在 held-out checkpoint/layer 上稳定优于 depth 与其他低成本基线；以 checkpoint/domain 为高层单位的复现成立；真实 domain shift 和更多量化设置保持增量预测价值；实际 fallback 同时降低损伤并满足系统成本门。

### S1｜fixed-support recovery 与 exact 2:4

- **当前判断：** 三评一致为 2/2/2，中位数 2；`2.675 pp → 0.118 pp` 目前只能作为两组聚合点的算术记录，尚未证明相同三个 support 经相同恢复后范围被压缩，也没有形成可部署的原生 exact-2:4 或 SparseForge 方法证据。
- **决策性阻断项：** ① before 可能是跨 seed 方法均值、after 是固定 mask 恢复点，缺少同一 support/checkpoint 的配对映射、逐 item 预测与多恢复 seeds，range 也没有联合不确定性；②推理时仍有稠密低秩 SLoRB 分支，尚未完成 fold→exact-2:4 reprojection、逐组验证和导出后质量/系统测量；③ SparseForge 的匹配预算 endpoint、组件消融与强基线缺失，5B 历史 checkpoint/语料/评测 provenance 又不可核验。
- **无需新实验的首要动作：** 若不能恢复一一映射，删除“recovery compresses support spread”的配对因果读法，只报告不可配对横截面范围；删除无逐项数据支撑的显著性结论；明确当前模型是 sparse base + dense branch，而非 deployed exact-2:4；把不可验证 5B 数字移出证据性表图，删除未检验猜想并重构为诚实的测量审计。
- **真正需要新证据的动作：** 为每个确切 support 提供相同实例的 0-token/625M 起终点、SHA、逐 item 预测和多个恢复 seeds；完成 fold、reprojection、groupwise 2/4 assertion，并重测质量、吞吐、内存；运行 matched-budget SparseForge 组件/AST/CAST/固定 mask 基线消融，使用可访问且许可清楚的语料与 checkpoint。
- **可升档的证据：** 联合层次分析证明同一 supports 的范围确实实质收缩；无活跃稠密分支的 exact-2:4 导出保留质量并产生真实硬件收益；匹配预算多 seed 消融显示 SparseForge 组件有可归因增量；核心资产可由第三方端到端复现。

## 单列的格式与重建阻断

### A1 超页

`qa_manifest.tsv` 记录 A1 的 polished PDF 共 23 页，主文内容延至第 13 页，不符合 9 页主文上限。视觉 QA 为 pass，构建为 `pass_with_nonfatal_float_warnings`，且来源是 exact LaTeX；因此这不是构建失败，而是明确的提交格式硬阻断。首要动作是至少压回 9 页主文，并在压缩后重新做引用、浮动体和匿名性检查；其中既往评审分数/元叙事应直接删除，而不是移入主文其他位置。

### A23 与 P3 的 PDF-only 重建

`qa_manifest.tsv` 将 A23（14 页、主文不超过 9 页）和 P3（8 页、主文不超过 9 页）都标为 `pdf_only_reconstruction`；两者视觉 QA 和 build 均为 pass。该状态表示当前可交付版不是权威原始源码的精确快照，不能把“构建成功”解释为源级可复现。后续任何修改都应先与作者权威源逐项对齐，保存 reconstruction map、字体/图表/引用差异和最终哈希；在此之前，不应把重建版当作可无损继续迭代的 canonical source。

## 其余 QA 备注

14 篇视觉 QA 均为 pass，构建均为 pass 或仅含非致命 warning。除上面三项外，`qa_manifest.tsv` 还将 A32、C1 标为 exact LaTeX snapshot 但需重新固定 assets；这属于交付/复现待办，不是评分数据不一致，也不应被混入科学负结果。
