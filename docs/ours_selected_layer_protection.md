# Ours：任务结束后的重要层累计保护

2026-10-20 更新：仅把入选层额外年龄保护由逐模块max改成任务加权求和。
保留原选层/计龄/冻结快照，boost=2、scale=18不变。旧max profile不改写。
当前模板采用首次观测初始化EMA。
未知任务H=1，首次成功C替换默认值，后续普通EMA、不做偏差修正、删除外部固定1。
另将默认alpha改为rwC/(rwC+Omega+eps)，alpha_max=null，不再额外裁剪到0.5。
不改客户端敏感度、功能性陈旧度、18次频率窗口、LoRA增量聚合或评估，不加入敏感度消融。
不改其他 baseline、DDP、本地训练、数据、系统配置及 Plan。

## 当前配置与旧方案对照

```yaml
history_update_mode: first_observation_ema
history_beta: 0.9
base_precision: 0.0
alpha_max: null
history_age_enabled: true
history_age_mode: selected_layers_sum
history_age_boost: 2.0
history_age_time_scale: 18.0
```

`history_age_mode` 仅控制年龄项：新模板用 `selected_layers_sum`；
显式 `selected_layers_max` 的旧 profile 仍按原max运行；
缺少该字段的旧 profile 继续使用旧 `all_modules_sum`，保留已有全层年龄方案作为对照。
`history_age_enabled: false` 关闭额外年龄保护，恢复基础保护。
历史更新模式与年龄模式独立：缺少 `history_update_mode` 的旧 profile 保留从0初始化、
按任务计数偏差修正、未知H=0以及外部固定1。不要只删除新模板的模式字段却保留base0。
不要把旧日志/检查点改名当成新算法结果；原状态不包含结束快照时不能推断新名单。

## 聚合流程

1. 校验实际 `(client_id, local_round)` 和 `update_id`。完整 Plan 必须覆盖全部配置客户端。
   任务所有客户端的**每个唯一计划作业成功纳入**后才结束；仍有未接受作业就不结束。
   拒绝不会扣 pending、不更新 EMA/频率窗口/接受计数，允许之后用同一作业 ID 重试；
   已成功作业再次到达直接报重复错误。不会自动新建重试或新增客户端上传。
2. 用聚合前历史计算基础保护与旧任务额外保护，计算原 alpha，执行一次原增量聚合。
3. 成功后推进接受计数并更新当前任务 EMA/频率窗口。若任务刚结束，冻结包含最后一次
   成功上传的有效 EMA `H_end`、结束计数和选层名单。新模式不再修正偏差，旧模式仍修正。
   不是保存最后一次 C；首次观测替换只发生一次。
4. 原评估间隔照常执行；冻结名单不产生额外模型更新、上传或 optimizer step。

没有按一个客户端结束、窗口内消失、GPU实际完成或上传等待时间判断任务结束。
旧 `all_modules_sum` 为兼容已有轨迹保留其原“已处理作业”完成口径；严格成功完成口径
是新模式的要求。若某个作业被拒绝且没有重试成功，该任务就一直保留未完成状态。

## 任务结束时选层

在任务自己的冻结历史上：

```text
E[q,m] = max(H_end[q,m] - 0.5, 0)
I[q,layer] = mean(E[q,layer,Q/K/V/O])
p[q,layer] = I[q,layer] / sum_layers I[q,layer]
M[q,layer] = 1[p[q,layer] > 1/L + 1e-8]
```

LLaVA-1.5-7B 的 L=32，平均占比1/32=3.125%。只按任务自己的历史选层，不指定8层，
不硬编码先前实验的层名单。完整 Q/K/V/O 组从 LoRA 元数据解析，32层128模块；
缺少投影组或名称不匹配明确报错，不猜测归属。
代码中的扣除底座取 `1-sensitivity_uniform_mix`，当前0.5；仅用于选层，
不会修改 C、EMA 或基础历史。无正超额量或没有显著高于平均的层时，名单为空，记录原因。
非有限历史直接报错。名单与 H_end 在结束后冻结，不被其他任务改写。

## 额外保护：逐模块累计

令 u 为本次聚合**之前**已成功接受的上传数，e_q为任务结束时的接受数：

```text
age[q] = u - e_q
b[q] = 2 * (1 - exp(-age[q]/18))
P[q,m] = b[q] * M[q,layer(m)] * H_end[q,m] / 6
P_sum[m] = sum_finished_tasks P[q,m]    # 无结束任务时为0
Omega_base[m] = sum_tasks H_before[q,m]/6   # 未观测任务H=1；没有外部固定1
Omega_new[m] = Omega_base[m] + P_sum[m]
```

不保留旧的全层年龄项，不做 L/K 补偿，不把总保护量归一化回旧方案。
基础历史仍多任务求和，新增项也按任务原权重求和。例如同模块两任务的需求为0.30和
0.20时，额外量为0.50，不再是0.30。入选层四个模块都可获得保护，但各自用自己的
H_end。六任务固定权重仍各1/6，不按结束任务数重新归一化，也不除以贡献任务数。
这不是跨上传反复加旧P：每次按当前年龄和固定H_end重新计算一次，避免重复计入。
由于结束后H不再更新，该公式也等价于将已结束任务在入选模块的H乘以(1+b[q])，
再与其他任务的H按固定1/6求和；不是对所有任务历史整体乘一个共享权重。

新模式要求 `base_precision=0`；`history_strength` 仍同时缩放基础历史和额外历史，
`history_lambda` 照常进入原alpha分母，二者默认均为1。选层扣除的0.5是客户端敏感度
的均匀下限，与删除的外部固定1不是一回事，仍保持不变。

任务在第e次接受更新结束，第e+1次聚合前年龄0、该任务额外量0；第e+2次才为正。
新任务结束不会重置旧任务年龄。最终一个任务结束后若无后续更新，不虚构保护迭代。
覆盖并集仍可扩大；相同轨迹上sum不小于max，重叠位置保护更强、alpha可能更低。
只改合成方式，不再同时增强boost或缩短scale，避免混淆因素；不能据alpha推断性能提升。

## 记录哪些数据

原 `events.jsonl/result_metadata/ours_diagnostics` 保留原字段，新历史诊断 schema为6；
旧选层模式仍为5。日志另记录历史模式、是否偏差修正、未知先验值及外部底座。

- `memory.finished_task_snapshot`：仅在新任务结束时记录任务、结束计数、最后上传ID、
  每客户端 planned/accepted/pending、H_end、每层分数/占比/mask、名单、阈值和选择原因。
- `memory.history_age_before/after`：结束位置、接受数、年龄、各任务名单、选层并集、
  每模块额外保护、各任务额外贡献、正贡献任务名单/数量、正额外模块数及统计。
  sum模式不记录虚构的最大来源；旧max模式仍保留 `max_source_by_module`。
- `historical_precision_unweighted_before/after`：实际基础Omega；
  `historical_precision_before/after`：基础+额外的总Omega。实际alpha对应before，
  不用更新后的历史回填本次alpha。
- 原有客户端C、r/w、增量和alpha日志保留。

导出命令不变：

```bash
python tools/export_ours_diagnostics.py runs/你的本次运行目录
```

原 update/module CSV 保留模式、名单并集、正额外数量、额外保护和层号。
sum模式新增 `extra_precision_by_task_before/after`、`contributors_by_module_before/after`；
module表新增 `history_age_task_demands_before/after`、`history_age_contributors_before/after`、
`history_age_contributor_count_before/after`，原max来源列在sum模式留空。
有任务结束记录时另写 `ours_task_completion_diagnostics.csv`，保存完整H_end与选层信息。
旧日志缺少新字段时留空，不伪造当时的选层或额外保护。
新历史 method state schema为5，旧模式保留4；年龄子状态仍为2，保存完整快照/名单/接受作业集合；
加载时校验快照、mask、最终EMA、客户端完成状态及计龄一致，不新增训练恢复工作流。

## 新实验启动

新建 profile 才会使用模板的新模式；不要直接复用旧 ours YAML，后者仍是旧模式。
数据、batch、epochs、rounds与系统设置都不变时可原样复用 Plan：

```bash
python tools/generate_system_profiles.py \
  --profile module_gate_sum_firstema_nocap_b2_t18_e1_bs4_ga4_r10_s42 \
  --reuse_plans_from 你的旧profile名称

bash scripts/run_one.sh ours 2 module_gate_sum_firstema_nocap_b2_t18_e1_bs4_ga4_r10_s42
```

旧 profile 名换成真实名称；新 profile 从当前模板生成全部常规方法配置，不复制旧快照的
超参数。先确认模板符合你的既有实验。数据/训练/系统条件不同则不要复用旧 Plan。
不改已有 profile、Plan、数据或运行目录，也没有自动上传 GitHub。
