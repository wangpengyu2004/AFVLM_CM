# Proposed method: Sensitivity-Aware Asynchronous LoRA Consolidation

本项目中的 `ours` 已实现为 **Sensitivity-Aware Asynchronous LoRA
Consolidation**。它面向固定任务身份客户端、LoRA-only 的异步联邦 VLM
指令微调。冻结的 LLaVA 主干不参与敏感性估计、通信或聚合。

## 1. 客户端上传内容

客户端 `k` 在 server version `s` 下载 LoRA 状态 `phi^s`，本地训练得到
`phi_k`。上传内容包括：

- LoRA 本地状态与 delta；
- `base_version = s` 和对应的不可变 `base_state = phi^s`；
- 固定任务 ID `q_k`；
- 每个 LoRA group 的非负敏感性 `C[k,g]`；
- 实际 optimizer steps、样本量和常规运行 metadata。

原始数据、梯度和冻结主干均不上传。已有 worker 使用正常 backward 产生的
LoRA 梯度估计敏感性，不增加第二次 backward。

## 2. 三种敏感性估计器

通过 `method.params.sensitivity_estimator` 选择。

### 2.1 `module_gate`（默认主方法）

对 LoRA module `l` 的有效更新 `Delta W_l = scale_l B_l A_l` 引入虚拟 gate：

```text
Delta W_l(z_l) = z_l scale_l B_l A_l
```

正常训练点为 `z_l = 1`。已有梯度满足：

```text
h_l = dL/dz_l
    = <grad_B_l L, B_l>_F
    = <grad_A_l L, A_l>_F
```

代码对 A、B 两个有限精度估计取平均，再在 local optimizer steps 上维护：

```text
C[k,l] = EMA(h_l^2)
```

该定义作用于完整 `B_l A_l`，在 `B <- cB, A <- A/c` 重参数化下保持不变。
每个 module 只上传一个 scalar。

### 2.2 `rank_gate`（消融）

把 `B A` 写成 rank-1 分量之和，并为每个 rank 引入 gate。敏感性为：

```text
h[l,r] = <grad_B[:,r], B[:,r]>
       = <grad_A[r,:], A[r,:]>
C[k,l,r] = EMA(h[l,r]^2)
```

服务端也按 rank 分片计算陈旧度和融合系数。A 的第 `r` 行与 B 的第 `r`
列使用同一个系数。该实现保留固定 LoRA rank；由于直接在 A/B 因子上插值，
融合后的 `BA` 并不等于两个 dense `BA` 的严格线性插值，这是固定-rank
工程实现需要在论文中披露的边界。

### 2.3 `adam_v`（效率消融）

复用 Adam 二阶矩思想，在每个 module 内对梯度平方取均值：

```text
v_g <- beta2 v_g + (1-beta2) mean(grad_g^2)
C[k,g] = v_g / (1-beta2^steps)
```

分组均值与先维护逐参数 `v` 再分组求均值在代数上等价，但无需保存一份额外的
逐元素二阶矩。它是 optimization-trajectory sensitivity surrogate，不宣称等于
endpoint Fisher。

三种 estimator 的正值都先以客户端内部 group mean 归一化，再按
`sensitivity_floor` 和 `sensitivity_ceiling` 裁剪，避免绝对尺度任意地改变
服务器精度权衡。

## 3. 功能性陈旧度

更新到达时当前 server 状态为 `phi^t`。对客户端上传的敏感性计算：

```text
D[t,k] = sum_g C[k,g] ||phi^t(g) - phi^s(g)||_2^2 / |g|
U[k]   = sum_g C[k,g] ||phi_k(g) - phi^s(g)||_2^2 / |g|

delta[k,t] = sqrt(D[t,k] / (U[k] + epsilon_update))
rho[k,t]   = exp(-gamma * min(delta[k,t], delta_max))
```

`D` 衡量客户端离线期间，当前 global 在“该客户端真正敏感的 LoRA group”上
移动了多少；`U` 是客户端自身有效更新能量。因此相同 version staleness 的两个
客户端可以获得不同可靠度。`version_staleness` 仍记录用于分析，但不直接决定
融合权重。

若 `U <= min_update_energy`、optimizer steps 不足或能量非有限，更新会被显式
拒绝，不增加 server version，避免用 epsilon 掩盖无效更新。

## 4. 按任务历史敏感性记忆

server 为每个任务 `q`、每个 group `g` 保存：

```text
Omega[q,g] <- memory_beta * Omega[q,g]
              + (1-memory_beta) * r_k * rho[k,t] * C[k,g]
```

`r_k` 由 `quality_weighting` 决定；主配置使用 `constant`，避免数据量差异在没有
额外校准时淹没敏感性。可选值为 `constant`、`num_samples`、
`sqrt_num_samples` 和 `optimizer_steps`。

当前历史保护精度为：

```text
Omega_t[g] = omega_prior + sum_q pi_q Omega[q,g]
```

默认 `pi_q` 是 system profile 中所有任务的固定等权权重，防止高频任务仅因到达
次数更多而支配记忆。也可通过完整的 `task_weights` 映射显式设置。`omega_prior`
提供 cold-start 保护。

## 5. 逐组精度融合

在一个不可变的 `phi^t` 与 `Omega_t` 快照上同时计算所有 group 系数：

```text
p_local[k,g] = r_k rho[k,t] C[k,g]

alpha[k,g] = p_local[k,g]
             / (p_local[k,g] + lambda_history Omega_t[g] + epsilon_precision)
alpha[k,g] <- min(alpha[k,g], alpha_max)

phi^(t+1)(g) = phi^t(g) + alpha[k,g] (phi_k(g) - phi^t(g))
```

成功应用后只更新来源任务 `q_k` 的历史记忆并令 server version 加一。每个 group
的更新是位于当前状态和客户端状态之间的凸组合；它防止单次 group overshoot，
但不构成在无限延迟或任意冲突任务下的全局收敛保证。

实现直接使用 Update 中已经冻结的 `base_state`，因此无需为每个旧 server version
永久保留完整模型副本。并行 worker、EDF 物理调度和虚拟 Plan 顺序不变。

## 6. 配置

默认配置位于 `configs/methods/ours.yaml`。主实验保持：

```yaml
method:
  name: ours
  params:
    sensitivity_estimator: module_gate
    sensitivity_beta: 0.95
    gamma: 1.0
    delta_max: 10.0
    lambda_history: 1.0
    memory_beta: 0.9
    omega_prior: 1.0
    alpha_max: 0.9
    quality_weighting: constant
    task_weights: null
```

`task_weights: null` 表示对 AFVLM-CM 的六个任务自动等权。若手动设置，必须完整
覆盖 `cls`、`caption`、`vqa`、`chart_vqa`、`visual_reasoning`、`grounding`，
并使用非负权重。

## 7. 运行

使用兼容配置：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/run_one.sh ours 2
```

推荐先基于当前 `configs/base.yaml` 创建不可变 profile，再运行：

```bash
python tools/generate_system_profiles.py \
  --profile ours_module_gate_e1_bs1_ga4_r10_s42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/run_one.sh ours 2 ours_module_gate_e1_bs1_ga4_r10_s42
```

将 setting 改为 `5` 或 `10` 即运行另外两个 AFVLM-CM 客户端规模。Rank-Gate 或
Adam-v 消融应修改 `configs/methods/ours.yaml`（或准备该实验的基础配置），然后用
新的、能表达 estimator 的 profile 名重新生成快照；不要修改已有 profile。

## 8. 输出与复现检查

`events.jsonl` 的 `result_metadata` 额外记录：

- version staleness 与 functional staleness；
- drift/update energy；
- reliability 与 quality weight；
- group alpha 的 mean/min/max；
- estimator 和 group 数量；
- 被拒绝更新的明确原因。

最终 checkpoint 的 method state 保存 `task_memory`、归一化 `task_weights` 和全部
方法参数。以下检查不加载 7B 权重、不开始训练：

```bash
python -m unittest discover -s tests -v
python -m compileall -q code scripts tools tests
python tools/validate_repository.py
ruff check code scripts tools tests
ruff format --check code scripts tools tests
```

## 9. 实现边界

- 默认主方法是 Module-Gate；Rank-Gate 与 Adam-v 是同一方法的 estimator 消融，
  不是三个独立注册方法。
- 仅支持 LoRA-only 联邦状态。若启用 `train_mm_projector` 或其他非 LoRA 参数，
  方法会在模型/服务器初始化阶段直接报错，避免无定义的混合分组。
- 敏感性是局部二次精度代理，不是完整 Hessian，也不等价于真实任务泛化重要性。
- 历史记忆只保存按任务、按 group 的 scalar，不保存原始数据或完整客户端梯度。
- 当前项目仍只保存最终训练 checkpoint，不提供任意中间异步事件恢复。
