# 解析式 KL 散度功能说明 (`kl_loss_type: analytic_kl`)

> **本文档适用于 verl `0.6.1` 用户。**
> 代码已直接应用在 [`feat/analytic-kl-0.6.1`](https://github.com/K1XE/verl/tree/feat/analytic-kl-0.6.1) 分支，
> 只需 wget 5 个文件即可使用，无需手动 patch。

---

## 概述

本功能为 verl 0.6.1 新增 `kl_loss_type: "analytic_kl"` 选项，支持对完整词表或 top-k token
进行**解析式**（analytic）KL 散度与 JSD 计算，覆盖以下两篇论文的方法：

- **OPCD**（arXiv:2602.12275）—— top-k 近似反向 KL，默认 k=256
- **OPSD**（arXiv:2601.18734）—— full-vocab 广义 JSD

实现采用 **OEL 预算架构**（One-time Estimation Layout）：在每个训练步骤进入 `update_policy` 之前，
先做一次 ref forward，将 student top-k indices 和 ref gathered log-probs 写入 batch，
`update_policy` 阶段直接读取，无需额外 ref forward。
这将 ref forward 次数从 K×M 次（K 个 PPO epoch × M 个 mini-batch）压缩为每步 **1 次**。

---

## 与现有 k1/k2/k3 的本质区别

| | k1 / k2 / k3 (`low_var_kl`) | `analytic_kl` |
|---|---|---|
| 输入 | `(B, L)` — 只有被采样那一个 token 的 log_prob | `(B, L, V)` — 完整词表 logits |
| 本质 | 单样本 MC 估计，有采样噪声 | 直接对词表（子集）求和，解析计算 |
| ref forward 时机 | rollout 时已算好 `ref_log_prob`，update_policy 无需 | OEL：update_policy 前预算 **1 次**（而非 K×M 次） |
| 显存开销 | 极低 | top-k: 中等；full-vocab: 较高 |
| KL 无偏性 | 采样噪声，期望无偏 | top-k 近似（k≥256 时误差通常 <1%）；k=0 全词表则精确 |

---

## 支持的散度模式（`topk_kl_mode`）

| 模式 | 数学公式 | top-k 选取方式 | 论文依据 |
|---|---|---|---|
| `reverse_kl`（**默认**）| KL(π_θ ‖ π_ref) | 按 **student** 概率取 top-k | OPCD arXiv:2602.12275 |
| `forward_kl` | KL(π_ref ‖ π_θ) | 按 **reference** 概率取 top-k | GKD arXiv:2306.13649 |
| `jsd` | JSD_β = β·KL(π_ref‖m) + (1-β)·KL(π_θ‖m)，m=β·π_ref+(1-β)·π_θ | student + reference top-k 取并集 | OPSD arXiv:2601.18734 |
| `symmetric_kl` | 0.5·[KL(π_θ‖π_ref) + KL(π_ref‖π_θ)] | student + reference top-k 取并集 | Jeffreys 散度 |

---

## top-k vs full-vocab

- **`topk_kl_k: 256`**（推荐）→ 只对 student/ref 概率最高的 256 个 token 求和
  OPCD 推荐设置，节省显存；在 32K 词表上误差通常 <1%。走 OEL 预算路径，每步仅 1 次 ref forward。

- **`topk_kl_k: 0`**（全词表）→ 对全部 V 个 token 求和，精确计算
  显存消耗约为 top-256 的 `V/256` 倍（32K 词表约 125×）。
  **当前实现暂不支持全词表走 OEL 预算路径**，需在 `DataParallelPPOActor.__init__` 中注入 `ref_module`，
  显存开销较大，**不推荐日常使用**。

---

## 应用方法（verl 0.6.1 用户）

### 前置条件

1. **确认 verl 0.6.1 已以 editable 模式安装**（必须）：

   ```bash
   # 查看 verl 的安装位置
   python -c "import verl; print(verl.__file__)"
   ```

   如果输出类似 `/shared/verl/verl/__init__.py`（指向你本地目录），说明已是 editable 安装，直接进入第 2 步。

   如果输出类似 `.../site-packages/verl/__init__.py`，则需要先切换为 editable 安装：

   ```bash
   # 找到 verl 源码目录（package 安装时通常无源码，需先 clone）
   git clone https://github.com/K1XE/verl.git /shared/verl
   cd /shared/verl && git checkout v0.6.1  # 切回 0.6.1 基线

   # 卸载旧安装，改为 editable
   pip uninstall verl -y
   pip install -e /shared/verl

   # 在每台机器的 conda 环境里各执行一次（共享存储只需 clone 一次）
   ```

2. **记下 verl 源码根目录路径**（下文称 `VERL`）：

   ```bash
   export VERL=$(python -c "import verl, os; print(os.path.dirname(os.path.dirname(verl.__file__)))")
   echo "VERL 路径：$VERL"   # 例如 /shared/verl
   ```

### wget 5 个文件

```bash
# GitHub raw 文件基础 URL
BASE=https://raw.githubusercontent.com/K1XE/verl/feat/analytic-kl-0.6.1

# verl 源码根目录（根据实际情况修改）
VERL=/shared/verl     # ← 改成你的实际路径

# 下载并替换 5 个文件
wget -O $VERL/verl/trainer/ppo/core_algos.py        $BASE/verl/trainer/ppo/core_algos.py
wget -O $VERL/verl/workers/actor/dp_actor.py         $BASE/verl/workers/actor/dp_actor.py
wget -O $VERL/verl/workers/fsdp_workers.py           $BASE/verl/workers/fsdp_workers.py
wget -O $VERL/verl/trainer/ppo/ray_trainer.py        $BASE/verl/trainer/ppo/ray_trainer.py
wget -O $VERL/verl/trainer/config/actor/actor.yaml   $BASE/verl/trainer/config/actor/actor.yaml
```

> **多机共享存储**：如果所有机器通过共享存储访问同一个 `VERL` 目录，
> 只需在 **任意一台**机器上执行上述 wget 命令，其他机器立即生效。
> 每台机器需要各自执行一次 `pip install -e $VERL`（注册 editable 指针），但代码只需下载一次。

### 验证安装成功

```bash
python -c "from verl.trainer.ppo.core_algos import topk_analytic_kl; print('OK:', topk_analytic_kl)"
```

输出类似 `OK: <function topk_analytic_kl at 0x...>` 即表示成功。

也可运行[快速验证脚本](#快速验证无需-gpu)确认所有散度模式正常工作。

---

## rllm 用户附加 patch（必读）

> 如果你用的是 [rllm](https://github.com/rllm-org/rllm)，完成上面的 wget 之后还需要手动 patch 一处代码。

### 原因

rllm 的 `AgentPPOTrainer` 继承自 `RayPPOTrainer`，但训练主循环是自己的 `fit_agent()` 方法，
**不调用** `ray_trainer.py` 的 `fit()`。
上面 wget 的 `ray_trainer.py` 里的 OEL 预算块因此永远不会被触发，
`update_policy` 里会找不到 `kl_topk_indices` 和 `ref_log_prob_topk`，导致 analytic_kl 静默失效或报错。

### 找到插入位置

在你的 `agent_ppo_trainer.py`（rllm 仓库）里搜索 `compute_ref_log_prob`，找到这一段：

```python
if self.use_reference_policy:
    # compute reference log_prob
    with marked_timer("ref", timing_raw):
        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)
```

### 在其正后方插入以下代码

（缩进与上面的 `if self.use_reference_policy:` 对齐）

```python
# OEL 预算：analytic_kl 专用，每步 1 次 actor+ref forward
# 将 update_policy 内的 K×M 次 ref forward 压缩为此处 1 次
actor_cfg = self.config.actor_rollout_ref.actor
if (
    getattr(actor_cfg, "use_kl_loss", False)
    and getattr(actor_cfg, "kl_loss_type", "") == "analytic_kl"
    and getattr(actor_cfg, "topk_kl_k", 256) > 0
    and self.use_reference_policy
):
    with marked_timer("analytic_kl_topk_precompute", timing_raw):
        topk_idx_proto = self.actor_wg.compute_kl_topk_indices(batch)
        batch = batch.union(topk_idx_proto)
        ref_topk_proto = self.ref_policy_wg.compute_ref_log_prob_topk(batch)
        batch = batch.union(ref_topk_proto)
```

> **注意**：`self.actor_wg` 是 rllm 里 actor worker group 的名字。
> 如果你的版本命名不同（如 `self.actor_rollout_wg`），替换掉。
> 如果你的版本没有 `marked_timer`，去掉 `with marked_timer(...):` 那行，只保留内部 3 行。

---

## 配置方法

在训练脚本或 yaml 的 `actor_rollout_ref.actor` 节中添加：

```yaml
actor_rollout_ref:
  model:
    use_fused_kernels: false   # ⚠️ 必须为 false，见"使用限制"

  actor:
    use_kl_loss: true
    kl_loss_type: analytic_kl    # ← 改这里
    kl_loss_coef: 0.001

    # analytic_kl 专用参数（其他 kl_loss_type 忽略以下字段）
    topk_kl_k: 256               # top-k 个 token；0 = 全词表（显存更大，见限制）
    topk_kl_mode: reverse_kl     # reverse_kl | forward_kl | jsd | symmetric_kl
    topk_kl_jsd_beta: 0.5        # 仅 jsd 模式使用，β=0.5 为对称 JSD
```

### 常用配置预设

**OPCD 风格（top-256 反向 KL，推荐）：**

```yaml
actor_rollout_ref:
  model:
    use_fused_kernels: false
  actor:
    use_kl_loss: true
    kl_loss_type: analytic_kl
    kl_loss_coef: 0.001
    topk_kl_k: 256
    topk_kl_mode: reverse_kl
```

**OPSD 风格（top-256 广义 JSD）：**

```yaml
actor_rollout_ref:
  model:
    use_fused_kernels: false
  actor:
    use_kl_loss: true
    kl_loss_type: analytic_kl
    kl_loss_coef: 0.001
    topk_kl_k: 256
    topk_kl_mode: jsd
    topk_kl_jsd_beta: 0.5
```

**OPSD 原始风格（full-vocab 广义 JSD，高显存，不推荐）：**

```yaml
actor_rollout_ref:
  model:
    use_fused_kernels: false
  actor:
    use_kl_loss: true
    kl_loss_type: analytic_kl
    kl_loss_coef: 0.001
    topk_kl_k: 0          # 0 = 全词表
    topk_kl_mode: jsd
    topk_kl_jsd_beta: 0.5
```

---

## 使用限制

### 1. `use_fused_kernels` 必须为 `false`

Fused kernel 路径不经过 `output.logits`，无法获取完整 logits 张量。
启用 `use_fused_kernels: true` 时会触发 AssertionError：

```
AssertionError: analytic_kl requires logits; ensure use_fused_kernels=False.
```

务必在配置文件中设置：

```yaml
actor_rollout_ref:
  model:
    use_fused_kernels: false
```

### 2. `topk_kl_k: 0`（全词表模式）不走 OEL 预算路径

全词表模式（`topk_kl_k: 0`）暂不支持 OEL 预算路径，仍需在 `DataParallelPPOActor.__init__` 中
注入 `ref_module`，并在 `fsdp_workers.py` 中做对应修改（参考旧版实现）。
显存开销极大，对于 32K 词表 + bsz=16 + L=20K 的场景可能需要数十 GB 额外显存。

**强烈推荐使用 `topk_kl_k: 256` 走 OEL 预算路径。**

### 3. `remove_padding=True` 路径未经测试

理论上可用，但目前尚未在 variable-length / remove_padding 场景中验证。
首次使用建议关闭 `remove_padding`，验证后再开启。

### 4. 显存说明（与原始 k3 的详细对比）

#### 背景：为什么需要 OEL 预算架构

未经优化的 analytic_kl（非 OEL）需要在 `update_policy` 里每个 mini-batch 都做一次完整 ref forward，
产生 `(B_mini, L, V)` ref logits 张量，额外 GPU 显存开销极大。
OEL 架构将 ref forward 从 `update_policy` 里搬出来，变成训练步开始前的 1 次预算。

#### CPU 内存：多出两个 batch 张量

| 对比 | 额外 CPU 内存 |
|------|--------------|
| k3（基线） | 0（`ref_log_prob` 已存在，(B,L) 共 256 KB @ B=64,L=1024） |
| top-256 OEL | `kl_topk_indices` + `ref_log_prob_topk`，约 **200 MB** @ B=64,L=1024 |
| top-256 OEL | 约 **800 MB** @ B=64,L=4096（长序列场景） |
| full-vocab OEL（V=32K） | 约 **8 GB** @ B=64,L=1024 |
| full-vocab OEL（V=150K，Qwen） | 约 **38 GB** |

> CPU 内存放在 batch 里随数据流动，不占 GPU 显存，对一般机器（256 GB 内存）影响可忽略（除全词表外）。

公式：`kl_topk_indices` = B × L × k × 8 B（int64），`ref_log_prob_topk` = B × L × k × 4 B（float32）

#### GPU 显存（update_policy 阶段，per micro-batch）

关键事实：**k3 和 analytic_kl（OEL）在 update_policy 阶段都需要计算完整 logits (B_micro, L, V)**，
这是获取 log_prob 所必须的，两者峰值相同。analytic_kl 仅额外多出：

| 额外张量 | Shape | GPU 显存 @ B_micro=2,L=1024,k=256 |
|---------|-------|-----------------------------------|
| `student_log_probs_topk` | (B_micro, L, k) float32 | 2 × 1024 × 256 × 4 B ≈ **2 MB** |
| `ref_log_prob_topk`（从 CPU 搬来） | (B_micro, L, k) float32 | ≈ **2 MB** |
| **合计** | | ≈ **4 MB** |

结论：**与 k3 相比，OEL 路径几乎不增加 GPU 显存**（4 MB vs logits 动辄数 GB）。

#### 计算开销（额外 forward pass 次数）

| 方案 | 每训练步额外 forward 次数 |
|------|--------------------------|
| k3 | 0（ref_log_prob 在 rollout 时已算好） |
| top-256 OEL（本实现） | **actor × 1 + ref × 1**（OEL 预算） |
| 非 OEL analytic_kl（假设 2 PPO epoch × 4 mini-batch） | actor × 0 + ref × **8** |

OEL 把 ref forward 从 K×M 次压缩为 1 次，代价是 1 次额外 actor forward（用于计算 student top-k indices）。

#### 汇总对比（B=64, L=1024, k=256, V=32K, 2 epoch × 4 mini-batch）

| 方案 | 额外 CPU 内存 | 额外 GPU 峰值（per micro-batch） | 额外 forward 次/步 |
|------|-------------|--------------------------------|-------------------|
| k3 基线 | 0 | 0 | 0 |
| **top-256 OEL（推荐）** | **~200 MB** | **~4 MB** | **actor×1 + ref×1** |
| top-256 非 OEL | 0 | ~500 MB（ref logits）× 8 次 | ref×8 |
| full-vocab OEL | ~8 GB | ~4 MB | actor×1 + ref×1 |
| full-vocab 非 OEL | 0 | ~8 GB × 8 次 | ref×8（每次全词表） |

**结论**：top-256 OEL 相对 k3 只多 ~200 MB CPU + 4 MB GPU，代价极低；
相对于非 OEL 的 analytic_kl，节省了 K×M 次 ref forward 和对应的 GPU 峰值显存。

---

## 修改的文件（共 5 个）

| 文件 | 修改内容 |
|------|---------|
| `verl/trainer/ppo/core_algos.py` | 新增 `topk_analytic_kl()` 函数，支持 4 种散度模式 × top-k/full-vocab |
| `verl/workers/actor/dp_actor.py` | OEL 预算架构：新增 `_forward_micro_batch_with_topk`、`compute_kl_topk_indices`、`compute_ref_log_prob_topk` 方法；`update_policy` 读取预算好的 ref log-probs 计算 analytic KL；import 新增 `topk_analytic_kl` |
| `verl/workers/fsdp_workers.py` | 新增 `compute_kl_topk_indices`（actor_rollout_wg 调用）、`compute_ref_log_prob_topk`（ref_policy_wg 调用）两个 dispatch 方法 |
| `verl/trainer/ppo/ray_trainer.py` | 训练循环 ref log prob 计算之后插入 OEL 预算步骤（`compute_kl_topk_indices` → `compute_ref_log_prob_topk`），仅 `analytic_kl` + `topk_kl_k > 0` 时触发 |
| `verl/trainer/config/actor/actor.yaml` | 新增 `topk_kl_k`、`topk_kl_mode`、`topk_kl_jsd_beta` 三个字段；`kl_loss_type` 注释更新 |

---

## 快速验证（无需 GPU）

在完成 wget 并验证 import 成功之后，运行以下脚本确认所有散度模式逻辑正确：

```python
import torch
from verl.trainer.ppo.core_algos import topk_analytic_kl

B, L, V = 2, 10, 1000   # 用小词表节省测试时间

print("=== 形状 & 非负性测试 ===")
for mode in ["reverse_kl", "forward_kl", "jsd", "symmetric_kl"]:
    for k in [64, 256, 0, None]:   # top-k 64 / 256 / full-vocab
        out = topk_analytic_kl(
            torch.randn(B, L, V),
            torch.randn(B, L, V),
            k=k, mode=mode
        )
        assert out.shape == (B, L), f"[FAIL] shape={out.shape}, mode={mode}, k={k}"
        assert (out >= -1e-4).all(), (
            f"[FAIL] 出现负值, mode={mode}, k={k}, min={out.min():.4f}"
        )
        print(f"  [PASS] mode={mode:12s}  k={str(k):4s}  mean={out.mean():.4f}")

print("\n=== k=0 与 k=None 等价性测试 ===")
stu = torch.randn(1, 3, V)
ref = torch.randn(1, 3, V)
for mode in ["reverse_kl", "forward_kl", "jsd", "symmetric_kl"]:
    a = topk_analytic_kl(stu, ref, k=0,    mode=mode)
    b = topk_analytic_kl(stu, ref, k=None, mode=mode)
    assert torch.allclose(a, b, atol=1e-5), f"[FAIL] k=0 vs k=None 不一致, mode={mode}"
    print(f"  [PASS] mode={mode:12s}  k=0 == k=None")

print("\n=== 同分布 KL 应接近 0 测试 ===")
same = torch.randn(1, 5, V)
for mode in ["reverse_kl", "jsd", "symmetric_kl"]:
    out = topk_analytic_kl(same, same, k=256, mode=mode)
    assert out.abs().max() < 1e-4, f"[FAIL] 同分布 KL 非零, mode={mode}, max={out.abs().max():.6f}"
    print(f"  [PASS] mode={mode:12s}  同分布 KL ≈ 0  (max={out.abs().max():.2e})")

print("\n所有测试通过 ✓")
```

预期输出（数值因随机而异，重要的是 `[PASS]`）：

```
=== 形状 & 非负性测试 ===
  [PASS] mode=reverse_kl    k=64    mean=0.xxxx
  [PASS] mode=reverse_kl    k=256   mean=0.xxxx
  ...（共 16 行）

=== k=0 与 k=None 等价性测试 ===
  [PASS] mode=reverse_kl    k=0 == k=None
  ...（共 4 行）

=== 同分布 KL 应接近 0 测试 ===
  [PASS] mode=reverse_kl    同分布 KL ≈ 0  (max=0.00e+00)
  ...（共 3 行）

所有测试通过 ✓
```

---

## 参考文献

- **OPCD**：Ye et al. (2026). *On-Policy Context Distillation for Language Models*.
  [arXiv:2602.12275](https://arxiv.org/abs/2602.12275)

- **OPSD**：Zhao et al. (2026). *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models*.
  [arXiv:2601.18734](https://arxiv.org/abs/2601.18734)

- **GKD**：Agarwal et al. (2024). *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*.
  [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)

- **Schulman KL blog**：[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)
  k1/k2/k3 单样本 MC 估计量的数学推导与对比

- **verl**：Sheng et al. (2024). *HybridFlow: A Flexible and Efficient RLHF Framework*.
  [arXiv:2409.19256](https://arxiv.org/abs/2409.19256) | [GitHub](https://github.com/volcengine/verl)
