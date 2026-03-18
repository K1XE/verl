# 解析式 KL 散度功能说明 (`kl_loss_type: analytic_kl`)

## 概述

本功能为 verl 新增 `kl_loss_type: "analytic_kl"` 选项，支持对完整词表或 top-k token
进行**解析式**（analytic）KL 散度与 JSD 计算，覆盖以下两篇论文的方法：

- **OPCD**（arXiv:2602.12275）—— top-k 近似反向 KL，默认 k=256
- **OPSD**（arXiv:2601.18734）—— full-vocab 广义 JSD

### 与现有 k1/k2/k3 的本质区别

| | k1 / k2 / k3 (`low_var_kl`) | `analytic_kl` |
|---|---|---|
| 输入 | `(B, L)` — 只有被采样那一个 token 的 log_prob | `(B, L, V)` — 完整词表 logits |
| 本质 | 单样本 MC 估计，有采样噪声 | 直接对词表（子集）求和，解析计算 |
| 需要 ref forward | 否（rollout 时已算好 `ref_log_prob`） | **是**（update_policy 阶段额外做一次 ref forward） |
| 显存开销 | 极低 | top-k: 中等；full-vocab: 较高 |

---

## 支持的散度模式（`topk_kl_mode`）

| 模式 | 数学公式 | top-k 选取方式 | 论文依据 |
|---|---|---|---|
| `reverse_kl`（**默认**）| KL(π_θ ‖ π_ref) | 按 **student** 概率取 top-k | OPCD arXiv:2602.12275 |
| `forward_kl` | KL(π_ref ‖ π_θ) | 按 **reference** 概率取 top-k | GKD arXiv:2306.13649 |
| `jsd` | JSD_β = β·KL(π_ref‖m) + (1-β)·KL(π_θ‖m)，m=β·π_ref+(1-β)·π_θ | student + reference top-k 取并集 | OPSD arXiv:2601.18734 |
| `symmetric_kl` | 0.5·[KL(π_θ‖π_ref) + KL(π_ref‖π_θ)] | student + reference top-k 取并集 | Jeffreys 散度 |

### top-k vs full-vocab

- `topk_kl_k: 256` → 只对 student/ref 概率最高的 256 个 token 求和（OPCD 推荐，节省显存）
- `topk_kl_k: 0` → 对全部 V 个 token 求和（精确但显存消耗约为 top-256 的 `V/256` 倍）

---

## 配置方法

在训练脚本或 yaml 的 `actor_rollout_ref.actor` 节中添加：

```yaml
actor_rollout_ref:
  actor:
    use_kl_loss: true
    kl_loss_type: analytic_kl    # ← 改这里
    kl_loss_coef: 0.001

    # analytic_kl 专用参数（其他 kl_loss_type 忽略以下字段）
    topk_kl_k: 256               # top-k 个 token；0 = 全词表
    topk_kl_mode: reverse_kl     # reverse_kl | forward_kl | jsd | symmetric_kl
    topk_kl_jsd_beta: 0.5        # 仅 jsd 模式使用，β=0.5 为对称 JSD
```

### 常用配置预设

**OPCD 风格（top-256 反向 KL）：**
```yaml
kl_loss_type: analytic_kl
topk_kl_k: 256
topk_kl_mode: reverse_kl
kl_loss_coef: 0.001
```

**OPSD 风格（full-vocab 广义 JSD）：**
```yaml
kl_loss_type: analytic_kl
topk_kl_k: 0            # 0 = 全词表
topk_kl_mode: jsd
topk_kl_jsd_beta: 0.5
kl_loss_coef: 0.001
```

---

## 使用限制

1. **`use_fused_kernels` 必须为 `false`**：fused kernel 路径不经过 `output.logits`，无法获取完整 logits，启用时会报 AssertionError。

2. **worker 必须同时承载 ref model（`_is_ref=True`）**：standard `ActorRolloutRefWorker` 默认满足此条件。

3. **`remove_padding=True` 路径未经测试**：理论可用但未验证，建议关闭 remove_padding 进行首次测试。

4. **显存**：每个 micro-batch 额外保存 `(B, L, V)` student logits + ref model 做一次 no_grad forward。对长序列（L > 2048）或大词表（V > 32000）压力较大，建议减小 `ppo_mini_batch_size`。

---

## 修改的文件（共 5 个）

| 文件 | 修改内容 |
|------|---------|
| `verl/trainer/ppo/core_algos.py` | 新增 `topk_analytic_kl()` 函数 |
| `verl/workers/actor/dp_actor.py` | OEL 预算方式：新增 `compute_kl_topk_indices`、`compute_ref_log_prob_topk` 方法；`_forward_micro_batch` 加 top-k gather 逻辑；`update_policy` 用预算 ref log-probs |
| `verl/workers/fsdp_workers.py` | 新增 `compute_kl_topk_indices`、`compute_ref_log_prob_topk` dispatch 方法 |
| `verl/trainer/ppo/ray_trainer.py` | 训练循环 ref forward 之后插入 OEL 预算步骤（一次性预算 top-k indices + ref log-probs） |
| `verl/trainer/config/actor/actor.yaml` | 新增 `topk_kl_k`、`topk_kl_mode`、`topk_kl_jsd_beta` 三个字段 |

---

## 手动 Patch 教程（verl 0.6.1 用户必读）

> ⚠️ **本分支基于 verl `0.8.0.dev`，公司环境如果是 0.6.1，请勿直接替换文件，
> 按照以下步骤逐一手动 patch。**

### Patch 1：`verl/trainer/ppo/core_algos.py`

**操作**：在文件末尾（或 `kl_penalty_forward` 函数之后）**添加**以下完整函数。
无需修改文件中其他任何代码。

```python
def topk_analytic_kl(
    student_logits,   # (B, L, V) student 模型的原始 logits
    ref_logits,       # (B, L, V) reference 模型的原始 logits
    k=256,            # top-k 个 token；None 或 0 表示使用全词表
    mode="reverse_kl",
    jsd_beta=0.5,
):
    """解析式 KL/JSD 散度，支持 top-k 近似或全词表精确计算。

    mode 可选：
        "reverse_kl"   KL(pi_theta || pi_ref)，top-k 按 student 概率选取   [OPCD]
        "forward_kl"   KL(pi_ref || pi_theta)，top-k 按 reference 概率选取 [GKD]
        "jsd"          广义 JSD，top-k 取两者并集                           [OPSD]
        "symmetric_kl" 0.5*(正向+反向 KL)，top-k 取两者并集
    """
    import torch
    import torch.nn.functional as F

    use_full = k is None or k <= 0

    student_log_probs = F.log_softmax(student_logits, dim=-1)  # (B, L, V)
    ref_log_probs     = F.log_softmax(ref_logits, dim=-1)      # (B, L, V)
    student_probs     = student_log_probs.exp()                 # (B, L, V)
    ref_probs         = ref_log_probs.exp()                     # (B, L, V)

    if mode == "reverse_kl":
        if use_full:
            return (student_probs * (student_log_probs - ref_log_probs)).sum(-1)
        _, idx = student_probs.topk(k, dim=-1)
        p   = student_probs.gather(-1, idx)
        lps = student_log_probs.gather(-1, idx)
        lpr = ref_log_probs.gather(-1, idx)
        return (p * (lps - lpr)).sum(-1)

    elif mode == "forward_kl":
        if use_full:
            return (ref_probs * (ref_log_probs - student_log_probs)).sum(-1)
        _, idx = ref_probs.topk(k, dim=-1)
        p   = ref_probs.gather(-1, idx)
        lpr = ref_log_probs.gather(-1, idx)
        lps = student_log_probs.gather(-1, idx)
        return (p * (lpr - lps)).sum(-1)

    elif mode == "jsd":
        beta = jsd_beta
        if use_full:
            m     = beta * ref_probs + (1.0 - beta) * student_probs
            log_m = m.clamp(min=1e-30).log()
            kl_r  = (ref_probs     * (ref_log_probs     - log_m)).sum(-1)
            kl_s  = (student_probs * (student_log_probs - log_m)).sum(-1)
            return beta * kl_r + (1.0 - beta) * kl_s
        _, si = student_probs.topk(k, dim=-1)
        _, ri = ref_probs.topk(k, dim=-1)
        idx   = torch.cat([si, ri], dim=-1)           # (B, L, 2k)
        ps    = student_probs.gather(-1, idx)
        pr    = ref_probs.gather(-1, idx)
        m     = beta * pr + (1.0 - beta) * ps
        log_m = m.clamp(min=1e-30).log()
        lps   = student_log_probs.gather(-1, idx)
        lpr   = ref_log_probs.gather(-1, idx)
        kl_r  = (pr * (lpr - log_m)).sum(-1)
        kl_s  = (ps * (lps - log_m)).sum(-1)
        return beta * kl_r + (1.0 - beta) * kl_s

    elif mode == "symmetric_kl":
        if use_full:
            fwd = (ref_probs     * (ref_log_probs     - student_log_probs)).sum(-1)
            rev = (student_probs * (student_log_probs - ref_log_probs)).sum(-1)
            return 0.5 * (fwd + rev)
        _, si  = student_probs.topk(k, dim=-1)
        _, ri  = ref_probs.topk(k, dim=-1)
        idx    = torch.cat([si, ri], dim=-1)
        ps     = student_probs.gather(-1, idx)
        pr     = ref_probs.gather(-1, idx)
        lps    = student_log_probs.gather(-1, idx)
        lpr    = ref_log_probs.gather(-1, idx)
        return 0.5 * ((pr * (lpr - lps)).sum(-1) + (ps * (lps - lpr)).sum(-1))

    else:
        raise ValueError(
            f"topk_analytic_kl: 未知 mode='{mode}'，"
            "请选择 'reverse_kl'、'forward_kl'、'jsd' 或 'symmetric_kl'。"
        )
```

---

### Patch 2：`verl/workers/actor/dp_actor.py`

**共 3 处修改。**

#### 修改 2-A：更新 import（文件顶部）

**找到**（大约在文件前 50 行）：
```python
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
```

**替换为**：
```python
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty, topk_analytic_kl
```

> 如果你的 0.6.1 版本 import 行写法稍有不同，只需在行末加上 `, topk_analytic_kl` 即可。

---

#### 修改 2-B：`DataParallelPPOActor.__init__` 加 `ref_module` 参数

**找到** `DataParallelPPOActor` 类的 `__init__` 方法（通常如下）：
```python
def __init__(self, config, actor_module, actor_optimizer=None):
    """When optimizer is None, it is Reference Policy"""
    super().__init__(config)
    self.actor_module = actor_module
    self.actor_optimizer = actor_optimizer
```

**替换为**（只加了一个参数 `ref_module=None` 和一行赋值）：
```python
def __init__(self, config, actor_module, actor_optimizer=None, ref_module=None):
    """When optimizer is None, it is Reference Policy"""
    super().__init__(config)
    self.actor_module = actor_module
    self.actor_optimizer = actor_optimizer
    self.ref_module = ref_module  # 用于 analytic_kl loss，非 analytic_kl 时始终为 None
```

> ⚠️ 注意：如果 0.6.1 的 `__init__` 里在 `self.actor_optimizer = ...` 之后还有
> `role = "Ref" if actor_optimizer is None else "Actor"` 等行，**只在这两行之后加一行
> `self.ref_module = ref_module`**，不要改动其他部分。

---

#### 修改 2-C：`_forward_micro_batch` 返回 logits

**找到** `_forward_micro_batch` 方法中的返回语句（大约如下）：
```python
outputs = {"log_probs": log_probs}
if calculate_entropy:
    outputs["entropys"] = entropy
return outputs
```

**替换为**（在 `return outputs` 前插入 2 行）：
```python
outputs = {"log_probs": log_probs}
if calculate_entropy:
    outputs["entropys"] = entropy
# 当使用 analytic_kl 时，额外返回 logits 供 KL 计算（仅非 fused kernel 路径有效）
if not self.use_fused_kernels and getattr(self.config, "kl_loss_type", "") == "analytic_kl":
    outputs["logits"] = logits  # shape: (bsz, response_length, vocab_size)
return outputs
```

> ⚠️ 注意：`logits` 变量在此处必须已定义。在 0.6.x 的非 fused kernel 路径中，
> `logits = output.logits` 之后会做 `logits.div_(temperature)` 和 slice 操作，
> 确认 `logits` 变量在你的版本中是否存在于同一作用域。
> 如果不存在，你需要将 `output.logits` 截取 response 段后单独存储。
>
> 典型代码形态（0.8.0.dev 版本）：
> ```python
> logits = output.logits                              # (bsz, seqlen, vocab)
> logits.div_(temperature)
> logits = logits[:, -response_length - 1 : -1, :]   # 截取 response 段
> log_probs = logprobs_from_logits(logits, micro_batch["responses"])
> ```
> 如果你的 0.6.1 版本形态相同，直接加上述两行即可。

---

#### 修改 2-D：`update_policy` 添加 `analytic_kl` 分支

**找到** `update_policy` 方法中处理 KL loss 的代码块（大约如下）：
```python
if self.config.use_kl_loss:
    ref_log_prob = model_inputs["ref_log_prob"]
    kld = kl_penalty(
        logprob=log_prob, ref_logprob=ref_log_prob,
        kl_penalty=self.config.kl_loss_type
    )
    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask,
                       loss_agg_mode=loss_agg_mode)
    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
    metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
    micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
```

**替换为**：
```python
if self.config.use_kl_loss:
    if getattr(self.config, "kl_loss_type", "") == "analytic_kl":
        # ── analytic_kl：需要完整词表 logits + ref model forward ──
        assert self.ref_module is not None, (
            "kl_loss_type='analytic_kl' 需要 ref_module 注入到 DataParallelPPOActor。\n"
            "请确认 ActorRolloutRefWorker 中 _is_ref=True，并已完成 fsdp_workers.py 的 Patch 3。"
        )
        assert "logits" in outputs, (
            "analytic_kl 需要 _forward_micro_batch 返回 logits，"
            "请确认 use_fused_kernels=False 且已完成 dp_actor.py 的 Patch 2-C。"
        )
        k_val    = getattr(self.config, "topk_kl_k", 256) or None   # 0 → None → 全词表
        kl_mode  = getattr(self.config, "topk_kl_mode", "reverse_kl")
        jsd_beta = getattr(self.config, "topk_kl_jsd_beta", 0.5)
        response_length = response_mask.shape[-1]

        with torch.no_grad():
            ref_out = self.ref_module(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                position_ids=model_inputs.get("position_ids"),
                use_cache=False,
            )
            # 截取 response 段，与 student logits 对齐（student logits 已在 _forward_micro_batch 中截取）
            ref_logits = ref_out.logits[:, -response_length - 1 : -1, :]

        kld = topk_analytic_kl(
            outputs["logits"], ref_logits,
            k=k_val, mode=kl_mode, jsd_beta=jsd_beta
        )
    else:
        # ── 原有逻辑不变 ──
        ref_log_prob = model_inputs["ref_log_prob"]
        kld = kl_penalty(
            logprob=log_prob, ref_logprob=ref_log_prob,
            kl_penalty=self.config.kl_loss_type
        )

    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask,
                       loss_agg_mode=loss_agg_mode)
    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
    metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
    micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
```

> ⚠️ 注意：上述代码中的变量名（`outputs`、`response_mask`、`loss_agg_mode`、
> `loss_scale_factor` 等）需要与你的 0.6.1 版本实际使用的变量名对齐。
> 如果 0.6.1 中变量名不同（如 `micro_batch_outputs` 而非 `outputs`），请相应修改。

---

### Patch 3：`verl/workers/fsdp_workers.py`

**只需添加 2 行。**

**找到** `_is_ref` 代码块中创建 `self.ref_policy` 的那一行（大约如下）：
```python
self.ref_policy = DataParallelPPOActor(
    config=self.config.ref, actor_module=self.ref_module_fsdp
)
```

**在其正下方插入 2 行**（注意缩进与上方代码保持一致）：
```python
self.ref_policy = DataParallelPPOActor(
    config=self.config.ref, actor_module=self.ref_module_fsdp
)
# 将 ref model 注入 actor，供 analytic_kl loss 使用
# （ref model 在 _is_actor 之后才构建，因此在此处回填）
if self._is_actor and getattr(self.config.actor, "kl_loss_type", "") == "analytic_kl":
    self.actor.ref_module = self.ref_module_fsdp
```

> ⚠️ 注意：在 0.6.1 中，`self.ref_policy` 的构建可能在方法的不同位置。
> 核心是找到 `self.ref_module_fsdp` 构建完成之后的位置插入上面两行。
> `self.ref_module_fsdp` 通常通过 `self._build_model_optimizer(...)` 返回，
> 确认它已赋值后再插入。

---

### Patch 4：`verl/trainer/config/actor/actor.yaml`

**找到**：
```yaml
kl_loss_type: low_var_kl
```

**替换为**（在原行后追加 4 行注释和配置）：
```yaml
# Type of KL loss. Options: "kl"(k1), "abs", "mse"(k2), "low_var_kl"(k3),
#   "analytic_kl" (top-k 或 full-vocab 解析式 KL/JSD；OPCD / OPSD)
kl_loss_type: low_var_kl

# analytic_kl 专用参数（kl_loss_type 为其他值时忽略）
# top-k 词表 token 数量；0 = 使用全词表（精确但显存更大）
topk_kl_k: 256
# 散度计算方式：reverse_kl | forward_kl | jsd | symmetric_kl
topk_kl_mode: reverse_kl
# jsd 模式的混合权重 β（0 < β < 1），β=0.5 为对称 JSD
topk_kl_jsd_beta: 0.5
```

---

## 快速验证（无需 GPU 即可运行）

在完成所有 patch 之后，运行以下 Python 脚本确认函数逻辑正确：

```python
import torch

# 直接把 Patch 1 中的函数粘贴到这里，或者从 verl 导入：
# from verl.trainer.ppo.core_algos import topk_analytic_kl

B, L, V = 2, 10, 1000  # 用小词表节省测试时间

for mode in ["reverse_kl", "forward_kl", "jsd", "symmetric_kl"]:
    for k in [64, 0, None]:   # top-k=64、full-vocab
        out = topk_analytic_kl(
            torch.randn(B, L, V),
            torch.randn(B, L, V),
            k=k, mode=mode
        )
        assert out.shape == (B, L), f"[FAIL] shape={out.shape}, mode={mode}, k={k}"
        assert (out >= -1e-4).all(), f"[FAIL] 出现负值, mode={mode}, k={k}, min={out.min():.4f}"
        print(f"[PASS] mode={mode:12s}  k={str(k):4s}  mean={out.mean():.4f}")

# k=0 与 k=None 结果应完全相同
a = topk_analytic_kl(torch.ones(1, 1, V), torch.ones(1, 1, V) * 2, k=0)
b = topk_analytic_kl(torch.ones(1, 1, V), torch.ones(1, 1, V) * 2, k=None)
assert torch.allclose(a, b), "[FAIL] k=0 与 k=None 结果不一致"
print("[PASS] k=0 == k=None")
print("\n所有测试通过。")
```

---

## 参考文献

- **OPCD**：Ye et al. (2026). *On-Policy Context Distillation for Language Models*. [arXiv:2602.12275](https://arxiv.org/abs/2602.12275)
- **OPSD**：Zhao et al. (2026). *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models*. [arXiv:2601.18734](https://arxiv.org/abs/2601.18734)
- **GKD**：Agarwal et al. (2024). *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- **Schulman KL blog**：[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)（k1/k2/k3 估计量的来源）
