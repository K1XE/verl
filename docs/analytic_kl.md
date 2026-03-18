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

2. **`topk_kl_k=0`（全词表模式）不走 OEL 预算路径**：仍需在 `DataParallelPPOActor.__init__` 注入 `ref_module`，并在 `fsdp_workers.py` 回填（参考旧实现），显存开销较大。**推荐使用 `topk_kl_k: 256` 走 OEL 预算路径。**

3. **`remove_padding=True` 路径未经测试**：理论可用但未验证，建议关闭 remove_padding 进行首次测试。

4. **显存（OEL 预算路径）**：预算阶段临时存储 `(B, L, k)` student indices + ref log-probs（top-256 约 328 MB @bsz=16,L=20k），`update_policy` 阶段仅需 student `(B, L, V)` logits，无额外 ref forward，相比旧版本大幅降低显存峰值。

---

## 修改的文件（共 5 个）

| 文件 | 修改内容 |
|------|---------|
| `verl/trainer/ppo/core_algos.py` | 新增 `topk_analytic_kl()` 函数 |
| `verl/workers/actor/dp_actor.py` | OEL 预算方式：新增 `compute_kl_topk_indices`、`compute_ref_log_prob_topk` 方法；`_forward_micro_batch` 加 top-k gather 逻辑；`update_policy` 用预算 ref log-probs（回退路径仍支持实时 ref forward） |
| `verl/workers/fsdp_workers.py` | 新增 `compute_kl_topk_indices`、`compute_ref_log_prob_topk` dispatch 方法 |
| `verl/trainer/ppo/ray_trainer.py` | 训练循环 ref forward 之后插入 OEL 预算步骤（一次性预算 top-k indices + ref log-probs，将 ref forward 从 K×M 次压缩为 1 次） |
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

**共 5 处修改（OEL 预算架构）。**

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

#### 修改 2-B：`_forward_micro_batch` 加参数 + 末尾 top-k 逻辑

**找到** `_forward_micro_batch` 的方法定义行，**在参数列表末尾追加 2 个参数**：

```python
# 改前（典型形态）：
def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):

# 改后：
def _forward_micro_batch(
    self,
    micro_batch,
    temperature,
    calculate_entropy=False,
    return_topk_indices=False,  # 新增：True 时只返回 top-k indices，不走 log_probs 路径
    kl_topk_k=256,              # 新增：top-k 数量
):
```

**找到** `_forward_micro_batch` 中的 `return outputs` 语句（即方法末尾），在其**正前方**插入以下代码块：

```python
# analytic_kl top-k helpers（仅非 fused kernel 路径）
if not self.use_fused_kernels and getattr(self.config, "kl_loss_type", "") == "analytic_kl":
    if return_topk_indices:
        # OEL 预算第一步：返回 student top-k indices 供 ref 侧 gather 使用
        log_probs_full = torch.nn.functional.log_softmax(logits, dim=-1)
        _, topk_idx = log_probs_full.topk(kl_topk_k, dim=-1)  # (B, L, k)
        outputs["kl_topk_indices"] = topk_idx
    elif "kl_topk_indices" in micro_batch:
        # OEL 主路径（update_policy 阶段）：用预算好的 indices gather student log-probs
        topk_idx = micro_batch["kl_topk_indices"].to(logits.device)
        log_probs_full = torch.nn.functional.log_softmax(logits, dim=-1)
        outputs["student_log_probs_topk"] = log_probs_full.gather(-1, topk_idx)  # (B, L, k)
    else:
        # 回退：full-vocab（k=0）或 indices 未预算时，保留完整 logits
        outputs["logits"] = logits  # (B, L, V)
return outputs
```

> ⚠️ 注意：`logits` 变量在此处须已定义（通常是 `logits = output.logits[:, -response_length-1:-1, :]` 经 temperature 缩放后的结果）。

---

#### 修改 2-C：新增 `compute_kl_topk_indices` 方法

在 `DataParallelPPOActor` 类中新增以下方法（建议放在 `compute_log_prob` 方法之后）：

```python
def compute_kl_topk_indices(self, data):
    """OEL 预算第一步：计算 student 对当前 batch 的 top-k token indices。

    返回 dict，键 ``kl_topk_indices``，shape (B, response_length, k)，dtype int64。
    """
    self.actor_module.eval()
    k = data.meta_info["kl_topk_k"]
    micro_batch_size = data.meta_info["micro_batch_size"]
    temperature = data.meta_info["temperature"]

    select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
    data = data.select(batch_keys=select_keys)
    micro_batches = data.split(micro_batch_size)

    idx_list = []
    for micro_batch in micro_batches:
        micro_batch = micro_batch.to(get_device_id())
        model_inputs = {**micro_batch.batch}
        with torch.no_grad():
            outputs = self._forward_micro_batch(
                model_inputs, temperature=temperature,
                return_topk_indices=True, kl_topk_k=k,
            )
        idx_list.append(outputs["kl_topk_indices"])
    return {"kl_topk_indices": torch.cat(idx_list, dim=0)}  # (B, L, k)
```

> ⚠️ 0.6.1 中 `get_device_id()` 可能写法不同（如直接写 `"cuda"`），对照同文件其他方法的写法即可。

---

#### 修改 2-D：新增 `compute_ref_log_prob_topk` 方法

紧接 `compute_kl_topk_indices` 之后新增：

```python
def compute_ref_log_prob_topk(self, data):
    """OEL 预算第二步：用 kl_topk_indices gather ref model log-probs。

    期望 ``data.batch`` 中含有 ``kl_topk_indices``（由 2-C 步骤生成）。
    返回 dict，键 ``ref_log_prob_topk``，shape (B, response_length, k)，dtype float32。
    """
    self.actor_module.eval()
    micro_batch_size = data.meta_info["micro_batch_size"]
    temperature = data.meta_info["temperature"]

    select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "kl_topk_indices"]
    data = data.select(batch_keys=select_keys)
    micro_batches = data.split(micro_batch_size)

    ref_lp_list = []
    for micro_batch in micro_batches:
        micro_batch = micro_batch.to(get_device_id())
        model_inputs = {**micro_batch.batch}
        with torch.no_grad():
            # kl_topk_indices 在 model_inputs 中，_forward_micro_batch 会走 gather 路径
            outputs = self._forward_micro_batch(model_inputs, temperature=temperature)
        ref_lp_list.append(outputs["student_log_probs_topk"])
    return {"ref_log_prob_topk": torch.cat(ref_lp_list, dim=0)}  # (B, L, k)
```

---

#### 修改 2-E：`update_policy` 两处修改

**第一处**：找到 `update_policy` 中的 `select_keys` 列表（含有 `"ref_log_prob"` 那段），在追加 `ref_log_prob` 之后**再追加** OEL tensors：

```python
if self.config.use_kl_loss:
    select_keys.append("ref_log_prob")
    # OEL 预算 tensors（analytic_kl 专用）
    if getattr(self.config, "kl_loss_type", "") == "analytic_kl":
        if "kl_topk_indices" in data.batch:
            select_keys.append("kl_topk_indices")
        if "ref_log_prob_topk" in data.batch:
            select_keys.append("ref_log_prob_topk")
```

**第二处**：找到 `update_policy` 中处理 KL loss 的代码块（大约如下）：

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
        k_val = getattr(self.config, "topk_kl_k", 256)
        if k_val > 0 and "ref_log_prob_topk" in model_inputs:
            # ── OEL 预算路径（推荐）：直接用预算好的 ref log-probs ──
            stu_lp = outputs["student_log_probs_topk"]                     # (B, L, k)
            ref_lp = model_inputs["ref_log_prob_topk"].to(stu_lp.device)   # (B, L, k)
            stu_p  = stu_lp.exp()
            kld    = (stu_p * (stu_lp - ref_lp)).sum(-1)                   # (B, L)
        else:
            # ── 回退路径：full-vocab（k=0）或未预算时，实时做 ref forward ──
            assert self.ref_module is not None, (
                "analytic_kl with k=0 requires ref_module injected into DataParallelPPOActor."
            )
            assert "logits" in outputs, (
                "analytic_kl fallback requires logits; ensure use_fused_kernels=False."
            )
            k_val_opt = k_val or None   # 0 → None → 全词表
            kl_mode   = getattr(self.config, "topk_kl_mode", "reverse_kl")
            jsd_beta  = getattr(self.config, "topk_kl_jsd_beta", 0.5)
            response_length = response_mask.shape[-1]
            with torch.no_grad():
                ref_out = self.ref_module(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    position_ids=model_inputs.get("position_ids"),
                    use_cache=False,
                )
                ref_logits = ref_out.logits[:, -response_length - 1 : -1, :]
            kld = topk_analytic_kl(
                outputs["logits"], ref_logits,
                k=k_val_opt, mode=kl_mode, jsd_beta=jsd_beta,
            )
    else:
        # ── 原有 k1/k2/k3 路径不变 ──
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

> ⚠️ 注意：上述代码中的变量名（`outputs`、`model_inputs`、`response_mask`、`loss_agg_mode` 等）
> 需与你的 0.6.1 版本实际使用的变量名对齐。

---

### Patch 3：`verl/workers/fsdp_workers.py`

**废弃旧实现（2 行 ref_module 注入），改为新增 2 个 dispatch 方法。**

**找到** `ActorRolloutRefWorker` 中 `compute_ref_log_prob` 方法的末尾，在其**正后方**插入以下两个方法：

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def compute_kl_topk_indices(self, data: DataProto) -> DataProto:
    """OEL 预算第一步：计算 student top-k indices（由 actor_rollout_wg 调用）。"""
    assert self._is_actor
    data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
    data.meta_info["max_token_len"]    = self.config.rollout.log_prob_max_token_len_per_gpu
    data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
    data.meta_info["temperature"]      = self.config.rollout.temperature
    data.meta_info["kl_topk_k"]        = self.config.actor.topk_kl_k

    with self.ulysses_sharding_manager:
        result = self.actor.compute_kl_topk_indices(data=data)
        output = DataProto.from_dict(tensors={"kl_topk_indices": result["kl_topk_indices"]})

    return output.to("cpu")


@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def compute_ref_log_prob_topk(self, data: DataProto) -> DataProto:
    """OEL 预算第二步：gather ref log-probs at top-k positions（由 ref_policy_wg 调用）。"""
    assert self._is_ref or self._is_lora
    data.meta_info["micro_batch_size"] = self.config.ref.log_prob_micro_batch_size_per_gpu
    data.meta_info["max_token_len"]    = self.config.ref.log_prob_max_token_len_per_gpu
    data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
    data.meta_info["temperature"]      = self.config.rollout.temperature

    with self.ulysses_sharding_manager:
        result = self.ref_policy.compute_ref_log_prob_topk(data=data)
        output = DataProto.from_dict(tensors={"ref_log_prob_topk": result["ref_log_prob_topk"]})

    return output.to("cpu")
```

> ⚠️ 注意：
> - `@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)` 是标准 dispatch 装饰器，与 `compute_ref_log_prob` 同级写法。
> - `self.ulysses_sharding_manager` 在 0.6.1 中写法可能是 `with self.sharding_manager:` 或类似，对照原有 `compute_ref_log_prob` 方法写法即可。
> - **LoRA 场景**（`_is_lora`）：`compute_ref_log_prob_topk` 需改为 `with self.actor.actor_module.disable_adapter():` 并调用 `self.actor.compute_ref_log_prob_topk`，参考同文件 LoRA 分支的 `compute_ref_log_prob` 写法。

---

### Patch 4：`verl/trainer/ppo/ray_trainer.py`

**找到** `fit()` 训练循环中 ref log prob 计算之后（通常是 `batch = batch.union(ref_log_prob)` 的下一行），**插入**以下代码块：

```python
# OEL 预算：一次性预算 student top-k indices + ref gathered log-probs
# 将 ref forward 从 update_policy 的 K×M 次压缩为训练循环里 1 次
actor_cfg = self.config.actor_rollout_ref.actor
if (
    getattr(actor_cfg, "use_kl_loss", False)
    and getattr(actor_cfg, "kl_loss_type", "") == "analytic_kl"
    and getattr(actor_cfg, "topk_kl_k", 256) > 0
    and self.use_reference_policy
):
    # Step 1: actor forward → student top-k indices (B, L, k)
    topk_idx_proto = self.actor_rollout_wg.compute_kl_topk_indices(batch)
    batch = batch.union(topk_idx_proto)
    # Step 2: ref forward → gathered ref log-probs (B, L, k)
    ref_topk_proto = self.ref_policy_wg.compute_ref_log_prob_topk(batch)
    batch = batch.union(ref_topk_proto)
```

> ⚠️ 注意：
> - `self.actor_rollout_wg` 和 `self.ref_policy_wg` 是 0.8.0.dev 的命名，0.6.1 中可能是 `self.actor_rollout_worker_group`、`self.ref_policy_worker_group` 或类似写法，对照原有 `compute_ref_log_prob` 的调用方写法即可。
> - `batch.union(...)` 在 0.6.1 中如果 API 不同（如 `batch.update(...)` 或直接赋值），相应调整。
> - 如果 0.6.1 无 `marked_timer`，直接去掉计时包装，保留内部 3 行代码即可。
> - 确保此块位于 `update_actor(batch)` 调用之前。

---

### Patch 5：`verl/trainer/config/actor/actor.yaml`

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
