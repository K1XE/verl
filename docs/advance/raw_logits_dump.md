# Raw Logits Dump for `compute_log_prob`

This note describes the raw-logits dump path added around the FSDP/FSDP2 `compute_log_prob` and `compute_ref_log_prob` passes.

## Goal

The purpose of this feature is to persist the full-vocabulary raw logits that are already produced inside the actor/ref forward pass, so they can be analyzed offline without adding another model forward.

The implementation is intentionally narrow:

- backend: FSDP / FSDP2
- worker: `ActorRolloutRefWorker`
- model path: `DataParallelPPOActor`
- user-facing caller: `AgentPPOTrainer` in downstream integrations such as `rllm`

## What is saved

The dump stores **raw logits**, not probabilities and not log-probabilities.

Each shard is a `torch.save(...)` payload that contains:

- `logits`
- `input_ids`
- `attention_mask`
- `responses`
- `response_mask`
- `position_ids`
- `uids` when provided by the caller
- metadata such as `role`, `step`, `temperature`, `position_scope`, `storage_format`

## Why it does not add another forward pass

The dump is written from `output.logits` that already exists in `_forward_micro_batch`.

The training path still computes:

- actor `old_log_probs` through `compute_log_prob`
- ref `ref_log_prob` through `compute_ref_log_prob`

The extra work is only:

1. cast the existing logits to the requested save dtype
2. copy them to CPU
3. serialize them to disk

Offline scripts can later compute softmax, log-softmax, top-k, or cross-run comparisons.

## Storage formats

### Dense path

Used when `use_remove_padding=False`.

- `position_scope=full_sequence`: save `(batch, sequence_length, vocab_size)`
- `position_scope=response_only`: save `(batch, response_length, vocab_size)`

### Ragged path

Used when `use_remove_padding=True`.

- valid-token rows are saved as `(num_valid_positions, vocab_size)`
- `flat_token_positions` maps each row back to `(sample, token_position)`
- this avoids re-padding the vocabulary tensor online just for visualization

## Constraints

- `use_fused_kernels=True` is not supported, because that path does not retain the raw vocabulary logits needed for export.
- The worker returns only lightweight dump records to the driver; the full logits never travel back through the controller.

## Integration points

- `verl/workers/actor/dp_actor.py`
  - captures `output.logits`
  - handles dense and remove-padding export
- `verl/workers/fsdp_workers.py`
  - threads dump requests through `compute_log_prob` / `compute_ref_log_prob`
  - returns per-shard records in `non_tensor_batch`

The driver-side manifesting and offline visualization live in the downstream trainer repo rather than in `verl` itself.
