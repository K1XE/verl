# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
import time
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty, topk_analytic_kl
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    @staticmethod
    def _resolve_dump_dtype(dtype_name: str) -> torch.dtype:
        normalized = str(dtype_name).lower()
        if normalized == "float16":
            return torch.float16
        if normalized == "bfloat16":
            return torch.bfloat16
        if normalized == "float32":
            return torch.float32
        raise ValueError(f"Unsupported raw logits dump dtype: {dtype_name}")

    @staticmethod
    def _extract_uids(micro_batch: dict[str, Any]) -> list[str] | None:
        if "uid" not in micro_batch:
            return None
        raw_uids = micro_batch["uid"]
        if hasattr(raw_uids, "tolist"):
            raw_uids = raw_uids.tolist()
        return [str(uid) for uid in raw_uids]

    @staticmethod
    def _compute_response_flat_positions(attention_mask: torch.Tensor, response_length: int) -> torch.Tensor:
        batch_size, seqlen = attention_mask.shape
        prompt_length = seqlen - response_length
        valid_lengths = attention_mask.sum(dim=-1).to(dtype=torch.long)
        flat_positions = []
        for batch_idx in range(batch_size):
            valid_length = int(valid_lengths[batch_idx].item())
            start = prompt_length - 1
            end = valid_length - 1
            if end <= start:
                continue
            local_positions = torch.arange(start, end, device=attention_mask.device, dtype=torch.long)
            flat_positions.append(batch_idx * seqlen + local_positions)
        if not flat_positions:
            return torch.empty(0, device=attention_mask.device, dtype=torch.long)
        return torch.cat(flat_positions, dim=0)

    def _save_raw_logits_dump(
        self,
        *,
        logits: torch.Tensor,
        micro_batch: dict[str, Any],
        dump_request: dict[str, Any] | None,
        micro_batch_index: int,
        storage_format: str,
        flat_token_positions: torch.Tensor | None = None,
    ) -> list[dict[str, Any]] | None:
        if dump_request is None:
            return None
        if self.use_fused_kernels:
            raise ValueError("Raw logits dump is not supported with use_fused_kernels=True")

        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        responses = micro_batch["responses"]
        position_ids = micro_batch["position_ids"]
        batch_size, seqlen = input_ids.shape
        response_length = responses.shape[-1]

        role = str(dump_request["role"])
        step = int(dump_request["step"])
        position_scope = str(dump_request["position_scope"])
        target_dtype = self._resolve_dump_dtype(dump_request["dtype"])
        rank = torch.distributed.get_rank()

        step_dir = Path(dump_request["path"]).expanduser() / f"step_{step:07d}" / role
        step_dir.mkdir(parents=True, exist_ok=True)
        file_path = step_dir / f"{role}_rank{rank:05d}_micro{micro_batch_index:04d}.pt"

        if storage_format == "dense":
            logits_to_save = logits[:, -response_length - 1 : -1, :] if position_scope == "response_only" else logits
            payload_flat_positions = None
        elif storage_format == "rmpad_ragged":
            if flat_token_positions is None:
                raise ValueError("flat_token_positions is required for rmpad raw logits dumps")
            if position_scope == "response_only":
                response_flat_positions = self._compute_response_flat_positions(attention_mask, response_length)
                selection_mask = torch.isin(flat_token_positions, response_flat_positions)
                logits_to_save = logits[selection_mask]
                payload_flat_positions = flat_token_positions[selection_mask]
            else:
                logits_to_save = logits
                payload_flat_positions = flat_token_positions
        else:
            raise ValueError(f"Unsupported raw logits dump storage format: {storage_format}")

        copy_start = time.perf_counter()
        cpu_logits = logits_to_save.detach().to(device="cpu", dtype=target_dtype)
        get_torch_device().synchronize()
        input_ids_cpu = input_ids.detach().to(device="cpu")
        attention_mask_cpu = attention_mask.detach().to(device="cpu")
        responses_cpu = responses.detach().to(device="cpu")
        position_ids_cpu = position_ids.detach().to(device="cpu")
        response_mask_cpu = attention_mask[:, -response_length:].detach().to(device="cpu")
        flat_token_positions_cpu = None
        if payload_flat_positions is not None:
            flat_token_positions_cpu = payload_flat_positions.detach().to(device="cpu")
        copy_time_s = time.perf_counter() - copy_start

        payload = {
            "schema_version": 1,
            "role": role,
            "step": step,
            "rank": rank,
            "micro_batch_index": micro_batch_index,
            "position_scope": position_scope,
            "storage_format": storage_format,
            "saved_dtype": str(dump_request["dtype"]).lower(),
            "temperature": float(dump_request["temperature"]),
            "sequence_shape": [batch_size, seqlen],
            "response_length": int(response_length),
            "input_ids": input_ids_cpu,
            "attention_mask": attention_mask_cpu,
            "responses": responses_cpu,
            "response_mask": response_mask_cpu,
            "position_ids": position_ids_cpu,
            "logits": cpu_logits,
            "uids": self._extract_uids(micro_batch),
        }
        if flat_token_positions_cpu is not None:
            payload["flat_token_positions"] = flat_token_positions_cpu

        write_start = time.perf_counter()
        torch.save(payload, file_path)
        write_time_s = time.perf_counter() - write_start

        record = {
            "role": role,
            "path": str(file_path),
            "step": step,
            "rank": rank,
            "micro_batch_index": micro_batch_index,
            "storage_format": storage_format,
            "position_scope": position_scope,
            "dtype": str(dump_request["dtype"]).lower(),
            "batch_size": batch_size,
            "sequence_length": seqlen,
            "response_length": int(response_length),
            "num_positions": int(cpu_logits.shape[0] if cpu_logits.dim() == 2 else cpu_logits.shape[0] * cpu_logits.shape[1]),
            "vocab_size": int(cpu_logits.shape[-1]) if cpu_logits.numel() > 0 else 0,
            "bytes_on_disk": file_path.stat().st_size,
            "copy_time_s": copy_time_s,
            "write_time_s": write_time_s,
        }
        return [record] * batch_size

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        raw_logits_dump_request: dict[str, Any] | None = None,
        micro_batch_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]] | None]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        if raw_logits_dump_request is not None and self.use_fused_kernels:
            raise ValueError("Raw logits dump is not supported with use_fused_kernels=True")
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            dump_records = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    if raw_logits_dump_request is not None:
                        dump_logits_rmpad = logits_rmpad
                        if self.use_ulysses_sp:
                            dump_logits_rmpad = gather_outputs_and_unpad(
                                dump_logits_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                        dump_records = self._save_raw_logits_dump(
                            logits=dump_logits_rmpad,
                            micro_batch=micro_batch,
                            dump_request=raw_logits_dump_request,
                            micro_batch_index=micro_batch_index,
                            storage_format="rmpad_ragged",
                            flat_token_positions=indices,
                        )
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits
                    if raw_logits_dump_request is not None:
                        dump_records = self._save_raw_logits_dump(
                            logits=logits,
                            micro_batch=micro_batch,
                            dump_request=raw_logits_dump_request,
                            micro_batch_index=micro_batch_index,
                            storage_format="dense",
                        )

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs, dump_records

    def _forward_micro_batch_with_topk(
        self,
        micro_batch,
        temperature: float,
        kl_topk_indices=None,
        kl_topk_k: int = 256,
        calculate_entropy: bool = False,
    ):
        """Forward pass that also computes analytic_kl top-k tensors (non-fused path only).

        Args:
            micro_batch: dict of model inputs
            temperature: sampling temperature
            kl_topk_indices: (B, L, k) int64 tensor. If None, computes and returns top-k indices.
                If provided, gathers student log-probs at those positions.
            kl_topk_k: number of top-k tokens (used only when kl_topk_indices is None)
            calculate_entropy: whether to compute entropy

        Returns:
            (entropy, log_probs, topk_result) where topk_result is:
                - kl_topk_indices (B, L, k) int64 if kl_topk_indices was None
                - student_log_probs_topk (B, L, k) float if kl_topk_indices was provided
        """
        assert not self.use_fused_kernels, (
            "analytic_kl is not supported with use_fused_kernels=True. Set use_fused_kernels=False."
        )

        response_length = micro_batch["responses"].size(-1)

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )

            logits = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (B, L, V)
            log_probs = logprobs_from_logits(logits, micro_batch["responses"])

            entropy = None
            if calculate_entropy:
                if not self.config.entropy_checkpointing:
                    entropy = verl_F.entropy_from_logits(logits)
                else:
                    # Clone before in-place log_softmax_ so checkpoint re-forward sees original logits
                    entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits.clone())

            if torch.is_grad_enabled():
                # update_policy path: need gradients through log_probs_full for KL loss backprop
                log_probs_full = logits.log_softmax(dim=-1)  # non-in-place, retains autograd graph
                if kl_topk_indices is None:
                    with torch.no_grad():
                        _, topk_result = log_probs_full.topk(kl_topk_k, dim=-1)  # (B, L, k) int64, no grad needed
                else:
                    topk_result = log_probs_full.gather(-1, kl_topk_indices.to(logits.device))  # (B, L, k) float, has grad
            else:
                # no-grad paths (compute_kl_topk_indices / compute_ref_log_prob_topk): save memory
                with torch.no_grad():
                    # in-place log_softmax: x -= logsumexp(x) reuses logits storage, avoids OOM
                    logits.sub_(torch.logsumexp(logits, dim=-1, keepdim=True))
                    log_probs_full = logits  # logits now holds log-probs in-place
                    if kl_topk_indices is None:
                        _, topk_result = log_probs_full.topk(kl_topk_k, dim=-1)  # (B, L, k) int64
                    else:
                        topk_result = log_probs_full.gather(-1, kl_topk_indices.to(logits.device))  # (B, L, k)

        return entropy, log_probs, topk_result

    def compute_kl_topk_indices(self, data: DataProto) -> dict:
        """OEL pre-computation step 1: compute student top-k token indices.

        Returns:
            dict with ``kl_topk_indices``: (B, response_length, k) int64 tensor.
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
                _, _, topk_idx = self._forward_micro_batch_with_topk(
                    model_inputs, temperature=temperature, kl_topk_k=k
                )
            idx_list.append(topk_idx)
        return {"kl_topk_indices": torch.cat(idx_list, dim=0)}  # (B, L, k)

    def compute_ref_log_prob_topk(self, data: DataProto) -> dict:
        """OEL pre-computation step 2: gather ref log-probs at student top-k positions.

        Expects ``kl_topk_indices`` in ``data.batch``.
        Returns:
            dict with ``ref_log_prob_topk``: (B, response_length, k) float32 tensor.
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
            kl_topk_indices = model_inputs.pop("kl_topk_indices")
            with torch.no_grad():
                _, _, ref_lp = self._forward_micro_batch_with_topk(
                    model_inputs, temperature=temperature, kl_topk_indices=kl_topk_indices
                )
            ref_lp_list.append(ref_lp)
        return {"ref_log_prob_topk": torch.cat(ref_lp_list, dim=0)}  # (B, L, k)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(
        self,
        data: DataProto,
        calculate_entropy=False,
        raw_logits_dump_request: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, Any]]]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if raw_logits_dump_request is not None and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        dump_records_lst = []
        for micro_batch_index, micro_batch in enumerate(micro_batches):
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, dump_records = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    raw_logits_dump_request=raw_logits_dump_request,
                    micro_batch_index=micro_batch_index,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if dump_records is not None:
                dump_records_lst.extend(dump_records)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if dump_records_lst:
                revert_indices = get_reverse_idx(list(chain.from_iterable(batch_idx_list)))
                dump_records_lst = [dump_records_lst[idx] for idx in revert_indices]

        return log_probs, entropys, dump_records_lst

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
            # OEL pre-computed tensors for analytic_kl
            if getattr(self.config, "kl_loss_type", "") == "analytic_kl":
                if "kl_topk_indices" in data.batch:
                    select_keys.append("kl_topk_indices")
                if "ref_log_prob_topk" in data.batch:
                    select_keys.append("ref_log_prob_topk")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    # Use topk-aware forward when analytic_kl OEL pre-computation is active
                    _use_analytic_topk = (
                        getattr(self.config, "kl_loss_type", "") == "analytic_kl"
                        and getattr(self.config, "topk_kl_k", 256) > 0
                        and "kl_topk_indices" in model_inputs
                    )
                    if _use_analytic_topk:
                        entropy, log_prob, _stu_lp_topk = self._forward_micro_batch_with_topk(
                            model_inputs,
                            temperature=temperature,
                            kl_topk_indices=model_inputs["kl_topk_indices"].to(get_device_id()),
                            calculate_entropy=calculate_entropy,
                        )
                    else:
                        entropy, log_prob = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                        _stu_lp_topk = None

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        if getattr(self.config, "kl_loss_type", "") == "analytic_kl":
                            k_val = getattr(self.config, "topk_kl_k", 256)
                            if k_val > 0 and _stu_lp_topk is not None and "ref_log_prob_topk" in model_inputs:
                                # OEL path: compute divergence from pre-gathered log-probs at student top-k positions.
                                # NOTE: OEL always pre-computes student top-k indices.
                                # - reverse_kl: exact top-k approximation (student top-k is optimal support).
                                # - forward_kl/jsd/symmetric_kl: uses student top-k as support, which is an
                                #   approximation (full correctness requires ref top-k or union). Acceptable
                                #   because student top-k captures most probability mass.
                                ref_lp = model_inputs["ref_log_prob_topk"].to(_stu_lp_topk.device)  # (B, L, k)
                                stu_p = _stu_lp_topk.exp()   # (B, L, k)
                                ref_p = ref_lp.exp()          # (B, L, k)
                                mode = getattr(self.config, "topk_kl_mode", "reverse_kl")
                                if mode == "reverse_kl":
                                    kld = (stu_p * (_stu_lp_topk - ref_lp)).sum(-1)  # (B, L)
                                elif mode == "forward_kl":
                                    kld = (ref_p * (ref_lp - _stu_lp_topk)).sum(-1)  # (B, L)
                                elif mode == "jsd":
                                    beta = getattr(self.config, "topk_kl_jsd_beta", 0.5)
                                    m = beta * ref_p + (1.0 - beta) * stu_p
                                    log_m = m.clamp(min=1e-30).log()
                                    kld = (beta * (ref_p * (ref_lp - log_m)) + (1.0 - beta) * (stu_p * (_stu_lp_topk - log_m))).sum(-1)
                                elif mode == "symmetric_kl":
                                    kld = 0.5 * ((ref_p * (ref_lp - _stu_lp_topk)) + (stu_p * (_stu_lp_topk - ref_lp))).sum(-1)
                                else:
                                    raise ValueError(f"Unknown topk_kl_mode: {mode!r}. Use reverse_kl/forward_kl/jsd/symmetric_kl.")
                            else:
                                raise AssertionError(
                                    "analytic_kl requires pre-computed kl_topk_indices and ref_log_prob_topk "
                                    "in batch. Ensure ray_trainer.py OEL patch is applied and topk_kl_k > 0."
                                )
                        else:
                            ref_log_prob = model_inputs["ref_log_prob"]
                            kld = kl_penalty(
                                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                            )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
