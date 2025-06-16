import time
import numpy as np
from slora.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from slora.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from slora.models.peft.layer_weights.lora_layer_weight import LoraLayerWeight
import torch
import torch.nn as nn
from typing import final
from typing import Tuple
from slora.common.infer_utils import init_bloc
from slora.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from slora.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from slora.models.peft.triton_kernel.lora.lora_prefill import lora_get_qkvo_fwd_shrink, lora_get_qkvo_fwd_expand
from slora.server.router.model_infer.naive_infer_adapter import NaiveInferAdapter
from slora.utils.infer_utils import mark_cost_time
from slora.utils.infer_utils import calculate_time, mark_start, mark_end
from slora._kernels import dispatch_bgmv
from ...server.router.mixed_req_queue import rprint
import hashlib
from slora.models.peft.alt_to_slora_kernel import dispatch_bgmv_pt, compare_tensors, dispatch_bgmv_pt_exact
import math

class LoraUnorderedBatchMixed:

    def __init__(self, base_model, adapters, infer_adapter=None, finetuning_adapter= None):
        self.base_model = base_model

        lora_layer_dim = [adapter.r if adapter is not None else 0 for adapter in adapters]
        self.max_lora_dim = max(lora_layer_dim)
        self.req_bins = torch.zeros(len(adapters), dtype=torch.long, device="cuda")
        self.finetuning_adapter = finetuning_adapter
        self.is_finetuning_batch = False

        if infer_adapter is not None:
            self.infer_adapter = infer_adapter
            if isinstance(infer_adapter, NaiveInferAdapter):
                self.key_buffer = infer_adapter.key_buffer
                self.value_buffer = infer_adapter.value_buffer
            else:
                self.key_buffer = infer_adapter.mem_manager.key_buffer
                self.value_buffer = infer_adapter.mem_manager.value_buffer
            for i, adapter in enumerate(adapters):
                # FIX ME @TODO: currently not supporting adapter is None
                if adapter is None: continue
                idx = infer_adapter.adapter_dirs.index(adapter.lora_dir)
                self.req_bins[i] = idx
        
        self.kv_embed_dim = base_model.tp_k_head_num_ * base_model.head_dim_
        self.last_adapter_combined_A = {}


    @torch.no_grad()
    def forward(
            self,
            batch_size, # number of request
            total_token_num,
            max_len_in_batch,
            input_ids, # 1D input tensor
            b_loc, # mapping to memory pool
            b_start_loc, # the start index of each request
            b_seq_len, # the current length of each request
            finetune_mask,
            interrupt_flag,
            is_prefill=True,
            use_bmm=True,
            no_lora_compute=False,
            ref_mask = None,
            no_lora_copy=False):

        rprint("LoraUnorderedBatchMixed: batchsize", batch_size)
        rprint("LoraUnorderedBatchMixed: total_token_num", total_token_num)
        rprint("LoraUnorderedBatchMixed: max_len_in_batch", max_len_in_batch)
        rprint("LoraUnorderedBatchMixed: input_ids shape", input_ids.shape)
        # Notice that batch_lora only support decoding
        assert len(b_loc) == len(b_start_loc) == len(b_seq_len)
        self.delta = []

        self.max_b_seq_len = torch.max(b_seq_len).item()

        if is_prefill:
            assert(len(self.req_bins)==len(b_seq_len))
            self.batch_req_bins = torch.repeat_interleave(self.req_bins, b_seq_len)
            # self.b_start_loc = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.long, device="cuda"), b_seq_len[:-1]]), dim=0)
            for _ in range(3):
                self.delta.append(torch.zeros((len(self.batch_req_bins), self.max_lora_dim), dtype=torch.float16, device="cuda"))
            return self._prefill(batch_size, total_token_num, max_len_in_batch,
                                 input_ids,
                                 b_loc, b_start_loc, b_seq_len, finetune_mask, ref_mask, no_lora_compute, interrupt_flag=interrupt_flag)
        else:
            for _ in range(3):
                self.delta.append(torch.zeros((len(b_seq_len), self.max_lora_dim), dtype=torch.float16, device="cuda"))
            return self._decode(batch_size, total_token_num, max_len_in_batch,
                                input_ids,
                                b_loc, b_start_loc, b_seq_len,
                                no_lora_compute, no_lora_copy)


    def _prefill(self, batch_size, total_token_num, max_len_in_batch,
                 input_ids,
                 b_loc, b_start_loc, b_seq_len, finetune_mask, ref_mask, no_lora_compute=False, interrupt_flag=None):

        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = True
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch

        assert (input_ids.shape[0] == total_token_num)
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])

        infer_state.finetune_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool, device="cuda")
        nr_finetuning_reqs = 0
        for i in range(batch_size):
            if finetune_mask[i] == 1:
                nr_finetuning_reqs += 1
                start = b_start_loc[i].item()
                length = b_seq_len[i].item()
                infer_state.finetune_mask[start : start + length] = True
        if ref_mask!=None:
            infer_state.ref_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool, device="cuda")
            for i in range(batch_size):
                if ref_mask[i] == 1:
                    nr_finetuning_reqs += 1
                    start = b_start_loc[i].item()
                    length = b_seq_len[i].item()
                    infer_state.ref_mask[start : start + length] = True

            # print(f"Ref mask: {torch.count_nonzero(infer_state.ref_mask)}")
            # print(f"Finetune mask: {torch.count_nonzero(infer_state.finetune_mask)}") 
            

            #print(f"ref_mask: {infer_state.ref_mask}")
            #print(f"finetune_mask: {infer_state.finetune_mask}")

        b_seq_len_numpy = b_seq_len.cpu().numpy()
        position_ids = torch.from_numpy(np.concatenate([
            np.arange(0, b_seq_len_numpy[i]) for i in range(len(b_seq_len_numpy))
            ], axis=0)).cuda()
        infer_state.position_cos = torch.index_select(
                self.base_model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
        infer_state.position_sin = torch.index_select(
                self.base_model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
        position_ids = None
        infer_state.b_loc = b_loc
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.mem_manager = self.base_model.mem_manager
        infer_state.prefill_mem_index = self.base_model.mem_manager.alloc(infer_state.total_token_num)
        infer_state.prefill_key_buffer = torch.empty(
                (infer_state.total_token_num, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")
        infer_state.prefill_value_buffer = torch.empty(
                (infer_state.total_token_num, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")
        init_bloc(b_loc, b_seq_len, max_len_in_batch, infer_state.prefill_mem_index)
        predict_logics = self._context_forward(input_ids, infer_state, no_lora_compute, interrupt_flag=interrupt_flag, nr_finetuning_reqs=nr_finetuning_reqs)
        return predict_logics


    def _decode(self, batch_size, total_token_num, max_len_in_batch,
                input_ids,
                b_loc, b_start_loc, b_seq_len, no_lora_compute=False, no_lora_copy=False):
        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])

        infer_state.b_loc = b_loc
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        
        infer_state.mem_manager = self.base_model.mem_manager

        alloc_mem = self.base_model.mem_manager.alloc_contiguous(batch_size)
        if alloc_mem is not None:
            infer_state.decode_is_contiguous = True
            infer_state.decode_mem_index = alloc_mem[0]
            infer_state.decode_mem_start = alloc_mem[1]
            infer_state.decode_mem_end = alloc_mem[2]
            b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index
        else:
            infer_state.decode_is_contiguous = False
            alloc_mem = self.base_model.mem_manager.alloc(batch_size)
            infer_state.decode_mem_index = alloc_mem
            infer_state.decode_key_buffer = torch.empty(
                    (batch_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                    dtype=torch.float16, device="cuda")
            infer_state.decode_value_buffer = torch.empty(
                    (batch_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                    dtype=torch.float16, device="cuda")
            b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index

        infer_state.init_some_extra_state(self.base_model, batch_size, total_token_num, max_len_in_batch,
                                          input_ids, b_loc, b_start_loc, b_seq_len, False)
        predict_logics = self._token_forward(input_ids, infer_state, no_lora_compute, no_lora_copy)
        return predict_logics

    def save_finetune_activations_to_buffer(self, layer_id, input_embs, infer_state):
        finetune_mask = infer_state.finetune_mask  # shape: [total_token_num]
        finetune_activations = input_embs[finetune_mask].clone()  # shape: [N, hidden_size]
        prev_total = sum(infer_state.mem_manager.request_token_info)
        num_new_tokens = finetune_activations.shape[0]
        # Step 2: write activations
        infer_state.mem_manager.finetune_activation_buffer[layer_id][prev_total : prev_total + num_new_tokens] = finetune_activations.clone()
        # Step 3: count finetuning tokens per request using finetune_mask
        if layer_id == self.base_model.layers_num - 1:
            b_start_loc = infer_state.b_start_loc
            b_seq_len = infer_state.b_seq_len
            batch_size = infer_state.batch_size
            for i in range(batch_size):
                start = b_start_loc[i].item()
                end = start + b_seq_len[i].item()
                n_finetune_tokens = finetune_mask[start:end].sum().item()
                if n_finetune_tokens > 0:
                    infer_state.mem_manager.request_token_info.append(n_finetune_tokens)
    
    def save_input_layer_output_to_buffer(self, input_embs, infer_state):
        finetune_mask = infer_state.finetune_mask  # shape: [total_token_num]
        finetune_activations = input_embs[finetune_mask].clone()  # shape: [N, hidden_size]
        prev_total = sum(infer_state.mem_manager.request_token_info)
        num_new_tokens = finetune_activations.shape[0]
        infer_state.mem_manager.input_layer_output[prev_total : prev_total + num_new_tokens] = finetune_activations.clone()

    def report_diff_percent(
        self,
        name: str,
        ours: torch.Tensor,
        slora: torch.Tensor,
        eps: float = 1e-4,       # what “near zero” means for the reference
        thresh: float = 1e-2     # 1% threshold for “bad” elements
    ):
        assert ours.shape == slora.shape, f"shape mismatch: {ours.shape} vs {slora.shape}"

        diff     = ours - slora
        abs_diff = diff.abs()
        ref_abs  = slora.abs()

        # L2 relative error
        rel_l2   = diff.norm() / (slora.norm() + eps)

        # mean and max absolute error
        mean_abs = abs_diff.mean().item()
        max_abs  = abs_diff.max().item()

        # fraction of “bad” elements (relative abs error > thresh)
        bad      = abs_diff > (thresh * (ref_abs + eps))
        frac_bad = float(bad.sum()) / bad.numel() * 100.0

        # fraction of reference elements that are essentially zero
        small    = ref_abs <= eps
        frac_small = float(small.sum()) / small.numel() * 100.0
        print()
        print(f"[{name}] shape={list(ours.shape)}, dtype={ours.dtype}")
        print(f"  L2-rel err: {rel_l2*100:6.2f}%   mean |Δ|: {mean_abs:.3e}   max |Δ|: {max_abs:.3e}")
        print(f"  >{thresh*100:.1f}% err: {frac_bad:5.2f}% of elements")
        print(f"  reference near-zero (<= {eps}): {frac_small:5.2f}%  ") 

    
    @final
    def _context_forward(self, input_ids, infer_state, no_lora_compute=False, interrupt_flag=None, nr_finetuning_reqs=0):
        if interrupt_flag[0] and nr_finetuning_reqs==infer_state.batch_size:
            print(f"\033[91mReceive interrupt before forward starts\033[0m")
            interrupt_flag[0] = False
            if infer_state.ref_mask is not None:
                infer_state.mem_manager.rewind_alignment_pool(int(nr_finetuning_reqs/2))
            else:
                infer_state.mem_manager.finetune_input_ids = infer_state.mem_manager.finetune_input_ids[0:-nr_finetuning_reqs]
            return None
        alt = False
        self.finetuning_adapter.load_to_gpu(prefetch=False, bmm=True)
        cuda_input_ids = input_ids
        rprint("Input ids shape", cuda_input_ids.shape)
        input_embs = self.base_model.pre_infer.context_forward(
                cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        self.save_input_layer_output_to_buffer(input_embs, infer_state)

        if interrupt_flag[0] and nr_finetuning_reqs==infer_state.batch_size:
            print(f"\033[91mReceive interrupt after input layer\033[0m")
            interrupt_flag[0] = False
            if infer_state.ref_mask is not None:
                infer_state.mem_manager.rewind_alignment_pool(int(nr_finetuning_reqs/2))
            else:
                infer_state.mem_manager.finetune_input_ids = infer_state.mem_manager.finetune_input_ids[0:-nr_finetuning_reqs]
            return None
        for i in range(self.base_model.layers_num):
            input_embs, q_alt, k_alt, v_alt, o_alt = self._lora_context_forward(i, input_embs, infer_state, no_lora_compute)
            if i==31:
                infer_state.mem_manager.saved_q = q_alt.clone()
                infer_state.mem_manager.saved_k = k_alt.view(k_alt.size(0), -1).clone()
                infer_state.mem_manager.saved_v = v_alt.view(v_alt.size(0), -1).clone()
                infer_state.mem_manager.saved_o = o_alt.clone()
            if interrupt_flag[0] and nr_finetuning_reqs==infer_state.batch_size:
                print(f"\033[91mReceive interrupt after transformer layer {i}\033[0m")
                interrupt_flag[0] = False
                if infer_state.ref_mask is not None:
                    infer_state.mem_manager.rewind_alignment_pool(int(nr_finetuning_reqs/2))
                else:
                    infer_state.mem_manager.finetune_input_ids = infer_state.mem_manager.finetune_input_ids[0:-nr_finetuning_reqs]
                return None
            self.save_ffn_input_to_buffer(i, input_embs, infer_state)
            layer_weight = self.base_model.trans_layers_weight[i]
            layer_infer = self.base_model.layers_infer[i]
            layer_infer._context_ffn(input_embs, infer_state, layer_weight)
            self.save_finetune_activations_to_buffer(i, input_embs, infer_state)
        
        if infer_state.ref_mask is not None:
            predict_logics, finetune_logits_per_request, ref_logits_per_request = self.base_model.post_infer.token_forward_alignment(
                    input_embs, infer_state, self.base_model.pre_post_weight)
            # for i in range(len(ref_logits_per_request)):
            #     print(f"ref_logits_per_request {i} shape: {ref_logits_per_request[i].shape}")
            #     print(f"finetune_logits_per_request {i} shape: {finetune_logits_per_request[i].shape}")
            infer_state.mem_manager.finetune_logits_per_request.extend(finetune_logits_per_request)
            infer_state.mem_manager.reference_logits_per_request.extend(ref_logits_per_request)
        else:
            predict_logics, finetune_logits_per_request = self.base_model.post_infer.token_forward_with_finetune_outputs(
                    input_embs, infer_state, self.base_model.pre_post_weight)
            infer_state.mem_manager.finetune_logits_per_request.extend(finetune_logits_per_request)
        return predict_logics


    @final
    def _token_forward(self, input_ids, infer_state, no_lora_compute=False, no_lora_copy=False):
        cuda_input_ids = input_ids
        input_embs = self.base_model.pre_infer.token_forward(
                cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        for i in range(self.base_model.layers_num):
            input_embs = self._lora_token_forward(i, input_embs, infer_state, no_lora_compute, no_lora_copy)
        predict_logics = self.base_model.post_infer.token_forward(
                input_embs, infer_state, self.base_model.pre_post_weight, return_logics=True)
        return predict_logics

    def save_ffn_input_to_buffer(self, layer_id, input_embs, infer_state):
        finetune_mask = infer_state.finetune_mask  # shape: [total_token_num]
        finetune_activations = input_embs[finetune_mask].clone()  # shape: [N, hidden_size]
        prev_total = sum(infer_state.mem_manager.request_token_info)
        num_new_tokens = finetune_activations.shape[0]
        infer_state.mem_manager.ffn_input_buffer[layer_id][prev_total : prev_total + num_new_tokens] = finetune_activations

    @final
    def _lora_context_forward(self, layer_id, input_embs, infer_state, no_lora_compute=False):
        q_alt, k_alt, v_alt, o = self._lora_context_attention(layer_id, input_embs, infer_state, no_lora_compute)
        # layer_weight = self.base_model.trans_layers_weight[layer_id]
        # layer_infer = self.base_model.layers_infer[layer_id]
        # self.save_ffn_input_to_buffer(layer_id, input_embs, infer_state)
        # layer_infer._context_ffn(input_embs, infer_state, layer_weight)
        return input_embs, q_alt, k_alt, v_alt, o


    @final
    # @calculate_time(show=True, min_cost_ms=0)
    def _lora_token_forward(self, layer_id, input_embs, infer_state, no_lora_compute=False, no_lora_copy=False):
        self._lora_token_attention(layer_id, input_embs, infer_state, no_lora_compute, no_lora_copy)
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # mark_start("token_ffn")
        layer_infer._token_ffn(input_embs, infer_state, layer_weight)
        # mark_end("token_ffn")
        return input_embs


    # @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _lora_context_attention(self, layer_id, input_embs, infer_state, no_lora_compute=False):
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # layer normalization
        input1 = layer_infer._att_norm(input_embs, infer_state, layer_weight)
        # fetch k, v
        cache_k, cache_v = layer_infer._pre_cache_kv(infer_state, layer_weight)
        # gen new q, k, v (batch different adapters)
        q, k_alt, v_alt = self._lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute)
        q_alt = q.clone() #TODO: remove this
        input1 = None
        layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        # compute attention
        o = layer_infer._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        q = None
        o = self._lora_get_o(layer_id, o, infer_state, no_lora_compute)
        # if self.world_size_ > 1:
        #     dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        # residual
        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        return q_alt, k_alt, v_alt, o


    # @calculate_time(show=True, min_cost_ms=0)
    # this impl dont to use @mark_cost_time
    def _lora_token_attention(self, layer_id, input_embs, infer_state, no_lora_compute=False, no_lora_copy=False):
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # layer normalization
        input1 = layer_infer._att_norm(input_embs, infer_state, layer_weight)
        # fetch k, v
        cache_k, cache_v = layer_infer._pre_cache_kv(infer_state, layer_weight)
        # gen new q, k, v (batch different adapters)
        q = self._batch_lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute, no_lora_copy)
        input1 = None
        layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        # compute attention
        o = layer_infer._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._batch_lora_get_o(layer_id, o, infer_state, no_lora_compute)
        # if self.world_size_ > 1:
        #     dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        return
    
    # @calculate_time(show=True, min_cost_ms=0)
    def _batch_lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False, no_lora_copy=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        # q (bs, H)
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.q_weight_)
         # @TODO: fix me, filter requests querying only base model
        assert(len(q)==len(self.req_bins))

        if not no_lora_compute:
            # mark_start("get_q")
            delta_qA = self.delta[0]
            dispatch_bgmv(delta_qA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 0, self.infer_adapter.a_scaling)
            dispatch_bgmv(q, delta_qA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 0, self.infer_adapter.a_scaling)
            # delta_qA = None
            # mark_end("get_q")

        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                          infer_state.position_cos, infer_state.position_sin)

        # k (bs, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))

        if not no_lora_compute:
            # mark_start("get_k")
            delta_kA = self.delta[1]
            dispatch_bgmv(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 1, self.infer_adapter.a_scaling)
            dispatch_bgmv(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                          delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 1, self.infer_adapter.a_scaling)
            # delta_kA = None
            # mark_end("get_k")

        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # v (bs, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))

        if not no_lora_compute:
            # mark_start("get_v")
            delta_vA = self.delta[2]
            dispatch_bgmv(delta_vA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 2, self.infer_adapter.a_scaling)
            dispatch_bgmv(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                          delta_vA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 2, self.infer_adapter.a_scaling)
            # delta_vA = None
            # mark_end("get_v")

        return q        
    

    def get_w_combined_A(self, a_loc, a_start, a_len, adapter_index, layer_id):
        start_idx = a_start[adapter_index].item()
        length = a_len[adapter_index].item()
        indices = a_loc[start_idx: start_idx + length]  # flat index range in key_buffer
        return self.key_buffer[layer_id][indices]
    
    
    def hash_tensor(self, tensor: torch.Tensor) -> str:
        return hashlib.sha256(tensor.detach().to('cpu').float().numpy().tobytes()).hexdigest()

    def _lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False)->torch.Tensor:
        # current_w_combined_A = self.get_w_combined_A(
        #     self.infer_adapter.a_loc, self.infer_adapter.a_start, self.infer_adapter.a_len, 0, layer_id)
        # print(f"hashed current_w_combined_A for layer {layer_id}, {self.hash_tensor(current_w_combined_A)}")
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        # q (S, H)
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_),
                     base_layer_weight.q_weight_)
        q_base = q.clone() #TODO: remove this
        #print(self.req_bins)
        assert(len(q)==len(self.batch_req_bins))
        # q = q_base + input * A * B * scaling
        # input: (S, H) A: (H, R) B: (R, H)
        if not no_lora_compute:
            # fix me: @TODO we need to filter out requests querying only base model
            delta_qA = self.delta[0]
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         delta_qA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         0, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_qA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         q, self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         0, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(                                             # 1️⃣ SHRINK  (A-side)
                    delta_qA,
                    input_embs.view(-1, base_layer_infer.embed_dim_),
                    self.key_buffer[layer_id],
                    self.infer_adapter.a_start,
                    self.infer_adapter.a_len,
                    self.infer_adapter.a_loc,
                    self.batch_req_bins,
                    0,                                   # qkvo
                    self.infer_adapter.a_scaling,
                )
                delta_qA_cuda = delta_qA.clone()         # keep for diff

                dispatch_bgmv(                                             # 2️⃣ EXPAND (B-side)
                    q,
                    delta_qA,
                    self.value_buffer[layer_id],
                    self.infer_adapter.a_start,
                    self.infer_adapter.a_len,
                    self.infer_adapter.a_loc,
                    self.batch_req_bins,
                    0,
                    self.infer_adapter.a_scaling,
                )
                q_cuda = q.clone()

                # delta_qA_pt = dispatch_bgmv_pt_exact(
                #     input_embs.view(-1, base_layer_infer.embed_dim_),
                #     self.key_buffer[layer_id],
                #     self.infer_adapter.a_start,
                #     self.infer_adapter.a_len,
                #     self.infer_adapter.a_loc,
                #     self.batch_req_bins,
                #     0,
                #     self.infer_adapter.a_scaling,
                #     first_launch=True
                # )                                       # shape [N , r]

                # q_pt = dispatch_bgmv_pt_exact(
                #     delta_qA_pt,                        # X  is  delta
                #     self.value_buffer[layer_id],
                #     self.infer_adapter.a_start,
                #     self.infer_adapter.a_len,
                #     self.infer_adapter.a_loc,
                #     self.batch_req_bins,
                #     0,
                #     self.infer_adapter.a_scaling,
                #     first_launch=False
                # ).to(input_embs.dtype)
                # q_pt.add_(q_base)                                     # shape [N , D]

                #self.report_diff_percent('q', q_cuda, q_pt)
            # delta_qA = None

        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                       infer_state.position_cos, infer_state.position_sin)


        # k (S, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        k_base = cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_).clone() #TODO: remove this
        if not no_lora_compute:
            delta_kA = self.delta[1]
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         delta_kA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         1, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_kA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                                         self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         1, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                            self.key_buffer[layer_id], 
                            self.infer_adapter.a_start, self.infer_adapter.a_len, 
                            self.infer_adapter.a_loc, self.batch_req_bins, 1, self.infer_adapter.a_scaling)
                delta_kA_cuda = delta_kA.clone()         # keep for diff
                dispatch_bgmv(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                            delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                            self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                            self.batch_req_bins, 1, self.infer_adapter.a_scaling)
                
                k_cuda = cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_).clone()

                # delta_kA_pt = dispatch_bgmv_pt_exact(
                #     input_embs.view(-1, base_layer_infer.embed_dim_),
                #     self.key_buffer[layer_id],
                #     self.infer_adapter.a_start,
                #     self.infer_adapter.a_len,
                #     self.infer_adapter.a_loc,
                #     self.batch_req_bins,
                #     1,
                #     self.infer_adapter.a_scaling,
                #     first_launch=True
                # )                                       # shape [N , r]

                # k_pt = dispatch_bgmv_pt_exact(
                #     delta_kA_pt,                        # X  is  delta
                #     self.value_buffer[layer_id],
                #     self.infer_adapter.a_start,
                #     self.infer_adapter.a_len,
                #     self.infer_adapter.a_loc,
                #     self.batch_req_bins,
                #     1,
                #     self.infer_adapter.a_scaling,
                #     first_launch=False
                # ).to(input_embs.dtype)
                # k_pt.add_(k_base)  
                #self.report_diff_percent('k', k_cuda, k_pt)
            # delta_kA = None

        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)
        k = cache_k.clone()

        # v (S, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        v_base = cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_).clone() #TODO: remove this
        if not no_lora_compute:
            delta_vA = self.delta[2]
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64 and len(infer_state.b_seq_len) >= 2:
            # if 1 ==0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         delta_vA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         2, self.max_lora_dim, self.max_b_seq_len)       
                lora_get_qkvo_fwd_expand(delta_vA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         cache_v.view(-1, base_model.tp_v_head_num_ * base_model.head_dim_), 
                                         self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         2, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(delta_vA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                            self.key_buffer[layer_id], 
                            self.infer_adapter.a_start, self.infer_adapter.a_len, 
                            self.infer_adapter.a_loc, self.batch_req_bins, 2, self.infer_adapter.a_scaling)
                delta_vA_cuda = delta_vA.clone()
                v_base = cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_).clone()
                dispatch_bgmv(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                            delta_vA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                            self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                            self.batch_req_bins, 2, self.infer_adapter.a_scaling)
                v_cuda = cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_).clone()
                # delta_vA_pt = dispatch_bgmv_pt_exact(
                #     input_embs.view(-1, base_layer_infer.embed_dim_),
                #     self.key_buffer[layer_id],
                #     self.infer_adapter.a_start,
                #     self.infer_adapter.a_len,
                #     self.infer_adapter.a_loc,
                #     self.batch_req_bins,
                #     2,
                #     self.infer_adapter.a_scaling,
                #     first_launch=True
                # )                                       # shape [N , r]
                # # v = wv*x + x*Bv.T * scaling * Av.T
                # v_pt = dispatch_bgmv_pt_exact(
                #     delta_vA_pt,                        # X  is  delta
                #     self.value_buffer[layer_id],
                #     self.infer_adapter.a_start,
                #     self.infer_adapter.a_len,
                #     self.infer_adapter.a_loc,
                #     self.batch_req_bins,
                #     2,
                #     self.infer_adapter.a_scaling,
                #     first_launch=False
                # ).to(input_embs.dtype)
                # v_pt.add_(v_base)  
                #self.report_diff_percent('v', v_cuda, v_pt)
        v = cache_v.clone()
            # delta_vA = None
        return q, k, v
    

    # @calculate_time(show=True, min_cost_ms=0)
    def _batch_lora_get_o(self, layer_id, input, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        
        o = torch.mm(input.view(-1, base_layer_infer.embed_dim_),
                          base_layer_weight.o_weight_)
        
        if not no_lora_compute:
            # mark_start("get_o")
            delta_oA = self.delta[0]
            dispatch_bgmv(delta_oA, input.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 3, self.infer_adapter.a_scaling)
            dispatch_bgmv(o, delta_oA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 3, self.infer_adapter.a_scaling)
            # delta_oA = None
            # mark_end("get_o")
        return o


    def _lora_get_o(self, layer_id, input, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        o = torch.mm(input.view(-1, base_layer_infer.embed_dim_),
                          base_layer_weight.o_weight_)
        if not no_lora_compute:
            delta_oA = self.delta[0]
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         delta_oA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         3, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_oA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         o, self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         3, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(delta_oA, input.view(-1, base_layer_infer.embed_dim_), 
                            self.key_buffer[layer_id], 
                            self.infer_adapter.a_start, self.infer_adapter.a_len, 
                            self.infer_adapter.a_loc, self.batch_req_bins, 3, self.infer_adapter.a_scaling)
                
                dispatch_bgmv(o, delta_oA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                            self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                            self.batch_req_bins, 3, self.infer_adapter.a_scaling)
                o_cuda = o.clone()
                    
        return o
    
    def attention_layer_forward_2(self, x_in, layer_id, infer_state):
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        lora_weight = self.finetuning_adapter.layers[layer_id]
        scaling = self.finetuning_adapter.scaling
        eps = self.base_model.backward_engine.eps_                    
        s= self.finetuning_adapter.scaling
        if lora_weight.w_combined is None:
            lora_weight.w_combined = lora_weight.w_combined_home.to("cuda")
        w_combined = lora_weight.w_combined_home

        w_q, w_k, w_v, w_o = (
            layer_weight.q_weight_,
            layer_weight.k_weight_,
            layer_weight.v_weight_,
            layer_weight.o_weight_,
        )
        w_attn_norm = layer_weight.att_norm_weight_

        # LoRA matrices
        l = lora_weight
        qA, qB = l.q_lora_A, l.q_lora_B
        kA, kB = l.k_lora_A, l.k_lora_B
        vA, vB = l.v_lora_A, l.v_lora_B
        oA, oB = l.o_lora_A, l.o_lora_B

        # rmsnorm(x)
        x_norm = rmsnorm_forward(x_in, w_attn_norm, eps=eps)

        #QKV  (base + LoRA)
        q_base = x_norm @ w_q
        q_lora = s * ((x_norm @ qA) @ qB)
        q = q_base + q_lora

        q = x_norm @ w_q.T + s * ((x_norm @ qB.T) @ qA.T)
        k = x_norm @ w_k.T + s * ((x_norm @ kB.T) @ kA.T)
        v = x_norm @ w_v.T + s * ((x_norm @ vB.T) @ vA.T)

        d_k = q.shape[-1]
        attn_scores = q @ k.T / d_k**0.5
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = attn_weights @ v  # [N, D]             

        # output proj (base + LoRA)
        o = attn_out @ w_o.T + s * ((attn_out @ oB.T) @ oA.T)

        o_base = attn_out @ w_o.T
        o_lora_B_out = attn_out @ oB.T
        o_lora = scaling * (o_lora_B_out @ oA.T)
        o = o_base + o_lora
        x_in = x_in + o.view(-1, self.base_model.layers_infer[layer_id].embed_dim_)
        
        return x_in

    def attention_layer_forward(self, x_in, layer_id, infer_state):
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        lora_weight = self.finetuning_adapter.layers[layer_id]
        eps = self.base_model.backward_engine.eps_                    
        scaling = self.finetuning_adapter.scaling
        base_layer_infer = self.base_model.layers_infer[layer_id]

        # Get base weights
        w_q = layer_weight.q_weight_
        w_k = layer_weight.k_weight_
        w_v = layer_weight.v_weight_
        w_o = layer_weight.o_weight_
        w_attn_norm = layer_weight.att_norm_weight_

        # RMSNorm
        x_norm = rmsnorm_forward(x_in, w_attn_norm, eps=eps)
       
        # Base projections
        q_base = torch.mm(x_norm.view(-1, base_layer_infer.embed_dim_), w_q)
        k_base = torch.mm(x_norm.view(-1, base_layer_infer.embed_dim_), w_k)
        v_base = torch.mm(x_norm.view(-1, base_layer_infer.embed_dim_), w_v)

        # === Load LoRA weight block ===
        if lora_weight.w_combined is None:
            lora_weight.w_combined = lora_weight.w_combined_home.to("cuda")

        w_combined = lora_weight.w_combined  # [2, 4r, H, Hd]
        r = w_combined.shape[1] // 4
        H, Hd = w_combined.shape[2], w_combined.shape[3]

        q_lora_A = w_combined[0, 0 : r].reshape(r, -1).T
        q_lora_B = w_combined[1, 0 : r].reshape(-1, r).T
        q_lora = torch.mm(x_norm, q_lora_A)
        q_lora = torch.mm(q_lora, q_lora_B).mul_(scaling)

        k_lora_A = w_combined[0, r : 2 * r].reshape(r, -1).T
        k_lora_B = w_combined[1, r : 2 * r].reshape(-1, r).T
        k_lora = torch.mm(x_norm, k_lora_A)
        k_lora = torch.mm(k_lora, k_lora_B).mul_(scaling)

        v_lora_A = w_combined[0, 2 * r : 3 * r].reshape(r, -1).T
        v_lora_B = w_combined[1, 2 * r : 3 * r].reshape(-1, r).T
        v_lora = torch.mm(x_norm, v_lora_A)
        v_lora = torch.mm(v_lora, v_lora_B).mul_(scaling)

        def rotary_emb_fwd_pt(q: torch.Tensor,
                      cos: torch.Tensor,
                      sin: torch.Tensor) -> None:
            T, H, D = q.shape
            Dh = D // 2
            q_flat = q.reshape(T * H, D)
            q_even = q_flat[:, :Dh] 
            q_odd  = q_flat[:, Dh:]
            cos_exp = cos[:, None, :].expand(T, H, Dh).reshape(T * H, Dh)
            sin_exp = sin[:, None, :].expand(T, H, Dh).reshape(T * H, Dh)
            q_even_orig = q_even.clone()
            q_even.mul_(cos_exp).addcmul_(q_odd, sin_exp, value=-1.0)
            q_odd.mul_(cos_exp).addcmul_(q_even_orig, sin_exp)

        q = q_base + q_lora
        rotary_emb_fwd_pt(q.view(-1, base_layer_infer.tp_q_head_num_, self.base_model.head_dim_), infer_state.position_cos, infer_state.position_sin)
        k = k_base + k_lora
        rotary_emb_fwd_pt(k.view(-1, base_layer_infer.tp_k_head_num_, self.base_model.head_dim_), infer_state.position_cos, infer_state.position_sin)
        v = v_base + v_lora


        S, D = q.shape
        H = base_layer_infer.tp_q_head_num_
        Hd = D // H

        q = q.view(S, H, Hd)
        k = k.view(S, H, Hd)
        v = v.view(S, H, Hd)
        attn_out = torch.empty_like(q)

        B = infer_state.b_seq_len.shape[0]
        scale = 1.0 / (Hd ** 0.5)

        for i in range(B):
            start = infer_state.b_start_loc[i].item()
            seqlen = infer_state.b_seq_len[i].item()
            end = start + seqlen

            q_i = q[start:end]  # [L, H, D]
            k_i = k[start:end]
            v_i = v[start:end]

            q_i = q_i.transpose(0, 1)  # [H, L, D]
            k_i = k_i.transpose(0, 1)
            v_i = v_i.transpose(0, 1)

            scores = torch.matmul(q_i, k_i.transpose(-1, -2)) * scale
            mask = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool, device=q.device))
            scores = scores.masked_fill(~mask, float('-inf'))
            probs = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, v_i)  # [H, L, D]
            ctx = ctx.transpose(0, 1).contiguous()  # [L, H, D]
            attn_out[start:end] = ctx

        attn_out = attn_out.view(S, D)  # [S, D]
        # Output projection (base + LoRA)
        o_base = torch.mm(attn_out.view(-1, base_layer_infer.embed_dim_), w_o)
        oA = w_combined[0, 3 * r : 4 * r].reshape(r, -1).T  # [D, r]
        oB = w_combined[1, 3 * r : 4 * r].reshape(-1, r).T  # [r, D]
        o_lora = (attn_out @ oA) @ oB * scaling
        o = o_base + o_lora
        # Residual connection
        x_out = x_in + o.view_as(x_in)
        
        return x_out, q.view(q.size(0), -1), k.view(k.size(0), -1), v.view(v.size(0), -1), attn_out, o

