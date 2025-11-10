import time
import numpy as np
from slora.common.unified_mem_allocator import PageType
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
from slora.utils.infer_utils import mark_cost_time, set_random_seed
from slora.utils.infer_utils import calculate_time, mark_start, mark_end
from slora._kernels import dispatch_bgmv
from ...server.router.mixed_req_queue import rprint
import hashlib
from slora.models.peft.alt_to_slora_kernel import dispatch_bgmv_pt, compare_tensors, dispatch_bgmv_pt_exact
import math


import torch
import hashlib

def tensor_hash(t: torch.Tensor, algo="sha256") -> str:
    h = hashlib.new(algo)
    h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class LoraUnorderedBatchMixed:
    def __init__(self, base_model, adapters, infer_adapter=None, finetuning_adapter= None, infer_adapter_alt=None, enable_unified_mem_manager=False):
        self.base_model = base_model

        lora_layer_dim = [adapter.r if adapter is not None else 0 for adapter in adapters]
        self.max_lora_dim = max(lora_layer_dim)
        self.req_bins = torch.zeros(len(adapters), dtype=torch.long, device="cuda")
        self.finetuning_adapter = finetuning_adapter
        self.is_finetuning_batch = False
        self.enable_unified_mem_manager = enable_unified_mem_manager
        
        if enable_unified_mem_manager:
            self.infer_adapter_alt = infer_adapter_alt
            for i, adapter in enumerate(adapters):
                if adapter is None: continue
                idx = infer_adapter_alt.adapter_dirs.index(adapter.lora_dir)
                self.req_bins[i] = idx
        else:
            self.infer_adapter = infer_adapter
            if isinstance(infer_adapter, NaiveInferAdapter):
                self.key_buffer = infer_adapter.key_buffer
                self.value_buffer = infer_adapter.value_buffer
            else:
                self.key_buffer = infer_adapter.mem_manager.key_buffer
                #print("Access key buffer from LoraUnorderedBatchMixed")
                self.value_buffer = infer_adapter.mem_manager.value_buffer
                #print("Access value buffer from LoraUnorderedBatchMixed")
            for i, adapter in enumerate(adapters):
                # FIX ME @TODO: currently not supporting adapter is None
                if adapter is None: continue
                idx = infer_adapter.adapter_dirs.index(adapter.lora_dir)
                self.req_bins[i] = idx

        
        self.kv_embed_dim = base_model.tp_k_head_num_ * base_model.head_dim_


    @torch.no_grad()
    def forward(
            self,
            batch_size, # number of request
            total_token_num,
            max_len_in_batch,
            input_ids, # 1D input tensor
            b_loc, # mapping to memory pool
            b_loc_key, # mapping to key memory pool
            b_loc_value, # mapping to value memory pool
            b_start_loc, # the start index of each request
            b_seq_len, # the current length of each request
            finetune_mask,
            is_prefill=True,
            use_bmm=True,
            no_lora_compute=False,
            ref_mask = None,
            no_lora_copy=False,
            prefill_interrupt_event=None,
            print_time_profile = False):

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

            out = self._prefill(batch_size, total_token_num, max_len_in_batch,
                                    input_ids,
                                    b_loc, b_loc_key, b_loc_value, b_start_loc, b_seq_len, 
                                    finetune_mask, ref_mask, no_lora_compute, prefill_interrupt_event)
            return out
        else:
            for _ in range(3):
                self.delta.append(torch.zeros((len(b_seq_len), self.max_lora_dim), dtype=torch.float16, device="cuda"))
            out = self._decode(batch_size, total_token_num, max_len_in_batch,
                                input_ids,
                                b_loc, b_loc_key, b_loc_value, b_start_loc, b_seq_len,
                                no_lora_compute, no_lora_copy, print_time_profile)
            #print(tensor_hash(out))
            return out

    # Prefill functions for inference and finetuning
    def _prefill(self, batch_size, total_token_num, max_len_in_batch,
                 input_ids,
                 b_loc, b_loc_key, b_loc_value, b_start_loc, b_seq_len, finetune_mask, ref_mask,
                   no_lora_compute=False, prefill_interrupt_event=None):

        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = True
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch

        assert (input_ids.shape[0] == total_token_num)
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])

        infer_state.finetune_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool, device="cuda")
        nr_finetuning_reqs = 0
        finetuning_start = input_ids.shape[0]
        if ref_mask!=None:
            infer_state.ref_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool, device="cuda")
        if finetune_mask is not None:
            for i in range(batch_size):
                if finetune_mask[i] == 1:
                    nr_finetuning_reqs += 1
                    start = b_start_loc[i].item()
                    length = b_seq_len[i].item()
                    if finetuning_start == input_ids.shape[0]:
                        finetuning_start = start
                    infer_state.finetune_mask[start : start + length] = True
            if ref_mask!=None:
                for i in range(batch_size):
                    if ref_mask[i] == 1:
                        nr_finetuning_reqs += 1
                        start = b_start_loc[i].item()
                        length = b_seq_len[i].item()
                        infer_state.ref_mask[start : start + length] = True

        b_seq_len_numpy = b_seq_len.cpu().numpy()
        position_ids = torch.from_numpy(np.concatenate([
            np.arange(0, b_seq_len_numpy[i]) for i in range(len(b_seq_len_numpy))
            ], axis=0)).cuda()
        infer_state.position_cos = torch.index_select(
                self.base_model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
        infer_state.position_sin = torch.index_select(
                self.base_model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
        position_ids = None
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        if self.enable_unified_mem_manager:
            infer_state.alt_mem_manager = self.base_model.alt_mem_manager
            infer_state.b_loc_key = b_loc_key
            infer_state.b_loc_value = b_loc_value
            infer_state.prefill_mem_index_key = self.base_model.alt_mem_manager.alloc(infer_state.total_token_num, PageType.KV_CACHE)
            infer_state.prefill_mem_index_value = self.base_model.alt_mem_manager.alloc(infer_state.total_token_num, PageType.KV_CACHE)
            infer_state.prefill_mem_index_cat = torch.cat([infer_state.prefill_mem_index_key, infer_state.prefill_mem_index_value], dim=0)
            init_bloc(infer_state.b_loc_key, b_seq_len, max_len_in_batch, infer_state.prefill_mem_index_key)
            init_bloc(infer_state.b_loc_value, b_seq_len, max_len_in_batch, infer_state.prefill_mem_index_value)
        else:
            infer_state.mem_manager = self.base_model.mem_manager
            infer_state.prefill_mem_index = self.base_model.mem_manager.alloc(infer_state.total_token_num)
            infer_state.b_loc = b_loc
            init_bloc(b_loc, b_seq_len, max_len_in_batch, infer_state.prefill_mem_index)

        infer_state.prefill_key_buffer = torch.empty(
                (infer_state.total_token_num, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")
        infer_state.prefill_value_buffer = torch.empty(
                (infer_state.total_token_num, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")
        predict_logics = self._context_forward(input_ids, infer_state, no_lora_compute, prefill_interrupt_event=prefill_interrupt_event)
        return predict_logics       

    def interrupt_and_clean(
        self,
        prefill_interrupt_event,
        infer_state,
        FFN_input_vpids: torch.Tensor = None,
        attention_input_vpids: torch.Tensor = None,
    ) -> bool:
        """
        Tensor-based version: if interrupted, free all given vpids (on GPU).
        """
        if prefill_interrupt_event is not None and prefill_interrupt_event.is_set():
            vpids_to_free = None
            if FFN_input_vpids is not None and attention_input_vpids is not None:
                vpids_to_free = torch.cat((FFN_input_vpids, attention_input_vpids))
            elif FFN_input_vpids is not None:
                vpids_to_free = FFN_input_vpids
            elif attention_input_vpids is not None:
                vpids_to_free = attention_input_vpids
            if vpids_to_free is not None and vpids_to_free.numel() > 0:
                print("Freeing vpids!")
                infer_state.alt_mem_manager.free(vpids_to_free)
                self.infer_adapter_alt.unpin_adapters_pages() 
                self.base_model.alt_mem_manager.reset_b_loc_kv(None, None)
            return True

        return False
    
    @torch.no_grad()
    def sanitize_logits(self, logits: torch.Tensor, clamp_min: float = -1e4, clamp_max: float = 1e4) -> torch.Tensor:
        return logits
        logits_fp32 = logits.float()
        logits_fp32 = torch.nan_to_num(logits_fp32, nan=0.0, posinf=0.0, neginf=0.0)
        logits_fp32 = torch.clamp(logits_fp32, min=clamp_min, max=clamp_max)

        if torch.isnan(logits_fp32).any() or torch.isinf(logits_fp32).any():
            print("⚠️ NaN or Inf remain in logits after cleanup!")
        return logits_fp32

    @final
    def _context_forward(self, input_ids, infer_state, no_lora_compute=False, prefill_interrupt_event=None):
        cuda_input_ids = input_ids
        input_embs = self.base_model.pre_infer.context_forward(
                cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        if self.interrupt_and_clean(prefill_interrupt_event, infer_state): 
            return None
        if torch.any(infer_state.finetune_mask):
            infer_state.alt_mem_manager.save_embedding_output(input_embs, infer_state)
        FFN_input_vpids = None
        attention_input_vpids = None
        for i in range(self.base_model.layers_num):
            input_embs = self._lora_context_forward(i, input_embs, infer_state, no_lora_compute)
            if self.interrupt_and_clean(prefill_interrupt_event, infer_state, FFN_input_vpids, attention_input_vpids): 
                return None
            if torch.any(infer_state.finetune_mask):
                FFN_input_vpids = infer_state.alt_mem_manager.save_activations_by_layer(i, input_embs, infer_state, 
                                                                            PageType.FFN_INPUT_ACTIVATION, FFN_input_vpids)
            self.base_model.layers_infer[i]._context_ffn(input_embs, infer_state, self.base_model.trans_layers_weight[i])            
            if self.interrupt_and_clean(prefill_interrupt_event, infer_state, FFN_input_vpids, attention_input_vpids): 
                return None
            if torch.any(infer_state.finetune_mask):
                attention_input_vpids = infer_state.alt_mem_manager.save_activations_by_layer(i, input_embs, infer_state, 
                                                                            PageType.ATTENTION_INPUT_ACTIVATION, attention_input_vpids)
        if self.interrupt_and_clean(prefill_interrupt_event, infer_state, FFN_input_vpids, attention_input_vpids): 
            return None
        if infer_state.ref_mask is not None:
            predict_logics, finetune_logits_per_request, ref_logits_per_request = self.base_model.post_infer.token_forward_alignment(
                    input_embs, infer_state, self.base_model.pre_post_weight)
            infer_state.alt_mem_manager.finetune_logits_per_request.extend(finetune_logits_per_request)
            infer_state.alt_mem_manager.reference_logits_per_request.extend(ref_logits_per_request)
        else:
            finetune_logits_per_request = []
            predict_logics = self.base_model.post_infer.token_forward_with_finetune_outputs(
                    input_embs, finetune_logits_per_request, infer_state, self.base_model.pre_post_weight)
            #predict_logics = self.sanitize_logits(predict_logics)
            if torch.any(infer_state.finetune_mask):
                infer_state.alt_mem_manager.write_to_logit_tensor(finetune_logits_per_request, FFN_input_vpids, attention_input_vpids)
        #print(f"Embedding time: {input_embs_layer_end_time - start:.3f}s")
        #print(f"Transformer layer time: {transformer_layer_end_time - input_embs_layer_end_time:.3f}s")
        #print(f"Output layer time: {output_layer_end_time - transformer_layer_end_time:.3f}s")
        #self.check_invalid_probs(predict_logics)
        return predict_logics

    @final
    def _lora_context_forward(self, layer_id, input_embs, infer_state, no_lora_compute=False):
        self._lora_context_attention(layer_id, input_embs, infer_state, no_lora_compute)
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
        if self.enable_unified_mem_manager:
            start = time.time()
            q = self._lora_get_qkv_alt(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute)
            #if layer_id==16: print(f"\tLayer {layer_id} get qkv time: {time.time() - start:.4f}s")
            input1 = None
            start = time.time()
            layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
            #if layer_id==16: print(f"\tLayer {layer_id} post cache kv time: {time.time() - start:.4f}s")
            # compute attention
            start = time.time()
            o = layer_infer._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
            #if layer_id==16: print(f"\tLayer {layer_id} attention kernel time: {time.time() - start:.4f}s")
            start = time.time()
            o = self._lora_get_o_alt(layer_id, o, infer_state, no_lora_compute)
            #if layer_id==16: print(f"\tLayer {layer_id} get o time: {time.time() - start:.4f}s")
        else:
            q, k, v = self._lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute)
            input1 = None
            layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
            o = layer_infer._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
            o = self._lora_get_o(layer_id, o, infer_state, no_lora_compute)

        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        return

    def _lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        # q (S, H)
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_),
                     base_layer_weight.q_weight_)
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

    def _lora_get_qkv_alt(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        # if layer_id==0:
        #     self.infer_adapter_alt.pin_adapters_pages()
        #if layer_id==16: print(f"\t\t\tLayer {layer_id} pin adapter pages time: {time.time() - start:.5f}s")
        # q (S, H)
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_),
                     base_layer_weight.q_weight_)
        buffer_address, a_start_lora, a_len_lora, gpu_a_loc_lora_a, gpu_a_loc_lora_b, a_scaling = \
                    self.infer_adapter_alt.get_lora_params_at_layer(layer_id)
        #if layer_id==16: print(f"\t\t\tLayer {layer_id} address conversion time: {time.time() - start:.5f}s")
        assert(len(q)==len(self.batch_req_bins))
        if not no_lora_compute:
            # fix me: @TODO we need to filter out requests querying only base model
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         buffer_address.view(-1, self.kv_embed_dim), 
                                         self.delta[0], gpu_a_loc_lora_a, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         0, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(self.delta[0], buffer_address.view(-1, self.kv_embed_dim), 
                                         q, a_scaling, 
                                         gpu_a_loc_lora_b, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         0, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(                                             
                    self.delta[0],
                    input_embs.view(-1, base_layer_infer.embed_dim_),
                    buffer_address,
                    a_start_lora,
                    a_len_lora,
                    gpu_a_loc_lora_a,
                    self.batch_req_bins,
                    0,                                   # qkvo
                    a_scaling,
                )

                dispatch_bgmv(                                             # 2️⃣ EXPAND (B-side)
                    q,
                    self.delta[0],
                    buffer_address,
                    a_start_lora,
                    a_len_lora,
                    gpu_a_loc_lora_b,
                    self.batch_req_bins,
                    0,
                    a_scaling,
                )

        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                          infer_state.position_cos, infer_state.position_sin)

        # k (S, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        if not no_lora_compute:
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         buffer_address.view(-1, self.kv_embed_dim), 
                                         self.delta[1], gpu_a_loc_lora_a, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         1, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(self.delta[1], buffer_address.view(-1, self.kv_embed_dim), 
                                         cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                                         a_scaling, 
                                         gpu_a_loc_lora_b, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         1, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(                                             
                    self.delta[1],
                    input_embs.view(-1, base_layer_infer.embed_dim_),
                    buffer_address,
                    a_start_lora,
                    a_len_lora,
                    gpu_a_loc_lora_a,
                    self.batch_req_bins,
                    1,                                   # qkvo
                    a_scaling,
                )

                dispatch_bgmv(                                             # 2️⃣ EXPAND (B-side)
                    cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                    self.delta[1],
                    buffer_address,
                    a_start_lora,
                    a_len_lora,
                    gpu_a_loc_lora_b,
                    self.batch_req_bins,
                    1,
                    a_scaling,
                )
               
        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # v (S, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
       
        if not no_lora_compute:
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64 and len(infer_state.b_seq_len) >= 2:
            # if 1 ==0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         buffer_address.view(-1, self.kv_embed_dim), 
                                         self.delta[2], gpu_a_loc_lora_a, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         2, self.max_lora_dim, self.max_b_seq_len)       
                lora_get_qkvo_fwd_expand(self.delta[2], buffer_address.view(-1, self.kv_embed_dim), 
                                         cache_v.view(-1, base_model.tp_v_head_num_ * base_model.head_dim_), 
                                         a_scaling, 
                                         gpu_a_loc_lora_b, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         2, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(                                             
                    self.delta[2],
                    input_embs.view(-1, base_layer_infer.embed_dim_),
                    buffer_address,
                    a_start_lora,
                    a_len_lora,
                    gpu_a_loc_lora_a,
                    self.batch_req_bins,
                    2,                                   # qkvo
                    a_scaling,
                )

                dispatch_bgmv(                                             # 2️⃣ EXPAND (B-side)
                    cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                     self.delta[2],
                    buffer_address,
                    a_start_lora,
                    a_len_lora,
                    gpu_a_loc_lora_b,
                    self.batch_req_bins,
                    2,
                    a_scaling,
                )
        return q

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

    def _lora_get_o_alt(self, layer_id, input, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        o = torch.mm(input.view(-1, base_layer_infer.embed_dim_),
                          base_layer_weight.o_weight_)
        if not no_lora_compute:
            delta_oA = self.delta[0]
            buffer_address, a_start_lora, a_len_lora, gpu_a_loc_lora_a, gpu_a_loc_lora_b, a_scaling = \
                self.infer_adapter_alt.get_lora_params_at_layer(layer_id)
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input.view(-1, base_layer_infer.embed_dim_), 
                                         buffer_address.view(-1, self.kv_embed_dim), 
                                         delta_oA, gpu_a_loc_lora_a, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         3, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_oA, buffer_address.view(-1, self.kv_embed_dim), 
                                         o, a_scaling, 
                                         gpu_a_loc_lora_b, a_start_lora, 
                                         a_len_lora, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         3, self.max_lora_dim, self.max_b_seq_len)
            else:
                
                dispatch_bgmv(delta_oA, input.view(-1, base_layer_infer.embed_dim_), 
                            buffer_address, 
                            a_start_lora, a_len_lora, 
                            gpu_a_loc_lora_a, self.batch_req_bins, 3, a_scaling)

                dispatch_bgmv(o, delta_oA, buffer_address, a_start_lora, 
                            a_len_lora, gpu_a_loc_lora_b, 
                            self.batch_req_bins, 3, a_scaling)   
        return o

    # Decoding functions for inference
    def _decode(self, batch_size, total_token_num, max_len_in_batch,
                input_ids, b_loc, b_loc_key, b_loc_value, 
                b_start_loc, b_seq_len, no_lora_compute=False, no_lora_copy=False, print_time_profile=False):
        start = time.time()
        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])
        infer_state.b_start_loc = b_start_loc 
        infer_state.b_seq_len = b_seq_len

        if self.enable_unified_mem_manager:
            infer_state.alt_mem_manager = self.base_model.alt_mem_manager
            infer_state.b_loc_key = b_loc_key
            infer_state.b_loc_value = b_loc_value
            #start = time.time()
            alloc_mem = self.base_model.alt_mem_manager.alloc_contiguous_kv(batch_size, PageType.KV_CACHE)
            #print(f"[decode] alloc_contiguous_kv time: {time.time() - start:.5f}s")
            if alloc_mem is not None:
                infer_state.decode_is_contiguous = True
                infer_state.decode_mem_index_key = alloc_mem[0]
                infer_state.decode_mem_start_key = alloc_mem[1]
                infer_state.decode_mem_end_key = alloc_mem[2]
                infer_state.decode_mem_index_value = alloc_mem[3]
                infer_state.decode_mem_start_value = alloc_mem[4]
                infer_state.decode_mem_end_value = alloc_mem[5]
                b_loc_key[:, max_len_in_batch - 1] = infer_state.decode_mem_index_key
                b_loc_value[:, max_len_in_batch - 1] = infer_state.decode_mem_index_value
                self.base_model.alt_mem_manager.reset_b_loc_kv(b_loc_key, b_loc_value)
            else:    
                infer_state.decode_is_contiguous = False
                infer_state.decode_mem_index_key = self.base_model.alt_mem_manager.alloc(batch_size, PageType.KV_CACHE)
                infer_state.decode_mem_index_value = self.base_model.alt_mem_manager.alloc(batch_size, PageType.KV_CACHE)
                infer_state.decode_mem_index_cat = torch.cat([infer_state.decode_mem_index_key, infer_state.decode_mem_index_value], dim=0)
                infer_state.decode_key_buffer = torch.empty(
                            (batch_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                            dtype=torch.float16, device="cuda")
                infer_state.decode_value_buffer = torch.empty(
                            (batch_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                            dtype=torch.float16, device="cuda")
                b_loc_key[:, max_len_in_batch - 1] = infer_state.decode_mem_index_key
                b_loc_value[:, max_len_in_batch - 1] = infer_state.decode_mem_index_value
                self.base_model.alt_mem_manager.reset_b_loc_kv(b_loc_key, b_loc_value)
        else:
            infer_state.b_loc = b_loc
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
        predict_logics = self._token_forward(input_ids, infer_state, no_lora_compute, no_lora_copy, print_time_profile=print_time_profile)
        # if print_time_profile:
        #     print(f"[forward engine]:total decode time {(time.time() - start):.5f}\n")
        return predict_logics
    
    @final
    def _token_forward(self, input_ids, infer_state, no_lora_compute=False, no_lora_copy=False, print_time_profile=True):
        cuda_input_ids = input_ids
        input_embs = self.base_model.pre_infer.token_forward(
                cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        for i in range(self.base_model.layers_num):
            input_embs = self._lora_token_forward(i, input_embs, infer_state, no_lora_compute, no_lora_copy)
           
        predict_logics = self.base_model.post_infer.token_forward(
                input_embs, infer_state, self.base_model.pre_post_weight, return_logics=True)
        #predict_logics = self.sanitize_logits(predict_logics)
        return predict_logics

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

    # @calculate_time(show=True, min_cost_ms=0)
    # this impl dont to use @mark_cost_time
    def _lora_token_attention(self, layer_id, input_embs, infer_state, no_lora_compute=False, no_lora_copy=False):
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # layer normalization
        input1 = layer_infer._att_norm(input_embs, infer_state, layer_weight)
        #self.check_invalid_probs(input1)
        # fetch k, v
        cache_k, cache_v = layer_infer._pre_cache_kv(infer_state, layer_weight)
        if self.enable_unified_mem_manager:
            q = self._batch_lora_get_qkv_alt(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute, no_lora_copy)
            input1 = None
            layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
            o = layer_infer._token_attention_kernel(q, infer_state, layer_weight, q_alt=q)
            q = None
            o = self._batch_lora_get_o_alt(layer_id, o, infer_state, no_lora_compute)
        else:
            q = self._batch_lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute, no_lora_copy)
            input1 = None
            layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
            o = layer_infer._token_attention_kernel(q, infer_state, layer_weight)
            q = None
            o = self._batch_lora_get_o(layer_id, o, infer_state, no_lora_compute)

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
    
    def _batch_lora_get_qkv_alt(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False, no_lora_copy=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        if layer_id==0:
            self.infer_adapter_alt.pin_adapters_pages()
        # q (bs, H)
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.q_weight_)
        assert(len(q)==len(self.req_bins))
        buffer_address, a_start_lora, a_len_lora, gpu_a_loc_lora_a, gpu_a_loc_lora_b, a_scaling = \
                    self.infer_adapter_alt.get_lora_params_at_layer(layer_id)
    
        if not no_lora_compute:
            dispatch_bgmv(                                             
                self.delta[0], input_embs.view(-1, base_layer_infer.embed_dim_),
                buffer_address, a_start_lora, a_len_lora, gpu_a_loc_lora_a,
                self.req_bins, 0, a_scaling,
            )

            dispatch_bgmv(                                             # 2️⃣ EXPAND (B-side)
                q, self.delta[0],
                buffer_address, a_start_lora, a_len_lora, gpu_a_loc_lora_b,
                self.req_bins, 0, a_scaling,
            )

        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                          infer_state.position_cos, infer_state.position_sin)

        # k (bs, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))

        if not no_lora_compute:
            dispatch_bgmv(                                             
                self.delta[1],
                input_embs.view(-1, base_layer_infer.embed_dim_),
                buffer_address,
                a_start_lora,
                a_len_lora,
                gpu_a_loc_lora_a,
                self.req_bins,
                1,                                   # qkvo
                a_scaling,
            )

            dispatch_bgmv(                                             # 2️⃣ EXPAND (B-side)
                cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                self.delta[1],
                buffer_address,
                a_start_lora,
                a_len_lora,
                gpu_a_loc_lora_b,
                self.req_bins,
                1,
                a_scaling,
            )
            # delta_kA = None
            # mark_end("get_k")

        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # v (bs, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))

        if not no_lora_compute:
            # mark_start("get_v")
            dispatch_bgmv(                                             
                self.delta[2],
                input_embs.view(-1, base_layer_infer.embed_dim_),
                buffer_address,
                a_start_lora,
                a_len_lora,
                gpu_a_loc_lora_a,
                self.req_bins,
                2,                                   # qkvo
                a_scaling,
            )

            dispatch_bgmv(                                             # 2️⃣ EXPAND (B-side)
                cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                    self.delta[2],
                buffer_address,
                a_start_lora,
                a_len_lora,
                gpu_a_loc_lora_b,
                self.req_bins,
                2,
                a_scaling,
            )  
        return q

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
    
    def _batch_lora_get_o_alt(self, layer_id, input, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        o = torch.mm(input.view(-1, base_layer_infer.embed_dim_),
                          base_layer_weight.o_weight_)
        #self.check_invalid_probs(o)
        if not no_lora_compute:
            # mark_start("get_o")
            buffer_address, a_start_lora, a_len_lora, gpu_a_loc_lora_a, gpu_a_loc_lora_b, a_scaling = \
                self.infer_adapter_alt.get_lora_params_at_layer(layer_id)
            delta_oA = self.delta[0]
            dispatch_bgmv(delta_oA, input.view(-1, base_layer_infer.embed_dim_), 
                            buffer_address, 
                            a_start_lora, a_len_lora, 
                            gpu_a_loc_lora_a, self.req_bins, 3, a_scaling)
            #self.check_invalid_probs(delta_oA)
            dispatch_bgmv(o, delta_oA, buffer_address, a_start_lora, 
                            a_len_lora, gpu_a_loc_lora_b, 
                            self.req_bins, 3, a_scaling) 


        if layer_id==self.base_model.layers_num-1:
            self.infer_adapter_alt.unpin_adapters_pages() 
   
        return o