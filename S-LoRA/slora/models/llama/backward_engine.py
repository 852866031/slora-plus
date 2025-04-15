from slora.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from slora.models.peft.layer_weights.lora_layer_weight import LoraLayerWeight
from slora.models.peft.lora_adapter import LoraTpPartAdapter
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from einops import rearrange
from slora.models.llama.infer_struct import LlamaInferStateInfo
from slora.models.llama.triton_kernel.rmsnorm import rmsnorm_backward, rmsnorm_forward
from slora.common.basemodel import PostLayerInferTpl
from slora.server.router.mixed_req_queue import rprint

class LlamaBackwardEngine():
    def __init__(self, mem_manager, network_config):
        self.mem_manager = mem_manager
        self.eps_ = network_config["rms_norm_eps"]
        self.vocab_size_ = network_config["vocab_size"]
        self.embed_dim_ = network_config["n_embed"]
    

    def get_logits_and_targets(self):
        logits_list = self.mem_manager.finetune_logits_per_request
        input_ids = self.mem_manager.get_concatenated_finetune_input_ids()  # shape [sum(T_i)]
        
        # Compute per-request input length from logits (add 1 to get back original token count)
        token_counts = [logits.shape[0] + 1 for logits in logits_list]  # [T_i - 1] → [T_i]
        input_ids_split = torch.split(input_ids, token_counts)

        # Align logits and targets
        results = []
        for logits, input_ids_i in zip(logits_list, input_ids_split):
            if logits.shape[0] < 1:
                continue  # skip empty logits

            # logits: [T-1, vocab]
            # input_ids_i[1:]: [T-1] — the tokens being predicted
            pred_logits = logits
            target_ids = input_ids_i[1:]

            results.append((pred_logits, target_ids))

        return results  # list of (logits: [T-1, vocab], targets: [T-1])
    
    def _logit_backward(self, logits_and_targets):
        all_logits = []
        all_targets = []

        for logits, targets in logits_and_targets:
            all_logits.append(logits)      # each [T_i - 1, vocab]
            all_targets.append(targets)    # each [T_i - 1]

        logits_cat = torch.cat(all_logits, dim=0)   # [N, vocab]
        targets_cat = torch.cat(all_targets, dim=0) # [N]

        # Compute ∇loss/∇logits manually (cross-entropy gradient)
        probs = torch.softmax(logits_cat, dim=-1)  # [N, vocab]
        probs[torch.arange(len(targets_cat)), targets_cat] -= 1
        probs /= len(targets_cat)  # normalize over total number of tokens

        return probs  # gradient of loss w.r.t logits: [N, vocab]

    
    def _post_layer_backward(self, logit_grad: torch.Tensor, layer_weight):
        lm_head_weight = layer_weight.lm_head_weight_           # [vocab_size, D]
        norm_weight = layer_weight.final_norm_weight_           # [D]
        transformer_out = self.mem_manager.get_finetune_activations(layer_id=-1)  # [N, D]
        rprint("transformer_out shape", transformer_out.shape)
        # Grad of the norm output
        grad_norm_out = logit_grad @ lm_head_weight  # [N, D]
        # Grad of the transformer outpu
        D = transformer_out.shape[-1]
        norm = transformer_out.norm(2, dim=-1, keepdim=True) / (D ** 0.5)
        grad_x_hat = grad_norm_out * norm_weight      # [N, D]
        norm_term = (transformer_out * grad_x_hat).sum(dim=-1, keepdim=True) / (norm + self.eps_)**2
        grad_transformer_out = (grad_x_hat - transformer_out * norm_term / D) / (norm + self.eps_)  # [N, D]
        return grad_transformer_out
    
    def _lora_context_backward(self, layer_id, output_grad: torch.Tensor, layer_weight: LlamaTransformerLayerWeight, finetuning_adapter: LoraTpPartAdapter):
        # backprop for a transformer layer, start from ffn
        ffn_input = self.mem_manager.get_ffn_input(layer_id)  # shape [N, D]
        grad_ffn_input = self._backprop_ffn(ffn_input, output_grad, layer_weight)
        rprint("grad_ffn_input shape", grad_ffn_input.shape)
        last_layer_input = self.mem_manager.get_finetune_activations(layer_id-1)  # shape [N, D]
        lora_weights = finetuning_adapter.layers[layer_id]
        # Backprop through LoRA
        grad_attn_input, grads = self._backpop_attention(last_layer_input, grad_ffn_input, lora_weights, layer_weight)
        rprint("grad_attn_input shape", grad_attn_input.shape)

        return grad_attn_input

    def _backpop_attention(self, 
                        last_layer_input: torch.Tensor, 
                        grad_ffn_input: torch.Tensor, 
                        lora_weight: LoraLayerWeight, 
                        layer_weight: LlamaTransformerLayerWeight):
        #lora_weight has field, q_lora_A, q_lora_B, k_lora_A, k_lora_B, v_lora_A, v_lora_ B o_lora_A, o_lora_B
        # lora_weight has field, att_norm_weight_, q_weight_, k_weight_, v_weight_, o_weight_    
        x = rmsnorm_forward(last_layer_input, layer_weight.att_norm_weight_, eps=self.eps_)
        rprint("lora_weight.q_lora_A shape", lora_weight.q_lora_A.shape)
        rprint("lora_weight.q_lora_B shape", lora_weight.q_lora_B.shape)
        rprint("lora_weight.k_lora_A shape", lora_weight.k_lora_A.shape)
        rprint("lora_weight.k_lora_B shape", lora_weight.k_lora_B.shape)
        rprint("lora_weight.v_lora_A shape", lora_weight.v_lora_A.shape)
        rprint("lora_weight.v_lora_B shape", lora_weight.v_lora_B.shape)
        rprint("lora_weight.o_lora_A shape", lora_weight.o_lora_A.shape)
        rprint("lora_weight.o_lora_B shape", lora_weight.o_lora_B.shape)
        # === Forward recomputation ===
        # QKV base + LoRA
        q_base = x @ layer_weight.q_weight_.T
        k_base = x @ layer_weight.k_weight_.T
        v_base = x @ layer_weight.v_weight_.T

        q_lora_B_out = x @ lora_weight.q_lora_B.T
        k_lora_B_out = x @ lora_weight.k_lora_B.T
        v_lora_B_out = x @ lora_weight.v_lora_B.T

        q = q_base + q_lora_B_out @ lora_weight.q_lora_A.T
        k = k_base + k_lora_B_out @ lora_weight.k_lora_A.T
        v = v_base + v_lora_B_out @ lora_weight.v_lora_A.T

        d_k = q.shape[-1]
        attn_scores = q @ k.T / d_k**0.5
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = attn_weights @ v  # [N, D]

        # === Output projection (base + LoRA) ===
        o_base = attn_out @ layer_weight.o_weight_.T
        o_lora_B_out = attn_out @ lora_weight.o_lora_B.T
        o_lora = o_lora_B_out @ lora_weight.o_lora_A.T
        o = o_base + o_lora  # final output of attention block

        # === Backprop ===
        grad_o = grad_ffn_input  # residual branch
        # Backprop through o = attn_out @ W_o^T + LoRA
        grad_attn_out_base = grad_o @ layer_weight.o_weight_        # from base
        grad_o_lora_A = grad_o.T @ o_lora_B_out                     # [D, r]
        grad_o_lora_B = (lora_weight.o_lora_A.T @ grad_o.T) @ attn_out  # [r, D]
        grad_attn_out_lora = (grad_o @ lora_weight.o_lora_A) @ lora_weight.o_lora_B
        grad_attn_out = grad_attn_out_base + grad_attn_out_lora     # [N, D]
        # Backprop through attn_out = attn_weights @ v
        grad_attn_weights = grad_attn_out @ v.T        # [N, N]
        grad_v = attn_weights.T @ grad_attn_out        # [N, D]
        # Backprop through softmax
        d_attn = grad_attn_weights * attn_weights
        d_attn_sum = d_attn.sum(dim=-1, keepdim=True)
        grad_attn_scores = (d_attn - attn_weights * d_attn_sum) / d_k**0.5
        grad_q = grad_attn_scores @ k
        grad_k = grad_attn_scores.T @ q
        # LoRA grads for Q, K, V
        grad_q_lora_A = grad_q.T @ q_lora_B_out
        grad_q_lora_B = (lora_weight.q_lora_A.T @ grad_q.T) @ x
        grad_k_lora_A = grad_k.T @ k_lora_B_out
        grad_k_lora_B = (lora_weight.k_lora_A.T @ grad_k.T) @ x
        grad_v_lora_A = grad_v.T @ v_lora_B_out
        grad_v_lora_B = (lora_weight.v_lora_A.T @ grad_v.T) @ x
        # Backprop to x through LoRA only
        grad_q_in = (grad_q @ lora_weight.q_lora_A) @ lora_weight.q_lora_B 
        grad_k_in = (grad_k @ lora_weight.k_lora_A) @ lora_weight.k_lora_B 
        grad_v_in = (grad_v @ lora_weight.v_lora_A) @ lora_weight.v_lora_B 
        grad_o_in = (grad_o @ lora_weight.o_lora_A) @ lora_weight.o_lora_B 
        grad_input_lora = grad_q_in + grad_k_in + grad_v_in + grad_o_in
        grad_attn_input = rmsnorm_backward(
            last_layer_input, grad_input_lora, layer_weight.att_norm_weight_, eps=self.eps_)

        grads = {
            "q_lora_A": grad_q_lora_A,
            "q_lora_B": grad_q_lora_B,
            "k_lora_A": grad_k_lora_A,
            "k_lora_B": grad_k_lora_B,
            "v_lora_A": grad_v_lora_A,
            "v_lora_B": grad_v_lora_B,
            "o_lora_A": grad_o_lora_A,
            "o_lora_B": grad_o_lora_B,
        }

        return grad_attn_input, grads


    def _backprop_ffn(self, ffn_input: torch.Tensor, output_grad: torch.Tensor, layer_weight):
        # Recompute RMSNorm
        input1 = rmsnorm_forward(ffn_input, layer_weight.ffn_norm_weight_, eps=self.eps_)
        # Forward again through FFN (no need to save intermediates)
        gate_in = input1 @ layer_weight.gate_proj
        gate_out = torch.nn.functional.silu(gate_in)
        up_out = input1 @ layer_weight.up_proj
        ffn1_out = gate_out * up_out
        # Backprop through down_proj
        grad_ffn1_out = output_grad @ layer_weight.down_proj.T
        # Backprop through elementwise mul
        grad_gate_out = grad_ffn1_out * up_out
        grad_up_out = grad_ffn1_out * gate_out
        # Backprop through up and gate proj
        grad_input_up = grad_up_out @ layer_weight.up_proj.T
        silu_grad = torch.sigmoid(gate_in) * (1 + gate_in * (1 - torch.sigmoid(gate_in)))
        grad_gate_in = grad_gate_out * silu_grad
        grad_input_gate = grad_gate_in @ layer_weight.gate_proj.T
        grad_input1 = grad_input_gate + grad_input_up  # [N, D]
        # Backprop through RMSNorm
        grad_ffn_input = rmsnorm_backward(ffn_input, grad_input1, layer_weight.ffn_norm_weight_, eps=self.eps_)
        return grad_ffn_input
    