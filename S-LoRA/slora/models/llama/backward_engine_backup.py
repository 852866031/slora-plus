from enum import Enum
import math
import time
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

class GradientResumePoint(Enum):
    LOSS = 0
    LOGIT = 1
    POST = 2
    LAYER = 3


class LlamaBackwardEngine():
    def __init__(self, mem_manager, network_config):
        self.mem_manager = mem_manager
        self.eps_ = network_config["rms_norm_eps"]
        self.vocab_size_ = network_config["vocab_size"]
        self.embed_dim_ = network_config["n_embed"]
        #self.last_logit_list = []
        #self.last_input_ids = None
        #self.last_logit_and_targets = []
        self.last_loss = None
        self.gradient_resume_point = GradientResumePoint.LOSS
        self.saved_logit_grad = None
        self.saved_grad_transformer_out = None
        self.resume_after_layer = -1
        self.saved_loss = None
        self.saved_total_tokens_to_process = None
        self.saved_seq_lens = None

    def _context_backward(self, base_model, finetuning_adapter, interrupt_flag):
        def run_layer_backward(start_idx, grad_transformer_out):
            for i in reversed(range(0, start_idx)):
                #time.sleep(0.1)  # TODO: remove this
                layer_weight = base_model.trans_layers_weight[i]
                grad_transformer_out = self._lora_context_backward(i, base_model, grad_transformer_out, finetuning_adapter, self.saved_seq_lens)
                if interrupt_flag[0]:
                    print(f"\033[91mReceive interrupt after layer {i}\033[0m")
                    interrupt_flag[0] = False
                    self.saved_grad_transformer_out = grad_transformer_out
                    self.resume_after_layer = i
                    self.gradient_resume_point = GradientResumePoint.LAYER
                    return False
            self.gradient_resume_point = GradientResumePoint.LOSS
            return True

        if self.gradient_resume_point == GradientResumePoint.LAYER:
            print(f"\033[91mGradient compute Resume after layer {self.resume_after_layer}\033[0m")
            return run_layer_backward(self.resume_after_layer, self.saved_grad_transformer_out), self.saved_loss, self.saved_total_tokens_to_process

        elif self.gradient_resume_point == GradientResumePoint.POST:
            print("\033[91mGradient compute Resume after post layer\033[0m")
            grad_transformer_out = self.saved_grad_transformer_out
            return run_layer_backward(base_model.layers_num, grad_transformer_out), self.saved_loss, self.saved_total_tokens_to_process

        elif self.gradient_resume_point == GradientResumePoint.LOGIT:
            print("\033[91mGradient compute Resume after logit\033[0m")
            grad_transformer_out = self._post_layer_backward(self.saved_logit_grad, base_model.pre_post_weight)
            if interrupt_flag[0]:
                print("\033[91mReceive interrupt after post_layer_grad\033[0m")
                interrupt_flag[0] = False
                self.saved_grad_transformer_out = grad_transformer_out
                self.gradient_resume_point = GradientResumePoint.POST
                return False, self.saved_loss, self.saved_total_tokens_to_process
            return run_layer_backward(base_model.layers_num, grad_transformer_out), self.saved_loss, self.saved_total_tokens_to_process

        else:
            print("\033[91mGradient compute from beginning\033[0m")
            return self._context_backward_logic(base_model, finetuning_adapter, interrupt_flag)

    def _context_backward_logic(self, base_model, finetuning_adapter, interrupt_flag):
        logits_and_targets, total_tokens_to_process, batch_seq_lens = self.get_logits_and_targets()
        self.saved_seq_lens = batch_seq_lens
        loss = self.compute_total_loss(logits_and_targets)
        self.saved_loss = loss
        self.saved_total_tokens_to_process = total_tokens_to_process
        print(f"\033[92mBackward Total Tokens: {total_tokens_to_process}, Loss: {loss:.12f}\033[0m")
        logit_grad = self._logit_backward(logits_and_targets)
        if interrupt_flag[0]: 
            print("\033[91mReceive interrupt after logit_grad\033[0m")
            interrupt_flag[0]= False
            self.saved_logit_grad = logit_grad
            self.gradient_resume_point = GradientResumePoint.LOGIT
            return False, loss, total_tokens_to_process
        grad_transformer_out = self._post_layer_backward(logit_grad, base_model.pre_post_weight)
        if interrupt_flag[0]: 
            print("\033[91mReceive interrupt after post_layer_grad\033[0m")
            interrupt_flag[0]= False
            self.saved_grad_transformer_out = grad_transformer_out
            self.gradient_resume_point = GradientResumePoint.POST
            return False, loss, total_tokens_to_process
        
        for i in reversed(range(base_model.layers_num)):
            grad_transformer_out = self._lora_context_backward(i, base_model, grad_transformer_out, finetuning_adapter, self.saved_seq_lens) 
            if interrupt_flag[0]: 
                print(f"\033[91mReceive interrupt after layer {i}\033[0m")
                interrupt_flag[0]= False
                self.saved_grad_transformer_out = grad_transformer_out
                self.resume_after_layer = i
                self.gradient_resume_point = GradientResumePoint.LAYER
                return False, loss, total_tokens_to_process
        self.gradient_resume_point = GradientResumePoint.LOSS
        return True, loss, total_tokens_to_process

    def get_logits_and_targets(self):
        logits_list = self.mem_manager.finetune_logits_per_request
        device = logits_list[0].device
        batch_seq_lens = torch.tensor(
            [logits.shape[0] for logits in logits_list],
            dtype=torch.long,
            device=device
        )
        input_ids = self.mem_manager.get_concatenated_finetune_input_ids()  # shape [sum(T_i)]
        total_tokens_to_process = input_ids.shape[0]

        # Compute per-request input length from logits (add 1 to get back original token count)
        token_counts = [logits.shape[0] + 1 for logits in logits_list]  # [T_i - 1] → [T_i]
        input_ids_split = torch.split(input_ids, token_counts)

        # Align logits and targets
        results = []
        for logits, input_ids_i in zip(logits_list, input_ids_split):
            if logits.shape[0] < 1:
                continue  # skip empty logits

            # Cast to float32 for numerical stability
            pred_logits = logits.float()          # [T-1, vocab]
            target_ids = input_ids_i[1:].long()   # [T-1], ensure targets are long for CE loss

            results.append((pred_logits, target_ids))

        return results, total_tokens_to_process, batch_seq_lens

    def compute_total_loss(self, logits_and_targets, ignore_index=-100):
        total_loss = 0.0
        total_tokens = 0
        for logits, targets in logits_and_targets:
            if logits.shape[0] != targets.shape[0]:
                raise ValueError(f"Logits and targets length mismatch: {logits.shape[0]} vs {targets.shape[0]}")

            # Compute CE loss   
            loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100, reduction='sum')
            total_loss += loss.item()
            valid_mask = targets != ignore_index
            total_tokens += valid_mask.sum().item()
        if total_tokens == 0:
            return torch.tensor(0.0)

        losses = torch.tensor(total_loss / total_tokens)

        if self.last_loss and torch.equal(losses, self.last_loss):
            print("losses equal to last loss")
        self.last_loss = losses.clone()


        return losses
    
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

    
    def _post_layer_backward_2(self, logit_grad: torch.Tensor, layer_weight):
        lm_head_weight = layer_weight.lm_head_weight_.float()           # [vocab_size, D]
        norm_weight = layer_weight.final_norm_weight_.float()          # [D]
        transformer_out = self.mem_manager.get_finetune_activations(layer_id=-1).float()    # [N, D]
        grad_norm_out = logit_grad @ lm_head_weight  # [N, D]
        D = transformer_out.shape[-1]
        norm = transformer_out.norm(2, dim=-1, keepdim=True) / (D ** 0.5)
        grad_x_hat = grad_norm_out * norm_weight      # [N, D]
        norm_term = (transformer_out * grad_x_hat).sum(dim=-1, keepdim=True) / (norm + self.eps_)**2
        grad_transformer_out = (grad_x_hat - transformer_out * norm_term / D) / (norm + self.eps_)  # [N, D]
        return grad_transformer_out
    
    def _post_layer_backward(self,
                            logit_grad: torch.Tensor,
                            layer_weight) -> torch.Tensor:
        lm_W     = layer_weight.lm_head_weight_.float()      # (V, D)
        norm_W   = layer_weight.final_norm_weight_.float()   # (D,)
        x = self.mem_manager.get_finetune_activations(layer_id=-1).float()                  # (N, D)
        g_y = logit_grad @ lm_W            # (N, D)
        D = x.shape[-1]
        r = x.norm(p=2, dim=-1, keepdim=True) / math.sqrt(D)   # (N, 1)
        g_x_hat = g_y * norm_W               # (N, D)
        dot     = (g_x_hat * x).sum(dim=-1, keepdim=True)      # (N, 1)
        grad_x = (g_x_hat - x * dot / (D * (r**2 + self.eps_))) / (r + self.eps_)
        return grad_x
    
    def _lora_context_backward(self, layer_id, base_model, output_grad: torch.Tensor, 
                                finetuning_adapter: LoraTpPartAdapter,
                                batch_seq_lens: torch.Tensor):
        # backprop for a transformer layer, start from ffn
        layer_weight = base_model.trans_layers_weight[layer_id]
        ffn_input = self.mem_manager.get_ffn_input(layer_id).float()      # shape [N, D]
        #print(f"Saved activation: ffn_input norm: {ffn_input.norm().item()}")
        grad_ffn_input = self._backprop_ffn(ffn_input, output_grad, layer_weight)
        # rprint("grad_ffn_input grad norm: {:.6f}".format(grad_ffn_input.norm().item()))
        if layer_id == 0:
            last_layer_input = self.mem_manager.get_input_layer_output().float()    # shape [N, D]
        else:
            last_layer_input = self.mem_manager.get_finetune_activations(layer_id-1).float()    # shape [N, D]
        #print(f"Saved activation: last_layer_input norm: {last_layer_input.norm().item()}")
        lora_weights = finetuning_adapter.layers[layer_id]
        # Backprop through LoRA
        grad_attn_input, grad_w_combined = self._backpop_attention_4(base_model, last_layer_input, grad_ffn_input, lora_weights, layer_weight, layer_id, finetuning_adapter.scaling, batch_seq_lens)
        return grad_attn_input
    
    def _backpop_attention_4(self, base_model,
                       last_layer_input: torch.Tensor, 
                       grad_ffn_input: torch.Tensor, 
                       lora_weight: LoraLayerWeight, 
                       layer_weight: LlamaTransformerLayerWeight,
                       layer_id: int,
                       scaling: any,
                       batch_seq_lens: torch.Tensor):

        compare_layer_id = 31
        base_layer_infer = base_model.layers_infer[layer_id]
        position_ids = torch.cat([
            torch.arange(0, batch_seq_lens[i], device=batch_seq_lens.device)
            for i in range(len(batch_seq_lens))
        ])
        
        position_cos = base_model._cos_cached.index_select(0, position_ids)  # [T, D/2]
        position_sin = base_model._sin_cached.index_select(0, position_ids)  # [T, D/2]
        # === Cast all weights to float32 ===
        w_q = layer_weight.q_weight_.float()
        w_k = layer_weight.k_weight_.float()
        w_v = layer_weight.v_weight_.float()
        w_o = layer_weight.o_weight_.float()
        w_attn_norm = layer_weight.att_norm_weight_.float()
        
        # RMSNorm
        last_layer_input_leaf = last_layer_input.float().detach().clone().requires_grad_()
        x_norm = rmsnorm_forward(last_layer_input_leaf, w_attn_norm, eps=self.eps_)
        if lora_weight.w_combined is None:
            print("###### LoRA weight is None, using LoRA weight from home")
            w_combined = lora_weight.w_combined_home.to("cuda").float() 
        else:
            w_combined = lora_weight.w_combined.float()  # [2, 4r, H, Hd]
        r = w_combined.shape[1] // 4
        H, Hd = w_combined.shape[2], w_combined.shape[3]

        w_combined_leaf = w_combined.detach().clone().requires_grad_()
        qA = w_combined_leaf[0, 0 : r].reshape(r, -1).T
        qB = w_combined_leaf[1, 0 : r].reshape(-1, r).T
        kA = w_combined_leaf[0, r : 2 * r].reshape(r, -1).T
        kB = w_combined_leaf[1, r : 2 * r].reshape(-1, r).T
        vA = w_combined_leaf[0, 2 * r : 3 * r].reshape(r, -1).T
        vB = w_combined_leaf[1, 2 * r : 3 * r].reshape(-1, r).T
        oA = w_combined_leaf[0, 3 * r : 4 * r].reshape(r, -1).T  # [D, r]
        oB = w_combined_leaf[1, 3 * r : 4 * r].reshape(-1, r).T  # [r, D]
        x_norm_leaf = x_norm.detach().clone().requires_grad_()

        def proj_lora(x, A, B):
            output = torch.mm(x, A)
            output = torch.mm(output, B).mul_(scaling)
            return output

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

        q_base = torch.mm(x_norm_leaf.view(-1, base_layer_infer.embed_dim_), w_q)
        k_base = torch.mm(x_norm_leaf.view(-1, base_layer_infer.embed_dim_), w_k)
        v_base = torch.mm(x_norm_leaf.view(-1, base_layer_infer.embed_dim_), w_v)

        q_  = q_base + proj_lora(x_norm_leaf, qA, qB)
        k_  = k_base + proj_lora(x_norm_leaf, kA, kB)
        v_  = v_base + proj_lora(x_norm_leaf, vA, vB)
        rotary_emb_fwd_pt(q_.view(-1, H, Hd), position_cos, position_sin)
        rotary_emb_fwd_pt(k_.view(-1, H, Hd), position_cos, position_sin)
        if layer_id == compare_layer_id:
            self.report_diff_percent("Recomputed q", q_, self.mem_manager.saved_q)
            self.report_diff_percent("Recomputed k", k_, self.mem_manager.saved_k)
            self.report_diff_percent("Recomputed v", v_, self.mem_manager.saved_v)

        S = x_norm.size(0) 
        D = x_norm.size(1)  
        qh, kh, vh = q_.view(S, H, Hd), k_.view(S, H, Hd), v_.view(S, H, Hd)
        ctx = torch.empty_like(qh)
        
        B = batch_seq_lens.shape[0]
        scale = 1.0 / (Hd ** 0.5)

        b_start_loc = torch.cat([torch.tensor([0], device=batch_seq_lens.device), batch_seq_lens.cumsum(dim=0)[:-1]])
        for i in range(B):
            st, ln = b_start_loc[i], batch_seq_lens[i]
            q_blk  = qh[st:st+ln].transpose(0, 1)          # [H,L,D]
            k_blk  = kh[st:st+ln].transpose(0, 1)
            v_blk  = vh[st:st+ln].transpose(0, 1)

            att = (q_blk @ k_blk.transpose(-1, -2)) * scale
            att.masked_fill_(torch.triu(torch.ones_like(att), 1).bool(), float('-inf'))
            att = torch.softmax(att, dim=-1)
            ctx_blk = (att @ v_blk).transpose(0, 1)        # [L,H,D]
            ctx[st:st+ln] = ctx_blk

        ctx_flat = ctx.reshape(S, D)

        o_base_ = torch.mm(ctx_flat, w_o)
        o_lora_ = proj_lora(ctx_flat, oA, oB)
        o_total = o_base_ + o_lora_
        if layer_id == compare_layer_id:
            self.report_diff_percent("Recomputed O", o_total, self.mem_manager.saved_o)

        input_embs = last_layer_input_leaf + o_total.view(-1, base_layer_infer.embed_dim_)

        #input_embs.add_(o_total.view(-1, base_layer_infer.embed_dim_))
        ffn_input = self.mem_manager.get_ffn_input(layer_id).float() 
        if layer_id == compare_layer_id:
            self.report_diff_percent("Recomputed ffn input", input_embs, ffn_input)

        grad_o = grad_ffn_input.float()
        input_embs.backward(grad_o)
        
        g = w_combined_leaf.grad
        max_norm = 1
        if g is not None:
            grad_norm = g.norm()                           # ‖g‖₂   (scalar tensor)
            if grad_norm > max_norm:                       # scale *in-place*
                g.mul_(max_norm / (grad_norm + 1e-6))

            # now copy into the fp32 master copy
        lora_weight.w_combined_home_fp32.grad = g.to(device=lora_weight.w_combined_home_fp32.device)
        return last_layer_input_leaf.grad, w_combined_leaf.grad

    def _backpop_attention_3(self, base_model,
                       last_layer_input: torch.Tensor, 
                       grad_ffn_input: torch.Tensor, 
                       lora_weight: LoraLayerWeight, 
                       layer_weight: LlamaTransformerLayerWeight,
                       layer_id: int,
                       scaling: any,
                       batch_seq_lens: torch.Tensor):

        compare_layer_id = 31
        base_layer_infer = base_model.layers_infer[layer_id]
        position_ids = torch.cat([
            torch.arange(0, batch_seq_lens[i], device=batch_seq_lens.device)
            for i in range(len(batch_seq_lens))
        ])
        
        position_cos = base_model._cos_cached.index_select(0, position_ids)  # [T, D/2]
        position_sin = base_model._sin_cached.index_select(0, position_ids)  # [T, D/2]
        # === Cast all weights to float32 ===
        w_q = layer_weight.q_weight_.float()
        w_k = layer_weight.k_weight_.float()
        w_v = layer_weight.v_weight_.float()
        w_o = layer_weight.o_weight_.float()
        w_attn_norm = layer_weight.att_norm_weight_.float()

        # RMSNorm
        x_norm = rmsnorm_forward(last_layer_input.float(), w_attn_norm, eps=self.eps_)
        if lora_weight.w_combined is None:
            print("###### LoRA weight is None, using LoRA weight from home")
            w_combined = lora_weight.w_combined_home.to("cuda").float() 
        else:
            w_combined = lora_weight.w_combined.float()  # [2, 4r, H, Hd]
        r = w_combined.shape[1] // 4
        H, Hd = w_combined.shape[2], w_combined.shape[3]

        q_lora_A = w_combined[0, 0 : r].reshape(r, -1).T
        q_lora_B = w_combined[1, 0 : r].reshape(-1, r).T
        k_lora_A = w_combined[0, r : 2 * r].reshape(r, -1).T
        k_lora_B = w_combined[1, r : 2 * r].reshape(-1, r).T
        v_lora_A = w_combined[0, 2 * r : 3 * r].reshape(r, -1).T
        v_lora_B = w_combined[1, 2 * r : 3 * r].reshape(-1, r).T
        oA = w_combined[0, 3 * r : 4 * r].reshape(r, -1).T  # [D, r]
        oB = w_combined[1, 3 * r : 4 * r].reshape(-1, r).T  # [r, D]

        qA = q_lora_A.detach().clone().requires_grad_()
        qB = q_lora_B.detach().clone().requires_grad_()
        kA = k_lora_A.detach().clone().requires_grad_()
        kB = k_lora_B.detach().clone().requires_grad_()
        vA = v_lora_A.detach().clone().requires_grad_()
        vB = v_lora_B.detach().clone().requires_grad_()
        oA = oA.detach().clone().requires_grad_()
        oB = oB.detach().clone().requires_grad_()
        x_norm_leaf = x_norm.detach().clone().requires_grad_()

        def proj_lora(x, A, B):
            output = torch.mm(x, A)
            output = torch.mm(output, B).mul_(scaling)
            return output

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

        q_base = torch.mm(x_norm_leaf.view(-1, base_layer_infer.embed_dim_), w_q)
        k_base = torch.mm(x_norm_leaf.view(-1, base_layer_infer.embed_dim_), w_k)
        v_base = torch.mm(x_norm_leaf.view(-1, base_layer_infer.embed_dim_), w_v)

        q_  = q_base + proj_lora(x_norm_leaf, qA, qB)
        k_  = k_base + proj_lora(x_norm_leaf, kA, kB)
        v_  = v_base + proj_lora(x_norm_leaf, vA, vB)
        rotary_emb_fwd_pt(q_.view(-1, H, Hd), position_cos, position_sin)
        rotary_emb_fwd_pt(k_.view(-1, H, Hd), position_cos, position_sin)
        if layer_id == compare_layer_id:
            self.report_diff_percent("Recomputed q", q_, self.mem_manager.saved_q)
            self.report_diff_percent("Recomputed k", k_, self.mem_manager.saved_k)
            self.report_diff_percent("Recomputed v", v_, self.mem_manager.saved_v)

        S = x_norm.size(0) 
        D = x_norm.size(1)  
        qh, kh, vh = q_.view(S, H, Hd), k_.view(S, H, Hd), v_.view(S, H, Hd)
        ctx = torch.empty_like(qh)
        
        B = batch_seq_lens.shape[0]
        scale = 1.0 / (Hd ** 0.5)

        b_start_loc = torch.cat([torch.tensor([0], device=batch_seq_lens.device), batch_seq_lens.cumsum(dim=0)[:-1]])
        for i in range(B):
            st, ln = b_start_loc[i], batch_seq_lens[i]
            q_blk  = qh[st:st+ln].transpose(0, 1)          # [H,L,D]
            k_blk  = kh[st:st+ln].transpose(0, 1)
            v_blk  = vh[st:st+ln].transpose(0, 1)

            att = (q_blk @ k_blk.transpose(-1, -2)) * scale
            att.masked_fill_(torch.triu(torch.ones_like(att), 1).bool(), float('-inf'))
            att = torch.softmax(att, dim=-1)
            ctx_blk = (att @ v_blk).transpose(0, 1)        # [L,H,D]
            ctx[st:st+ln] = ctx_blk

        ctx_flat = ctx.reshape(S, D)

        o_base_ = torch.mm(ctx_flat, w_o)
        o_lora_ = proj_lora(ctx_flat, oA, oB)
        o_total = o_base_ + o_lora_
        if layer_id == compare_layer_id:
            self.report_diff_percent("Recomputed O", o_total, self.mem_manager.saved_o)

        input_embs = last_layer_input.clone().float()
        input_embs.add_(o_total.view(-1, base_layer_infer.embed_dim_))
        ffn_input = self.mem_manager.get_ffn_input(layer_id).float() 
        if layer_id == compare_layer_id:
            self.report_diff_percent("Recomputed ffn input", input_embs, ffn_input)

        grad_o = grad_ffn_input.float()
        input_embs.backward(grad_o)
        grad_attn_input = x_norm_leaf.grad  
        grad_attn_input = rmsnorm_backward(
            last_layer_input.float(), grad_attn_input, w_attn_norm, eps=self.eps_)
        # only these gradients are needed, skip any unnecessary computation for other gradients

        def fill_combined(dst_slice, grad_mat, is_A):
            if is_A:                                          # [D,r] → [r,H,Hd]
                tmp = grad_mat.T.reshape(r, H, Hd)
            else:                                             # [r,D] → [r,H,Hd]
                tmp = grad_mat.reshape(r, H, Hd)
            dst_slice.copy_(tmp)

        grad_w_combined = torch.zeros_like(w_combined) 
        # --- Q / K / V / O blocks ----------------------------------------
        fill_combined(grad_w_combined[0, 0*r : 1*r], qA.grad, is_A=True)   # Q-A
        fill_combined(grad_w_combined[1, 0*r : 1*r], qB.grad, is_A=False)  # Q-B

        fill_combined(grad_w_combined[0, 1*r : 2*r], kA.grad, is_A=True)   # K-A
        fill_combined(grad_w_combined[1, 1*r : 2*r], kB.grad, is_A=False)  # K-B

        fill_combined(grad_w_combined[0, 2*r : 3*r], vA.grad, is_A=True)   # V-A
        fill_combined(grad_w_combined[1, 2*r : 3*r], vB.grad, is_A=False)  # V-B

        fill_combined(grad_w_combined[0, 3*r : 4*r], oA.grad, is_A=True)   # O-A
        fill_combined(grad_w_combined[1, 3*r : 4*r], oB.grad, is_A=False)  # O-B

        baseline = 1e-6
        norm = grad_attn_input.norm().item()
        if norm < baseline:
            scale = baseline / (norm + 1e-10)
            grad_attn_input *= scale
            #print(f"🔧 Rescaled output_grad by {scale:.2f}")
        return grad_attn_input, grad_w_combined
    
    def _backpop_attention_2(self, base_model,
                       last_layer_input: torch.Tensor, 
                       grad_ffn_input: torch.Tensor, 
                       lora_weight: LoraLayerWeight, 
                       layer_weight: LlamaTransformerLayerWeight,
                       layer_id: int,
                       scaling: any,
                       batch_seq_lens: torch.Tensor):

        base_layer_infer = base_model.layers_infer[layer_id]
        position_ids = torch.cat([
            torch.arange(0, batch_seq_lens[i], device=batch_seq_lens.device)
            for i in range(len(batch_seq_lens))
        ])

        position_cos = torch.index_select(
                base_model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
        position_sin = torch.index_select(
                base_model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
        
        # === Cast all weights to float32 ===
        w_q = layer_weight.q_weight_.float().clone()
        w_k = layer_weight.k_weight_.float().clone()
        w_v = layer_weight.v_weight_.float().clone()
        w_o = layer_weight.o_weight_.float().clone()
        w_attn_norm = layer_weight.att_norm_weight_.float().clone()

        # RMSNorm
        x_norm = rmsnorm_forward(last_layer_input.float().clone(), w_attn_norm, eps=self.eps_)
       
        # Base projections
        q_base = torch.mm(x_norm.view(-1, base_layer_infer.embed_dim_), w_q)
        k_base = torch.mm(x_norm.view(-1, base_layer_infer.embed_dim_), w_k)
        v_base = torch.mm(x_norm.view(-1, base_layer_infer.embed_dim_), w_v)

        # if lora_weight.w_combined is None:
        #     print("###### LoRA weight is None, using LoRA weight from home")
        #     lora_weight.w_combined = lora_weight.w_combined_home.to("cuda")

        if lora_weight.w_combined is None:
            print("###### LoRA weight is None, using LoRA weight from home")
            w_combined = lora_weight.w_combined_home.to("cuda").float() 
        else:
            w_combined = lora_weight.w_combined.float()  # [2, 4r, H, Hd]

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
        rotary_emb_fwd_pt(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_), position_cos, position_sin)
        k = k_base + k_lora
        rotary_emb_fwd_pt(k.view(-1, base_layer_infer.tp_k_head_num_, base_model.head_dim_), position_cos, position_sin)
        v = v_base + v_lora

        if layer_id == 31:
            self.report_diff_percent("Recomputed q", q, self.mem_manager.saved_q)
            self.report_diff_percent("Recomputed k", k, self.mem_manager.saved_k)
            self.report_diff_percent("Recomputed v", v, self.mem_manager.saved_v)

        S, D = q.shape
        H = base_layer_infer.tp_q_head_num_
        Hd = D // H
        q = q.view(S, H, Hd)
        k = k.view(S, H, Hd)
        v = v.view(S, H, Hd)
        attn_out = torch.empty_like(q)

        B = batch_seq_lens.shape[0]
        scale = 1.0 / (Hd ** 0.5)

        b_start_loc = torch.cat([torch.tensor([0], device=batch_seq_lens.device), batch_seq_lens.cumsum(dim=0)[:-1]])
        for i in range(B):
            start = b_start_loc[i].item()
            seqlen = batch_seq_lens[i].item()
            end = start + seqlen

            q_i = q[start:end].transpose(0, 1)  # [H, L, D]
            k_i = k[start:end].transpose(0, 1)
            v_i = v[start:end].transpose(0, 1)

            scores = torch.matmul(q_i, k_i.transpose(-1, -2)) * scale
            mask = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool, device=q.device))
            scores = scores.masked_fill(~mask, float('-inf'))
            probs = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, v_i).transpose(0, 1).contiguous()  # [L, H, D]
            attn_out[start:end] = ctx

        attn_out = attn_out.view(S, D)
        # Output projection (base + LoRA)
        o_base = torch.mm(attn_out.view(-1, base_layer_infer.embed_dim_), w_o)
        oA = w_combined[0, 3 * r : 4 * r].reshape(r, -1).T  # [D, r]
        oB = w_combined[1, 3 * r : 4 * r].reshape(-1, r).T  # [r, D]
        o_lora = (attn_out @ oA) @ oB * scaling
        o = o_base + o_lora
        if layer_id == 31:
            self.report_diff_percent("Recomputed O", o, self.mem_manager.saved_o)

        g_o = grad_ffn_input.float()
        N, D = attn_out.shape
        g_o = grad_ffn_input.float()                              # [N, D]
        # ---- LoRA-O part -----------------------------------------
        U     = attn_out @ oA               # [N, r]
        grad_oB = (U.T @ g_o) * scaling     # dL/d oB      [r, D]
        grad_U  =  g_o @ oB.T * scaling     # [N, r]
        grad_oA = attn_out.T @ grad_U       # dL/d oA      [D, r]

        # ---- base W_O path ---------------------------------------
        grad_ctx = g_o @ w_o.T              # dL/d attn_out  [N, D]
        grad_ctx += grad_U @ oA.T           # add LoRA pathway

        # ===== back-prop through (masked) self-attention ==========

        # reshape helpers
        H    = base_layer_infer.tp_q_head_num_
        Hd   = D // H
        q    = q.view(N, H, Hd)
        k    = k.view(N, H, Hd)
        v    = v.view(N, H, Hd)
        grad_ctx = grad_ctx.view(N, H, Hd)

        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)

        # per-request slices
        start_locs = torch.cat([torch.tensor([0], device=batch_seq_lens.device),
                                batch_seq_lens.cumsum(0)[:-1]])

        scale_att = 1.0 / (Hd ** 0.5)
        for start, seqlen in zip(start_locs.tolist(), batch_seq_lens.tolist()):
            end = start + seqlen
            # blocks: (H, L, Hd)
            q_blk = q[start:end].transpose(0, 1)
            k_blk = k[start:end].transpose(0, 1)
            v_blk = v[start:end].transpose(0, 1)
            g_blk = grad_ctx[start:end].transpose(0, 1)   # dL/d context

            # forward attn pieces
            att   = (q_blk @ k_blk.transpose(-1, -2)) * scale_att
            att.masked_fill_(torch.triu(torch.ones_like(att), 1).bool(), float('-inf'))
            P     = torch.softmax(att, dim=-1)

            # ---- back-prop ----
            # dL/d V
            grad_v_blk = P.transpose(-1, -2) @ g_blk
            # dL/d P
            grad_P = g_blk @ v_blk.transpose(-1, -2)

            # softmax backward (row-wise)
            row_sum = (grad_P * P).sum(-1, keepdim=True)
            grad_att = (grad_P - row_sum) * P * scale_att

            # dL/d Q, K
            grad_q_blk = grad_att @ k_blk
            grad_k_blk = grad_att.transpose(-1, -2) @ q_blk

            # accumulate
            grad_q[start:end] += grad_q_blk.transpose(0, 1)
            grad_k[start:end] += grad_k_blk.transpose(0, 1)
            grad_v[start:end] += grad_v_blk.transpose(0, 1)

        # ===== undo rotary for Q & K =========================================
        def rotary_backward(g, cos, sin):
            T, Hh, Dd = g.shape
            Dh = Dd // 2
            g = g.reshape(T * Hh, Dd)
            ge, go = g[:, :Dh], g[:, Dh:]
            cos_e = cos[:, None, :].expand(T, Hh, Dh).reshape(T * Hh, Dh)
            sin_e = sin[:, None, :].expand(T, Hh, Dh).reshape(T * Hh, Dh)
            # inverse rotation
            ge_new =  ge * cos_e + go * sin_e
            go_new = -ge * sin_e + go * cos_e
            g[:, :Dh] = ge_new
            g[:, Dh:] = go_new
            return g.view(T, Hh, Dd)

        grad_q = rotary_backward(grad_q, position_cos, position_sin)
        grad_k = rotary_backward(grad_k, position_cos, position_sin)

        # ===== gradients w.r.t. LoRA Q/K/V and input =========================
        def lora_backward(grad_out, A, B):
            # Y = ((X·A)·B)·s
            grad_B = ( (x_norm @ A).T @ grad_out) * scaling      # [r, D]
            grad_A = x_norm.T @ (grad_out @ B.T * scaling)       # [D, r]
            grad_X = (grad_out @ B.T) @ A.T * scaling            # [N, D]
            return grad_A, grad_B, grad_X

        grad_qA, grad_qB, gx1 = lora_backward(grad_q.view(N, D), q_lora_A, q_lora_B)
        grad_kA, grad_kB, gx2 = lora_backward(grad_k.view(N, D), k_lora_A, k_lora_B)
        grad_vA, grad_vB, gx3 = lora_backward(grad_v.view(N, D), v_lora_A, v_lora_B)

        # base-path gradients to input
        gx_base  = grad_q.view(N, D) @ w_q.T
        gx_base += grad_k.view(N, D) @ w_k.T
        gx_base += grad_v.view(N, D) @ w_v.T

        grad_input_total = gx1 + gx2 + gx3 + gx_base            # [N, D]

        # ===== back through RMSNorm to residual input ========================
        grad_attn_input = rmsnorm_backward(
                last_layer_input.float(), grad_input_total, w_attn_norm, eps=self.eps_)

        # ===== pack gradients for LoRA matrices ==============================
        grads = {
            f"layer_{layer_id}.q_lora_A_home": grad_qA,  # [D, r]
            f"layer_{layer_id}.q_lora_B_home": grad_qB,  # [r, D]
            f"layer_{layer_id}.k_lora_A_home": grad_kA,
            f"layer_{layer_id}.k_lora_B_home": grad_kB,
            f"layer_{layer_id}.v_lora_A_home": grad_vA,
            f"layer_{layer_id}.v_lora_B_home": grad_vB,
            f"layer_{layer_id}.o_lora_A_home": grad_oA,  # [D, r]
            f"layer_{layer_id}.o_lora_B_home": grad_oB,  # [r, D]
        }

        def fill_combined(dst_slice, grad_mat, is_A):
            if is_A:                                          # [D,r] → [r,H,Hd]
                tmp = grad_mat.T.reshape(r, H, Hd)
            else:                                             # [r,D] → [r,H,Hd]
                tmp = grad_mat.reshape(r, H, Hd)
            dst_slice.copy_(tmp)

        grad_w_combined = torch.zeros_like(w_combined) 
        # --- Q / K / V / O blocks ----------------------------------------
        fill_combined(grad_w_combined[0, 0*r : 1*r], grad_qA, is_A=True)   # Q-A
        fill_combined(grad_w_combined[1, 0*r : 1*r], grad_qB, is_A=False)  # Q-B

        fill_combined(grad_w_combined[0, 1*r : 2*r], grad_kA, is_A=True)   # K-A
        fill_combined(grad_w_combined[1, 1*r : 2*r], grad_kB, is_A=False)  # K-B

        fill_combined(grad_w_combined[0, 2*r : 3*r], grad_vA, is_A=True)   # V-A
        fill_combined(grad_w_combined[1, 2*r : 3*r], grad_vB, is_A=False)  # V-B

        fill_combined(grad_w_combined[0, 3*r : 4*r], grad_oA, is_A=True)   # O-A
        fill_combined(grad_w_combined[1, 3*r : 4*r], grad_oB, is_A=False)  # O-B

        return grad_attn_input, grad_w_combined


    def _backpop_attention(self, 
                       last_layer_input: torch.Tensor, 
                       grad_ffn_input: torch.Tensor, 
                       lora_weight: LoraLayerWeight, 
                       layer_weight: LlamaTransformerLayerWeight,
                       layer_id: int,
                       scaling: any, 
                       batch_seq_lens: list):

        # === Cast all weights to float32 ===
        lw = layer_weight  # alias
        lw_q = lw.q_weight_.float()
        lw_k = lw.k_weight_.float()
        lw_v = lw.v_weight_.float()
        lw_o = lw.o_weight_.float()
        lw_att_norm = lw.att_norm_weight_.float()

        lora = lora_weight  # alias
        lora_qA = lora.q_lora_A.float()
        lora_qB = lora.q_lora_B.float()
        lora_kA = lora.k_lora_A.float()
        lora_kB = lora.k_lora_B.float()
        lora_vA = lora.v_lora_A.float()
        lora_vB = lora.v_lora_B.float()
        lora_oA = lora.o_lora_A.float()
        lora_oB = lora.o_lora_B.float()

        x = rmsnorm_forward(last_layer_input.float(), lw_att_norm, eps=self.eps_)

        # === Forward recomputation ===
        q_base = x @ lw_q.T
        k_base = x @ lw_k.T
        v_base = x @ lw_v.T

        q_lora_B_out = x @ lora_qB.T
        k_lora_B_out = x @ lora_kB.T
        v_lora_B_out = x @ lora_vB.T

        q = q_base + scaling * (q_lora_B_out @ lora_qA.T)
        k = k_base + scaling * (k_lora_B_out @ lora_kA.T)
        v = v_base + scaling * (v_lora_B_out @ lora_vA.T)

        d_k = q.shape[-1]
        attn_scores = q @ k.T / d_k**0.5
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = attn_weights @ v  # [N, D]

        # === Output projection (base + LoRA) ===
        o_base = attn_out @ lw_o.T
        o_lora_B_out = attn_out @ lora_oB.T
        o_lora = scaling * (o_lora_B_out @ lora_oA.T)
        o = o_base + o_lora

        # === Backprop ===
        grad_o = grad_ffn_input.float()

        grad_attn_out_base = grad_o @ lw_o
        grad_o_lora_A = grad_o.T @ o_lora_B_out
        grad_o_lora_B = (lora_oA.T @ grad_o.T) @ attn_out
        grad_attn_out_lora = scaling * ((grad_o @ lora_oA) @ lora_oB)
        grad_attn_out = grad_attn_out_base + grad_attn_out_lora

        grad_attn_weights = grad_attn_out @ v.T
        grad_v = attn_weights.T @ grad_attn_out

        d_attn = grad_attn_weights * attn_weights
        d_attn_sum = d_attn.sum(dim=-1, keepdim=True)
        grad_attn_scores = (d_attn - attn_weights * d_attn_sum) / d_k**0.5

        grad_q = grad_attn_scores @ k
        grad_k = grad_attn_scores.T @ q

        grad_q_lora_A = scaling * (grad_q.T @ q_lora_B_out)
        grad_q_lora_B = scaling * ((lora_qA.T @ grad_q.T) @ x)

        grad_k_lora_A = scaling * (grad_k.T @ k_lora_B_out)
        grad_k_lora_B = scaling * ((lora_kA.T @ grad_k.T) @ x)

        grad_v_lora_A = scaling * (grad_v.T @ v_lora_B_out)
        grad_v_lora_B = scaling * ((lora_vA.T @ grad_v.T) @ x)

        grad_o_lora_A = scaling * (grad_o.T @ o_lora_B_out)
        grad_o_lora_B = scaling * ((lora_oA.T @ grad_o.T) @ attn_out)

        grad_q_in = scaling * ((grad_q @ lora_qA) @ lora_qB)
        grad_k_in = scaling * ((grad_k @ lora_kA) @ lora_kB)
        grad_v_in = scaling * ((grad_v @ lora_vA) @ lora_vB)

        grad_q_base = grad_q @ lw_q
        grad_k_base = grad_k @ lw_k
        grad_v_base = grad_v @ lw_v

        grad_input_total = grad_q_base + grad_k_base + grad_v_base + grad_q_in + grad_k_in + grad_v_in + grad_o

        grad_attn_input = rmsnorm_backward(
            last_layer_input.float(), grad_input_total, lw_att_norm, eps=self.eps_)

        grads = {
            f"layer_{layer_id}.q_lora_A_home": grad_q_lora_A,
            f"layer_{layer_id}.q_lora_B_home": grad_q_lora_B,
            f"layer_{layer_id}.k_lora_A_home": grad_k_lora_A,
            f"layer_{layer_id}.k_lora_B_home": grad_k_lora_B,
            f"layer_{layer_id}.v_lora_A_home": grad_v_lora_A,
            f"layer_{layer_id}.v_lora_B_home": grad_v_lora_B,
            f"layer_{layer_id}.o_lora_A_home": grad_o_lora_A,
            f"layer_{layer_id}.o_lora_B_home": grad_o_lora_B,
        }

        baseline = 1e-6
        norm = grad_attn_input.norm().item()
        if norm < baseline:
            scale = baseline / (norm + 1e-10)
            grad_attn_input *= scale
            #print(f"🔧 Rescaled output_grad by {scale:.2f}")

        return grad_attn_input, grads

    def _backprop_ffn_2(self, ffn_input: torch.Tensor, output_grad: torch.Tensor, layer_weight):
        # Recompute RMSNorm
        input1 = rmsnorm_forward(ffn_input.float(), layer_weight.ffn_norm_weight_.float(), eps=self.eps_)
        # Forward again through FFN (no need to save intermediates)
        gate_in = input1 @ layer_weight.gate_proj.float()  
        gate_out = torch.nn.functional.silu(gate_in)
        up_out = input1 @ layer_weight.up_proj.float() 
        ffn1_out = gate_out * up_out
        # Backprop through down_proj
        grad_ffn1_out = output_grad @ layer_weight.down_proj.float().T
        # Backprop through elementwise mul
        grad_gate_out = grad_ffn1_out * up_out
        grad_up_out = grad_ffn1_out * gate_out
        # Backprop through up and gate proj
        grad_input_up = grad_up_out @ layer_weight.up_proj.float() .T
        silu_grad = torch.sigmoid(gate_in) * (1 + gate_in * (1 - torch.sigmoid(gate_in)))
        grad_gate_in = grad_gate_out * silu_grad
        grad_input_gate = grad_gate_in @ layer_weight.gate_proj.float() .T
        grad_input1 = grad_input_gate + grad_input_up + output_grad  # [N, D]
        # Backprop through RMSNorm
        grad_ffn_input = rmsnorm_backward(ffn_input, grad_input1, layer_weight.ffn_norm_weight_.float(), eps=self.eps_)
        return grad_ffn_input

    def _backprop_ffn(
            self,
            ffn_input: torch.Tensor,          # x  (dtype can be fp16/bf16/fp32)
            output_grad: torch.Tensor,        # ∂L/∂y  with y = x + FFN(...)
            layer_weight,
    ):
        eps = self.eps_
        # ---- 1. forward re-compute (in float32 for stability) -------------
        w_rms = layer_weight.ffn_norm_weight_.float()
        w_gate = layer_weight.gate_proj.float()
        w_up   = layer_weight.up_proj.float()
        w_down = layer_weight.down_proj.float()

        x      = ffn_input.float()            # (N, D)
        x_norm = rmsnorm_forward(x, w_rms, eps=eps)          # (N, D)

        gate_in  = x_norm @ w_gate            # (N, M)
        gate_out = torch.nn.functional.silu(gate_in)

        up_out   = x_norm @ w_up              # (N, M)
        ffn_mid  = gate_out * up_out          # (N, M)
        grad_ffn_mid = output_grad.float() @ w_down.t()      # (N, M)
        grad_gate_out = grad_ffn_mid * up_out                # (N, M)
        grad_up_out   = grad_ffn_mid * gate_out
        grad_x_norm_up   = grad_up_out   @ w_up.t()          # (N, D)
        sig      = torch.sigmoid(gate_in)
        silu_grad = sig * (1 + gate_in * (1 - sig))          # d SiLU / d gate_in
        grad_gate_in     = grad_gate_out * silu_grad         # (N, M)
        grad_x_norm_gate = grad_gate_in @ w_gate.t()         # (N, D)
        grad_x_norm = grad_x_norm_up + grad_x_norm_gate      # (N, D)
        grad_from_norm = rmsnorm_backward(
            x,                     # original pre-norm input
            grad_x_norm,           # gradient w.r.t x_norm
            w_rms,
            eps=eps,
        )                          # (N, D)
        grad_ffn_input = grad_from_norm + output_grad.float()
        return grad_ffn_input
    
    @torch.no_grad()
    def report_diff_percent(
        self,
        name: str,
        ours: torch.Tensor,
        slora: torch.Tensor,
        eps: float = 1e-4,       # what “near zero” means for the reference
        thresh: float = 1e-2     # 1% threshold for “bad” elements
    ):
        if ours.shape != slora.shape:
            return

        diff     = ours - slora
        abs_diff = diff.abs()
        ref_abs  = slora.abs()

        # L2 relative error
        rel_l2   = diff.norm() / (slora.norm() + eps)

        # mean and max absolute error
        mean_abs = abs_diff.detach().mean().item()
        max_abs  = abs_diff.max().item()

        # fraction of “bad” elements (relative abs error > thresh)
        bad      = abs_diff > (thresh * (ref_abs + eps))
        frac_bad = float(bad.sum()) / bad.numel() * 100.0

        # fraction of reference elements that are essentially zero
        small    = ref_abs <= eps
        frac_small = float(small.sum()) / small.numel() * 100.0
        print()
        print(f"[{name}] shape={list(ours.shape)}, dtype={ours.dtype}, L2-rel err: {rel_l2*100:6.2f}%")

    