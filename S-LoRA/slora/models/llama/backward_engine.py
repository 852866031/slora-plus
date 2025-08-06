# -----------------------------------------------------------------------------
# LlamaBackwardEngine – Back‑propagation support utilities for S‑LoRA
# -----------------------------------------------------------------------------
# This module contains everything required to compute analytical gradients for a
# LoRA‑augmented Llama model **without** replaying the entire forward graph.

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
    """Which *checkpoint* inside the backward pipeline we last finished.

    This makes the backward pass resumeable, which is crucial when the engine is
    interrupted by higher‑priority inference work.  In practice `interrupt_flag`
    is set by the outer scheduler; on resume we skip the already computed part
    and continue from the stored tensors.
    """
    LOSS = 0
    LOGIT = 1
    POST = 2
    LAYER = 3


class LlamaBackwardEngine():
    def __init__(self, mem_manager, network_config):
        # Runtime references / constants
        self.mem_manager = mem_manager
        self.eps_ = network_config["rms_norm_eps"]
        self.vocab_size_ = network_config["vocab_size"]
        self.embed_dim_ = network_config["n_embed"]
        # *Mutable* state – these fields let us resume through backward.
        self.last_loss = None
        self.gradient_resume_point = GradientResumePoint.LOSS
        self.saved_logit_grad = None
        self.saved_grad_transformer_out = None
        self.resume_after_layer = -1
        self.saved_loss = None
        self.saved_total_tokens_to_process = None
        self.saved_seq_lens = None
        # Alignment training hyper‑params 
        self.is_alignment = False
        self.alpha:   float = 0.5   # curvature of utility
        self.lambdas: float = 2      # loss-aversion (λ)
        self.beta:    float = 0.02
        self.buffer = []
    
    def setup_alignment(self, is_alignment: bool, 
                        alpha: float = 0.5, 
                        lambdas: float = 2, 
                        beta: float = 0.02):
        self.is_alignment = is_alignment
        self.alpha = alpha
        self.lambdas = lambdas
        self.beta = beta
        print(f"\033[92mAlignment mode set to {self.is_alignment}, alpha: {self.alpha}, lambdas: {self.lambdas}, beta: {self.beta}\033[0m")
        
    # Top‑level backward orchestration entry‑point 
    def _context_backward(self, base_model, finetuning_adapter, interrupt_flag):
        """Resumable backward function.

        Parameters
        ----------
        base_model : `Llama`
            The frozen *backbone* model (weights & per-layer meta).
        finetuning_adapter : `LoraTpPartAdapter`
            The LoRA adapter we are currently fine-tuning.
        interrupt_flag : `list[bool]` (size-1)
            Mutable flag set by the scheduler **during** backward to tell us to
            pause ASAP. 
        """
        # Helper: execute a *slice* of layers [0 .. start_idx‑1] in reverse order
        def run_layer_backward(start_idx, grad_transformer_out):
            for i in reversed(range(0, start_idx)):
                grad_transformer_out = self._lora_context_backward(i, base_model, grad_transformer_out, finetuning_adapter, self.saved_seq_lens)
                if i == start_idx-1:
                    print(f"Backwarding layer {i}", end="")
                elif i == 0:
                    print(", layer{i}")
                else:
                    print(", layer{i}", end="")
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
    
    # Full backward (no resume state)
    def _context_backward_logic(self, base_model, finetuning_adapter, interrupt_flag):
        if self.is_alignment:
            logit_grad, kto_loss, batch_completion_loss, total_tokens_to_process, batch_seq_lens = self.get_grad_logits_alignment()
            loss = (kto_loss, batch_completion_loss)
        else:
            logits_and_targets, total_tokens_to_process, batch_seq_lens = self.get_logits_and_targets()
            logit_grad = self._logit_backward(logits_and_targets)
            loss = self.compute_total_loss(logits_and_targets)
            print(f"\033[92mBackward Total Tokens: {total_tokens_to_process}, Loss: {loss:.12f}\033[0m")

        self.saved_seq_lens = batch_seq_lens
        self.saved_loss = loss
        self.saved_total_tokens_to_process = total_tokens_to_process
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
            if i == base_model.layers_num-1:
                    print(f"Backwarding layer {i}", end="")
            elif i == 0:
                print(f", layer{i}")
            else:
                print(f", layer{i}", end="")
            if interrupt_flag[0]: 
                print(f"\033[91mReceive interrupt after layer {i}\033[0m")
                interrupt_flag[0]= False
                self.saved_grad_transformer_out = grad_transformer_out
                self.resume_after_layer = i
                self.gradient_resume_point = GradientResumePoint.LAYER
                return False, loss, total_tokens_to_process
        self.gradient_resume_point = GradientResumePoint.LOSS
        return True, loss, total_tokens_to_process

    # Alignment‑specific logit gradient
    @torch.no_grad()
    def get_grad_logits_alignment(self): 
        alpha = self.alpha
        lambdas = self.lambdas
        beta = self.beta
        pol_logits_list = self.mem_manager.finetune_logits_per_request
        ref_logits_list = self.mem_manager.reference_logits_per_request
        masks_list      = self.mem_manager.alignment_completion_masks
        labels          = self.mem_manager.alignment_labels
        input_ids       = self.mem_manager.get_concatenated_finetune_input_ids()

        device = pol_logits_list[0].device
        B      = len(pol_logits_list)

        token_counts = [lg.shape[0] + 1 for lg in pol_logits_list]
        ids_split    = torch.split(input_ids, token_counts)

        loss_items, grad_chunks, completion_nlls = [], [], []
        delta_stack, kl_stack = [], []

        for logits_pol, logits_ref, mask, ids, lbl in zip(
                pol_logits_list, ref_logits_list, masks_list, ids_split, labels):

            tgt    = ids[1:].long()
            mask_f = mask.float()
            n_tok  = mask_f.sum().clamp(min=1.0)

            # -------- log-probs --------------------------------------------------
            p_lp = torch.log_softmax(logits_pol.float(), -1)
            r_lp = torch.log_softmax(logits_ref.float(), -1)

            lp_pol = (mask_f * p_lp.gather(1, tgt[:, None]).squeeze(1)).sum() / n_tok
            lp_ref = (mask_f * r_lp.gather(1, tgt[:, None]).squeeze(1)).sum() / n_tok
            delta  = lp_pol - lp_ref                                   # scalar

            # -------- collect Δ and KL for stats -------------------------------
            delta_stack.append(delta)
            kl_tok = (p_lp.exp() * (p_lp - r_lp)).sum(-1)             # [Ti-1]
            kl_stack.append((kl_tok * mask_f).sum() / n_tok)

            # -------- prospect utility (softplus) ------------------------------
            if lbl == 1:          # good
                u  = torch.nn.functional.softplus(alpha * delta) - torch.log(torch.tensor(2.0))
                du = alpha * torch.sigmoid(alpha * delta)
            else:                 # bad
                u  = lambdas * (torch.nn.functional.softplus(-alpha * delta) -
                                torch.log(torch.tensor(2.0)))
                du = -lambdas * alpha * torch.sigmoid(-alpha * delta)

            loss_i = -u + beta * delta
            loss_items.append(loss_i)
            dloss_dDelta = -du + beta

            # -------- gradient wrt logits --------------------------------------
            probs = torch.softmax(logits_pol.float(), -1)
            grad  = probs.clone()
            grad[torch.arange(grad.size(0)), tgt] -= 1
            grad *= mask_f[:, None]
            grad *= dloss_dDelta / B
            grad_chunks.append(grad)

            # -------- completion CE (monitor) ----------------------------------
            tok_lp_pol = p_lp.gather(1, tgt[:, None]).squeeze(1)
            completion_nlls.append(-(tok_lp_pol * mask_f).sum() / n_tok)

        # ---------- aggregate ---------------------------------------------------
        grad_logits_concat = torch.cat(grad_chunks, dim=0)
        total_loss         = torch.stack(loss_items).mean().cpu()
        batch_comp_loss    = torch.stack(completion_nlls).mean().cpu()

        delta_tensor = torch.stack(delta_stack)
        kl_tensor    = torch.stack(kl_stack)        
        self.buffer.append([total_loss, delta_tensor, kl_tensor, grad_logits_concat, batch_comp_loss])
        print(
            f"KTO Loss: {total_loss:.6f} | "
            f"Δ mean/std/max: {delta_tensor.mean():+.3f} / "
            f"{delta_tensor.std():.3f} / {delta_tensor.max():+.3f} | "
            f"KL mean: {kl_tensor.mean():.3f} | "
            f"Grad-norm: {grad_logits_concat.norm():.3f} | "
            f"Completion Loss: {batch_comp_loss:.3f}")
        total_tokens_to_process = input_ids.shape[0]
        batch_seq_lens = torch.tensor(
            [lg.shape[0] for lg in pol_logits_list],
            dtype=torch.long, device=device
        )

        return grad_logits_concat, total_loss, batch_comp_loss, total_tokens_to_process, batch_seq_lens
    
    def print_reset_log(self):
        Batch = 1
        for total_loss, delta_tensor, kl_tensor, grad_logits_concat, batch_comp_loss in self.buffer:
            print(
            f"\033[92m Batch {Batch} | "
            f"KTO Loss: {total_loss:.6f} | "
            f"Δ mean/std/max: {delta_tensor.mean():+.3f} / "
            f"{delta_tensor.std():.3f} / {delta_tensor.max():+.3f} | "
            f"KL mean: {kl_tensor.mean():.3f} | "
            f"Grad-norm: {grad_logits_concat.norm():.3f} | "
            f"Completion Loss: {batch_comp_loss:.3f}"
            f"\033[0m")
            Batch += 1
        self.buffer = []

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

    # For SFT, compute the total loss of the batch
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
    
    # Backprop the gradient to the logit 
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

    
    # Backprop the gradient to the output layer
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
    
    # Backprop through LoRA augmented transformer layer
    def _lora_context_backward(self, layer_id, base_model, output_grad: torch.Tensor, 
                                finetuning_adapter: LoraTpPartAdapter,
                                batch_seq_lens: torch.Tensor):
        # backprop for a transformer layer, start from ffn
        layer_weight = base_model.trans_layers_weight[layer_id]
        ffn_input = self.mem_manager.get_ffn_input(layer_id).float()      # shape [N, D]
        grad_ffn_input = self._backprop_ffn(ffn_input, output_grad, layer_weight)
        if layer_id == 0:
            last_layer_input = self.mem_manager.get_input_layer_output().float()    # shape [N, D]
        else:
            last_layer_input = self.mem_manager.get_finetune_activations(layer_id-1).float()    # shape [N, D]
        lora_weights = finetuning_adapter.layers[layer_id]
        # Backprop through LoRA
        
        grad_attn_input, grad_w_combined = self._backpop_attention(base_model, last_layer_input, grad_ffn_input, lora_weights, layer_weight, layer_id, finetuning_adapter.scaling, batch_seq_lens)
        return grad_attn_input
    
    # Backprop through the LoRA augmented attention block, only this function uses the autograd of pytorch
    def _backpop_attention(self, base_model,
                       last_layer_input: torch.Tensor, 
                       grad_ffn_input: torch.Tensor, 
                       lora_weight: LoraLayerWeight, 
                       layer_weight: LlamaTransformerLayerWeight,
                       layer_id: int,
                       scaling: any,
                       batch_seq_lens: torch.Tensor):
        """
        Back-propagate *only through the attention sub-block* of a single
        transformer layer that has been augmented with **one LoRA adapter**.
        The key parameter is:
        the *combined* LoRA weight tensor ``w_combined`` that is a packed version of lora qkvo
            (shape ``[2, 4 r, H, Hd]`` as described in the paper / original repo)
        We want to directly get the gradient w.r.t. this tensor.

        The routine is mainly in three parts:
        1. Create a clone of w_combined tensor for gradient computation.
        2. Use the clone to rematerialise the missing activations (RMS-norm, QKV, rotary, masked
        causal attention, output-proj) in **FP32** for numerical stability. (recompute forward pass)
        3. Step 2 require grad, so the autograd graph is built automatically. Then we can use pytorch's
        autograd to directly compute the gradient w.r.t. the combined LoRA weight tensor.

        After this, we clip the gradient or safety and writes it into the original FP32 “master” copy that is saved

        in the LoRA adapter object.
        ----------
        Parameters
        ----------
        base_model :
            The full Llama model object (provides cached rotary tables, layer meta).
        last_layer_input : torch.Tensor
            Activations *entering* this attention block  
            shape = ``[tokens, D]`` (mixed precision, will be cast to fp32).
        grad_ffn_input : torch.Tensor
            Gradient that arrives *after* the FFN residual add  
            (i.e. ∂L/∂ attn_out_residual). Same shape as ``last_layer_input``.
        lora_weight : LoraLayerWeight
            Holds the LoRA weights for this layer. We only need
            ``w_combined`` (or the “home” backup if it is still off-GPU).
        layer_weight : LlamaTransformerLayerWeight
            The frozen base weights (Q/K/V/O + layer norms, etc.).
        layer_id : int
            Index of the current transformer layer (for debug / diff printing).
        scaling : float
            α / r from the LoRA paper  the scalar applied to B @ A branch.
        batch_seq_lens : torch.Tensor
            Per-request **sequence lengths** inside the **flattened token batch**
            produced by the router (shape ``[B]``).

        ----------
        Returns
        -------
        grad_attn_input : torch.Tensor
            ∂L/∂ input_to_this_layer  to be propagated into the lower layer.
        grad_w_combined : torch.Tensor | None
            Gradient for the packed LoRA weight tensor
            (same shape as ``w_combined``).  Note: because we immediately copy
            this gradient into ``w_combined_home_fp32.grad`` the caller usually
            does not need the return value, but it is provided for diagnostic
            purposes.

        ----------
        Implementation notes
        --------------------
        * The function uses **torch.autograd** on small leaf-clones
        (``x_norm_leaf`` & sliced LoRA matrices) so that PyTorch builds the
        backward graph for us – this is simpler and less error-prone than
        hand-deriving every Jacobian.
        * Rotary embedding is executed by the local helper
        ``rotary_emb_fwd_pt`` (identical numerics to S-LoRA's Triton kernel).
        * Causal masking is re-applied *per request* using the original
        sequence-length list; this yields bit-for-bit outputs equal to the
        forward pass.
        * After the backward call we perform an **in-place gradient clip**
        (max-norm = 1) **before** copying to the FP32 master weight.
        * Debug helpers ``report_diff_percent`` can be toggled by changing
        ``compare_layer_id`` to visualise discrepancies versus the forward
        cache.
        """
        
         # ---------------------------
        # Debug / comparison settings
        # ---------------------------
        compare_layer_id = 31 #layer to compare with the saved intermediate result saved during forward pass

        # Grab layer-specific inference metadata (e.g. embed_dim) --------------
        base_layer_infer = base_model.layers_infer[layer_id]

        # Build *flattened* position index for **all** tokens in this micro‑batch
        position_ids = torch.cat([
            torch.arange(0, batch_seq_lens[i], device=batch_seq_lens.device)
            for i in range(len(batch_seq_lens))
        ])
        position_cos = base_model._cos_cached.index_select(0, position_ids)  # [T, D/2]
        position_sin = base_model._sin_cached.index_select(0, position_ids)  # [T, D/2]


        # ========================  Weight re‑materialisation  =================
        # Everything is explicitly cast to *float32* for numerical robustness
        # ---------------------------------------------------------------------
        w_q = layer_weight.q_weight_.float()
        w_k = layer_weight.k_weight_.float()
        w_v = layer_weight.v_weight_.float()
        w_o = layer_weight.o_weight_.float()
        w_attn_norm = layer_weight.att_norm_weight_.float()
        
        # -------------------- 1️⃣  RMSNorm (x → x_norm) -----------------------
        last_layer_input_leaf = last_layer_input.float().detach().clone().requires_grad_()
        x_norm = rmsnorm_forward(last_layer_input_leaf, w_attn_norm, eps=self.eps_)

         # -------------------- 2️⃣  LoRA weight materialisation -----------------
        if lora_weight.w_combined is None:
            print("###### LoRA weight is None, using LoRA weight from home")
            w_combined = lora_weight.w_combined_home.to("cuda").float() 
        else:
            w_combined = lora_weight.w_combined.float()  # [2, 4r, H, Hd]
        r = w_combined.shape[1] // 4 # Derived LoRA hyper‑params
        H, Hd = w_combined.shape[2], w_combined.shape[3]

        # To use autograd: create *leaf* clones so autograd tracks them separately
        w_combined_leaf = w_combined.detach().clone().requires_grad_()
        # Unpack the packed [2,4r,H,Hd] tensor into the individual A/B matrices.
        # Shapes follow the original LoRA paper:  A is [D,r]  and  B is [r,D].
        qA = w_combined_leaf[0, 0 : r].reshape(r, -1).T
        qB = w_combined_leaf[1, 0 : r].reshape(-1, r).T
        kA = w_combined_leaf[0, r : 2 * r].reshape(r, -1).T
        kB = w_combined_leaf[1, r : 2 * r].reshape(-1, r).T
        vA = w_combined_leaf[0, 2 * r : 3 * r].reshape(r, -1).T
        vB = w_combined_leaf[1, 2 * r : 3 * r].reshape(-1, r).T
        oA = w_combined_leaf[0, 3 * r : 4 * r].reshape(r, -1).T  # [D, r]
        oB = w_combined_leaf[1, 3 * r : 4 * r].reshape(-1, r).T  # [r, D]
         # To use autograd: Leaf clone of x_norm so we get ∂L/∂x_norm later 
        x_norm_leaf = x_norm.detach().clone().requires_grad_()

         # Helper: apply LoRA projection  x · A · B  * α/r ----------------------
        def proj_lora(x, A, B):
            output = torch.mm(x, A)
            output = torch.mm(output, B).mul_(scaling)
            return output

         # Helper: *PyTorch* reference rotary embedding:
         # Note that this is an exact duplicate of the Triton kernel in pytorch
         # I did not use the Triton kernel here because I want pytorch to build the autograd graph
         # Here could be a part for optimization since this is done in pytorch.
         # The original triton kernel is in slora/models/llama/triton_kernel/rotary_emb.py
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

        # -------------------- 3️⃣  Linear projections Q K V --------------------
        q_base = torch.mm(x_norm_leaf.view(-1, base_layer_infer.embed_dim_), w_q)
        k_base = torch.mm(x_norm_leaf.view(-1, base_layer_infer.embed_dim_), w_k)
        v_base = torch.mm(x_norm_leaf.view(-1, base_layer_infer.embed_dim_), w_v)

        q_  = q_base + proj_lora(x_norm_leaf, qA, qB)
        k_  = k_base + proj_lora(x_norm_leaf, kA, kB)
        v_  = v_base + proj_lora(x_norm_leaf, vA, vB)
        rotary_emb_fwd_pt(q_.view(-1, H, Hd), position_cos, position_sin)
        rotary_emb_fwd_pt(k_.view(-1, H, Hd), position_cos, position_sin)
        # if layer_id == compare_layer_id:
        #     self.report_diff_percent("Recomputed q", q_, self.mem_manager.saved_q)
        #     self.report_diff_percent("Recomputed k", k_, self.mem_manager.saved_k)
        #     self.report_diff_percent("Recomputed v", v_, self.mem_manager.saved_v)

        # -------------------- 4️⃣  Masked causal attention --------------------
        S = x_norm.size(0) 
        D = x_norm.size(1)  
        qh, kh, vh = q_.view(S, H, Hd), k_.view(S, H, Hd), v_.view(S, H, Hd)
        ctx = torch.empty_like(qh)
        
        B = batch_seq_lens.shape[0]
        scale = 1.0 / (Hd ** 0.5)

        # Pre‑compute token start offsets for each request
        # This following loop is to compute attention for each request in the batch
        # This cannot be directly applied to the whole batch because
        # the attention mask is different for each request in the batch
        # This could be a part for potential optimization
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

        # Flatten back to [tokens, D]
        ctx_flat = ctx.reshape(S, D)

        # -------------------- 5️⃣  Output projection (O) ----------------------
        o_base_ = torch.mm(ctx_flat, w_o)
        o_lora_ = proj_lora(ctx_flat, oA, oB)
        o_total = o_base_ + o_lora_
        # if layer_id == compare_layer_id:
        #     self.report_diff_percent("Recomputed O", o_total, self.mem_manager.saved_o)

        # Residual add: x_prev + Attn_out
        input_embs = last_layer_input_leaf + o_total.view(-1, base_layer_infer.embed_dim_)

        #input_embs.add_(o_total.view(-1, base_layer_infer.embed_dim_))
        ffn_input = self.mem_manager.get_ffn_input(layer_id).float() 
        # if layer_id == compare_layer_id:
        #     self.report_diff_percent("Recomputed ffn input", input_embs, ffn_input)

        # -------------------- 6️⃣  Backward pass ------------------------------
        grad_o = grad_ffn_input.float()
        input_embs.backward(grad_o)
        
        # -------------------- 7️⃣  Gradient clipping & copy -------------------
        g = w_combined_leaf.grad
        max_norm = 1
        if g is not None:
            grad_norm = g.norm()                           # ‖g‖₂   (scalar tensor)
            if grad_norm > max_norm:                       # scale *in-place*
                g.mul_(max_norm / (grad_norm + 1e-6))

        # now copy into the fp32 master copy
        lora_weight.w_combined_home_fp32.grad = g.to(device=lora_weight.w_combined_home_fp32.device)
        return last_layer_input_leaf.grad, w_combined_leaf.grad

    # Backprop through the FFN sub-block of a transformer layer
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

    