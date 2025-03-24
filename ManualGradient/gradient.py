import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import math
import time
import torch.backends.mkl
import torch.utils.benchmark as benchmark

class CustomBackwardFunction(torch.autograd.Function):
    supported_gradient_functions = {
        "linear": lambda dZ, A_prev, W: CustomBackwardFunction.compute_linear_gradients(dZ, A_prev, W),
    }

    supported_activation_functions = {
        "relu": lambda dZ, A_prev: CustomBackwardFunction.compute_relu_gradients(dZ, A_prev),
    }

    @staticmethod
    def forward(ctx, layers_info, *tensors):
        ctx.save_for_backward(*tensors)
        ctx.layers_info = layers_info  # Save layer dictionary
        return tensors[-1] # Return final output (last activation)

    @staticmethod
    def backward(ctx, grad_output):
        print("Backward called")
        saved_tensors = list(ctx.saved_tensors)
        layers_info = ctx.layers_info 
        grads = []
        dA = grad_output
        num_params = sum(i for (_, i, _) in layers_info)
        params = saved_tensors[:num_params]
        activations = saved_tensors[num_params:]
        param_idx = len(params) 
        act_idx = len(activations)-1
        
        for layer_type, num_weights, activation_type, in reversed(layers_info):
            start_idx = param_idx - num_weights
            layer_params = params[start_idx:param_idx]
            param_idx = start_idx  # move the pointer
            act_idx -= 1
            A_prev = activations[act_idx]
            
            compute_gradient = CustomBackwardFunction.supported_gradient_functions.get(layer_type)
            activation_gradient = CustomBackwardFunction.supported_activation_functions.get(activation_type)
            if compute_gradient is None:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            
            if activation_gradient is not None:
                dA = activation_gradient(dA, activations[act_idx+1])
            param_grads, dA = compute_gradient(dA, A_prev, layer_params)
            grads = param_grads + grads
        
        all_grads = [None] + grads + [None]*len(activations)
        return tuple(all_grads)
    
    
    
    @staticmethod
    def compute_linear_gradients(dZ, A_prev, layer_params):
        W, _ = layer_params
        dW = A_prev.T @ dZ
        db = dZ.sum(dim=0)
        dA_prev = dZ @ W.T
        return [dW, db], dA_prev
    
    @staticmethod
    def compute_relu_gradients(dZ, A_prev):
        """
        Computes gradient of ReLU activation: dA = dZ * (A_prev > 0)
        """
        dA_prev = dZ.clone()
        dA_prev[A_prev <= 0] = 0 
        return dA_prev
    

class TransformerCustomBackwardFunction(torch.autograd.Function):
    supported_gradient_functions = {
        "embedding": lambda dZ, A_prev, W: TransformerCustomBackwardFunction.compute_embedding_gradients(dZ, A_prev, W),
        "decoder": lambda dZ, A_prev, W: TransformerCustomBackwardFunction.compute_decoder_gradients(dZ, A_prev, W),
        "output": lambda dZ, A_prev, W: TransformerCustomBackwardFunction.compute_output_gradients(dZ, A_prev, W),
    }

    @staticmethod
    def forward(ctx, layers_info, *tensors):
        ctx.save_for_backward(*tensors)
        ctx.layers_info = layers_info  # Save layer dictionary
        '''
        for i, t in enumerate(tensors):
            print(f"{i+1} Expected Tensor shape: ", t.shape)
        '''
        return tensors[-1]  # Return final output (last activation)

    @staticmethod
    def backward(ctx, grad_output):
        #print("Backward called")
        saved_tensors = list(ctx.saved_tensors)
        layers_info = ctx.layers_info
        grads = []
        batch_size, seq_len, vocab_size = saved_tensors[-1].shape
        dA = grad_output.view(batch_size, seq_len, vocab_size)
        #print("Grad output shape: ", dA.shape)
        num_params = sum(i for (_, i, _) in layers_info)
        params = saved_tensors[:num_params]
        activations = saved_tensors[num_params:]

        layer_data = []
        param_idx = 0
        act_idx = 0

        for layer_type, num_weights, activation_type in layers_info:
            layer_params = params[param_idx:param_idx + num_weights]
            param_idx += num_weights
            A_prev = activations[act_idx]
            act_idx += 1
            layer_data.append((layer_type, layer_params, A_prev, activation_type))

        
        # Iterate through the layers in reverse order to compute gradients
        for layer_type, layer_params, A_prev, activation_type in reversed(layer_data):
            print(f"Backward Layer type: {layer_type}, Num weights: {len(layer_params)}")
            #print("Layer type: ", layer_type)
            compute_gradient = TransformerCustomBackwardFunction.supported_gradient_functions.get(layer_type)
            if compute_gradient is None:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            
            #print(f"Computing gradient with dZ {dA.shape}, A_prev {A_prev.shape},  parameter shape {layer_params[0].shape}", )
            param_grads, dA = compute_gradient(dA, A_prev, layer_params)
            grads = param_grads + grads
            #print("Gradient computed for layer type ", layer_type)

        all_grads = [None] + grads + [None] * len(activations)
        '''
        for i, grad in enumerate(all_grads):
            print(f"{i} Gradient shape: ", grad.shape if grad is not None else None)
        '''
        return tuple(all_grads)

    @staticmethod
    def compute_embedding_gradients(dZ, A_prev, layer_params):
        _, embedding_dim = layer_params[0].shape
        d_embedding_weight = torch.zeros_like(layer_params[0])
        d_embedding_weight.index_add_(0, A_prev.long().reshape(-1), dZ.reshape(-1, embedding_dim))
        return [d_embedding_weight], None    
    
    
    @staticmethod
    def compute_decoder_gradients(dZ, A_prev, layer_params):
        (
            W_q, W_k, W_v, W_o,
            ln1_weight, ln1_bias,
            W1, b1, W2, b2,
            ln2_weight, ln2_bias
        ) = layer_params
        
        batch_size, seq_len, d_model = A_prev.shape
        nhead = W_q.shape[0] // (d_model // W_o.shape[0])
        head_dim = d_model // nhead

        # ----- Rematerialization -----
        attn_start = time.time()

        Q = A_prev @ W_q
        K = A_prev @ W_k
        V = A_prev @ W_v

        Q_heads = Q.view(batch_size, -1, nhead, d_model // nhead).transpose(1, 2)
        K_heads = K.view(batch_size, -1, nhead, d_model // nhead).transpose(1, 2)
        V_heads = V.view(batch_size, -1, nhead, d_model // nhead).transpose(1, 2)

        scores = (Q_heads @ K_heads.transpose(-2, -1)) / math.sqrt(Q_heads.size(-1))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output_heads = attn_weights @ V_heads
        attn_output = attn_output_heads.transpose(1, 2).contiguous().view(batch_size, -1, d_model)
        attn_out_proj = attn_output @ W_o
        attn_time = time.time() - attn_start
        print(f"\tAttention Rematerialization done in {attn_time:.4f} seconds")
        # LayerNorm1
        ln1_start = time.time()
        Z1 = A_prev + attn_out_proj
        mean1 = Z1.mean(dim=-1, keepdim=True)
        var1 = Z1.var(dim=-1, unbiased=False, keepdim=True)
        Z1_norm = (Z1 - mean1) / torch.sqrt(var1 + 1e-5)
        Z1_out = ln1_weight * Z1_norm + ln1_bias
        ln1_time = time.time() - ln1_start
        print(f"\tLayerNorm 1 Rematerialization done in {ln1_time:.4f} seconds")


        # Feedforward network
        ffn_start = time.time()
        FF_intermediate = F.relu(Z1_out @ W1 + b1)
        FF_output = FF_intermediate @ W2 + b2
        ffn_time = time.time() - ffn_start
        print(f"\tFFN Rematerialization done in {ffn_time:.4f} seconds")


        # LayerNorm2 (nn.LayerNorm initialized with params)
        ln2_start = time.time()
        Z2 = Z1_out + FF_output
        mean2 = Z2.mean(dim=-1, keepdim=True)
        var2 = Z2.var(dim=-1, unbiased=False, keepdim=True)
        Z2_norm = (Z2 - mean2) / torch.sqrt(var2 + 1e-5)
        Z2_out = ln2_weight * Z2_norm + ln2_bias
        ln2_time = time.time() - ln2_start
        print(f"\tLayerNorm 2 Rematerialization done in {ln2_time:.4f} seconds")

        # ----- Backward pass -----
        # LayerNorm2 gradients
        start = time.time()
        dZ2_norm = dZ * ln2_weight
        dln2_weight = (dZ * Z2_norm).sum(dim=[0,1])
        dln2_bias = dZ.sum(dim=[0,1])

        dvar2 = (dZ2_norm * (Z2 - mean2) * (-0.5) * (var2 + 1e-5)**(-1.5)).sum(-1, keepdim=True)
        dmean2 = (dZ2_norm * (-1) / torch.sqrt(var2 + 1e-5)).sum(-1, keepdim=True) + \
                dvar2 * (-2 / d_model) * (Z2 - mean2).sum(-1, keepdim=True)
        dZ2 = dZ2_norm / torch.sqrt(var2 + 1e-5) + dvar2 * 2 * (Z2 - mean2) / d_model + dmean2 / d_model

        dZ1_out = dZ2.clone()
        dFF_output = dZ2.clone()
        print(f"\tLayerNorm 2 gradient done in {time.time() - start:.4f} seconds")

        # Feed-forward gradients
        start = time.time()
        dW2 = (FF_intermediate.transpose(1,2) @ dFF_output).sum(0)
        db2 = dFF_output.sum(dim=[0,1])
        dFF_intermediate = dFF_output @ W2.T
        dFF_intermediate[FF_intermediate <= 0] = 0

        dW1 = (Z1_out.transpose(1,2) @ dFF_intermediate).sum(0)
        db1 = dFF_intermediate.sum(dim=[0,1])
        dZ1_out += dFF_intermediate @ W1.T
        print(f"\tFFN gradient done in {time.time() - start:.4f} seconds")
        # LayerNorm1 gradients
        start = time.time()
        dZ1_norm = dZ1_out * ln1_weight
        dln1_weight = (dZ1_out * Z1_norm).sum(dim=[0,1])
        dln1_bias = dZ1_out.sum(dim=[0,1])

        dvar1 = (dZ1_norm * (Z1 - mean1) * (-0.5) * (var1 + 1e-5)**(-1.5)).sum(-1, keepdim=True)
        dmean1 = (dZ1_norm * (-1) / torch.sqrt(var1 + 1e-5)).sum(-1, keepdim=True) + \
                dvar1 * (-2 / d_model) * (Z1 - mean1).sum(-1, keepdim=True)
        dZ1 = dZ1_norm / torch.sqrt(var1 + 1e-5) + dvar1 * 2 * (Z1 - mean1) / d_model + dmean1 / d_model

        dA_prev = dZ1.clone()
        dattention_proj = dZ1.clone()
        print(f"\tLayerNorm 1 gradient done in {time.time() - start:.4f} seconds")
        # Attention output projection gradient
        start = time.time()
        dW_o = (attn_output.reshape(-1, d_model).T @ dattention_proj.reshape(-1, d_model))
        dattention_output = dattention_proj @ W_o.T
        dattention_output_heads = dattention_output.view(batch_size, seq_len, nhead, head_dim).transpose(1, 2)
        # Attention weight gradients
        dattn_weights = dattention_output_heads @ V_heads.transpose(-2, -1)
        dV_heads = attn_weights.transpose(-2, -1) @ dattention_output_heads
        dscores = attn_weights * (dattn_weights - (dattn_weights * attn_weights).sum(dim=-1, keepdim=True))
        dscores /= math.sqrt(head_dim)
        dQ_heads = dscores @ K_heads
        dK_heads = dscores.transpose(-2, -1) @ Q_heads
        dQ = dQ_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        dK = dK_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        dV = dV_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Compute gradients for W_q, W_k, W_v
        dW_q = (A_prev.reshape(-1, d_model).T @ dQ.reshape(-1, d_model))
        dW_k = (A_prev.reshape(-1, d_model).T @ dK.reshape(-1, d_model))
        dW_v = (A_prev.reshape(-1, d_model).T @ dV.reshape(-1, d_model))
        print(f"\tAttention gradient done in {time.time() - start:.4f} seconds")
        # Aggregate dA_prev
        dA_prev += dQ @ W_q.T + dK @ W_k.T + dV @ W_v.T

        gradients = [
            dW_q, dW_k, dW_v, dW_o,
            dln1_weight, dln1_bias,
            dW1, db1, dW2, db2,
            dln2_weight, dln2_bias
        ]

        return gradients, dA_prev
    @staticmethod
    def compute_output_gradients(dZ, A_prev, layer_params):
        W, b = layer_params

        # Gradient w.r.t. weights
        dW = torch.einsum('bsi,bsj->ij', dZ, A_prev)

        # Gradient w.r.t. biases (sum over batch and seq_len dimensions)
        db = dZ.sum(dim=(0, 1))

        # Gradient w.r.t. previous activation
        dA_prev = torch.einsum('bsi,ij->bsj', dZ, W)
        
        return [dW, db], dA_prev