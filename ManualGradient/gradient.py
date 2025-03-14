import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


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

    supported_activation_functions = {
        "layer_norm": lambda dZ, A_prev: TransformerCustomBackwardFunction.compute_layer_norm_gradients(dZ, A_prev),
    }

    @staticmethod
    def forward(ctx, layers_info, *tensors):
        ctx.save_for_backward(*tensors)
        ctx.layers_info = layers_info  # Save layer dictionary
        print(f"{0} Expected Tensor shape: ", None)
        for i, t in enumerate(tensors):
            print(f"{i+1} Expected Tensor shape: ", t.shape)
        return tensors[-1]  # Return final output (last activation)

    @staticmethod
    def backward(ctx, grad_output):
        print("Backward called")
        saved_tensors = list(ctx.saved_tensors)
        layers_info = ctx.layers_info
        grads = []
        batch_size, seq_len, vocab_size = saved_tensors[-1].shape
        dA = grad_output.view(batch_size, seq_len, vocab_size)
        print("Grad output shape: ", dA.shape)
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
        
        for layer_type, layer_params, A_prev, activation_type in layer_data:
            print("Layer type: ", layer_type)
            print("Activation type: ", activation_type)
            print("Layer params count: ", len(layer_params))
            print("Previous Activation shape", A_prev.shape)

        # Iterate through the layers in reverse order to compute gradients
        for layer_type, layer_params, A_prev, activation_type in reversed(layer_data):
            print("Layer type: ", layer_type)
            compute_gradient = TransformerCustomBackwardFunction.supported_gradient_functions.get(layer_type)
            activation_gradient = TransformerCustomBackwardFunction.supported_activation_functions.get(activation_type)
            if compute_gradient is None:
                raise ValueError(f"Unsupported layer type: {layer_type}")

            if activation_gradient is not None:
                dA = activation_gradient(dA, A_prev)
            
            print(f"Computing gradient with dZ {dA.shape}, A_prev {A_prev.shape},  parameter shape {layer_params[0].shape}", )
            param_grads, dA = compute_gradient(dA, A_prev, layer_params)
            grads = param_grads + grads
            print("Gradient computed for layer type ", layer_type)

        all_grads = [None] + grads + [None] * len(activations)
        for i, grad in enumerate(all_grads):
            print(f"{i} Gradient shape: ", grad.shape if grad is not None else None)
        return tuple(all_grads)

    @staticmethod
    def compute_embedding_gradients(dZ, A_prev, layer_params):
        _, embedding_dim = layer_params[0].shape
        d_embedding_weight = torch.zeros_like(layer_params[0])
        d_embedding_weight.index_add_(0, A_prev.long().reshape(-1), dZ.reshape(-1, embedding_dim))
        return [d_embedding_weight], None

    @staticmethod
    def compute_decoder_gradients(dZ, A_prev, layer_params):
        W_q, W_k, W_v, W_o, W1, b1, W2, b2 = layer_params

        batch_size, seq_len, d_model = dZ.shape
        dZ_ffn = dZ.clone()
        hidden_ffn = torch.relu(A_prev @ W1 + b1)
        dW2 = torch.einsum('bsd,bsf->fd', dZ_ffn, hidden_ffn)
        db2 = dZ_ffn.sum(dim=(0,1))
        d_hidden = (dZ_ffn @ W2.T) * (hidden_ffn > 0).float()

        # Gradient of feedforward network
        dW1 = torch.einsum('bsi,bsf->if', A_prev, d_hidden)       
        db1 = d_hidden.sum(dim=(0,1))
        dA_prev_ffn = d_hidden @ W1.T

        # Gradient of multi-head attention  
        d_attn_output = dZ.clone()
        dW_o = torch.einsum('bsd,bsi->di', d_attn_output, A_prev)
        d_attn_hidden = d_attn_output @ W_o.T
        dW_q = torch.einsum('bsd,bsi->di', d_attn_hidden, A_prev)
        dW_k = torch.einsum('bsd,bsi->di', d_attn_hidden, A_prev)
        dW_v = torch.einsum('bsd,bsi->di', d_attn_hidden, A_prev)
        dA_prev_attn = d_attn_hidden @ (W_q + W_k + W_v).T
        dA_prev = dA_prev_ffn + dA_prev_attn
        grads = [dW_q, dW_k, dW_v, dW_o, dW1, db1, dW2, db2]
        for i, grad in enumerate(grads):
            print(f"{i} Decoder Gradient shape: ", grad.shape)
        return grads, dA_prev


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
    
    @staticmethod
    def compute_layer_norm_gradients(dZ, A_prev):
        dA_prev = dZ.clone()
        return dA_prev