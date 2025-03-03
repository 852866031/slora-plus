import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class CustomBackwardFunction_1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W1, b1, W2, b2, W3, b3, activations):
        ctx.save_for_backward(W1, b1, W2, b2, W3, b3, *activations)
        return activations[-1]  

    @staticmethod
    def backward(ctx, grad_output):
        W1, b1, W2, b2, W3, b3, X, A1, A2, A3 = ctx.saved_tensors
        # Compute gradients for W3 and b3
        dZ3 = grad_output  # dL/dA3
        dW3, db3, dA2 = CustomBackwardFunction_1.compute_linear_gradients(dZ3, A2, W3)
        # Compute gradients for W2 and b2
        dZ2 = dA2 * (A2 > 0)  # ReLU gradient
        dW2, db2, dA1 = CustomBackwardFunction_1.compute_linear_gradients(dZ2, A1, W2)
        # Compute gradients for W1, b1
        dZ1 = dA1 * (A1 > 0)  # ReLU gradient
        dW1, db1, _ = CustomBackwardFunction_1.compute_linear_gradients(dZ1, X, W1)
        # Return all gradients
        return dW1, db1, dW2, db2, dW3, db3, None, None, None, None  # None for non-learnable tensors
    
    @staticmethod
    def compute_linear_gradients(dZ, A_prev, W):
        dW = A_prev.T @ dZ
        db = dZ.sum(dim=0)
        dA_prev = dZ @ W.T  # Backpropagate to the previous layer
        return dW, db, dA_prev



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
        
        alter = list(CustomBackwardFunction.backward_1(ctx, grad_output))
        for i in range(len(alter)):
            if all_grads[i+1]== None and alter[i]==None or torch.equal(all_grads[i+1], alter[i]):
                continue
            else:
                print("Sth wrong")
                break

        return tuple(all_grads)
    
    @staticmethod
    def backward_1(ctx, grad_output):
        W1, b1, W2, b2, W3, b3, X, A1, A2, A3 = ctx.saved_tensors
        # Compute gradients for W3 and b3
        dZ3 = grad_output  # dL/dA3
        dW3, db3, dA2 = CustomBackwardFunction_1.compute_linear_gradients(dZ3, A2, W3)
        # Compute gradients for W2 and b2
        dZ2 = dA2 * (A2 > 0)  # ReLU gradient
        dW2, db2, dA1 = CustomBackwardFunction_1.compute_linear_gradients(dZ2, A1, W2)
        # Compute gradients for W1, b1
        dZ1 = dA1 * (A1 > 0)  # ReLU gradient
        dW1, db1, _ = CustomBackwardFunction_1.compute_linear_gradients(dZ1, X, W1)
        # Return all gradients
        return dW1, db1, dW2, db2, dW3, db3, None, None, None, None  # None for non-learnable tensors
    
    
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
