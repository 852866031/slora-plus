import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from gradient import CustomBackwardFunction, TransformerCustomBackwardFunction
import math

class BackwardCache:
    def __init__(self):
        self.activations = [] 
        self.num_saved_requests = 0 
    
    def add(self, X, A1, A2, A3, labels):
        if labels is not None and labels.numel() > 0:
            self.activations.append((X.clone(), A1.clone(), A2.clone(), A3.clone(), labels.clone()))
            self.num_saved_requests += labels.shape[0]

    def get_batch(self, backward_batchsize, flush=False):
        if not flush and self.num_saved_requests != backward_batchsize:
            return None
        
        if not self.activations:
            return None
        
        X_train, A1_train, A2_train, A3_train, labels_train = zip(*self.activations)
        num_requests = self.num_saved_requests
        self.activations.clear()
        self.num_saved_requests = 0 

        return num_requests, (
            torch.cat(X_train),
            torch.cat(A1_train),
            torch.cat(A2_train),
            torch.cat(A3_train),
            torch.cat(labels_train),
        )


class MLP:
    def __init__(self, manual_mlp):
        # Initialize weights and biases with requires_grad enabled
        self.W1 = manual_mlp.W1.detach().clone().requires_grad_()
        self.b1 = manual_mlp.b1.detach().clone().requires_grad_()
        self.W2 = manual_mlp.W2.detach().clone().requires_grad_()
        self.b2 = manual_mlp.b2.detach().clone().requires_grad_()
        self.W3 = manual_mlp.W3.detach().clone().requires_grad_()
        self.b3 = manual_mlp.b3.detach().clone().requires_grad_()
    
    def params(self):
        return self.W1, self.b1, self.W2, self.b2, self.W3, self.b3


    def forward(self, X):
        """Standard forward pass."""
        Z1 = X @ self.W1 + self.b1
        A1 = F.relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = F.relu(Z2)
        Z3 = A2 @ self.W3 + self.b3
        A3 = F.softmax(Z3, dim=1)
        return A3


    def train_step(self, optimizer, X, y):
        """Perform a single training step."""
        optimizer.zero_grad()  # Reset gradients
        A3 = self.forward(X)  # Forward pass with filtering
        criterion = nn.CrossEntropyLoss()
        loss = criterion(A3, y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        return loss.item()
    
    def evaluate(self, X):
        return self.forward(X)

class MLPManual:
    def __init__(self, input_size=4, hidden_size1=10, hidden_size2=8, output_size=3):
        # Initialize weights and biases
        self.W1 = torch.randn(input_size, hidden_size1, dtype=torch.float32).requires_grad_()
        self.b1 = torch.zeros(hidden_size1, dtype=torch.float32).requires_grad_()
        self.W2 = torch.randn(hidden_size1, hidden_size2, dtype=torch.float32).requires_grad_()
        self.b2 = torch.zeros(hidden_size2, dtype=torch.float32).requires_grad_()
        self.W3 = torch.randn(hidden_size2, output_size, dtype=torch.float32).requires_grad_()
        self.b3 = torch.zeros(output_size, dtype=torch.float32).requires_grad_()
        self.layers_info = [("linear", 2, "relu"), ("linear", 2, "relu"), ("linear", 2, "None")]
        self.backward_cache = BackwardCache()  # Use BackwardCache instead of activation_cache
    
    def evaluate(self, X):
        with torch.no_grad():
            Z1 = X @ self.W1 + self.b1
            A1 = F.relu(Z1)

            Z2 = A1 @ self.W2 + self.b2
            A2 = F.relu(Z2)

            Z3 = A2 @ self.W3 + self.b3
            A3 = F.softmax(Z3, dim=1)
        
        return A3

    def forward_no_grad_train_filter(self, X, train_filter, y):
        with torch.no_grad():
            Z1 = X @ self.W1 + self.b1
            A1 = F.relu(Z1)

            Z2 = A1 @ self.W2 + self.b2
            A2 = F.relu(Z2)

            Z3 = A2 @ self.W3 + self.b3
            A3 = F.softmax(Z3, dim=1)

            # Select only training samples based on train_filter
            train_indices = train_filter.nonzero(as_tuple=True)[0]  # Get indices of training samples
            filtered_X = X[train_indices]
            filtered_A1 = A1[train_indices]
            filtered_A2 = A2[train_indices]
            filtered_A3 = A3[train_indices]
            filtered_labels = y[train_indices]

            self.backward_cache.add(filtered_X, filtered_A1, filtered_A2, filtered_A3, filtered_labels)
        return A3, filtered_labels.shape[0]
    
    def params(self):
        return self.W1, self.b1, self.W2, self.b2, self.W3, self.b3

    def print_params(self):
        """Prints the current weights and biases."""
        for name, params in self.model_dict.items():
            print(f"{name} - W: {params[0]}")
            print(f"{name} - b: {params[1]}")

    def manual_backward_with_optimizer(self, optimizer, batch_data):
        X_train, A1_train, A2_train, A3_train, labels_train = batch_data
        A1_train = A1_train.detach().requires_grad_()
        A2_train = A2_train.detach().requires_grad_()
        
        A3_linked = CustomBackwardFunction.apply(
            self.layers_info, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, X_train, A1_train, A2_train, A3_train
        )

        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(A3_linked, labels_train)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    

class TransformerDecoderManual:
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # Initialize weights and biases for the multi-head attention
        self.W_q = torch.randn(d_model, d_model, dtype=torch.float32).requires_grad_()
        self.W_k = torch.randn(d_model, d_model, dtype=torch.float32).requires_grad_()
        self.W_v = torch.randn(d_model, d_model, dtype=torch.float32).requires_grad_()
        self.W_o = torch.randn(d_model, d_model, dtype=torch.float32).requires_grad_()

        # Initialize weights and biases for the feedforward network
        self.W1 = torch.randn(d_model, dim_feedforward, dtype=torch.float32).requires_grad_()
        self.b1 = torch.zeros(dim_feedforward, dtype=torch.float32).requires_grad_()
        self.W2 = torch.randn(dim_feedforward, d_model, dtype=torch.float32).requires_grad_()
        self.b2 = torch.zeros(d_model, dtype=torch.float32).requires_grad_()

        self.ln1_weight = torch.ones(d_model).requires_grad_()
        self.ln1_bias = torch.zeros(d_model).requires_grad_()
        self.ln2_weight = torch.ones(d_model).requires_grad_()
        self.ln2_bias = torch.zeros(d_model).requires_grad_()
        self.dropout_layer = nn.Dropout(dropout)

    def layer_norm1(self, x, eps=1e-5):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + eps)
        return self.ln1_weight * normalized + self.ln1_bias
    
    def layer_norm2(self, x, eps=1e-5):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + eps)
        return self.ln2_weight * normalized + self.ln2_bias

    def scaled_dot_product_attention(self, Q, K, V):
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)
        return attn_weights @ V

    def multi_head_attention(self, Q, K, V):
        batch_size = Q.size(0)
        Q = Q @ self.W_q
        K = K @ self.W_k
        V = V @ self.W_v

        Q = Q.view(batch_size, -1, self.nhead, self.d_model // self.nhead).transpose(1, 2)
        K = K.view(batch_size, -1, self.nhead, self.d_model // self.nhead).transpose(1, 2)
        V = V.view(batch_size, -1, self.nhead, self.d_model // self.nhead).transpose(1, 2)

        attn_output = self.scaled_dot_product_attention(Q, K, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return attn_output @ self.W_o

    def feedforward(self, X):
        return self.dropout_layer(F.relu(X @ self.W1 + self.b1)) @ self.W2 + self.b2

    def forward(self, tgt):
        tgt2 = self.multi_head_attention(tgt, tgt, tgt)
        tgt = self.layer_norm1(tgt + self.dropout_layer(tgt2))
        tgt2 = self.feedforward(tgt)
        tgt = self.layer_norm2(tgt + self.dropout_layer(tgt2))
        return tgt
    
    def manual_forward(self, A_prev):
        W_q = self.W_q 
        W_k = self.W_k
        W_v = self.W_v 
        W_o = self.W_o
        W_norm1 = self.ln1_weight
        b_norm1 = self.ln1_bias 
        W1 = self.W1
        b1 = self.b1
        W2 = self.W2
        b2 = self.b2
        W_norm2 = self.ln2_weight
        b_norm2 = self.ln2_bias 
        
        batch_size, seq_len, d_model = A_prev.shape
        nhead = W_q.shape[0] // (d_model // W_o.shape[0])

        # ----- Forward pass (with nn.LayerNorm) -----
        Q = A_prev @ W_q
        K = A_prev @ W_k
        V = A_prev @ W_v

        Q_heads = Q.view(batch_size, -1, self.nhead, self.d_model // self.nhead).transpose(1, 2)
        K_heads = K.view(batch_size, -1, self.nhead, self.d_model // self.nhead).transpose(1, 2)
        V_heads = V.view(batch_size, -1, self.nhead, self.d_model // self.nhead).transpose(1, 2)

        scores = (Q_heads @ K_heads.transpose(-2, -1)) / math.sqrt(Q_heads.size(-1))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output_heads = attn_weights @ V_heads
        attn_output = attn_output_heads.transpose(1, 2).contiguous().view(batch_size, -1, d_model)
        attn_out_proj = attn_output @ W_o

        # LayerNorm1
        eps = 1e-5
        x1 = A_prev + attn_out_proj
        x1_mean = x1.mean(dim=-1, keepdim=True)
        x1_var = x1.var(dim=-1, unbiased=False, keepdim=True)
        Z1_norm = (x1 - x1_mean) / torch.sqrt(x1_var + eps)
        Z1_out = self.ln1_weight * Z1_norm + self.ln1_bias

        # Feedforward network
        FF_intermediate = F.relu(Z1_out @ W1 + b1)
        FF_output = FF_intermediate @ W2 + b2

        # LayerNorm2 (nn.LayerNorm initialized with params)
        x2 = Z1_out + FF_output
        x2_mean = x2.mean(dim=-1, keepdim=True)
        x2_var = x2.var(dim=-1, unbiased=False, keepdim=True)
        Z2_norm = (x2 - x2_mean) / torch.sqrt(x2_var + eps)
        Z2_out = self.ln2_weight * Z2_norm + self.ln2_bias

        ### Forward pass 
        self.dropout_layer.eval()
        tgt = A_prev
        tgt2 = self.multi_head_attention(tgt, tgt, tgt)
        tgt3 = self.layer_norm1(tgt + self.dropout_layer(tgt2))
        tgt4 = self.feedforward(tgt3)
        tgt5 = self.layer_norm2(tgt3 + self.dropout_layer(tgt4))

        def compare_tensors(manual_tensor, auto_tensor):
            """Compare two tensors and return comparison results"""
            
            max_diff = torch.max(torch.abs(manual_tensor - auto_tensor)).item()
            mean_diff = torch.mean(torch.abs(manual_tensor - auto_tensor)).item()
            is_close = torch.allclose(manual_tensor, auto_tensor, rtol=1e-4, atol=1e-4)
            return {
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'matches': is_close
            }

        return torch.equal(attn_out_proj, tgt2) and torch.equal(Z1_out, tgt3) and torch.equal(FF_output, tgt4) and torch.equal(Z2_out, tgt5)


    def params(self):
        return [
            self.W_q, self.W_k, self.W_v, self.W_o,
            self.ln1_weight, self.ln1_bias,
            self.W1, self.b1, self.W2, self.b2,
            self.ln2_weight, self.ln2_bias,
        ]
    
class TransformerBackwardCache:
    def __init__(self):
        self.activations = []
        self.num_saved_requests = 0

    def add(self, input_ids, activations, labels):
        if input_ids is not None and input_ids.numel() > 0:
            self.activations.append((input_ids.clone(), [act.clone() for act in activations], labels.clone()))
            self.num_saved_requests += input_ids.shape[0]

    def get_batch(self, backward_batchsize, flush=False):
        if not flush and self.num_saved_requests != backward_batchsize:
            return None

        if not self.activations:
            return None

        input_ids, activations, labels = zip(*self.activations)
        num_requests = self.num_saved_requests
        self.activations.clear()
        self.num_saved_requests = 0

        # Flatten the list of activations
        activations = [torch.cat([act[i] for act in activations], dim=0) for i in range(len(activations[0]))]
        #print("Batch Generated: Size: ", num_requests)
        #print("Activations num: ", len(activations))
        return num_requests, (
            torch.cat(input_ids),
            *activations,
            torch.cat(labels)
        )

class SimpleTransformerLM:
    def __init__(self, vocab_size, output_size, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.output_size = output_size
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(d_model, 5000)
        self.decoder_layers = [
            TransformerDecoderManual(d_model, nhead, dim_feedforward, dropout),
        ]
        self.output_layer = torch.nn.Linear(d_model, self.output_size)
        self.backward_cache = TransformerBackwardCache()
        self.layers_info = [
            ("embedding", 1, None),
            ("decoder", 12, None),
            ("output", 2, None)
        ]

    def copy_params_from(self, other):
        """Copy parameters from another SimpleTransformerLM instance."""
        # Copy embedding parameters
        self.embedding.weight.data.copy_(other.embedding.weight.data)
        
        # Copy decoder layer parameters
        for self_layer, other_layer in zip(self.decoder_layers, other.decoder_layers):
            # Copy multi-head attention parameters
            self_layer.W_q.data.copy_(other_layer.W_q.data)
            self_layer.W_k.data.copy_(other_layer.W_k.data)
            self_layer.W_v.data.copy_(other_layer.W_v.data)
            self_layer.W_o.data.copy_(other_layer.W_o.data)
            
            # Copy feedforward network parameters
            self_layer.W1.data.copy_(other_layer.W1.data)
            self_layer.b1.data.copy_(other_layer.b1.data)
            self_layer.W2.data.copy_(other_layer.W2.data)
            self_layer.b2.data.copy_(other_layer.b2.data)
            
            # Copy layer norm parameters
            self_layer.ln1_weight.data.copy_(other_layer.ln1_weight.data)
            self_layer.ln1_bias.data.copy_(other_layer.ln1_bias.data)
            self_layer.ln2_weight.data.copy_(other_layer.ln2_weight.data)
            self_layer.ln2_bias.data.copy_(other_layer.ln2_bias.data)
        
        # Copy output layer parameters
        self.output_layer.weight.data.copy_(other.output_layer.weight.data)
        self.output_layer.bias.data.copy_(other.output_layer.bias.data)
        
        # Copy positional encoding
        self.positional_encoding.data.copy_(other.positional_encoding.data)

    def _generate_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        embeddings = self.embedding(input_ids) + self.positional_encoding[:, :seq_len, :]
        tgt = embeddings
        for layer in self.decoder_layers:
            tgt = layer.forward(tgt)
        logits = self.output_layer(tgt)
        return logits

    def forward_no_grad_train_filter(self, input_ids, train_filter, labels):
        with torch.no_grad():
            seq_len = input_ids.size(1)
            embeddings = self.embedding(input_ids) + self.positional_encoding[:, :seq_len, :]
            #print(f"Embedding layer input shape: {input_ids.shape}, output shape: {embeddings.shape}")
            tgt = embeddings
            activations = [tgt]
            for i, layer in enumerate(self.decoder_layers):
                input_shape = tgt.shape
                tgt = layer.forward(tgt)
                output_shape = tgt.shape
                #print(f"Decoder layer {i+1} input shape: {input_shape}, output shape: {output_shape}")
                activations.append(tgt)
            logits = self.output_layer(tgt)
            #print(f"Output layer input shape: {tgt.shape}, output shape: {logits.shape} \n")
            activations.append(logits)
            
            # Save activations and labels for samples marked with 1 in train_filter
            for i in range(input_ids.size(0)):
                if train_filter[i] == 1:
                    self.backward_cache.add(input_ids[i:i+1], [act[i:i+1] for act in activations], labels[i:i+1])
                    
        return logits

    def manual_backward_with_optimizer(self, optimizer, batch_data):
        input_ids, *activations, labels = batch_data
        activations = [torch.Tensor.float(input_ids)] + activations
        activations = [act.detach().requires_grad_() for act in activations]

        # Use TransformerCustomBackwardFunction to compute gradients
        layers_info = self.layers_info
        params = self.params()
        #print("Added Params: ", len(params))
        #print("Added Activations: ", len(activations))
        logits = TransformerCustomBackwardFunction.apply(layers_info, *params, *activations)
        logits = logits[:, -1, :]  # Shape: (batch_size, output_size)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.output_size), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def params(self):
        params = [self.embedding.weight]
        for layer in self.decoder_layers:
            params.extend(layer.params())
        params.append(self.output_layer.weight)
        params.append(self.output_layer.bias)
        return params
    
    def dict_params(self):
        params_dict = {
            'embedding': {
                'weight': self.embedding.weight
            },
            'decoder_layers': {},
            'output_layer': {
                'weight': self.output_layer.weight,
                'bias': self.output_layer.bias
            }
        }
        
        # Organize decoder layers
        for i, layer in enumerate(self.decoder_layers):
            params_dict['decoder_layers'][f'layer_{i}'] = {
                'attention': {
                    'W_q': layer.W_q,
                    'W_k': layer.W_k,
                    'W_v': layer.W_v,
                    'W_o': layer.W_o
                },
                'feedforward': {
                    'W1': layer.W1,
                    'b1': layer.b1,
                    'W2': layer.W2,
                    'b2': layer.b2
                },
                'layer_norm1': {
                    'weight': layer.ln1_weight,
                    'bias': layer.ln1_bias
                },
                'layer_norm2': {
                    'weight': layer.ln2_weight,
                    'bias': layer.ln2_bias
                }
            }
        
        return params_dict