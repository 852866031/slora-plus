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

        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

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

    def params(self):
        return [
            self.W_q, self.W_k, self.W_v, self.W_o,
            self.W1, self.b1, self.W2, self.b2
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
        print("Batch Generated: Size: ", num_requests)
        print("Activations num: ", len(activations))
        return num_requests, (
            torch.cat(input_ids),
            *activations,
            torch.cat(labels)
        )

class SimpleTransformerLM:
    def __init__(self, vocab_size, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(d_model, 5000)
        self.decoder_layers = [
            TransformerDecoderManual(d_model, nhead, dim_feedforward, dropout),
            TransformerDecoderManual(d_model, nhead, dim_feedforward, dropout)
        ]
        self.output_layer = torch.nn.Linear(d_model, vocab_size)
        self.backward_cache = TransformerBackwardCache()
        self.layers_info = [
            ("embedding", 1, None),
            ("decoder", 8, "layer_norm"),
            ("decoder", 8, "layer_norm"),
            ("output", 2, None)
        ]

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
            print(f"Embedding layer input shape: {input_ids.shape}, output shape: {embeddings.shape}")
            tgt = embeddings
            activations = [tgt]
            for i, layer in enumerate(self.decoder_layers):
                input_shape = tgt.shape
                tgt = layer.forward(tgt)
                output_shape = tgt.shape
                print(f"Decoder layer {i+1} input shape: {input_shape}, output shape: {output_shape}")
                activations.append(tgt)
            logits = self.output_layer(tgt)
            print(f"Output layer input shape: {tgt.shape}, output shape: {logits.shape} \n")
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
        print("Added Params: ", len(params))
        print("Added Activations: ", len(activations))
        logits = TransformerCustomBackwardFunction.apply(layers_info, *params, *activations)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))

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