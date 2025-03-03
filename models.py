import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from gradient import CustomBackwardFunction, CustomBackwardFunction_1

class BackwardCache:
    def __init__(self):
        self.activations = [] 
        self.num_saved_requests = 0 
    
    def add(self, X, A1, A2, A3, labels):
        if labels is not None and labels.numel() > 0:
            self.activations.append((X.clone(), A1.clone(), A2.clone(), A3.clone(), labels.clone()))
            self.num_saved_requests += labels.shape[0]
    
    def get_batch(self, backward_batchsize):
        if self.num_saved_requests != backward_batchsize:
            return None
        
        X_train, A1_train, A2_train, A3_train, labels_train = zip(*self.activations)
        self.activations.clear()
        self.num_saved_requests = 0  # Reset counter
        
        return (
            torch.cat(X_train),
            torch.cat(A1_train),
            torch.cat(A2_train),
            torch.cat(A3_train),
            torch.cat(labels_train),
        )

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
    

    def forward_no_grad_train_filter(self, X, train_filter, y):
        with torch.no_grad():
            Z1 = X @ self.W1 + self.b1
            A1 = F.relu(Z1)

            Z2 = A1 @ self.W2 + self.b2
            A2 = F.relu(Z2)

            Z3 = A2 @ self.W3 + self.b3
            A3 = F.softmax(Z3, dim=1)

            # Filter activations based on train_filter
            filtered_A1 = A1 * train_filter[:, None]
            filtered_A2 = A2 * train_filter[:, None]
            filtered_A3 = A3 * train_filter[:, None]

            train_labels = y * train_filter.long()  # Keep labels only for training samples
            self.backward_cache.add(X, filtered_A1, filtered_A2, filtered_A3, train_labels)
        
        return A3
    
    def params(self):
        return self.W1, self.b1, self.W2, self.b2, self.W3, self.b3

    def print_params(self):
        """Prints the current weights and biases."""
        for name, params in self.model_dict.items():
            print(f"{name} - W: {params[0]}")
            print(f"{name} - b: {params[1]}")


    def forward_no_grad(self, X):
        with torch.no_grad():
            Z1 = X @ self.W1 + self.b1
            A1 = F.relu(Z1)

            Z2 = A1 @ self.W2 + self.b2
            A2 = F.relu(Z2)

            Z3 = A2 @ self.W3 + self.b3
            A3 = F.softmax(Z3)

            
            self.activation_cache.append((X.clone(), A1.clone(), A2.clone(), A3.clone()))

        return A3  # No grad is tracked
    
    def forward_no_grad_train_filter(self, X, train_filter, y):
        with torch.no_grad():
            Z1 = X @ self.W1 + self.b1
            A1 = F.relu(Z1)

            Z2 = A1 @ self.W2 + self.b2
            A2 = F.relu(Z2)

            Z3 = A2 @ self.W3 + self.b3
            A3 = F.softmax(Z3, dim=1)

            # Filter activations based on train_filter
            filtered_A1 = A1 * train_filter[:, None]
            filtered_A2 = A2 * train_filter[:, None]
            filtered_A3 = A3 * train_filter[:, None]

            train_labels = y * train_filter.long()  # Keep labels only for training samples
            self.activation_cache.append((X.clone(), filtered_A1, filtered_A2, filtered_A3, train_labels))
        
        return A3
    
    def manual_backward_with_optimizer(self, train_batch_size, optimizer):
        batch_data = self.form_backward_batch(train_batch_size)
        if batch_data[0] is None:
            return None
        
        X_train, A1_train, A2_train, A3_train, labels_train = batch_data
        
        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(A3_train, labels_train)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def manual_backward_with_optimizer(self, optimizer):
        batch_data = self.form_backward_batch()
        # Retrieve activations
        X_train, A1_train, A2_train, A3_train, labels_train = batch_data
        A1_train = A1_train.detach().requires_grad_()
        A2_train = A2_train.detach().requires_grad_()
        # A3_linked = CustomBackwardFunction_1.apply(
        #     self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, [X, A1, A2, A3]
        # )

        A3_linked = CustomBackwardFunction.apply(
            self.layers_info, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, X_train, A1_train, A2_train, A3_train
        )

        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(A3_linked, labels_train)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        return loss.item()