import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from models import MLPManual, MLP
from tqdm import tqdm


def load_iris_batches(batch_size=32):
    # Load the iris dataset
    iris = load_iris()
    X = torch.tensor(iris.data, dtype=torch.float32)
    y = torch.tensor(iris.target, dtype=torch.long)

    # Normalize features
    scaler = StandardScaler()
    X = torch.tensor(scaler.fit_transform(X), dtype=torch.float32)

    # Calculate the number of batches
    num_samples = X.shape[0]
    num_batches = num_samples // batch_size
    batches = []

    # Create batches
    for batch_idx in range(num_batches):
        indices = torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size)
        batch_X = X[indices]
        batch_y = y[indices]
        batches.append((batch_X, batch_y))
    return batches


def load_iris_data(batch_size=32, train_count=8):
    iris = load_iris()
    X, y = iris.data, iris.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train = np.tile(X, (4, 1))  # Repeat dataset 4 times for training
    y_train = np.tile(y, 4)  # Repeat labels accordingly
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    
    X_test = torch.tensor(X, dtype=torch.float32)  # Test dataset (no repetition)
    y_test = torch.tensor(y, dtype=torch.long)
    
    num_batches = len(X_train) // batch_size
    batches = []
    total_training = 0
    total_inference = 0
    
    for batch_idx in range(num_batches):
        indices = torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size)
        batch_X = X_train[indices]
        batch_y = y_train[indices]
        
        train_filter = torch.zeros(batch_size, dtype=torch.float32)
        repeat_idx = (batch_idx % 4)  # Determine which repeat (0, 1, 2, or 3)
        train_start = repeat_idx * train_count
        train_indices = torch.arange(train_start, train_start + train_count)
        train_filter[train_indices] = 1  # Assign training samples
        
        batches.append((batch_X, batch_y, train_filter))
        total_training += train_count
        total_inference += batch_size - train_count
    
    return batches, total_training, total_inference, (X_test, y_test)

def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        predictions = model.evaluate(X_test)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(predictions, y_test).item()
        
        correct = (predictions.argmax(dim=1) == y_test).sum().item()
        accuracy = correct / y_test.shape[0]
    
    return loss, accuracy


def mix_serving(model, train_loader, forward_batch_size, backward_batch_size, total_training_requests, total_inference_requests, X_test, y_test, learning_rate=0.001, epochs=10):
    optimizer = torch.optim.Adam(model.params(), lr=learning_rate)
    print("Total Training:" + str(total_training_requests * epochs))
    print("Total Inference:" + str(total_inference_requests * epochs))
    loss_history = []
    accuracy_history = []
    with tqdm(total=total_training_requests * epochs, desc="Training Progress") as train_pbar, \
         tqdm(total=total_inference_requests * epochs, desc="Inference Progress") as infer_pbar:
        for _ in range(epochs):
            for batch_X, batch_y, train_filter in train_loader:
                # Forward pass
                _, num_saved_requests = model.forward_no_grad_train_filter(batch_X, train_filter, batch_y)
                
                # Update inference progress
                infer_pbar.update(forward_batch_size-num_saved_requests)
                
                # Try forming a backward batch
                batch_info = model.backward_cache.get_batch(backward_batch_size)
                if batch_info is not None:
                    num_requests, backward_batch = batch_info
                    loss = model.manual_backward_with_optimizer(optimizer, backward_batch)
                    train_pbar.update(num_requests)
            # Flush remaining activations
            batch_info = model.backward_cache.get_batch(backward_batch_size, flush=True)
            if batch_info is not None:
                    num_requests, backward_batch = batch_info
                    loss = model.manual_backward_with_optimizer(optimizer, backward_batch)
                    train_pbar.update(num_requests)
             # Evaluate the model
            test_loss, test_accuracy = evaluate_model(model, X_test, y_test)
            loss_history.append(test_loss)
            accuracy_history.append(test_accuracy)
        # Plot loss and accuracy
        
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss_history, marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Test Loss Over Epochs')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, accuracy_history, marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Over Epochs')



def main():
    batch_size = 32
    train_count = 8  # Number of training samples per batch
    epochs = 200
    learning_rate = 0.001
    train_batch_size = 32

    train_loader, total_training, total_inference, test_data = load_iris_data(batch_size=batch_size, train_count=train_count)
    X_test, y_test = test_data
    model = MLPManual()
    model_1 = MLP(model)
    mix_serving(model, train_loader, batch_size, train_batch_size, total_training, total_inference, X_test, y_test, learning_rate, epochs)

    optimizer = optim.Adam(model_1.params(), lr=0.001)  # Set learning rate to 0.001
    # Training loop with mini-batches
    epochs = 200
    loss_history = []
    accuracy_history = []
    batches_1 = load_iris_batches()
    for epoch in range(epochs):
        for batch_X, batch_y in batches_1:
            model_1.train_step(optimizer, batch_X, batch_y)

        test_loss, test_accuracy = evaluate_model(model_1, X_test, y_test)
        loss_history.append(test_loss)
        accuracy_history.append(test_accuracy)

    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss_history, marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('MLP Test Loss Over Epochs')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, accuracy_history, marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('MLP Test Accuracy Over Epochs')
    plt.show()

if __name__ == "__main__":
    main()


