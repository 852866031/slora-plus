import os
import pandas as pd
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from models import SimpleTransformerLM
import matplotlib.pyplot as plt
import numpy as np
import time



def get_sms_spam_dataset(num_samples=None):
    filename = "sms_spam.csv"
    
    # Check if the file exists locally
    if os.path.exists(filename):
        print(f"Loading dataset from {filename}...")
        df = pd.read_csv(filename)
    else:
        print("Downloading SMS Spam dataset...")
        dataset = load_dataset("sms_spam")
        df = pd.DataFrame(dataset["train"])
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
    
    # Select a subset if num_samples is provided
    if num_samples is not None:
        df = df.sample(n=num_samples, random_state=42)
    
    return df.to_dict(orient='records')

def train_custom_tokenizer(dataset, vocab_size=5000):
    texts = [sample['sms'] for sample in dataset]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tokenizer.train_from_iterator(texts, trainer)
    
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))]
    )
    
    return tokenizer

def tokenize_dataset(dataset, tokenizer):
    texts = [sample['sms'] for sample in dataset]
    labels = torch.tensor([1 if sample['label'] == 'spam' else 0 for sample in dataset])
    tokenized = [tokenizer.encode(text).ids for text in texts]
    
    max_length = 64
    input_ids = torch.zeros(len(tokenized), max_length, dtype=torch.long)
    for i, token_list in enumerate(tokenized):
        length = min(len(token_list), max_length)
        input_ids[i, :length] = torch.tensor(token_list[:length])
    
    return input_ids, labels

def create_dataloader(input_ids, labels, batch_size=16):
    dataset = TensorDataset(input_ids, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(dataloader, vocab_size, num_epochs=5, learning_rate=1e-4, batch_size=32):    
    model = SimpleTransformerLM(vocab_size=vocab_size, output_size=2)
    optimizer = optim.Adam(model.params(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    print("Starting training...")
    loss_list = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_input_ids, batch_labels in dataloader:
            logits = model.forward(batch_input_ids)  # Shape: (batch_size, seq_len, output_size)
            logits = logits[:, -1, :]  # Take the last token's logits for classification
            
            loss = loss_fn(logits, batch_labels)  # Ensure correct shape alignment
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        loss_list.append(epoch_loss)
    
    print("Training complete.")
    return model, loss_list

def train_model_manual(dataloader, vocab_size, num_epochs=5, learning_rate=1e-4, batch_size=32):
    # Initialize model and optimizer
    model = SimpleTransformerLM(vocab_size=vocab_size, output_size=2)
    optimizer = optim.Adam(model.params(), lr=learning_rate)
    loss_list = []
    print("Starting training with manual backward...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_input_ids, batch_labels in dataloader:
            # Create train filter (all 1s since we're training on all samples)
            train_filter = torch.ones(batch_input_ids.size(0))
            
            # Forward pass with activation caching
            logits = model.forward_no_grad_train_filter(batch_input_ids, train_filter, batch_labels)
            print("Forward pass complete.")
            # Get cached activations and perform manual backward
            batch_data = model.backward_cache.get_batch(batch_size, flush=True)
            if batch_data is not None:
                num_requests, activation_data = batch_data
                loss = model.manual_backward_with_optimizer(optimizer, activation_data)
                print("Backward pass complete.")
                epoch_loss += loss
        
        print(f"----- Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        loss_list.append(epoch_loss)
    
    print("Training complete.")
    return model, loss_list

def plot_training_loss(loss_list, title="Training Loss Over Time"):
    losses = np.array(loss_list)
    plt.figure(figsize=(10, 6))
    plt.xticks(range(1, len(losses)+1, 1))
    plt.plot(range(1, len(losses)+1, 1), losses, 'b-', label='Training Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
                    
num_samples = 32*10
batch_size=32
dataset = get_sms_spam_dataset(num_samples)
tokenizer = train_custom_tokenizer(dataset, vocab_size=5000)  # Train custom tokenizer
vocab_size = tokenizer.get_vocab_size()
print(f"Using Reduced Vocabulary Size: {vocab_size}")
input_ids, labels = tokenize_dataset(dataset, tokenizer)
dataloader = create_dataloader(input_ids, labels, batch_size)
start = time.time()
model, loss_list = train_model_manual(dataloader, vocab_size)
print(f"Manual Training done in {time.time() - start:.4f} seconds")
start = time.time()
model, loss_list_1 = train_model(dataloader, vocab_size)
print(f"Torch Training done in {time.time() - start:.4f} seconds")
print(loss_list)
plot_training_loss(loss_list)
plot_training_loss(loss_list_1)