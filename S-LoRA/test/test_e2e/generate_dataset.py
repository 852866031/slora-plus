from datasets import load_dataset
import random

# Load the dataset
dataset = load_dataset("dair-ai/emotion", split="train")

# Label ID to emotion name mapping
id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Shuffle the dataset
dataset = dataset.shuffle(seed=30)

# Format each sample with the label name
samples = [
    f"{ex['text']} <label> {id2label[ex['label']]}"
    for ex in dataset.select(range(100))
]

# Save to file
with open("emotion.txt", "w", encoding="utf-8") as f:
    for line in samples:
        f.write(line + "\n")

print("Saved 100 labeled emotion samples to emotion.txt")