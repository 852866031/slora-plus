import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Define simple model (e.g., two parameters)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(3, 3))
        self.weight2 = nn.Parameter(torch.randn(3, 3))

model = SimpleModel()

# Step 2: Create optimizer for model parameters
optimizer = optim.Adam([model.weight1, model.weight2], lr=0.01)

# Step 3: Manually assign random gradients
with torch.no_grad():
    model.weight1.grad = torch.randn_like(model.weight1)
    model.weight2.grad = torch.randn_like(model.weight2)

print("Before update:")
print("Weight1:\n", model.weight1)
print("Weight2:\n", model.weight2)

# Step 4: Optimizer step
optimizer.step()
optimizer.zero_grad()  # Don't forget to zero gradients!

print("\nAfter update:")
print("Weight1:\n", model.weight1)
print("Weight2:\n", model.weight2)