import unittest
import torch
import torch.optim as optim
from models import SimpleTransformerLM

class TestSimpleTransformerLM(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.d_model = 512
        self.nhead = 8
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.model = SimpleTransformerLM(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )
        self.optimizer = optim.Adam(self.model.params(), lr=0.001)
        self.input_ids = torch.randint(0, self.vocab_size, (16, 10))  # batch_size=32, seq_len=10
        self.labels = torch.randint(0, self.vocab_size, (16, 10))  # batch_size=32, seq_len=10
        self.train_filter = torch.tensor([1]*16)  # First 8 samples are for training

    def test_forward_pass(self):
        logits = self.model.forward(self.input_ids)
        self.assertEqual(logits.shape, (16, 10, self.vocab_size))

    def test_forward_no_grad_train_filter(self):
        logits = self.model.forward_no_grad_train_filter(self.input_ids, self.train_filter, self.labels)
        self.assertEqual(logits.shape, (16, 10, self.vocab_size))
        self.assertEqual(len(self.model.backward_cache.activations), 16)

    def test_backward_pass(self):
        logits = self.model.forward(self.input_ids)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.vocab_size), self.input_ids.view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.assertTrue(all(param.grad is not None for param in self.model.params()))

    def test_manual_backward_with_optimizer(self):
        self.model.forward_no_grad_train_filter(self.input_ids, self.train_filter, self.labels)
        batch_data = self.model.backward_cache.get_batch(16, flush=True)[1]
        
        # Save initial weights
        initial_weights = [param.clone() for param in self.model.params()]
        
        loss = self.model.manual_backward_with_optimizer(self.optimizer, batch_data)
        self.assertIsInstance(loss, float)
        
        # Print weight changes
        for i, param in enumerate(self.model.params()):
            weight_change = torch.norm(param - initial_weights[i]).item()
            print(f"Weight change for parameter {i}: {weight_change}")

if __name__ == '__main__':
    unittest.main()