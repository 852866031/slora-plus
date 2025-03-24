import unittest
import torch
import torch.optim as optim
from models import SimpleTransformerLM

class TestSimpleTransformerLM(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.output_size = 2
        self.d_model = 512
        self.nhead = 8
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.model = SimpleTransformerLM(
            vocab_size=self.vocab_size,
            output_size = self.output_size,
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )
        self.optimizer = optim.Adam(self.model.params(), lr=0.001)
        self.input_ids = torch.randint(0, self.vocab_size, (16, 10))  # batch_size=32, seq_len=10
        self.labels = torch.randint(0, self.output_size, (16, 10))  # batch_size=32, seq_len=10
        self.train_filter = torch.tensor([1]*16)  # First 8 samples are for training

    def test_forward_pass(self):
        logits = self.model.forward(self.input_ids)
        self.assertEqual(logits.shape, (16, 10, self.output_size))
    
    def test_manual_forward(self):
        tensor_random = torch.randn(16, 10, 512)
        equal = self.model.decoder_layers[0].manual_forward(tensor_random)
        self.assertEqual(equal, True)

    def test_forward_no_grad_train_filter(self):
        logits = self.model.forward_no_grad_train_filter(self.input_ids, self.train_filter, self.labels)
        self.assertEqual(logits.shape, (16, 10, self.output_size))
        self.assertEqual(len(self.model.backward_cache.activations), 16)

    def test_backward_pass(self):
        logits = self.model.forward(self.input_ids)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.output_size), self.labels.view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.assertTrue(all(param.grad is not None for param in self.model.params()))
    '''
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
    '''
    
    def test_copy_params_from(self):
        torch.manual_seed(42)  # For reproducibility
        vocab_size = 100
        input_ids = torch.randint(0, vocab_size, (2, 10))

        # Create two instances of SimpleTransformerLM
        model1 = SimpleTransformerLM(vocab_size, self.output_size)
        model2 = SimpleTransformerLM(vocab_size, self.output_size)

        # Copy parameters from model1 to model2
        model2.copy_params_from(model1)

        for layer in model1.decoder_layers:
            layer.dropout_layer.eval()
        for layer in model2.decoder_layers:
            layer.dropout_layer.eval()

        # Verify all parameters are identical
        self.assertTrue(torch.equal(model1.embedding.weight, model2.embedding.weight))
        
        for layer1, layer2 in zip(model1.decoder_layers, model2.decoder_layers):
            # Check multi-head attention parameters
            self.assertTrue(torch.equal(layer1.W_q, layer2.W_q))
            self.assertTrue(torch.equal(layer1.W_k, layer2.W_k))
            self.assertTrue(torch.equal(layer1.W_v, layer2.W_v))
            self.assertTrue(torch.equal(layer1.W_o, layer2.W_o))
            
            # Check feedforward network parameters
            self.assertTrue(torch.equal(layer1.W1, layer2.W1))
            self.assertTrue(torch.equal(layer1.b1, layer2.b1))
            self.assertTrue(torch.equal(layer1.W2, layer2.W2))
            self.assertTrue(torch.equal(layer1.b2, layer2.b2))
            
            # Check layer norm parameters
            self.assertTrue(torch.equal(layer1.ln1_weight, layer2.ln1_weight))
            self.assertTrue(torch.equal(layer1.ln1_bias, layer2.ln1_bias))
            self.assertTrue(torch.equal(layer1.ln2_weight, layer2.ln2_weight))
            self.assertTrue(torch.equal(layer1.ln2_bias, layer2.ln2_bias))

        self.assertTrue(torch.equal(model1.output_layer.weight, model2.output_layer.weight))
        self.assertTrue(torch.equal(model1.output_layer.bias, model2.output_layer.bias))
        self.assertTrue(torch.equal(model1.positional_encoding, model2.positional_encoding))


        # Compare forward pass results
        with torch.no_grad():
            output1 = model1.forward(input_ids)
            output2 = model2.forward(input_ids)
            self.assertTrue(torch.allclose(output1, output2, rtol=1e-5, atol=1e-5))

    def test_gradient_computation(self):
        torch.manual_seed(5)
        self.vocab_size = 100
        self.batch_size = 4
        self.seq_len = 8
        
        # Create two identical models
        self.model_manual = SimpleTransformerLM(self.vocab_size, self.output_size)
        self.model_auto = SimpleTransformerLM(self.vocab_size, self.output_size)
        self.model_auto.copy_params_from(self.model_manual)
        
        # Set models to eval mode to disable dropout
        for layer in self.model_manual.decoder_layers:
            layer.dropout_layer.eval()
        for layer in self.model_auto.decoder_layers:
            layer.dropout_layer.eval()
        
        # Create sample input
        self.input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.labels = torch.randint(0, self.output_size, (self.batch_size, self.seq_len))
        self.train_filter = torch.ones(self.batch_size)

        # Forward pass with manual gradients
        self.model_manual.forward_no_grad_train_filter(self.input_ids, self.train_filter, self.labels)
        batch_data = self.model_manual.backward_cache.get_batch(self.batch_size, flush=True)[1]
        optimizer_manual = torch.optim.Adam(self.model_manual.params())
        loss_manual = self.model_manual.manual_backward_with_optimizer(optimizer_manual, batch_data)
        manual_dict = self.model_manual.dict_params()

        # Forward pass with automatic gradients
        optimizer_auto = torch.optim.Adam(self.model_auto.params())
        output_auto = self.model_auto.forward(self.input_ids)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_auto = loss_fn(output_auto.view(-1, self.output_size), self.labels.view(-1))
        optimizer_auto.zero_grad()
        loss_auto.backward()
        auto_dict = self.model_auto.dict_params()

        def compare_tensors(manual_tensor, auto_tensor, path):
            if manual_tensor.grad is None and auto_tensor.grad is None:
                return f"{path}: Both gradients are None"
            elif manual_tensor.grad is None:
                return f"{path}: Manual gradients are None"
            elif auto_tensor.grad is None:
                return f"{path}: Auto gradients are None"
            
            mean_diff = torch.mean(torch.abs(manual_tensor.grad - auto_tensor.grad)).item()
            is_close = torch.allclose(manual_tensor.grad, auto_tensor.grad, rtol=1, atol=1)
            
            return {
                'path': path,
                'mean_diff': mean_diff,
                'matches': is_close
            }

        def compare_dicts(manual_dict, auto_dict, path=""):
            results = []
            
            for key, manual_value in manual_dict.items():
                current_path = f"{path}/{key}" if path else key
                auto_value = auto_dict[key]
                
                if isinstance(manual_value, dict):
                    results.extend(compare_dicts(manual_value, auto_value, current_path))
                elif torch.is_tensor(manual_value):
                    results.append(compare_tensors(manual_value, auto_value, current_path))
            
            return results

        comparison_results = compare_dicts(manual_dict, auto_dict)
        print("\nGradient Comparison Results:")
        print("-" * 80)
        print(f"{'Parameter Path':<50} {'Mean Diff':<12}")
        print("-" * 80)
        
        all_match = True
        for result in comparison_results:
            if isinstance(result, str):
                print(result)
                all_match = False
            else:
                print(f"{result['path']:<50}{result['mean_diff']:<12.6f}")
                if not result['matches']:
                    all_match = False
        
        print("-" * 80)
        
if __name__ == '__main__':
    unittest.main()