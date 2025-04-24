import torch

class LoRAAdamOptimizer:
    def __init__(
        self,
        finetuning_adapter,
        lr=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
        max_grad_norm=None,
    ):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.state = {}
        self.step_count = 0

        not_on_gpu = getattr(finetuning_adapter.layers[0], 'q_lora_A', None) is None
        for layer_id, layer in enumerate(finetuning_adapter.layers):
            for name in [
                "q_lora_A", "q_lora_B",
                "k_lora_A", "k_lora_B",
                "v_lora_A", "v_lora_B",
                "o_lora_A", "o_lora_B"
            ]:
                if not_on_gpu:
                    param = getattr(layer, name + "_home")
                else:
                    param = getattr(layer, name)

                # Use float32 for m and v
                self.state[f"{layer_id}.{name}.m"] = torch.zeros_like(param, dtype=torch.float32)
                self.state[f"{layer_id}.{name}.v"] = torch.zeros_like(param, dtype=torch.float32)
        print(f"LoRAAdamOptimizer initialized.")

    def update(self, grads: dict, lora_weights, layer_id: int):
        for name, grad in grads.items():
            key = f"{layer_id}.{name}"
            param = getattr(lora_weights, name)

            # Convert gradient to float32
            grad_fp32 = grad.float()

            # === NaN/Inf check for gradients ===
            if torch.isnan(grad_fp32).any() or torch.isinf(grad_fp32).any():
                print(f"[Warning] Gradient for {key} contains NaN or Inf. Skipping update.")
                continue

            # === Optional gradient clipping ===
            if self.max_grad_norm is not None:
                grad_norm = grad_fp32.norm()
                if grad_norm > self.max_grad_norm:
                    scale = float(self.max_grad_norm / (grad_norm + 1e-6))
                    grad_fp32.mul_(scale)

            # Update moving averages
            m = self.state[f"{key}.m"]
            v = self.state[f"{key}.v"]

            m.mul_(self.beta1).add_(grad_fp32, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad_fp32, grad_fp32, value=1 - self.beta2)

            m_hat = m / (1 - self.beta1 ** self.step_count)
            v_hat = v / (1 - self.beta2 ** self.step_count)

            # === Avoid NaN in update step ===
            denom = v_hat.sqrt() + self.eps
            delta_w = self.lr * m_hat / denom
            delta_w = torch.nan_to_num(delta_w, nan=0.0, posinf=1.0, neginf=-1.0)

            # === Final safety check ===
            if torch.isnan(delta_w).any() or torch.isinf(delta_w).any():
                print(f"[Warning] delta_w for {key} contains NaN or Inf. Skipping update.")
                continue

            # Apply update
            param.data -= delta_w.to(param.dtype)


    def step(self):
        self.step_count += 1

    def load_to_gpu(self):
        for key in self.state:
            if isinstance(self.state[key], torch.Tensor):
                self.state[key] = self.state[key].to("cuda")