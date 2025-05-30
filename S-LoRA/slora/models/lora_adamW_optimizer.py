import torch
import math

class ManualAdamW:
    def __init__(self, param_dict: dict[str, torch.Tensor], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 decay_rate=0.95, min_lr=1e-6):
        self.params = param_dict  # dict[name -> param tensor]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.decay_rate = decay_rate
        self.min_lr = min_lr

        self.state = {
            name: {
                "step": 0,
                "exp_avg": torch.zeros_like(p, dtype=torch.float32),
                "exp_avg_sq": torch.zeros_like(p, dtype=torch.float32),
            }
            for name, p in self.params.items()
        }
    
    @torch.no_grad()
    def step(self, grad_dict: dict[str, torch.Tensor]):
        total_norm_sq = 0.0
        total_drift_sq = 0.0
        found_inf = False
        found_nan = False

        for name, p in self.params.items():
            if name not in grad_dict:
                continue

            # Cast gradient to float32 for stable computation
            g = grad_dict[name].to(device=p.device, dtype=torch.float32)
            if torch.isnan(g).any():
                found_nan = True
            if torch.isinf(g).any():
                found_inf = True
            if found_inf or found_nan:
                print(f"[ManualAdamW] Non-finite gradient in {name}, continuing anyway")

            total_norm_sq += g.norm(2).item() ** 2

            state = self.state[name]
            state["step"] += 1

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            beta1, beta2 = self.betas

            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            step = state["step"]
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            denom = exp_avg_sq.sqrt().add_(self.eps)
            denom = torch.clamp(denom, min=1e-5)

            # === Decayed learning rate ===
            decayed_lr = max(self.min_lr, self.lr * (self.decay_rate ** step))
            step_size = decayed_lr * math.sqrt(bias_correction2) / bias_correction1

            if self.weight_decay > 0:
                p.data.add_(p.data, alpha=-decayed_lr * self.weight_decay)

            # Perform update in float32 and cast result to param dtype
            update = (-step_size) * (exp_avg / denom)
            total_drift_sq += update.pow(2).sum().item()
            p.data.add_(update.to(dtype=p.dtype))

        grad_norm = math.sqrt(total_norm_sq)
        weight_drift = math.sqrt(total_drift_sq)
        print(f"[ManualAdamW] Total weight drift this step: {weight_drift:.6f}")
        return grad_norm, found_inf, found_nan

    def load_to_gpu(self):
        for name, p in self.params.items():
            self.params[name] = p.to('cuda')
            self.state[name]["exp_avg"] = self.state[name]["exp_avg"].to('cuda')
            self.state[name]["exp_avg_sq"] = self.state[name]["exp_avg_sq"].to('cuda')


import torch
import math
from typing import Dict

class ManualAdamW_2:
    """
    A *simple, self-contained* AdamW optimiser that:
      • keeps all internal statistics in fp32
      • works with params stored in fp16 / bf16
      • supports optional weight–decay and exponential LR-decay
    """

    def __init__(
        self,
        params: Dict[str, torch.Tensor],
        lr: float = 2e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        lr_decay: float = 0.0,      # 0 → fixed LR ; e.g. 0.05 ≈ 5 % / step
        min_lr: float = 1e-6,
    ):
        self.params = params               # {name: parameter tensor}
        self.lr0 = lr
        self.betas = betas
        self.eps = eps
        self.wd = weight_decay
        self.lr_decay = lr_decay
        self.min_lr = min_lr

        # fp32 state-tensors (same device as their parameter)
        self.state = {}
        for name, p in params.items():
            dev = p.device
            self.state[name] = dict(
                step      = 0,
                exp_avg   = torch.zeros_like(p, dtype=torch.float32, device=dev),
                exp_avg_sq= torch.zeros_like(p, dtype=torch.float32, device=dev),
            )

    # -------------------------------------------------------------

    @torch.no_grad()
    def step(self, grad_dict: Dict[str, torch.Tensor]) -> None:
        drift_sq_total = 0.0

        for name, p in self.params.items():

            if name not in grad_dict:          # param frozen / no grad
                continue

            g = grad_dict[name].to(device=p.device)  # stable math
            has_nan = torch.isnan(g).any().item()
            has_inf = torch.isinf(g).any().item()
            if has_nan or has_inf:
                msg = f"[ManualAdamW] {'NaN' if has_nan else ''}{'/Inf' if has_inf else ''} in grad of {name}"
                print(msg + "  ➜  zeroing gradient")
                g = torch.zeros_like(g)


            if torch.isfinite(g).logical_not().any():
                print(f"[AdamWManual] non-finite grad in {name} skipped")
                continue

            st      = self.state[name]
            m, v    = st["exp_avg"], st["exp_avg_sq"]
            step    = st["step"] = st["step"] + 1

            beta1, beta2 = self.betas

            m.mul_(beta1).add_(g, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            lr_t = max(self.min_lr,
                       self.lr0 / (1.0 + self.lr_decay * step))

            denom = v.sqrt().add_(self.eps)
            step_size = lr_t * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)

            if self.wd:
                p.data.add_(p.data, alpha=-lr_t * self.wd)

            update = -step_size * (m / denom)          # fp32
            drift_sq_total += update.pow(2).sum().item()

            p.data.add_(update.to(p.dtype))

        print(f"[AdamWManual] ΔW L2 = {math.sqrt(drift_sq_total):.6f}")

    # ------------------------------------------------------------------
    def zero_state(self) -> None:
        for st in self.state.values():
            st["exp_avg"].zero_()
            st["exp_avg_sq"].zero_()