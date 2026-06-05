"""Abstract base class for per-model backward services.

The backward service runs inside the child subprocess spawned by
``backward_process.py``. The child's recv loop dispatches each
``{"op": "step", "acts": ...}`` message to ``BackwardService.step`` and
sends the returned grad dict back over the IPC channel.

Phase 6 only exercises the IPC round-trip — see
``Llama3BackwardService`` for the synthetic-grad stub. Real per-model
backward kernels arrive in Phase 7.
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch


class BackwardService(ABC):
    def __init__(self, model_name: str, device: str, mps_pct: int = 10):
        self.model_name = model_name
        self.device = device
        self.mps_pct = int(mps_pct)

    @abstractmethod
    def step(self, activations: Dict) -> Dict[str, torch.Tensor]:
        """Compute gradients from captured activations.

        Returns a dict of tensors keyed identically to the activations dict.
        """

    @abstractmethod
    def apply_grads(self, grad_dict: Dict) -> None:
        """Apply accumulated grads to the FT adapter (optimizer step)."""

    def start(self) -> None:
        """Optional lifecycle hook called once after construction."""

    def stop(self) -> None:
        """Optional lifecycle hook called before the subprocess exits."""
