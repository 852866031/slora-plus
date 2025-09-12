import threading, queue, time
from typing import Dict, Set, Tuple, Optional
import torch
import torch.nn as nn

class OptimizerWorker(threading.Thread):
    _STOP = object()

    def __init__(self, finetuning_optimizer, finetuning_adapter, finetuning_adapter_tracker, daemon=True):
        super().__init__(daemon=daemon, name="OptimizerWorker")
        self.finetuning_optimizer = finetuning_optimizer
        self.finetuning_adapter = finetuning_adapter
        self._q: queue.Queue = queue.Queue()
        self._evt = threading.Event()
        self._stop = threading.Event()
        self.finetuning_adapter_tracker = finetuning_adapter_tracker

    def enqueue(self, grads) -> None:
        self._q.put(grads)
        self._evt.set()
        self.finetuning_adapter_tracker.set(self.finetuning_adapter.lora_dir, False)

    def stop(self) -> None:
        self._q.put(self._STOP)
        self._evt.set()
        self.join()

    # ---------------- Device reporting helpers ----------------

    def _gather_optimizer_devices(self) -> Tuple[Set[str], Set[Tuple[str, Optional[int]]]]:
        """Return (device_types, detailed_devices) from optimizer params + state."""
        dev_types: Set[str] = set()
        detailed: Set[Tuple[str, Optional[int]]] = set()

        opt = self.finetuning_optimizer
        # Param devices
        for group in opt.param_groups:
            for p in group.get("params", []):
                if not torch.is_tensor(p):
                    continue
                d = p.device
                dev_types.add(d.type)
                detailed.add((d.type, d.index if d.type == "cuda" else None))
                # State tensors (e.g., exp_avg, exp_avg_sq)
                st = opt.state.get(p, {})
                for v in st.values():
                    if torch.is_tensor(v):
                        dv = v.device
                        dev_types.add(dv.type)
                        detailed.add((dv.type, dv.index if dv.type == "cuda" else None))
        return dev_types, detailed

    def _gather_adapter_devices(self) -> Tuple[Set[str], Set[Tuple[str, Optional[int]]]]:
        """Return (device_types, detailed_devices) from finetuning_adapter params (if it’s an nn.Module)."""
        dev_types: Set[str] = set()
        detailed: Set[Tuple[str, Optional[int]]] = set()

        mod = self.finetuning_adapter
        if isinstance(mod, nn.Module):
            for p in mod.parameters(recurse=True):
                d = p.device
                dev_types.add(d.type)
                detailed.add((d.type, d.index if d.type == "cuda" else None))
        return dev_types, detailed

    def _report_compute_device(self) -> None:
        """Print a human-friendly summary of where updates will run."""
        opt_types, opt_detailed = self._gather_optimizer_devices()
        adp_types, adp_detailed = self._gather_adapter_devices()

        all_types = opt_types.union(adp_types)
        all_detailed = opt_detailed.union(adp_detailed)

        # Build message
        if not all_types:
            msg = "Device: Unknown (no parameters found)"
        elif all_types == {"cpu"}:
            msg = "Device: CPU (all optimizer and adapter tensors on CPU)"
        elif all_types == {"cuda"}:
            # List unique cuda indices if any
            cuda_ids = sorted({idx for (t, idx) in all_detailed if t == "cuda"})
            if len(cuda_ids) == 1:
                msg = f"Device: GPU (cuda:{cuda_ids[0]})"
            else:
                ids_str = ", ".join(f"cuda:{i}" for i in cuda_ids)
                msg = f"Device: GPU (multiple devices: {ids_str})"
        else:
            # Mixed CPU/GPU
            # Optional: show brief breakdown
            parts = []
            if "cpu" in all_types:
                parts.append("CPU")
            cuda_ids = sorted({idx for (t, idx) in all_detailed if t == "cuda"})
            if cuda_ids:
                parts.append("GPU " + ", ".join(f"cuda:{i}" for i in cuda_ids))
            msg = "Device: Mixed (" + " | ".join(parts) + ")"

        print(f"[OptimizerWorker] {msg}")

    # ---------------- Main loop ----------------

    def run(self) -> None:
        while not self._stop.is_set():
            self._evt.wait()
            self._evt.clear()

            # Report devices at wake-up
            #self._report_compute_device()

            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
                if item is self._STOP:
                    self._stop.set()
                    break

                grads = item
                start = time.time()
                self.finetuning_optimizer.step()
                self.finetuning_optimizer.zero_grad(set_to_none=True)
                self.finetuning_adapter.unpack_all_combined_weights()
                print("Parameter update duration:", time.time() - start)

            self.finetuning_adapter_tracker.set(self.finetuning_adapter.lora_dir, True)