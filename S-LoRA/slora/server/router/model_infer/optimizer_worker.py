import threading, queue, time
from typing import Dict

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

    def run(self) -> None:
        while not self._stop.is_set():
            self._evt.wait()         
            self._evt.clear()
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
                print("Parameter update duration:", time.time()-start)
                
            self.finetuning_adapter_tracker.set(self.finetuning_adapter.lora_dir, True)