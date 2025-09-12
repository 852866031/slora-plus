import threading
import queue
import torch
from enum import Enum

class ActivationWriter(threading.Thread):
    def __init__(self, mem_manager, max_queue_size=64):
        super().__init__(daemon=True, name="ActivationWriter")
        self.mem_manager = mem_manager
        self.task_queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()

        # One vpids list per PageType
        self.vpids_by_type = {}

        self.start()

    def stop(self):
        self._stop_event.set()
        self.task_queue.put(None)  # Sentinel to unblock the thread
        self.join()

    def enqueue(self, layer_id, payload, infer_state, page_type):
        if layer_id >=-1:
            finetune_mask = infer_state.finetune_mask
            # nonzero = torch.nonzero(finetune_mask, as_tuple=False)
            # start_idx = nonzero[0].item() if nonzero.numel() > 0 else None
            # # if start_idx is not None and torch.all(finetune_mask[start_idx:]):
            finetune_mask = infer_state.finetune_mask  # shape: [total_token_num]
            finetune_activations = payload[finetune_mask].clone()
            # else:
            # finetune_activations = input_embs[finetune_mask].clone()
            self.task_queue.put((layer_id, finetune_activations, page_type))
        elif layer_id == -2: #logits or info
            self.task_queue.put((layer_id, payload, page_type))
        return
    

    def run(self):
        while not self._stop_event.is_set():
            task = self.task_queue.get()
            if task is None:
                break  # Sentinel value to exit
            layer_id, payload, page_type = task
            if layer_id == -1: #embedding layer
                num_tokens = payload.size(0)
                prev_total = sum(self.mem_manager.request_token_info)
                num_new_tokens = payload.shape[0]
                self.mem_manager.embedding_output[prev_total : prev_total + num_new_tokens] = payload
            elif layer_id == -2: #logits
                accumlate_len = sum(self.mem_manager.request_token_info)
                for logit in payload:
                    n = logit.size(0)
                    self.mem_manager.logit_tensor[accumlate_len:accumlate_len + n, :].copy_(logit)
                    accumlate_len += n
                    self.mem_manager.request_token_info.append(n)
            else:
                num_tokens = payload.size(0)
                if layer_id == 0:
                    vpids = self.mem_manager.alloc(num_tokens, page_type)
                    self.vpids_by_type[page_type] = vpids
                else:
                    vpids = self.vpids_by_type.get(page_type, None)
                    if vpids is None or len(vpids) != num_tokens:
                        raise RuntimeError(f"No valid vpids for PageType {page_type} at layer {layer_id}")

                self.mem_manager.pin_pages(vpids)
                self.mem_manager.copy_rows_to_layer(layer_id, vpids, payload)
                self.mem_manager.unpin_pages(vpids)