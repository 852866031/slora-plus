from enum import Enum, auto
import hashlib
import queue
from typing import List, Tuple
import numpy as np
from pyparsing import Iterable
import torch
from collections import deque
import triton
import triton.language as tl
import datetime, os, errno
import threading
import nvtx 
import time


def tensor_hash(t: torch.Tensor, algo="sha256") -> str:
    h = hashlib.new(algo)
    h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()
    
def get_tensor_size_kb(numel: int, dtype: torch.dtype) -> float:
        dtype_size_map = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float64: 8,
            torch.int8: 1,
            torch.int16: 2,
            torch.int32: 4,
            torch.int64: 8,
            torch.bool: 1,
        }
        bytes_per_element = dtype_size_map[dtype]
        return (numel * bytes_per_element) / 1024


class PageType(Enum):
    KV_CACHE        = auto()
    ADAPTER_WEIGHT  = auto()
    ATTENTION_INPUT_ACTIVATION     = auto() 
    FFN_INPUT_ACTIVATION      = auto()
    EMBEDDING = auto()
class PageTable:
    """Tensor-based page table for fast vectorized lookups."""

    _DEV_FREE = 0
    _DEV_GPU  = 1
    _DEV_CPU  = 2

    def __init__(self, max_gpu_slots: int, initial_max_vpid: int = None, device: str = "cuda"):
        if initial_max_vpid is None:
            initial_max_vpid = max_gpu_slots * 8  # headroom

        self.vpid_counter = 5
        self.device = device

        # VPID metadata tensors
        self.vpid_device   = torch.zeros(initial_max_vpid, dtype=torch.long,  device=device)
        self.vpid_phys_idx = torch.full((initial_max_vpid,), -1, dtype=torch.long, device=device)
        self.vpid_type     = torch.full((initial_max_vpid,), -1, dtype=torch.long, device=device)

        # GPU slot → VPID mapping
        self.gpu_slot_to_vpid = torch.full((max_gpu_slots,), -1, dtype=torch.long, device=device)

        self.cpu_resident_count = 0

        # Free-slot queue
        self.free_phys_pages_queue = queue.Queue(maxsize=max_gpu_slots)
        for i in range(max_gpu_slots):
            self.free_phys_pages_queue.put(i)

    # ---------- internal helpers ----------

    def _ensure_vpid_capacity(self, vpid: int):
        if vpid < self.vpid_device.numel():
            return
        new_cap = max(vpid + 1, self.vpid_device.numel() * 2)

        def grow(old: torch.Tensor, fill, dtype=None):
            dtype = dtype or old.dtype
            new = torch.full((new_cap,), fill, dtype=dtype, device=old.device)
            new[:old.numel()] = old
            return new

        self.vpid_device   = grow(self.vpid_device,   self._DEV_FREE, dtype=torch.int8)
        self.vpid_phys_idx = grow(self.vpid_phys_idx, -1,             dtype=torch.int32)
        self.vpid_type     = grow(self.vpid_type,     -1,             dtype=torch.int16)

    @staticmethod
    def _dev_int_to_label(code: int) -> str:
        return {1: "gpu", 2: "cpu"}.get(code, "free")

    # ---------- public API ----------

    def get_next_vpid(self) -> int:
        vpid = self.vpid_counter
        self.vpid_counter += 1
        self._ensure_vpid_capacity(vpid)
        return vpid

    def set_type(self, vpid: int, ptype: PageType):
        self._ensure_vpid_capacity(vpid)
        self.vpid_type[vpid] = int(ptype.value)

    def get_type(self, vpid: int) -> PageType:
        val = int(self.vpid_type[vpid].item())
        if val < 0:
            raise KeyError(f"Type for vpid {vpid} not set.")
        return PageType(val)

    def set_gpu_mapping(self, vpid: int, gpu_idx: int):
        self._ensure_vpid_capacity(vpid)
        # clear previous slot if any
        prev_dev = int(self.vpid_device[vpid])
        if prev_dev == self._DEV_GPU:
            prev_slot = int(self.vpid_phys_idx[vpid])
            if prev_slot >= 0:
                self.gpu_slot_to_vpid[prev_slot] = -1

        self.vpid_device[vpid]   = self._DEV_GPU
        self.vpid_phys_idx[vpid] = gpu_idx
        self.gpu_slot_to_vpid[gpu_idx] = vpid

    def set_cpu_mapping(self, vpid: int):
        self._ensure_vpid_capacity(vpid)
        if int(self.vpid_device[vpid]) == self._DEV_GPU:
            prev_slot = int(self.vpid_phys_idx[vpid])
            if prev_slot >= 0:
                self.gpu_slot_to_vpid[prev_slot] = -1
        self.vpid_device[vpid]   = self._DEV_CPU
        self.vpid_phys_idx[vpid] = vpid  # placeholder

    def get_location_index(self, vpid: int) -> Tuple[str, int]:
        dev = int(self.vpid_device[vpid])
        if dev == self._DEV_GPU:
            return ("gpu", int(self.vpid_phys_idx[vpid]))
        elif dev == self._DEV_CPU:
            return ("cpu", vpid)
        raise KeyError(f"VPID {vpid} has no mapping (free).")

    def remove(self, vpid: int):
        dev = int(self.vpid_device[vpid])
        idx = int(self.vpid_phys_idx[vpid])
        if dev == self._DEV_GPU and idx >= 0:
            self.gpu_slot_to_vpid[idx] = -1
        self.vpid_device[vpid]   = self._DEV_FREE
        self.vpid_phys_idx[vpid] = -1
        self.vpid_type[vpid]     = -1

    def vpids_to_types(self, vpids: Iterable[int]) -> List[PageType]:
        vpids_t = torch.as_tensor(vpids, device=self.device, dtype=torch.long)
        vals = self.vpid_type[vpids_t]
        uniq_vals = torch.unique(vals)
        return [PageType(v) for v in uniq_vals.tolist() if v >= 0]

    def evictable_gpu_slots(self):
        """Yields (gpu_idx, vpid, page_type) for every page resident on GPU."""
        occupied = torch.nonzero(self.gpu_slot_to_vpid != -1, as_tuple=False).flatten().tolist()
        for gpu_idx in occupied:
            vpid = int(self.gpu_slot_to_vpid[gpu_idx])
            yield gpu_idx, vpid, self.get_type(vpid)


class UnifiedMemoryAllocator:
    def __init__(self, head_num, head_dim, vocab_size, layer_num: int, max_pool_size: int, dtype=torch.float16, device='cuda', log_path=None):
        self.head_dim = head_dim
        self.head_num = head_num
        self.hidden_dim  = head_num * head_dim
        self.layer_num   = layer_num
        self.device      = device
        self.dtype      = dtype
        self.vocab_size = vocab_size
        self.tot_size = int(max_pool_size * 1024 * 1024 / self.layer_num / get_tensor_size_kb(self.head_num * self.head_dim, self.dtype))
        self.gpu_pools = [
            torch.empty((self.tot_size, self.head_num, self.head_dim),
                        device=self.device, dtype=self.dtype)
            for _ in range(self.layer_num)
        ]
        self.cpu_pools = [{} for _ in range(self.layer_num)]
        
        # mem_state[i] == 0 → free, 1 → occupied (GPU only)
        # self.mem_state = torch.zeros(self.tot_size,
        #                              dtype=torch.int8, device='cpu')
        initial_max_vpid = self.tot_size * 4
        self.pinned_pages = torch.zeros(
            initial_max_vpid, dtype=torch.bool, device=self.device)
        # Central page table
        self.page_table = PageTable(
            max_gpu_slots=self.tot_size,
            initial_max_vpid=initial_max_vpid,   # headroom
            device="cuda"
        )       
        self.vpid_device   = self.page_table.vpid_device
        self.vpid_phys_idx = self.page_table.vpid_phys_idx
        # Finetuning-related
        self.request_token_info = []    # request_token_info = [num_finetune_tokens_request_1, ...]
        self.activation_page_indices = []  # list of tuples: (FFN_input_phys_ids, ATTENTION_input_phys_ids)
        self.finetune_input_ids = []
        self.alignment_completion_masks = []
        self.alignment_labels= []
        self.finetune_logits_per_request = []
        self.reference_logits_per_request = []
        # Logging
        self.log_path = log_path
        #self.logging_enabled = log_path is not None
        self.logging_enabled = False
        self.accessed_layers = None
        self.last_accessed_vpids = None
        # Thread-safety
        self.page_table_lock = threading.RLock()  # serialize page table accesses
        self.log_lock = threading.RLock()  # serialize file writes (optional)
        self.thread_pool_dict = {}
        self.thread_count = 0
        if self.logging_enabled:
            self.t0_ns = time.perf_counter_ns()
            if os.path.exists(self.log_path):
                os.remove(self.log_path)
        self.shared_transformer_out_activations = None
        self.shared_attention_out_activations = None
        self.embedding_output = None
        self.max_finetuning_tokens = 512
        self.init_shared_activation_memory()
        self.gpu_b_loc_key   = None
        self.gpu_b_loc_value = None
    
    def init_shared_activation_memory(self):
        self.shared_transformer_out_activations = [
            torch.zeros((self.max_finetuning_tokens, self.head_num * self.head_dim),
                         dtype=self.dtype, device=self.device)
            for _ in range(self.layer_num)
        ]
        self.shared_attention_out_activations = [
            torch.zeros((self.max_finetuning_tokens, self.head_num * self.head_dim),
                         dtype=self.dtype, device=self.device)
            for _ in range(self.layer_num)
        ]
        self.embedding_output = torch.zeros((self.max_finetuning_tokens, self.head_num * self.head_dim),
                                        dtype=self.dtype, device=self.device)
        
        self.concat_input_ids = torch.zeros(self.max_finetuning_tokens*2, dtype=torch.int64, device=self.device) 
        self.logit_tensor = torch.zeros((self.max_finetuning_tokens, self.vocab_size), dtype=self.dtype, device=self.device)
    
    def share_activation_dict(self):
        return {
            "logit_tensor": self.logit_tensor,
            "concat_input_ids": self.concat_input_ids,
            "transformer_out_activations": self.shared_transformer_out_activations,
            "attention_out_activations": self.shared_attention_out_activations,
            "input_layer_output": self.embedding_output
        }
    
    def export_requests_info(self):
        with self.page_table_lock:
            self.get_concatenated_finetune_input_ids()
            self.saved_layer_0_activations = None
            for layer_id in range(self.layer_num):
                self.fill_activations_by_layer(layer_id, PageType.FFN_INPUT_ACTIVATION, 
                                            self.shared_attention_out_activations[layer_id])
                self.fill_activations_by_layer(layer_id, PageType.ATTENTION_INPUT_ACTIVATION, 
                                            self.shared_transformer_out_activations[layer_id])
            requests_info_dict = {
                "request_token_info": self.request_token_info,
                "finetuning_logits_per_request": self.finetune_logits_per_request,
            }
            return requests_info_dict
    
    def _num_free_gpu_slots(self) -> int:
        """Return number of currently free GPU slots."""
        with self.page_table_lock:
            # Queue tracks available slots; use its current size.
            return self.page_table.free_phys_pages_queue.qsize()

    def _get_free_gpu_slots(self, num_needed: int) -> list[int]:
        """
        Return a list of `num_needed` free GPU slot indices.
        Raise RuntimeError if not enough slots are available.
        """
        with self.page_table_lock:
            if self.page_table.free_phys_pages_queue.qsize() < num_needed:
                raise RuntimeError(
                    f"Requested {num_needed} free slots, "
                    f"but only {self.page_table.free_phys_pages_queue.qsize()} available."
                )
            return [self.page_table.free_phys_pages_queue.get() for _ in range(num_needed)]

        
    def alloc(self, num_pages: int, page_type: PageType) -> torch.Tensor:
        """
        Allocate `num_pages` GPU pages for the given page type.
        Returns a tensor of vpid (int64) allocated.
        """
        with self.page_table_lock:
            free_now = self._num_free_gpu_slots()
            shortfall = max(0, num_pages - free_now)
            if shortfall:
                self._pages_out(self._find_victim_pages(shortfall))
            # get free physical slots
            free_slots = torch.tensor(
                [self.page_table.free_phys_pages_queue.get() for _ in range(num_pages)],
                dtype=torch.long, device="cuda"
            )
            # generate vpid tensor range
            start_vpid = self.page_table.vpid_counter
            vpids = torch.arange(start_vpid, start_vpid + num_pages, dtype=torch.long, device="cuda")
            self.page_table.vpid_counter += num_pages

            # set page types
            self.page_table.vpid_type[vpids] = int(page_type.value)
            # map them to GPU
            self.page_table.vpid_device[vpids] = self.page_table._DEV_GPU
            self.page_table.vpid_phys_idx[vpids] = free_slots

            # update reverse mapping table
            self.page_table.gpu_slot_to_vpid[free_slots] = vpids
            return vpids
        
    def alloc_contiguous_kv(self, need_size: int, page_type: PageType):
        """
        Allocate contiguous GPU pages for K/V.

        Fast path: one contiguous block of size 2*need_size (split into K/V halves).
        Fallback: two disjoint contiguous blocks of size need_size each.

        Returns:
            ((vpids_k, start_k, end_k), (vpids_v, start_v, end_v))
            or None if not enough contiguous space exists.
        """
        with self.page_table_lock:
            gpu_slot_to_vpid = self.page_table.gpu_slot_to_vpid
            free_mask = (gpu_slot_to_vpid == -1)
            free_mask_np = free_mask.cpu().numpy()
            type_code = int(page_type.value)
            DEV_GPU = self.page_table._DEV_GPU

            need_total = need_size * 2
            run = 0
            start_double = None

            # Pass 1: try to find one big contiguous region
            for i, free in enumerate(free_mask_np):
                if free:
                    run += 1
                    if run == need_total:
                        start_double = i - need_total + 1
                        break
                else:
                    run = 0

            if start_double is not None:
                # ✅ Single large block case
                start_idx = start_double
                end_idx = start_idx + need_total
                contig_all = torch.arange(start_idx, end_idx, dtype=torch.long, device="cuda")

                # Allocate all VPIDs in one batch
                vpids_all = torch.tensor(
                    [self.page_table.free_phys_pages_queue.get() for _ in range(need_total)],
                    dtype=torch.long, device="cuda"
                )

                # Write metadata once
                self.page_table.vpid_type[vpids_all] = type_code
                self.page_table.vpid_device[vpids_all] = DEV_GPU
                self.page_table.vpid_phys_idx[vpids_all] = contig_all
                gpu_slot_to_vpid[contig_all] = vpids_all
                self.pinned_pages[vpids_all] = True

                # Split into K/V halves
                vpids_k = vpids_all[:need_size]
                vpids_v = vpids_all[need_size:]
                start_k, end_k = start_idx, start_idx + need_size
                start_v, end_v = end_idx - need_size, end_idx

                return vpids_k, start_k, end_k, vpids_v, start_v, end_v

            # ❌ Fallback: find two disjoint contiguous runs
            found_blocks = []
            run = 0
            for i, free in enumerate(free_mask_np):
                if free:
                    run += 1
                    if run == need_size:
                        s = i - need_size + 1
                        e = i + 1
                        found_blocks.append((s, e))
                        run = 0  # restart for disjoint block
                        if len(found_blocks) == 2:
                            break
                else:
                    run = 0

            if len(found_blocks) < 2:
                return None

            (start_k, end_k), (start_v, end_v) = found_blocks
            contig_k = torch.arange(start_k, end_k, dtype=torch.long, device="cuda")
            contig_v = torch.arange(start_v, end_v, dtype=torch.long, device="cuda")

            vpids_k = torch.tensor(
                [self.page_table.free_phys_pages_queue.get() for _ in range(need_size)],
                dtype=torch.long, device="cuda"
            )
            vpids_v = torch.tensor(
                [self.page_table.free_phys_pages_queue.get() for _ in range(need_size)],
                dtype=torch.long, device="cuda"
            )

            for vpids, contig_idxs in ((vpids_k, contig_k), (vpids_v, contig_v)):
                self.page_table.vpid_type[vpids] = type_code
                self.page_table.vpid_device[vpids] = DEV_GPU
                self.page_table.vpid_phys_idx[vpids] = contig_idxs
                gpu_slot_to_vpid[contig_idxs] = vpids
                self.pinned_pages[vpids] = True

            return vpids_k, start_k, end_k, vpids_v, start_v, end_v

    def free(self, vpids: torch.Tensor):
        """Free a tensor of vpids and release associated GPU slots."""
        with self.page_table_lock:
            # Normalize to tensor
            if not isinstance(vpids, torch.Tensor):
                vpids = torch.as_tensor(vpids, dtype=torch.long, device="cuda")
            if vpids.numel() == 0:
                return
            if vpids.device.type != "cuda":
                vpids = vpids.to("cuda")
            vpid_device = self.page_table.vpid_device
            vpid_phys_idx = self.page_table.vpid_phys_idx
            gpu_mask = (vpid_device[vpids] == self.page_table._DEV_GPU)
            if gpu_mask.any():
                gpu_vpids = vpids[gpu_mask]
                gpu_idxs = vpid_phys_idx[gpu_vpids].to("cpu")
                for idx in gpu_idxs.tolist():
                    self.page_table.free_phys_pages_queue.put(idx)
                self.page_table.gpu_slot_to_vpid[gpu_idxs] = -1
            cpu_mask  = (vpid_device[vpids] == self.page_table._DEV_CPU)
            self.page_table.cpu_resident_count -= int(cpu_mask.sum().item())
            for layer_pool in self.cpu_pools:
                for vpid in vpids.tolist():
                    layer_pool.pop(vpid, None)
            self.pinned_pages[vpids] = False
            vpid_device[vpids] = self.page_table._DEV_FREE
            vpid_phys_idx[vpids] = -1
            self.page_table.vpid_type[vpids] = 0  # optional: 0 means unused
            used_gpu = self.tot_size - self._num_free_gpu_slots()
            cpu_pages = self.page_table.cpu_resident_count
            #print(f"After FREE: used_gpu_slots={used_gpu}, cpu_pages={cpu_pages}")

    def _find_victim_pages(self, num_needed: int, priority: dict = None):
        with self.page_table_lock:
            if priority is None:
                priority = {
                    PageType.FFN_INPUT_ACTIVATION: 0,
                    PageType.ATTENTION_INPUT_ACTIVATION: 1,
                    PageType.KV_CACHE: 2,
                    PageType.ADAPTER_WEIGHT: 3,
                }

            # ── 1. Identify all GPU-resident vpids
            vpid_device = self.page_table.vpid_device
            vpid_type   = self.page_table.vpid_type
            phys_idx    = self.page_table.vpid_phys_idx
            gpu_mask    = (vpid_device == self.page_table._DEV_GPU)

            gpu_vpids   = torch.nonzero(gpu_mask, as_tuple=False).flatten()
            if gpu_vpids.numel() == 0:
                raise RuntimeError("No GPU pages to evict.")

            # ── 2. Filter out pinned pages
            unpinned_mask = ~self.pinned_mask[gpu_vpids]
            evictable_vpids = gpu_vpids[unpinned_mask]

            if evictable_vpids.numel() < num_needed:
                raise RuntimeError(
                    f"Not enough evictable GPU pages: needed {num_needed}, found {evictable_vpids.numel()}."
                )

            # ── 3. Map types → numeric priorities
            vpid_types = vpid_type[evictable_vpids]
            type_names = [PageType(t.item()) for t in vpid_types]  # for logging
            priorities = torch.tensor([priority[PageType(t.item())] for t in vpid_types], dtype=torch.long)

            # ── 4. Sort by priority
            sorted_idx = torch.argsort(priorities)
            victims_vpids = evictable_vpids[sorted_idx[:num_needed]]
            victims_gpu_idxs = phys_idx[victims_vpids].to("cuda")

            # ── 5. Logging and paging-out
            if self.logging_enabled:
                victim_counts = {}
                for t in type_names:
                    victim_counts[t] = victim_counts.get(t, 0) + 1
                self._log(f"Victim pages selected: {victim_counts}")

            self._pages_out(victims_gpu_idxs.tolist())
            return victims_gpu_idxs
    
    def _pages_out(self, gpu_idxs: torch.Tensor):
        """Evict the specified GPU slots to CPU storage."""
        with self.page_table_lock:
            if not isinstance(gpu_idxs, torch.Tensor):
                gpu_idxs = torch.as_tensor(gpu_idxs, dtype=torch.long, device="cuda")
            if gpu_idxs.numel() == 0:
                return

            print("##### Paging out", gpu_idxs.numel(), "pages from GPU to CPU #####")
            # ── 1. Resolve vpids and filter valid entries ──────────────────────────────
            vpid = self.page_table.gpu_slot_to_vpid[gpu_idxs]
            valid_mask = vpid >= 0
            vpid = vpid[valid_mask]
            gpu_idxs = gpu_idxs[valid_mask]
            if vpid.numel() == 0:
                return

            # ── 2. Copy data to CPU pools ─────────────────────────────────────────────
            for layer in range(self.layer_num):
                layer_gpu = self.gpu_pools[layer]
                layer_cpu = self.cpu_pools[layer]
                # Copy each tensor to CPU if not already cached
                for i, vid in enumerate(vpid.tolist()):
                    if vid not in layer_cpu:
                        layer_cpu[vid] = layer_gpu[gpu_idxs[i]].to("cpu").clone()

            # ── 3. Update mapping tensors ─────────────────────────────────────────────
            self.page_table.vpid_device[vpid] = self.page_table._DEV_CPU
            self.page_table.vpid_phys_idx[vpid] = -1
            self.page_table.gpu_slot_to_vpid[gpu_idxs] = -1
            self.page_table.cpu_resident_count += vpid.numel()
            # Return GPU slots to free queue
            for idx in gpu_idxs.tolist():
                self.page_table.free_phys_pages_queue.put(idx)

    def _pages_in(self, vpids: torch.Tensor, priority=None):
        """Move CPU-resident pages back to GPU slots."""
        with self.page_table_lock:
            if not isinstance(vpids, torch.Tensor):
                vpids = torch.as_tensor(vpids, dtype=torch.long, device="cuda")
            if vpids.numel() == 0:
                return

            self._log(f"PAGING IN: {vpids.numel()} CPU pages → GPU")

            # ── 1. Ensure enough GPU slots are available ─────────────────────────────
            free_now = self._num_free_gpu_slots()
            num_needed = vpids.numel() - free_now
            if num_needed > 0:
                self._pages_out(self._find_victim_pages(num_needed, priority))

            # ── 2. Allocate free slots ────────────────────────────────────────────────
            free_slots = torch.as_tensor(self._get_free_gpu_slots(vpids.numel()), dtype=torch.long)

            # ── 3. Copy tensors from CPU → GPU ────────────────────────────────────────
            for layer in range(self.layer_num):
                layer_gpu = self.gpu_pools[layer]
                layer_cpu = self.cpu_pools[layer]
                for vid, gpu_idx in zip(vpids.tolist(), free_slots.tolist()):
                    cpu_tensor = layer_cpu[vid]
                    layer_gpu[gpu_idx].copy_(cpu_tensor)

            # ── 4. Update mapping tensors ─────────────────────────────────────────────
            self.page_table.vpid_device[vpids] = self.page_table._DEV_GPU
            self.page_table.vpid_phys_idx[vpids] = free_slots
            self.page_table.gpu_slot_to_vpid[free_slots] = vpids
            self.page_table.cpu_resident_count -= vpids.numel()

            # Return the slots for convenience
            return free_slots

    def alloc_cpu(self, num_pages: int, page_type: PageType) -> torch.Tensor:
        """Allocate `num_pages` pages directly in CPU memory."""
        with self.page_table_lock:
            self._log(f"ALLOC_CPU: {num_pages} pages type={page_type.name}")
            start_vpid = self.page_table.vpid_counter
            vpids = torch.arange(start_vpid, start_vpid + num_pages, dtype=torch.long)
            self.page_table.vpid_counter += num_pages

            # ── 1. Set metadata ────────────────────────────────────────────────
            self.page_table.vpid_type[vpids] = int(page_type.value)
            self.page_table.vpid_device[vpids] = self.page_table._DEV_CPU
            self.page_table.vpid_phys_idx[vpids] = -1

            # ── 2. Allocate actual tensors for each layer ───────────────────────
            for layer in range(self.layer_num):
                layer_pool = self.cpu_pools[layer]
                for vpid in vpids.tolist():
                    layer_pool[vpid] = torch.empty(
                        (self.head_num, self.head_dim),
                        dtype=self.dtype,
                        device="cuda"
                    )
            return vpids

    def pin_pages(self, vpids):
        with self.page_table_lock:
            self.pinned_pages.index_fill_(0, vpids, True)
        # with self.page_table_lock:
        #     if not isinstance(vpids, torch.Tensor):
        #         raise ValueError("vpids must be a torch.Tensor")
        #     if vpids.numel() == 0:
        #         return
        #     # set mask positions to True
            

    def unpin_pages(self, vpids):
        with self.page_table_lock:
            self.pinned_pages.index_fill_(0, vpids, False)
        # """Unmark pages as pinned, allowing eviction."""
        # with self.page_table_lock:
        #     if not isinstance(vpids, torch.Tensor):
        #         raise ValueError("vpids must be a torch.Tensor")
        #     if vpids.numel() == 0:
        #         return
        #     # set mask positions to False
            
    
    def get_concatenated_finetune_input_ids(self):
        if not self.finetune_input_ids:
            return self.concat_input_ids[:0]
        cat_ids = torch.cat(self.finetune_input_ids, dim=0).to(self.concat_input_ids.device)
        n = cat_ids.numel()
        if n > self.concat_input_ids.numel():
            raise ValueError(f"concat_input_ids capacity {self.concat_input_ids.numel()} < needed {n}")
        self.concat_input_ids[:n].copy_(cat_ids)
        return self.concat_input_ids[:n]
    
    def write_to_logit_tensor(self, logits, FFN_input_vpids, attention_input_vpids):
        self.finetune_logits_per_request.extend(logits)
        accumlate_len = sum(self.request_token_info)
        for logit in logits:
            n = logit.size(0)
            #self.logit_tensor[accumlate_len:accumlate_len + n, :].copy_(logit)
            accumlate_len += n
            self.request_token_info.append(n)
        self.activation_page_indices.append((FFN_input_vpids, attention_input_vpids))

    # def fill_activations_by_layer(self, layer_id, page_type, dest):
    #     """
    #     Tensorized GPU version:
    #     Gather all activations of a given PageType for a layer into dest[:total_tokens].
    #     Assumes:
    #     • self.page_table tensors are up-to-date and GPU-resident
    #     • VPIDs and physical indices are tensor-based
    #     """
    #     if len(self.request_token_info) == 0:
    #         return None

    #     total_tokens = sum(self.request_token_info)
    #     vpid_type   = self.page_table.vpid_type.to(self.device, non_blocking=True)
    #     vpid_device = self.page_table.vpid_device.to(self.device, non_blocking=True)
    #     vpid_phys   = self.page_table.vpid_phys_idx.to(self.device, non_blocking=True)
    #     DEV_CPU     = self.page_table._DEV_CPU
    #     DEV_GPU     = self.page_table._DEV_GPU

    #     # 1) Select all VPIDs of this type
    #     type_code = int(page_type.value)
    #     matching_vpids_t = torch.nonzero(vpid_type == type_code, as_tuple=False).flatten()
    #     if matching_vpids_t.numel() == 0:
    #         raise ValueError(f"No pages of type {page_type.name} exist for layer {layer_id}.")

    #     # 2) Ensure they are all on GPU (page-in CPU ones)
    #     cpu_mask = vpid_device[matching_vpids_t] == DEV_CPU
    #     if cpu_mask.any():
    #         cpu_vpids_t = matching_vpids_t[cpu_mask]
    #         self.pin_pages(cpu_vpids_t)
    #         self._pages_in(cpu_vpids_t)  # tensor-based paging-in
    #         # refresh device status
    #         vpid_device[cpu_vpids_t] = DEV_GPU

    #     # 3) Sanity check
    #     if matching_vpids_t.numel() != total_tokens:
    #         raise ValueError(
    #             f"Expected {total_tokens} pages for layer {layer_id}, "
    #             f"found {matching_vpids_t.numel()} with type {page_type.name}"
    #         )

    #     matching_vpids_t, _ = torch.sort(matching_vpids_t)
    #     phys_idx = vpid_phys[matching_vpids_t]
    #     layer_pool = self.gpu_pools[layer_id]
    #     assert torch.all((phys_idx >= 0) & (phys_idx < self.tot_size)), "Invalid phys_idx range"
    #     flat = layer_pool.index_select(0, phys_idx).reshape(total_tokens, -1)

    #     if dest.device != flat.device or dest.dtype != flat.dtype:
    #         dest = dest.to(device=flat.device, dtype=flat.dtype)
    #     dest[:total_tokens].copy_(flat, non_blocking=True)
    #     self.unpin_pages(matching_vpids_t)

    #     return dest
    
    def fill_activations_by_layer(self, layer_id, page_type, dest):
        """
        Gather all activations of a given PageType for a layer into dest[:total_tokens],
        following the logical order recorded in self.activation_page_indices.
        """
        if len(self.request_token_info) == 0:
            return None

        total_tokens = sum(self.request_token_info)
        type_code = int(page_type.value)
        vpid_type = self.page_table.vpid_type.to(self.device, non_blocking=True)
        vpid_device = self.page_table.vpid_device.to(self.device, non_blocking=True)
        vpid_phys = self.page_table.vpid_phys_idx.to(self.device, non_blocking=True)
        DEV_CPU = self.page_table._DEV_CPU
        DEV_GPU = self.page_table._DEV_GPU

        # 1) Get all VPIDs of this type
        matching_vpids_t = torch.nonzero(vpid_type == type_code, as_tuple=False).flatten()
        if matching_vpids_t.numel() == 0:
            raise ValueError(f"No pages of type {page_type.name} exist for layer {layer_id}.")

        # 2) Ensure they’re all on GPU
        cpu_mask = vpid_device[matching_vpids_t] == DEV_CPU
        if cpu_mask.any():
            cpu_vpids_t = matching_vpids_t[cpu_mask]
            self.pin_pages(cpu_vpids_t)
            self._pages_in(cpu_vpids_t)
            vpid_device[cpu_vpids_t] = DEV_GPU

        # 3) Sanity check
        if matching_vpids_t.numel() != total_tokens:
            raise ValueError(
                f"Expected {total_tokens} pages for layer {layer_id}, "
                f"found {matching_vpids_t.numel()} with type {page_type.name}"
            )

        # 4) Reorder according to activation_page_indices (per-request order)
        ordered_vpids = []
        for ffn_vpids, attn_vpids in self.activation_page_indices:
            if page_type == PageType.FFN_INPUT_ACTIVATION and ffn_vpids is not None:
                ordered_vpids.append(ffn_vpids)
            elif page_type == PageType.ATTENTION_INPUT_ACTIVATION and attn_vpids is not None:
                ordered_vpids.append(attn_vpids)
        if not ordered_vpids:
            raise ValueError(f"No activation pages recorded for {page_type.name}.")

        ordered_vpids_t = torch.cat(ordered_vpids, dim=0)
        assert ordered_vpids_t.numel() == total_tokens, (
            f"Activation page mismatch: expected {total_tokens}, got {ordered_vpids_t.numel()}"
        )

        # 5) Use the ordered mapping for copy
        phys_idx = vpid_phys[ordered_vpids_t]
        layer_pool = self.gpu_pools[layer_id]
        assert torch.all((phys_idx >= 0) & (phys_idx < self.tot_size)), "Invalid phys_idx range"

        flat = layer_pool.index_select(0, phys_idx).reshape(total_tokens, -1)

        # 6) Copy to destination
        if dest.device != flat.device or dest.dtype != flat.dtype:
            dest = dest.to(device=flat.device, dtype=flat.dtype)
        dest[:total_tokens].copy_(flat, non_blocking=True)

        self.unpin_pages(matching_vpids_t)
        return dest
    
    def assert_vpid_on_gpu(self, vpids):
        """
        Vectorized check that all given VPIDs are on GPU.
        """
        if isinstance(vpids, torch.Tensor):
            vpids_t = vpids.to('cuda', dtype=torch.long)
        else:
            vpids_t = torch.as_tensor(vpids, dtype=torch.long, device='cuda')

        devs = self.page_table.vpid_device[vpids_t]
        bad_mask = (devs != self.page_table._DEV_GPU)
        if bad_mask.any():
            # Keep the same error shape as before
            non_gpu = vpids_t[bad_mask].tolist()
            raise RuntimeError(f"These vpids are not on GPU: {non_gpu}")
    
    def get_thread_id(self, id):
        if id in self.thread_pool_dict.keys():
            return self.thread_pool_dict[id]
        else:
            self.thread_pool_dict[id] = self.thread_count
            self.thread_count += 1
            return self.thread_pool_dict[id] 

    def _now_ns(self) -> int:
        """Nanoseconds since mem-manager start (monotonic, high resolution)."""
        return time.perf_counter_ns() - self.t0_ns
    
    def page_size_kb(self) -> float:
        t = self.gpu_pools[0][0]
        s_kb = t.numel() * t.element_size() / 1024.0
        return s_kb*self.layer_num
    
    def copy_rows_to_layer(self, layer_id: int, vpids, rows: torch.Tensor):
        """
        Copy `rows` into self.gpu_pools[layer_id][vpids], paging-in any CPU pages first.
        """
        self.layer_accum_log(layer_id, vpids, "COPY_ROWS: ")

        # Normalize vpids → 1-D tensor on CPU
        vpids_t = (vpids if isinstance(vpids, torch.Tensor)
                else torch.as_tensor(vpids, dtype=torch.long, device="cuda"))
        N = vpids_t.numel()
        if N == 0:
            return

        # Normalize row shape → [N, head_num, head_dim]
        expected_flat = self.head_num * self.head_dim
        if rows.ndim == 2:
            if rows.shape[1] != expected_flat:
                raise ValueError(f"rows.shape[1] must be {expected_flat}, got {rows.shape[1]}")
            rows = rows.view(N, self.head_num, self.head_dim)
        elif rows.ndim == 3 and rows.shape != (N, self.head_num, self.head_dim):
            raise ValueError(f"rows.shape must be [{N},{self.head_num},{self.head_dim}]")

        # ── ensure residency ───────────────────────────────────────────────
        devs = self.page_table.vpid_device[vpids_t]
        cpu_mask = (devs == self.page_table._DEV_CPU)
        if cpu_mask.any():
            cpu_vpids = vpids_t[cpu_mask].tolist()
            self._pages_in(cpu_vpids)

        # ── translate → physical GPU indices ────────────────────────────────
        gpu_idx = self.page_table.vpid_phys_idx[vpids_t].to(self.device, dtype=torch.long, non_blocking=True)
        self.assert_vpid_on_gpu(vpids_t)

        # ── write rows ─────────────────────────────────────────────────────
        if rows.device != self.device or rows.dtype != self.gpu_pools[layer_id].dtype:
            rows = rows.to(device=self.device, dtype=self.gpu_pools[layer_id].dtype,
                        non_blocking=True)
        assert torch.all((gpu_idx >= 0) & (gpu_idx < self.tot_size)), "Invalid gpu_idx range"
        self.gpu_pools[layer_id].index_copy_(0, gpu_idx, rows)
        
    
    def to_gpu_index(self, vpids) -> torch.Tensor:
        """
        Vectorized: guarantee residency and return physical GPU slot indices.
        """
        with self.page_table_lock:
            if self.page_table.cpu_resident_count != 0:
                cpu_mask = (self.page_table.vpid_device[vpids] == self.page_table._DEV_CPU)
                if cpu_mask.any(): self._pages_in(vpids[cpu_mask])
            return self.page_table.vpid_phys_idx[vpids]

    def reset_b_loc_kv(self, b_loc_key: torch.Tensor, b_loc_value: torch.Tensor):
        with self.page_table_lock:
            if b_loc_key is None or b_loc_value is None:
                self.gpu_b_loc_key   = None
                self.gpu_b_loc_value = None
                return
            vpid_phys_idx = self.page_table.vpid_phys_idx  # tensor[int] on CPU
            self.gpu_b_loc_key   = vpid_phys_idx[b_loc_key]
            self.gpu_b_loc_value = vpid_phys_idx[b_loc_value]
            self.gpu_b_loc_key[self.gpu_b_loc_key==-1] = 0
            self.gpu_b_loc_value[self.gpu_b_loc_value==-1] = 0
            return

    def prepare_b_locs_for_layer(
        self,
        b_loc_key:   torch.Tensor,
        b_loc_value: torch.Tensor,
        b_seq_len:   torch.Tensor,
        layer_id:    int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with self.page_table_lock:
            # if self.gpu_b_loc_key is None or self.gpu_b_loc_value is None:
            #     self.reset_b_loc_kv(b_loc_key, b_loc_value)
            vpid_phys_idx = self.page_table.vpid_phys_idx  # tensor[int] on CPU
            self.gpu_b_loc_key   = vpid_phys_idx[b_loc_key]
            self.gpu_b_loc_value = vpid_phys_idx[b_loc_value]
            self.gpu_b_loc_key[self.gpu_b_loc_key==-1] = 0
            self.gpu_b_loc_value[self.gpu_b_loc_value==-1] = 0
            return self.gpu_pools[layer_id], self.gpu_b_loc_key, self.gpu_b_loc_value

    def save_activations_by_layer(self, layer_id, input_embs, infer_state, page_type, vpids=None):
        #self._log(f"SAVE_ACTIVATIONS_BY_LAYER: layer {layer_id}, type {page_type.name}")
        finetune_mask = infer_state.finetune_mask  # shape: [total_token_num]
        #finetune_activations = input_embs[finetune_mask].clone()  # shape: [N, hidden_size]
        finetune_activations = input_embs[finetune_mask].clone()
        # nonzero = torch.nonzero(finetune_mask, as_tuple=False)
        # start_idx = nonzero[0].item() if nonzero.numel() > 0 else None
        # finetune_activations = input_embs[start_idx:]
        num_new_tokens = finetune_activations.shape[0]
        if vpids is None:
            # Allocate new pages
            vpids = self.alloc(num_new_tokens, page_type)
        else:
            # Reuse existing pages
            if len(vpids) != num_new_tokens:
                raise ValueError(f"Expected {num_new_tokens} vpids, got {len(vpids)}")
        self.pin_pages(vpids)  # Ensure pages are resident on GPU
        self.copy_rows_to_layer(layer_id, vpids, finetune_activations)
        self.unpin_pages(vpids)  # Unpin after copying
        return vpids
    
    def save_embedding_output(self, input_embs, infer_state):
        finetune_mask = infer_state.finetune_mask  # shape: [total_token_num]
        #finetune_activations = input_embs[finetune_mask].clone()  # shape: [N, hidden_size]
        finetune_activations = input_embs[finetune_mask]
        prev_total = sum(self.request_token_info)
        num_new_tokens = finetune_activations.shape[0]
        self.embedding_output[prev_total : prev_total + num_new_tokens] = finetune_activations
    
    def update_request_token_info(self, infer_state):
        b_start = infer_state.b_start_loc
        b_len = infer_state.b_seq_len
        finetune_mask = infer_state.finetune_mask
        # Preallocate output list
        request_token_info = []
        for i in range(infer_state.batch_size):
            s, l = b_start[i], b_len[i]
            if l > 0:
                count = finetune_mask[s:s + l].sum().item()
                if count > 0:
                    request_token_info.append(count)
        self.request_token_info.extend(request_token_info)
    
    def rewind_alignment_pool(self, rewind_size):
        self.finetune_input_ids = self.finetune_input_ids[0:-rewind_size]
        self.alignment_completion_masks = self.alignment_completion_masks[0:-rewind_size]
        self.alignment_labels = self.alignment_labels[0:-rewind_size]
        return

    def get_input_layer_output(self):
        if not self.request_token_info:
            return None  # No activations saved
        total_tokens = sum(self.request_token_info)
        return self.embedding_output[:total_tokens]

    def get_finetune_activations(self, layer_id):
        return self.get_activations_by_layer(layer_id, PageType.ATTENTION_INPUT_ACTIVATION)
    
    def get_ffn_input(self, layer_id):
        return self.get_activations_by_layer(layer_id, PageType.FFN_INPUT_ACTIVATION)
    
    def get_activations_by_layer(self, layer_id, page_type):
        """
        Collect and return all activations of a given PageType for the specified layer.
        Returns a contiguous tensor of shape [total_tokens, head_num * head_dim].
        """
        if len(self.request_token_info) == 0:
            return None

        total_tokens = sum(self.request_token_info)
        type_code = int(page_type.value)

        # ── 1. Filter VPIDs by type (tensor mask, no dicts)
        mask = (self.page_table.vpid_type == type_code)
        if not torch.any(mask):
            raise ValueError(f"No VPIDs found for page type {page_type.name}")
        matching_vpids_t = torch.nonzero(mask, as_tuple=False).flatten()

        # ── 2. Determine which pages are on GPU or in CPU cache ──────────────
        vpid_device = self.page_table.vpid_device
        is_gpu_t = (vpid_device[matching_vpids_t] == self.page_table._DEV_GPU)

        present_vpids = matching_vpids_t[is_gpu_t].tolist()
        cpu_layer_pool = self.cpu_pools[layer_id]

        # add CPU-resident ones that exist in cpu_pool
        if (~is_gpu_t).any():
            cpu_candidates = matching_vpids_t[~is_gpu_t].tolist()
            for vpid in cpu_candidates:
                if vpid in cpu_layer_pool:
                    present_vpids.append(vpid)

        if len(present_vpids) != total_tokens:
            raise ValueError(
                f"Expected {total_tokens} pages for layer {layer_id}, "
                f"but found {len(present_vpids)} for type {page_type.name}"
            )

        present_vpids.sort()
        self.pin_pages(present_vpids)

        # ── 3. Page in any CPU-resident pages ───────────────────────────────
        vpids_t = torch.as_tensor(present_vpids, dtype=torch.long, device="cuda")
        cpu_mask = (self.page_table.vpid_device[vpids_t] == self.page_table._DEV_CPU)
        if cpu_mask.any():
            cpu_vpids = vpids_t[cpu_mask].tolist()
            priority = {
                PageType.KV_CACHE:  0,
                PageType.ADAPTER_WEIGHT: 1,
                PageType.ATTENTION_INPUT_ACTIVATION: 2,
                PageType.FFN_INPUT_ACTIVATION: 3,
            } if page_type == PageType.FFN_INPUT_ACTIVATION else {
                PageType.KV_CACHE:  0,
                PageType.ADAPTER_WEIGHT: 1,
                PageType.FFN_INPUT_ACTIVATION: 2,
                PageType.ATTENTION_INPUT_ACTIVATION: 3,
            }
            self._pages_in(cpu_vpids, priority=priority)

        # ── 4. Extract GPU slot indices and copy data ───────────────────────
        gpu_idx_t = self.page_table.vpid_phys_idx[vpids_t].to(self.device, non_blocking=True)
        self.assert_vpid_on_gpu(present_vpids)

        rows = (
            self.gpu_pools[layer_id]
            .index_select(0, gpu_idx_t)
            .reshape(total_tokens, -1)
            .clone()
        )

        self.unpin_pages(present_vpids)
        return rows
    
    def reset_activation_pool(self):
        """
        Clear all activation-related states and free any PageType.ACTIVATION pages.
        """
        with self.page_table_lock:
            # ── Reset training/inference bookkeeping ────────────────────────
            self.request_token_info.clear()
            self.finetune_input_ids.clear()
            self.alignment_completion_masks.clear()
            self.alignment_labels.clear()
            self.finetune_logits_per_request.clear()
            self.reference_logits_per_request.clear()
            self.activation_page_indices.clear()

            # ── Free any activation-related VPIDs ───────────────────────────
            vpid_type = self.page_table.vpid_type
            mask = (vpid_type == int(PageType.ATTENTION_INPUT_ACTIVATION.value)) | \
                (vpid_type == int(PageType.FFN_INPUT_ACTIVATION.value))
            target_vpids_t = torch.nonzero(mask, as_tuple=False).flatten()

            if target_vpids_t.numel() > 0:
                self.unpin_pages(target_vpids_t)
                self.free(target_vpids_t)

            self._log("RESET ACTIVATION POOL DONE")
    
    def _log(self, msg: str, print_count=True):
        if not self.logging_enabled:
            return
        # compute live statistics
        used_gpu = self.tot_size - self._num_free_gpu_slots()
        cpu_pages = sum(
            1 for loc, _ in self.page_table.vpid_to_phys.values()
            if loc == 'cpu'
        )
        ts_ns = self._now_ns()
        line = f"[{ts_ns}] {msg.ljust(25)} "
        if print_count:
            line += (f"\t|GPU {used_gpu}/{self.tot_size:<4} "
                    f"CPU {cpu_pages}| Page Size {self.page_size_kb():.2f} KB |")
            line += f"(Pinned pages: {len(self.pinned_pages)})"
            line += f"(Thread ID: {self.get_thread_id(threading.get_ident())})"
        # make parent dir once
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        with self.log_lock:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    def layer_accum_log(self, layer_id: int, vpids, msg: str, ignore_vpid_changes: bool = False):
        if not self.logging_enabled:
            return
        if self.accessed_layers is None or self.last_accessed_vpids is None:
            self.accessed_layers = [layer_id]
            self.last_accessed_vpids = vpids.copy()
            return

        is_disjoint = abs(layer_id - self.accessed_layers[-1]) != 1
        is_vpid_changed = not ignore_vpid_changes and vpids != self.last_accessed_vpids
        is_last_layer = len(self.accessed_layers) != 1 and (layer_id == self.layer_num - 1 or layer_id == 0)

        self.accessed_layers.append(layer_id)
        self.last_accessed_vpids = vpids.copy()

        if is_disjoint or is_vpid_changed or is_last_layer:
            msg += f" accessed layers {self.accessed_layers}"
            if vpids!=None:
                msg += f" | vpids={len(vpids)}"
            msg += f"(Pinned pages: {len(self.pinned_pages)})"
            msg += f"(Thread ID: {self.get_thread_id(threading.get_ident())})"
            self._log(msg)
            self.accessed_layers = None
            self.last_accessed_vpids = None
        
    