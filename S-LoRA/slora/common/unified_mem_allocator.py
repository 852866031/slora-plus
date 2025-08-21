from enum import Enum, auto
import torch
from collections import deque

import triton
import triton.language as tl
import datetime, os, errno
import threading
import nvtx 
import time

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

class PageTable:
    def __init__(self, max_gpu_slots: int):
        self.vpid_counter      = 5
        self.vpid_to_phys      = {}                 # vpid → ('gpu'|'cpu', idx)
        self.vpid_to_type      = {}                 # vpid → PageType
        self.gpu_slot_to_vpid  = [None] * max_gpu_slots

    def get_next_vpid(self) -> int:
        vpid = self.vpid_counter
        self.vpid_counter += 1
        return vpid

    def set_type(self, vpid: int, ptype: PageType):
        self.vpid_to_type[vpid] = ptype

    def get_type(self, vpid: int) -> PageType:
        return self.vpid_to_type[vpid]

    def set_gpu_mapping(self, vpid: int, gpu_idx: int):
        self.vpid_to_phys[vpid]  = ('gpu', gpu_idx)
        self.gpu_slot_to_vpid[gpu_idx] = vpid

    def set_cpu_mapping(self, vpid: int):
        self.vpid_to_phys[vpid] = ('cpu', vpid)

    def get_location_index(self, vpid: int):
        return self.vpid_to_phys[vpid]

    def remove(self, vpid: int):
        loc, idx = self.vpid_to_phys[vpid]
        if loc == 'gpu':
            self.gpu_slot_to_vpid[idx] = None
        del self.vpid_to_phys[vpid]
        del self.vpid_to_type[vpid]
    
    def vpids_to_types(self, vpids):
        """
        Returns a list of unique PageType for the given vpids.
        """
        types = set()
        for vpid in vpids:
            types.add(self.get_type(vpid))
        return list(types)

    def evictable_gpu_slots(self):
        """
        Yields (gpu_idx, vpid, page_type) for every page resident on GPU.
        """
        for idx, vpid in enumerate(self.gpu_slot_to_vpid):
            if vpid is not None:
                yield idx, vpid, self.get_type(vpid)

class UnifiedMemoryAllocator:
    def __init__(self, head_num, head_dim, layer_num: int, max_pool_size: int, dtype=torch.float16, device='cuda', log_path=None):
        self.head_dim = head_dim
        self.head_num = head_num
        self.hidden_dim  = head_num * head_dim
        self.layer_num   = layer_num
        self.device      = device
        self.dtype      = dtype
        self.tot_size = int(max_pool_size * 1024 * 1024 / self.layer_num / get_tensor_size_kb(self.head_num * self.head_dim, self.dtype))

        self.gpu_pools = [
            torch.empty((self.tot_size, self.head_num, self.head_dim),
                        device=self.device, dtype=self.dtype)
            for _ in range(self.layer_num)
        ]
        self.cpu_pools = [{} for _ in range(self.layer_num)]
        
        self.embedding_output = torch.empty((self.tot_size, self.head_num * self.head_dim), 
                                        dtype=self.dtype, device="cuda") 


        # mem_state[i] == 0 → free, 1 → occupied (GPU only)
        self.mem_state = torch.zeros(self.tot_size,
                                     dtype=torch.int8, device='cpu')

        self.pinned_pages = set()

        # Central page table
        self.page_table = PageTable(max_gpu_slots=self.tot_size)
        # Finetune task parameters
        self.request_token_info = []    # request_token_info = [num_finetune_tokens_request_1, ...]
        self.finetune_input_ids = []
        self.alignment_completion_masks = []
        self.alignment_labels= []
        self.finetune_logits_per_request = []
        self.reference_logits_per_request = []
        # Logging
        self.log_path = log_path
        self.logging_enabled = log_path is not None
        self.accessed_layers = None
        self.last_accessed_vpids = None
        # Thread-safety
        self.lock = threading.RLock()      # serialize allocator state changes
        self.log_lock = threading.RLock()  # serialize file writes (optional)
        if self.logging_enabled:
            self.t0_ns = time.perf_counter_ns()
            if os.path.exists(self.log_path):
                os.remove(self.log_path)
    
    
    
    def _now_ns(self) -> int:
        """Nanoseconds since mem-manager start (monotonic, high resolution)."""
        return time.perf_counter_ns() - self.t0_ns
    
    def page_size_kb(self) -> float:
        t = self.gpu_pools[0][0]
        s_kb = t.numel() * t.element_size() / 1024.0
        return s_kb*self.layer_num
    
    def assert_vpid_on_gpu(self, vpids):
        if isinstance(vpids, torch.Tensor):
            vpids = vpids.tolist()

        non_gpu = [vpid for vpid in vpids if self.page_table.get_location_index(vpid)[0] != 'gpu']
        if non_gpu:
            raise RuntimeError(f"These vpids are not on GPU: {non_gpu}")
    
    def _log(self, msg: str, print_count=True):
        if not self.logging_enabled:
            return
        with self.log_lock:
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
            # make parent dir once
            try:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    def layer_accum_log(self, layer_id: int, vpids, msg: str, ignore_vpid_changes: bool = False):
        if not self.logging_enabled:
            return
        with self.log_lock:
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
                self._log(msg)
                self.accessed_layers = None
                self.last_accessed_vpids = None

    def _num_free_gpu_slots(self) -> int:
        # mem_state is on CPU, so .sum() is cheap
        with self.lock:
            return int(self.tot_size - self.mem_state.sum().item())

    def _find_free_gpu_slots(self, num_needed: int) -> list[int]:
        """
        Return a list of `num_needed` free GPU slot indices.
        Raise RuntimeError if not enough slots are available.
        """
        with self.lock:
            free_slots = [i for i, state in enumerate(self.mem_state) if state == 0]
            if len(free_slots) < num_needed:
                raise RuntimeError(f"Requested {num_needed} free slots, but only found {len(free_slots)}")
            return free_slots[:num_needed]

    def alloc(self, num_pages: int, page_type: PageType):
        with self.lock:
            free_now   = self._num_free_gpu_slots()
            shortfall  = max(0, num_pages - free_now)

            if shortfall:                                  
                self._pages_out(self._find_victim_pages(shortfall))
            
            free_slots = self._find_free_gpu_slots(num_pages) 
            vpids = []
            for gpu_idx in free_slots:
                vpid = self.page_table.get_next_vpid()
                self.page_table.set_type(vpid, page_type)
                self.page_table.set_gpu_mapping(vpid, gpu_idx)

                self.mem_state[gpu_idx] = 1
                vpids.append(vpid)
            self._log(f"ALLOC: #pages={num_pages}, type={page_type.name}")
            return vpids

    def free(self, vpids):
        with self.lock:
            if isinstance(vpids, torch.Tensor):
                vpids = vpids.tolist()
            num_gpu = 0
            cpu_layers = self.cpu_pools  # cache reference

            for vpid in vpids:
                location, idx = self.page_table.get_location_index(vpid)

                if location == 'gpu':
                    # Free GPU slot
                    self.mem_state[idx] = 0
                    num_gpu += 1

                # Remove from all CPU layers if exists
                for layer_pool in cpu_layers:
                    layer_pool.pop(vpid, None)  # safe, no KeyError

                self.page_table.remove(vpid)

            self._log(f"FREE: {len(vpids)} pages, {num_gpu} on GPU, vpids: {vpids}")

    def _find_victim_pages(self, num_needed: int, priority: dict = None):
        with self.lock:
            if priority is None:
                priority = {
                    PageType.FFN_INPUT_ACTIVATION: 0,
                    PageType.ATTENTION_INPUT_ACTIVATION: 1,
                    PageType.KV_CACHE:  2,
                    PageType.ADAPTER_WEIGHT: 3
                }

            candidates = []
            for gpu_idx, vpid, ptype in self.page_table.evictable_gpu_slots():
                if vpid in self.pinned_pages:
                    continue
                candidates.append((priority[ptype], gpu_idx))

            candidates.sort()              # lower priority value → evict first
            victims = [gpu_idx for _, gpu_idx in candidates[:num_needed]]

            if len(victims) < num_needed:
                self._log(f"ERROR: Not enough evictable GPU pages. "
                        f"Needed {num_needed}, found {len(victims)}.")
                raise RuntimeError("Not enough evictable GPU pages.")
            else:
                if self.logging_enabled:
                    # count victim pages types:
                    victim_type_counts = {}
                    for gpu_idx in victims:
                        vpid = self.page_table.gpu_slot_to_vpid[gpu_idx]
                        ptype = self.page_table.get_type(vpid)
                        if ptype in victim_type_counts:
                            victim_type_counts[ptype] += 1
                        else:
                            victim_type_counts[ptype] = 1
                    self._log(f"_find_victim_pages: Victim pages: {victim_type_counts}")
                self._pages_out(victims)
            return victims


    def _pages_out(self, gpu_idxs):
        with self.lock:
            self._log(f"PAGING OUT: {len(gpu_idxs)} GPU slots {gpu_idxs} to CPU")
            for gpu_idx in gpu_idxs:
                vpid = self.page_table.gpu_slot_to_vpid[gpu_idx]

                for layer in range(self.layer_num):
                    if vpid not in self.cpu_pools[layer]:
                        # Copy to CPU only if not already cached
                        cpu_tensor = self.gpu_pools[layer][gpu_idx].to('cpu').clone()
                        self.cpu_pools[layer][vpid] = cpu_tensor

                # Book-keeping: release GPU slot
                self.page_table.set_cpu_mapping(vpid)
                self.page_table.gpu_slot_to_vpid[gpu_idx] = None
                self.mem_state[gpu_idx] = 0

    def _pages_in(self, vpids, priority = None):
        with self.lock:
            self._log(f"PAGING IN: {len(vpids)} CPU pages {vpids} to GPU")
            num_needed = len(vpids) - self._num_free_gpu_slots()
            if num_needed > 0:
                self._pages_out(self._find_victim_pages(num_needed, priority))

            free_slots = self._find_free_gpu_slots(len(vpids))
            for vpid, gpu_idx in zip(vpids, free_slots):
                for layer in range(self.layer_num):
                    cpu_tensor = self.cpu_pools[layer][vpid]  # Must exist
                    self.gpu_pools[layer][gpu_idx].copy_(cpu_tensor)

                self.page_table.set_gpu_mapping(vpid, gpu_idx)
                self.mem_state[gpu_idx] = 1
    
    def alloc_cpu(self, num_pages: int, page_type: PageType):
        with self.lock:
            self._log(f"ALLOC_CPU: size={num_pages}, type={page_type.name}")
            vpids = []
            for _ in range(num_pages):
                vpid = self.page_table.get_next_vpid()
                self.page_table.set_type(vpid, page_type)
                self.page_table.set_cpu_mapping(vpid)

                # one row per layer, shape == (head_num, head_dim)
                for layer in range(self.layer_num):
                    self.cpu_pools[layer][vpid] = torch.empty(
                        (self.head_num, self.head_dim),
                        dtype=torch.float32, device='cpu')
                vpids.append(vpid)
            return vpids
    
    def pin_pages(self, vpids):
        with self.lock:
            #self._log(f"PIN PAGES: {len(vpids)} pages to GPU")
            if isinstance(vpids, torch.Tensor):
                vpids = vpids.tolist()
            self.pinned_pages.update(vpids)

    def unpin_pages(self, vpids):
        with self.lock:
            #self._log(f"UNPIN PAGES: {len(vpids)} pages from GPU")
            if isinstance(vpids, torch.Tensor):
                vpids = vpids.tolist()
            for vpid in vpids:
                self.pinned_pages.discard(vpid)

    def copy_rows_to_layer(self,
                        layer_id: int,
                        vpids,
                        rows: torch.Tensor):
        """
        Copy `rows` into self.gpu_pools[layer_id][vpids]. 
        Must pin/unpin pages before/after use.

        Accepts `rows` in either shape:
            • [N, head_num, head_dim]
            • [N, head_num * head_dim]   (will be reshaped internally)

        Where:
            N           == len(vpids)       == 4 * r
            head_num    == self.head_num
            head_dim    == self.head_dim
        """
        with self.lock:
            #self.layer_accum_log(layer_id, vpids, "COPY_ROWS: ")
            if isinstance(vpids, torch.Tensor):
                vpids = vpids.tolist()

            N = len(vpids)
            flat_needed = self.head_num * self.head_dim

            if rows.ndim == 3:
                assert rows.shape == (N, self.head_num, self.head_dim), \
                    f"rows should be [N,{self.head_num},{self.head_dim}]"
            elif rows.ndim == 2:
                assert rows.shape == (N, flat_needed), \
                    f"rows should be [N,{flat_needed}] when 2-D"
                rows = rows.view(N, self.head_num, self.head_dim)  # reshape once
            else:
                raise ValueError("rows must be 2-D or 3-D")

            cpu_vpids = [v for v in vpids
                        if self.page_table.get_location_index(v)[0] == 'cpu']
            if cpu_vpids:
                self._pages_in(cpu_vpids)
            self.assert_vpid_on_gpu(vpids)
            gpu_idx = torch.tensor(
                [self.page_table.get_location_index(v)[1] for v in vpids],
                dtype=torch.long, device=self.device)
            self.gpu_pools[layer_id].index_copy_(0, gpu_idx, rows.to(self.device, dtype=self.gpu_pools[layer_id].dtype))
    
    def to_gpu_index(self, vpids) -> torch.Tensor:
        """
        !!!! Must be used with pin/unpin to avoid eviction !!!!
        Given a list / 1-D tensor of virtual page IDs, guarantee they are
        resident on GPU and return a tensor of *physical* slot indices
        (same order).
        Parameters
        ----------
        vpids : List[int] | torch.Tensor[int64]
            Virtual page IDs to translate.

        Returns
        -------
        gpu_idx : torch.Tensor[int64]   (device = self.device)
            Row indices such that
                gpu_pools[layer_id][gpu_idx[i]]   ←→   vpids[i]
        """
        with self.lock:
            #self._log(f"TO GPU INDEX")
            if isinstance(vpids, torch.Tensor):
                vpids = vpids.tolist()
            cpu_vpids = [v for v in vpids
                        if self.page_table.get_location_index(v)[0] == "cpu"]
            if cpu_vpids:
                self._pages_in(cpu_vpids)        # will evict if space is short
            self.assert_vpid_on_gpu(vpids)     # ensure all are on GPU
            gpu_idx = torch.tensor(
                [self.page_table.get_location_index(v)[1] for v in vpids],
                dtype=torch.long,
                device=self.device,
            )
            return gpu_idx


    def prepare_b_locs_for_layer(
        self,
        b_loc_key:   torch.Tensor,   # [batch, max_len]  (int64 vpids)
        b_loc_value: torch.Tensor,   # [batch, max_len]  (int64 vpids)
        b_seq_len:   torch.Tensor,   # [batch]           (int32 lengths)
        layer_id:    int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        • Pages-in every VPID that occurs in the *valid* (right-aligned) part
        of `b_loc_key` + `b_loc_value`.
        • Translates those VPIDs to physical GPU-slot indices.
        • Returns:
            ( gpu_pool_for_layer,  gpu_b_loc_key,  gpu_b_loc_value, vpid_to_unpin )

        gpu_b_loc_* have the same shape as the inputs but contain *physical*
        row numbers; padding positions are left as-is (0).
        Important: vpid_to_unpin is a list of VPIDs that should be unpinned after use.
        """
        msg = f"PREPARE_B_LOCS:"
        msg += f"b_loc_key_shape={b_loc_key.shape}, "
        msg += f"b_loc_value_shape={b_loc_value.shape}, "
        msg += f"b_seq_len_shape={b_seq_len.shape}"
        #self.layer_accum_log(layer_id, [], msg, ignore_vpid_changes=True)
        with self.lock:
            vpids = torch.unique(
                torch.cat((b_loc_key.flatten(), b_loc_value.flatten()))
            )
            vpids = vpids[vpids != 0]          # drop padding zeros
        
            if vpids.numel():                  # only if real vpids exist
                # pages-in & translate → gpu rows
                self.pin_pages(vpids)
                gpu_rows = self.to_gpu_index(vpids)   # guarantees residency
                # fast LUT (size = max_vpid+1)  0 maps to 0 by default
                lut = torch.zeros(int(vpids.max()) + 1,
                                dtype=torch.long, device='cuda')
                lut[vpids] = gpu_rows
            else:
                # nothing to map: return zero tensors
                zero_k = torch.zeros_like(b_loc_key)
                zero_v = torch.zeros_like(b_loc_value)
                return self.gpu_pools[layer_id], zero_k, zero_v, []

            # ─────────────────────────────────────────────────────────────
            # 2. Create zero-filled output tensors and scatter slot indices
            # ─────────────────────────────────────────────────────────────
            gpu_b_loc_key   = torch.zeros_like(b_loc_key)
            gpu_b_loc_value = torch.zeros_like(b_loc_value)

            nz_mask_key   = b_loc_key   != 0
            nz_mask_value = b_loc_value != 0

            gpu_b_loc_key  [nz_mask_key]   = lut[b_loc_key  [nz_mask_key]]
            gpu_b_loc_value[nz_mask_value] = lut[b_loc_value[nz_mask_value]]

            vpid_to_unpin = vpids

            return self.gpu_pools[layer_id], gpu_b_loc_key, gpu_b_loc_value, vpid_to_unpin
    
    def report_diff_percent(
        self,
        name: str,
        ours: torch.Tensor,
        slora: torch.Tensor,
        eps: float = 1e-4,       # what “near zero” means for the reference
        thresh: float = 1e-2,     # 1% threshold for “bad” elements
        layer_id: int = -1
    ):
        assert ours.shape == slora.shape, f"shape mismatch: {ours.shape} vs {slora.shape}"

        diff     = ours - slora
        abs_diff = diff.abs()
        ref_abs  = slora.abs()

        # L2 relative error
        rel_l2   = diff.norm() / (slora.norm() + eps)

        # mean and max absolute error
        mean_abs = abs_diff.mean().item()
        max_abs  = abs_diff.max().item()

        # fraction of “bad” elements (relative abs error > thresh)
        bad      = abs_diff > (thresh * (ref_abs + eps))
        frac_bad = float(bad.sum()) / bad.numel() * 100.0

        # fraction of reference elements that are essentially zero
        small    = ref_abs <= eps
        frac_small = float(small.sum()) / small.numel() * 100.0
        msg = "\n"
        if layer_id >= 0:
            msg += f"[{name} at layer {layer_id}] shape={list(ours.shape)}, dtype={ours.dtype}\n"
        else:
            msg += f"[{name}] shape={list(ours.shape)}, dtype={ours.dtype}\n"
        msg += f"  L2-rel err: {rel_l2*100:6.2f}%   mean |Δ|: {mean_abs:.3e}   max |Δ|: {max_abs:.3e}\n"
        msg += f"  >{thresh*100:.1f}% err: {frac_bad:5.2f}% of elements\n"
        msg += f"  reference near-zero (<= {eps}): {frac_small:5.2f}%  \n"
        self._log(msg, print_count=False)
        return msg
    
    def save_activations_by_layer(self, layer_id, input_embs, infer_state, page_type, vpids=None):
        with self.lock:
            finetune_mask = infer_state.finetune_mask  # shape: [total_token_num]
            finetune_activations = input_embs[finetune_mask].clone()  # shape: [N, hidden_size]
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
        with self.lock:
            finetune_mask = infer_state.finetune_mask  # shape: [total_token_num]
            finetune_activations = input_embs[finetune_mask].clone()  # shape: [N, hidden_size]
            prev_total = sum(self.request_token_info)
            num_new_tokens = finetune_activations.shape[0]
            self.embedding_output[prev_total : prev_total + num_new_tokens] = finetune_activations
    
    def update_request_token_info(self, infer_state):
        with self.lock:
            finetune_mask = infer_state.finetune_mask
            b_start_loc = infer_state.b_start_loc
            b_seq_len = infer_state.b_seq_len
            batch_size = infer_state.batch_size
            for i in range(batch_size):
                start = b_start_loc[i].item()
                end = start + b_seq_len[i].item()
                n_finetune_tokens = finetune_mask[start:end].sum().item()
                if n_finetune_tokens > 0:
                    self.request_token_info.append(n_finetune_tokens)
    
    def rewind_alignment_pool(self, rewind_size):
        with self.lock:
            self.finetune_input_ids = self.finetune_input_ids[0:-rewind_size]
            self.alignment_completion_masks = self.alignment_completion_masks[0:-rewind_size]
            self.alignment_labels = self.alignment_labels[0:-rewind_size]
            return
    
    def get_concatenated_finetune_input_ids(self):
        return torch.cat(self.finetune_input_ids, dim=0).to('cuda')

    def get_input_layer_output(self):
        with self.lock:
            if not self.request_token_info:
                return None  # No activations saved
            total_tokens = sum(self.request_token_info)
            return self.embedding_output[:total_tokens]

    def get_finetune_activations(self, layer_id):
        return self.get_activations_by_layer(layer_id, PageType.ATTENTION_INPUT_ACTIVATION)
    
    def get_ffn_input(self, layer_id):
        return self.get_activations_by_layer(layer_id, PageType.FFN_INPUT_ACTIVATION)


    def get_activations_by_layer(self, layer_id, page_type):
        with self.lock:
            if len(self.request_token_info) == 0:
                return None
            total_tokens = sum(self.request_token_info)

            # 1. Filter vpids by type
            matching_vpids = [
                vpid for vpid, typ in self.page_table.vpid_to_type.items()
                if typ == page_type
            ]

            # 2. Collect unique valid vpids for this layer
            seen = set()
            present_vpids = []
            cpu_layer_pool = self.cpu_pools[layer_id]

            for vpid in matching_vpids:
                loc, idx = self.page_table.get_location_index(vpid)
                if vpid in seen:
                    continue
                if loc == 'gpu' and idx is not None and 0 <= idx < self.tot_size:
                    present_vpids.append(vpid)
                    seen.add(vpid)
                elif loc == 'cpu' and vpid in cpu_layer_pool:
                    present_vpids.append(vpid)
                    seen.add(vpid)

            if len(present_vpids) != total_tokens:
                raise ValueError(
                    f"Expected {total_tokens} unique pages for layer {layer_id}, "
                    f"but found {len(present_vpids)} with type {page_type.name}"
                )
            present_vpids.sort() 
            self.pin_pages(present_vpids)  # Ensure all are resident on GPU
            # 3. Page in CPU-resident pages (non-destructive)
            cpu_vpids = [v for v in present_vpids if self.page_table.get_location_index(v)[0] == "cpu"]

            if cpu_vpids:
                if page_type == PageType.FFN_INPUT_ACTIVATION:
                    priority = {
                        PageType.KV_CACHE:  0,
                        PageType.ADAPTER_WEIGHT: 1,
                        PageType.ATTENTION_INPUT_ACTIVATION: 2,  
                        PageType.FFN_INPUT_ACTIVATION: 3,
                    }
                else:
                    priority = {
                        PageType.KV_CACHE:  0,
                        PageType.ADAPTER_WEIGHT: 1,
                        PageType.FFN_INPUT_ACTIVATION: 2,  
                        PageType.ATTENTION_INPUT_ACTIVATION: 3,
                    }
                self._pages_in(cpu_vpids, priority=priority)

            # 4. Extract GPU slot indices
            self.assert_vpid_on_gpu(present_vpids)
            gpu_indices = [self.page_table.get_location_index(v)[1] for v in present_vpids]
            rows = self.gpu_pools[layer_id][gpu_indices].view(total_tokens, -1).clone() # shape: [token_num, head_num * head_dim]
            self.unpin_pages(present_vpids)
            return rows  

    def reset_activation_pool(self):
        """
        Reset the activation pool by clearing all saved activations.
        This is useful for starting a new inference or finetuning session.
        """
        with self.lock:
            self._log("RESET ACTIVATION POOL")
            self.request_token_info = []
            self.finetune_input_ids = []
            self.alignment_completion_masks = []
            self.alignment_labels = []
            self.finetune_logits_per_request = []
            self.reference_logits_per_request = []
            self.embedding_output = torch.empty((self.tot_size*10, self.head_num * self.head_dim), 
                                            dtype=self.dtype, device="cuda") 
            target_types = {PageType.ATTENTION_INPUT_ACTIVATION, PageType.FFN_INPUT_ACTIVATION}
            target_vpids = [
                vpid for vpid, typ in self.page_table.vpid_to_type.items()
                if typ in target_types
            ]
            if target_vpids:
                self.free(target_vpids)
            self._log("RESET ACTIVATION POOL DONE")
            torch.cuda.synchronize()
    
    