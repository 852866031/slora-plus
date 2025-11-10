import torch
import threading
from enum import Enum, auto


def get_tensor_size_kb(numel: int, dtype: torch.dtype) -> float:
    """Helper to compute tensor element size in KB."""
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
    FREE = 0
    KV_CACHE = auto()
    ADAPTER_WEIGHT = auto()
    ATTENTION_INPUT_ACTIVATION = auto()
    FFN_INPUT_ACTIVATION = auto()
    EMBEDDING = auto()


class UnifiedMemoryAllocator:
    """
    Simplified allocator assuming everything fits in GPU memory.
    Uses global free bitmap and page_type_map shared by all layers.
    All layers allocate/free the same physical pages together.
    """

    def __init__(self, head_num, head_dim, vocab_size, layer_num: int,
                 max_pool_size: int, dtype=torch.float16, device="cuda", log_path=None):
        self.head_dim = head_dim
        self.head_num = head_num
        self.hidden_dim = head_num * head_dim
        self.layer_num = layer_num
        self.device = device
        self.dtype = dtype
        self.vocab_size = vocab_size

        # total number of slots per layer
        self.tot_size = int(
            max_pool_size * 1024 * 1024
            / self.layer_num
            / get_tensor_size_kb(self.head_num * self.head_dim, self.dtype)
        )

        # contiguous tensor pool for each layer
        self.gpu_pools = [
            torch.empty((self.tot_size, self.head_num, self.head_dim),
                        device=self.device, dtype=self.dtype)
            for _ in range(self.layer_num)
        ]

        # bitmaps shared by all layers
        self.page_type_map = torch.zeros(self.tot_size, dtype=torch.long, device=self.device)
        self.free_bitmap = torch.ones(self.tot_size, dtype=torch.bool, device=self.device)  # True=free

        self.page_table_lock = threading.RLock()

        # bookkeeping
        self.request_token_info = []
        self.activation_page_indices = []  # list of tuples: (FFN_input_phys_ids, ATTENTION_input_phys_ids)
        self.finetune_input_ids = []
        self.finetune_logits_per_request = []
        self.reference_logits_per_request = []
        self.shared_transformer_out_activations = None
        self.shared_attention_out_activations = None
        self.embedding_output = None
        self.max_finetuning_tokens = 512
        self.init_shared_activation_memory()

    def _num_free_gpu_slots(self) -> int:
        """
        Return the number of currently free GPU slots (global across all layers).
        """
        with self.page_table_lock:
            return int(self.free_bitmap.sum().item())
    
    def alloc(self, num_pages: int, page_type: PageType) -> torch.Tensor:
        """
        Allocate `num_pages` GPU slots globally across all layers.
        Returns tensor of physical indices shared by all layers.
        """
        with self.page_table_lock:
            free_idx = torch.nonzero(self.free_bitmap, as_tuple=False).flatten()
            if free_idx.numel() < num_pages:
                raise RuntimeError(
                    f"Not enough free pages: need {num_pages}, have {free_idx.numel()}."
                )

            alloc_ids = free_idx[:num_pages]
            self.free_bitmap[alloc_ids] = False
            self.page_type_map[alloc_ids] = int(page_type.value)
            return alloc_ids

    def free(self, phys_ids: torch.Tensor):
        """
        Free pages globally for all layers.
        """
        with self.page_table_lock:
            if not isinstance(phys_ids, torch.Tensor):
                phys_ids = torch.as_tensor(phys_ids, dtype=torch.long, device=self.device)

            self.page_type_map[phys_ids] = int(PageType.FREE.value)
            self.free_bitmap[phys_ids] = True

    def alloc_contiguous_kv(self, need_size: int, page_type: PageType):
        with self.page_table_lock:
            free_mask = self.free_bitmap
            free_idx = torch.nonzero(free_mask, as_tuple=False).flatten()
            if free_idx.numel() < 2 * need_size:
                return None  # not enough total free slots
            # find contiguous free runs
            diffs = free_idx[1:] - free_idx[:-1]
            # gaps = 1 means contiguous, 0 means break
            # create run IDs for consecutive groups
            run_starts = torch.cat((
                torch.tensor([0], device=self.device),
                torch.nonzero(diffs != 1, as_tuple=False).flatten() + 1
            ))
            run_ends = torch.cat((run_starts[1:], torch.tensor([free_idx.numel()], device=self.device)))
            # scan for a contiguous run of at least 2 * need_size
            start_idx = None
            for s, e in zip(run_starts.tolist(), run_ends.tolist()):
                run_len = e - s
                if run_len >= 2 * need_size:
                    start_idx = free_idx[s].item()
                    break

            if start_idx is None:
                return None  # no large enough contiguous segment found

            end_idx = start_idx + 2 * need_size
            phys_all = torch.arange(start_idx, end_idx, dtype=torch.long, device=self.device)

            # mark them as used
            self.free_bitmap[phys_all] = False
            self.page_type_map[phys_all] = int(page_type.value)

            # split into K/V halves
            phys_k = phys_all[:need_size]
            phys_v = phys_all[need_size:]
            return phys_k, start_idx, start_idx + need_size, phys_v, start_idx + need_size, end_idx


    def reset_activation_pool(self):
        """
        Clears activation-related states and frees activation pages globally.
        """
        with self.page_table_lock:
            self.request_token_info.clear()
            self.finetune_input_ids.clear()
            self.finetune_logits_per_request.clear()
            self.reference_logits_per_request.clear()
            self.activation_page_indices.clear()

            mask = (self.page_type_map == int(PageType.ATTENTION_INPUT_ACTIVATION.value)) | (
                self.page_type_map == int(PageType.FFN_INPUT_ACTIVATION.value)
            )
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            if idx.numel() > 0:
                for pool in self.gpu_pools:
                    pool[idx].zero_()
                self.page_type_map[idx] = int(PageType.FREE.value)
                self.free_bitmap[idx] = True


    def page_size_kb(self) -> float:
        t = self.gpu_pools[0][0]
        s_kb = t.numel() * t.element_size() / 1024.0
        return s_kb * self.layer_num
    
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

    def get_concatenated_finetune_input_ids(self):
        if not self.finetune_input_ids:
            return self.concat_input_ids[:0]
        cat_ids = torch.cat(self.finetune_input_ids, dim=0).to(self.concat_input_ids.device)
        n = cat_ids.numel()
        if n > self.concat_input_ids.numel():
            raise ValueError(f"concat_input_ids capacity {self.concat_input_ids.numel()} < needed {n}")
        self.concat_input_ids[:n].copy_(cat_ids)
        return self.concat_input_ids[:n]
    
    def copy_rows_to_layer(self, layer_id: int, phys_ids, rows: torch.Tensor):
        """
        Copy `rows` into self.gpu_pools[layer_id][vpids], paging-in any CPU pages first.
        """
        assert rows.dim() == 2 or rows.dim() == 3, \
            f"Expected 2D or 3D tensor for rows, got shape {rows.shape}"
        assert layer_id < self.layer_num, f"Invalid layer_id {layer_id}"
        if rows.device != self.device or rows.dtype != self.gpu_pools[layer_id].dtype:
            rows = rows.to(device=self.device, dtype=self.gpu_pools[layer_id].dtype,non_blocking=True)
        # ensure shape consistency
        rows_reshaped = rows.view(-1, self.head_num, self.head_dim)
        self.gpu_pools[layer_id][phys_ids] = rows_reshaped


    def save_activations_by_layer(self, layer_id, input_embs, infer_state, page_type, phys_ids=None):
        finetune_mask = infer_state.finetune_mask  # shape: [total_token_num]
        #finetune_activations = input_embs[finetune_mask].clone()
        finetune_activations = input_embs[finetune_mask]
        num_new_tokens = finetune_activations.shape[0]
        if phys_ids is None:
            phys_ids = self.alloc(num_new_tokens, page_type)
        else:
            if len(phys_ids) != num_new_tokens:
                raise ValueError(f"Expected {num_new_tokens} phys_ids, got {len(phys_ids)}")
        self.gpu_pools[layer_id][phys_ids] = finetune_activations.view(-1, self.head_num, self.head_dim)
        return phys_ids
        
    def save_embedding_output(self, input_embs, infer_state):
        finetune_mask = infer_state.finetune_mask  # shape: [total_token_num]
        finetune_activations = input_embs[finetune_mask]
        prev_total = sum(self.request_token_info)
        num_new_tokens = finetune_activations.shape[0]
        self.embedding_output[prev_total : prev_total + num_new_tokens] = finetune_activations
    
    def write_to_logit_tensor(self, logits, FFN_input_pids, attention_input_pids):
        #self.finetune_logits_per_request.extend(logits)
        accumlate_len = sum(self.request_token_info)
        for logit in logits:
            n = logit.size(0)
            accumlate_len += n
            self.request_token_info.append(n)
        self.activation_page_indices.append((FFN_input_pids, attention_input_pids))
        flat_logits = torch.cat(logits, dim=0)
        total_tokens = flat_logits.size(0)
        end_pos = min(accumlate_len, self.logit_tensor.size(0))
        self.logit_tensor[accumlate_len - total_tokens:end_pos].copy_(flat_logits, non_blocking=True)
    

    def export_requests_info(self):
        self.get_concatenated_finetune_input_ids()
        self.saved_layer_0_activations = None
        for layer_id in range(self.layer_num):
            self.fill_activations_by_layer(layer_id, PageType.FFN_INPUT_ACTIVATION, 
                                         self.shared_attention_out_activations[layer_id])
            self.fill_activations_by_layer(layer_id, PageType.ATTENTION_INPUT_ACTIVATION, 
                                         self.shared_transformer_out_activations[layer_id])
        requests_info_dict = {
            "request_token_info": self.request_token_info,
            #"finetuning_logits_per_request": self.finetune_logits_per_request,
        }
        return requests_info_dict

    def fill_activations_by_layer(self, layer_id, page_type, dest):
        """
        Gather activations of the given PageType for a specific layer, in the same
        order as requests were recorded in self.activation_page_indices.

        Args:
            layer_id (int): Which layer's pool to read from.
            page_type (PageType): Which activation type to export (FFN_INPUT_ACTIVATION or ATTENTION_INPUT_ACTIVATION).
            dest (torch.Tensor): Destination tensor to fill with shape [max_finetuning_tokens, hidden_dim].

        Returns:
            dest (torch.Tensor): Filled up to total_tokens rows.
        """
        if not self.activation_page_indices or len(self.request_token_info) == 0:
            return None

        total_tokens = sum(self.request_token_info)
        layer_pool = self.gpu_pools[layer_id]
        collected = []
        # Map PageType → index position in the tuple
        # (FFN_INPUT_ACTIVATION, ATTENTION_INPUT_ACTIVATION)
        idx_pos = 0 if page_type == PageType.FFN_INPUT_ACTIVATION else 1

        # Collect activation tensors per request
        for ffn_phys_ids, attn_phys_ids in self.activation_page_indices:
            phys_ids = ffn_phys_ids if idx_pos == 0 else attn_phys_ids
            if phys_ids is not None and len(phys_ids) > 0:
                collected.append(layer_pool.index_select(0, phys_ids))

        if not collected:
            raise ValueError(f"No activations found for {page_type.name} in layer {layer_id}.")

        # Concatenate in request order and flatten to [total_tokens, hidden_dim]
        flat = torch.cat(collected, dim=0).reshape(total_tokens, -1)

        # Copy into destination tensor
        if dest.device != flat.device or dest.dtype != flat.dtype:
            dest = dest.to(device=flat.device, dtype=flat.dtype)
        dest[:total_tokens].copy_(flat, non_blocking=True)

        return dest

    def prepare_b_locs_for_layer(
        self,
        b_loc_key:   torch.Tensor,
        b_loc_value: torch.Tensor,
        b_seq_len:   torch.Tensor,
        layer_id:    int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with self.page_table_lock:
            return self.gpu_pools[layer_id], b_loc_key, b_loc_value
    
    def to_gpu_index(self, vpids) -> torch.Tensor:
        with self.page_table_lock:
            return vpids
    
    def pin_pages(self, vpids):
        return
    
    def unpin_pages(self, vpids):
        return
    
    def alloc_cpu(self, num_pages: int, page_type: PageType) -> torch.Tensor:
        raise NotImplementedError("UnifiedMemoryAllocator does not support CPU allocation.")
    
    def reset_b_loc_kv(self, b_loc_key, b_loc_value):
        return