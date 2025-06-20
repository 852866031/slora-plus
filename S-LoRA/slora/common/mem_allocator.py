import gc
import torch
from typing import List, Tuple
from slora.server.router.mixed_req_queue import rprint
import torch.nn.functional as F

# TODO: will it slow down the program?
def suffix_cumsum(tensor, dim=-1, dtype=torch.int32):
    return torch.cumsum(tensor.flip(dim), dim, dtype=torch.int32).flip(dim)


class MemoryAllocator:
    def __init__(self, tot_size, cache_size, dtype, head_num, head_dim, layer_num):
        assert tot_size >= cache_size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.cell_size = head_num * head_dim

        self.tot_size = tot_size
        self.cache_size = cache_size

        self.reset_all_pool()
        rprint("MemoryAllocator initialized, \n\ttot_size", tot_size)
        rprint("\thead_dim", head_dim)
        rprint("\tlayer_num", layer_num)
        rprint("\tdtype", dtype)

    def get_memory_size(self):
        dsize = 2 if self.dtype == torch.float16 else None
        return 2 * self.layer_num * self.tot_size * self.cell_size * dsize
  

    def alloc(self, need_size):
        #print("self.can_use_mem_size", self.can_use_mem_size)
        #print("need_size", need_size)
        if need_size > self.can_use_mem_size:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size}')
        
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        select_index = torch.logical_and(self._mem_cum_sum <= need_size, self.mem_state == 1)
        select_index = self.indexes[select_index]
        self.mem_state[select_index] = 0
        self.can_use_mem_size -= len(select_index)
        return select_index


    def alloc_contiguous(self, need_size):
        if need_size > self.can_use_mem_size:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size}')
        
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        loc_sums = self._mem_cum_sum[need_size - 1:self.tot_size] - self._mem_cum_sum[0:self.tot_size - need_size + 1] + self.mem_state[0:self.tot_size - need_size + 1]
        can_used_loc = self.indexes[0:self.tot_size - need_size + 1][loc_sums == need_size]
        if can_used_loc.shape[0] == 0:
            # print(f'warn no enough pool space: to contiguous need_size {need_size} left_size {self.can_use_mem_size}')
            return None
        start_loc = can_used_loc[0]
        select_index = self.indexes[start_loc : start_loc + need_size]
        
        self.mem_state[select_index] = 0
        self.can_use_mem_size -= need_size
        start = start_loc.item()
        end = start + need_size
        return select_index, start, end


    def alloc_strip(self, need_block, block_size):
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        loc_sums = self._mem_cum_sum[block_size - 1:self.tot_size] - self._mem_cum_sum[0:self.tot_size - block_size + 1] + self.mem_state[0:self.tot_size - block_size + 1]
        loc_use = (loc_sums == block_size)
        torch.cumsum(loc_use, dim=0, dtype=torch.int32, out=loc_sums)

        block_start = torch.empty((loc_use.shape[0]), dtype=torch.int32, device="cuda")
        block_start[0] = loc_use[0]
        block_start[1:] = (loc_use[:-1] == 0) & (loc_use[1:] == 1)

        cum_max, _ = torch.cummax(block_start, dim=0)
        # (diff % block_size == 0) & loc_use
        mask = block_size - 1
        loc_use = (((loc_sums - cum_max) & mask) == 0) & loc_use
        can_use_loc = self.indexes[0:self.tot_size - block_size + 1][loc_use == 1]
        if can_use_loc.shape[0] < need_block:
            raise Exception(f"no enough pool space for alloc_strip, "
                            f"need {need_block} blocks, {can_use_loc.shape[0]} left")
        can_use_loc = can_use_loc[:need_block]
        select_index = torch.empty((block_size, need_block), dtype=torch.int32, device="cuda")
        for i in range(block_size):
            select_index[i] = can_use_loc + i
        select_index = select_index.T.reshape(-1)

        self.mem_state[select_index] = 0
        self.can_use_mem_size -= select_index.shape[0]
        return select_index


    def alloc_grid(self, need_grid, grid_size):
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        loc_sums = self._mem_cum_sum[grid_size - 1:self.tot_size] - self._mem_cum_sum[0:self.tot_size - grid_size + 1] + self.mem_state[0:self.tot_size - grid_size + 1]
        loc_use = (loc_sums == grid_size)

        mask = grid_size - 1
        loc_use = ((self.indexes[:self.tot_size - grid_size + 1] & mask) == 0) & loc_use
        can_use_loc = self.indexes[0:self.tot_size - grid_size + 1][loc_use == 1]
        if can_use_loc.shape[0] < need_grid:
            raise Exception(f"no enough pool space for alloc_strip, "
                            f"need {need_grid} grids, {can_use_loc.shape[0]} left")
        can_use_loc = can_use_loc[:need_grid]
        select_index = torch.empty((grid_size, need_grid), dtype=torch.int32, device="cuda")
        for i in range(grid_size):
            select_index[i] = can_use_loc + i
        select_index = select_index.T.reshape(-1)

        self.mem_state[select_index] = 0
        self.can_use_mem_size -= select_index.shape[0]
        return select_index


    def alloc_prefix(self, need_size):
        assert False
        if need_size > self.can_use_mem_size_prefix:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size_prefix}')
        
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        select_index = torch.logical_and(self._mem_cum_sum <= need_size, self.mem_state == 1)
        select_index = self.indexes[select_index]
        self.mem_state[select_index] = 0
        self.can_use_mem_size_prefix -= len(select_index)
        return select_index
    

    def alloc_contiguous_prefix(self, need_size):
        assert False
        if need_size > self.can_use_mem_size_prefix:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size_prefix}')
        
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        loc_sums = self._mem_cum_sum[need_size - 1:self.cache_size] - self._mem_cum_sum[0:self.cache_size - need_size + 1] + self.mem_state[0:self.cache_size - need_size + 1]
        can_used_loc = self.indexes[0:self.cache_size - need_size + 1][loc_sums == need_size]
        if can_used_loc.shape[0] == 0:
            # print(f'warn no enough pool space: to contiguous need_size {need_size} left_size {self.can_use_mem_size_prefix}')
            return None
        start_loc = can_used_loc[0]
        select_index = self.indexes[start_loc : start_loc + need_size]
        
        self.mem_state[select_index] = 0
        self.can_use_mem_size_prefix -= need_size
        start = start_loc.item()
        end = start + need_size
        return select_index, start, end


    def alloc_suffix(self, need_size):
        assert False
        if need_size > self.can_use_mem_size_suffix:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size_suffix}')
            return None
        
        self._mem_cum_sum = suffix_cumsum(self.mem_state, dim=0, dtype=torch.int32)
        select_index = torch.logical_and(self._mem_cum_sum <= need_size, self.mem_state == 1)
        select_index = self.indexes[select_index]
        self.mem_state[select_index] = 0
        self.can_use_mem_size_suffix -= len(select_index)
        return select_index
    

    def alloc_contiguous_suffix(self, need_size):
        assert False
        if need_size > self.can_use_mem_size_suffix:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size_suffix}')
            return None
        
        self._mem_cum_sum = suffix_cumsum(self.mem_state, dim=0, dtype=torch.int32)
        assert len(self._mem_cum_sum) == self.cache_size
        loc_sums = (self._mem_cum_sum[0:self.cache_size - need_size + 1] - self._mem_cum_sum[need_size - 1:] +
                    self.mem_state[need_size - 1:])
        can_used_loc = self.indexes[0:self.cache_size - need_size + 1][loc_sums == need_size]
        if can_used_loc.shape[0] == 0:
            # print(f'warn no enough pool space: to contiguous need_size {need_size} left_size {self.can_use_mem_size_suffix}')
            return None
        start_loc = can_used_loc[0]
        select_index = self.indexes[start_loc : start_loc + need_size]
        
        self.mem_state[select_index] = 0
        self.can_use_mem_size_suffix -= need_size
        start = start_loc.item()
        end = start + need_size
        return select_index, start, end
 
    
    def free(self, free_index):
        """_summary_

        Args:
            free_index (torch.Tensor): _description_
        """
        #print("free_index len", len(free_index))
        self.can_use_mem_size += free_index.shape[0]
        # self.can_use_mem_size_prefix += torch.sum(free_index < self.cache_size)
        # self.can_use_mem_size_suffix += torch.sum(free_index >= self.cache_size)
        self.mem_state[free_index] = 1

        # if self.can_use_mem_size_prefix + self.can_use_mem_size_suffix == self.tot_size:
        #     print(f"freed all gpu mem size {self.tot_size}")
        # print(f"free state {self.can_use_mem_size_prefix} + {self.can_use_mem_size_suffix} all {self.tot_size}")
        return
    
    def free_all(self):
        self.mem_state[:] = 1
        self.can_use_mem_size = self.tot_size
        # self.can_use_mem_size_prefix = self.cache_size
        # self.can_use_mem_size_suffix = self.tot_size - self.cache_size
    

    def delete_all_pool(self):
        self.mem_state = None
        self._mem_cum_sum = None
        self.indexes = None
        self.can_use_mem_size = 0
        # self.can_use_mem_size_prefix = 0
        # self.can_use_mem_size_suffix = 0
        self.buffer = None
        gc.collect()

    def delete_all_cache(self):
        self.delete_all_pool()


    def reset_all_pool(self):
        self.mem_state = torch.ones((self.tot_size,), dtype=torch.bool, device="cuda")
        self._mem_cum_sum = torch.empty((self.tot_size,), dtype=torch.int32, device="cuda")
        self.indexes = torch.arange(0, self.tot_size, dtype=torch.long, device="cuda")
        self.can_use_mem_size = self.tot_size
        # self.can_use_mem_size_prefix = self.cache_size
        # self.can_use_mem_size_suffix = self.tot_size - self.cache_size
        self.key_buffer = [torch.empty((self.tot_size, self.head_num, self.head_dim),
                                       dtype=self.dtype, device="cuda")
                           for _ in range(self.layer_num)]
        self.value_buffer = [torch.empty((self.tot_size, self.head_num, self.head_dim),
                                       dtype=self.dtype, device="cuda")
                           for _ in range(self.layer_num)]
        
        rprint("Reset total size", self.tot_size)
        self.finetune_activation_buffer = [torch.empty((self.tot_size*10, self.head_num * self.head_dim), 
                                        dtype=self.dtype, device="cuda") 
                            for _ in range(self.layer_num)]
        self.input_layer_output = torch.empty((self.tot_size*10, self.head_num * self.head_dim), 
                                        dtype=self.dtype, device="cuda") 
        self.ffn_input_buffer = [torch.empty((self.tot_size*10, self.head_num * self.head_dim), 
                                        dtype=self.dtype, device="cuda") 
                            for _ in range(self.layer_num)]
        
        # attention -> ffn
        # [layer_id, slot_id, head_num * head_dim]
        self.request_token_info = []
        # mem_manager.request_token_info = [num_finetune_tokens_request_1, ...]
        self.finetune_input_ids = [] #List[input_ids_tensor]
        self.alignment_completion_masks = []
        self.finetune_logits_per_request = []
        self.reference_logits_per_request = []
        self.alignment_labels= []
        self.request_token_info_checkpoint = None
        self.saved_q = None
        self.saved_k = None
        self.saved_v = None
        self.saved_o = None
        
        #TODO: merge all possible buffers into one
    
    def rewind_alignment_pool(self, rewind_size):
        self.finetune_input_ids = self.finetune_input_ids[0:-rewind_size]
        self.alignment_completion_masks = self.alignment_completion_masks[0:-rewind_size]
        self.alignment_labels = self.alignment_labels[0:-rewind_size]
        return

    def checkpoint_request_token_info(self):
        self.request_token_info_checkpoint = self.request_token_info[:]
    
    def rewind_request_token_info(self):
        self.request_token_info = self.request_token_info_checkpoint[:]
        self.request_token_info_checkpoint = None

    def reset_activation_pool(self):
        self.request_token_info = []
        self.finetune_input_ids = []
        self.finetune_logits_per_request = []
        self.reference_logits_per_request = []
        self.alignment_completion_masks = []
        self.alignment_labels= []
        self.finetune_activation_buffer = [torch.empty((self.tot_size*10, self.head_num * self.head_dim), 
                                        dtype=self.dtype, device="cuda") 
                            for _ in range(self.layer_num)]
        self.input_layer_output = torch.empty((self.tot_size*10, self.head_num * self.head_dim), 
                                        dtype=self.dtype, device="cuda") 
        self.ffn_input_buffer = [torch.empty((self.tot_size*10, self.head_num * self.head_dim), 
                                        dtype=self.dtype, device="cuda") 
                            for _ in range(self.layer_num)]
        
        
    def get_input_layer_output(self):
        if not self.request_token_info:
            return None  # No activations saved

        total_tokens = sum(self.request_token_info)
        return self.input_layer_output[:total_tokens]

    def get_finetune_activations(self, layer_id):
        """
        Return all saved activations for a given layer as a batch tensor.
        """
        if not self.request_token_info:
            return None  # No activations saved

        total_tokens = sum(self.request_token_info)
        return self.finetune_activation_buffer[layer_id][:total_tokens]

    def get_ffn_input(self, layer_id):
        """
        Return all saved activations for a given layer as a batch tensor.
        """
        if not self.request_token_info:
            return None  # No activations saved

        total_tokens = sum(self.request_token_info)
        return self.ffn_input_buffer[layer_id][:total_tokens]

    def get_concatenated_finetune_input_ids(self):
        return torch.cat(self.finetune_input_ids, dim=0).to('cuda')

    def print_finetune_activation_summary(self):
        GREEN = '\033[92m'
        RESET = '\033[0m'

        total_layers = len(self.finetune_activation_buffer)
        total_tokens = sum(self.request_token_info)

        print(f"{GREEN}=== Finetune Activation Summary ==={RESET}")
        print(f"{GREEN}Layers: {total_layers}{RESET}")
        print(f"{GREEN}Total finetune tokens per layer saved: {total_tokens}{RESET}")
        print(f"{GREEN}Total requests: {len(self.request_token_info)}{RESET}")
        print(f"{GREEN}Buffer shape per layer: {self.finetune_activation_buffer[0].shape if total_layers > 0 else 'N/A'}{RESET}")
        print(f"{GREEN}=== End of Summary ==={RESET}")

    def reset_all_cache(self):
        self.reset_all_pool()
