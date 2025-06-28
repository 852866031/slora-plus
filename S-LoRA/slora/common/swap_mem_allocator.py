import torch
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum, auto
import random

class PageType(Enum):
    KV_CACHE = auto()
    ADAPTER_WEIGHT = auto()
    ACTIVATION = auto()

@dataclass
class PageEntry:
    status: int  # 0 = free, 1 = used
    data_type: Optional[PageType] = None
    layer: Optional[int] = None
    request_id: Optional[int] = None
    gpu_resident: bool = True
    last_used: int = 0

class UnifiedMemoryAllocator:
    def __init__(self, tot_size: int, dtype: torch.dtype, hidden_dim: int, layer_num: int):
        self.tot_size = tot_size
        self.dtype = dtype
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.time_counter = 0

        self.pool = torch.empty((tot_size, hidden_dim), dtype=dtype, device="cuda")
        self.cpu_storage: Dict[int, torch.Tensor] = {}

        self.page_table: List[PageEntry] = [PageEntry(status=0) for _ in range(tot_size)]

        self.mem_state = torch.ones((tot_size,), dtype=torch.bool, device="cuda")
        self.indexes = torch.arange(0, tot_size, dtype=torch.long, device="cuda")
        self._mem_cum_sum = torch.empty((tot_size,), dtype=torch.int32, device="cuda")

    def _find_victim_pages(self, size: int) -> List[int]:
        priority_order = [PageType.ACTIVATION, PageType.KV_CACHE, PageType.ADAPTER_WEIGHT]
        used_pages = [(i, entry) for i, entry in enumerate(self.page_table) if entry.status == 1 and entry.gpu_resident]
        used_pages.sort(key=lambda x: x[1].last_used)

        selected = []
        for priority in priority_order:
            for i, entry in used_pages:
                if entry.data_type == priority:
                    selected.append(i)
                    if len(selected) >= size:
                        return selected
        return selected

    def _evict_pages(self, page_ids: List[int]):
        data = self.pool[torch.tensor(page_ids, device="cuda")].detach().cpu().clone()
        for i, pid in enumerate(page_ids):
            self.cpu_storage[pid] = data[i]
            self.page_table[pid].gpu_resident = False
            self.mem_state[pid] = 1

    def alloc(self, kind: PageType, size: int, layer: Optional[int] = None, request_id: Optional[int] = None) -> torch.Tensor:
        self.time_counter += 1

        if torch.sum(self.mem_state).item() < size:
            victims = self._find_victim_pages(size)
            if len(victims) < size:
                raise RuntimeError("Cannot evict enough pages to satisfy allocation")
            self._evict_pages(victims[:size])

        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        select_index = torch.logical_and(self._mem_cum_sum <= size, self.mem_state == 1)
        page_ids = self.indexes[select_index][:size]

        self.mem_state[page_ids] = 0
        for idx in page_ids.tolist():
            self.page_table[idx] = PageEntry(1, kind, layer, request_id, True, self.time_counter)

        return page_ids

    def alloc_contiguous(self, kind: PageType, size: int, layer: Optional[int] = None, request_id: Optional[int] = None) -> Optional[Tuple[torch.Tensor, int, int]]:
        self.time_counter += 1
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        loc_sums = self._mem_cum_sum[size - 1:self.tot_size] - self._mem_cum_sum[0:self.tot_size - size + 1] + self.mem_state[0:self.tot_size - size + 1]
        can_use_locs = self.indexes[0:self.tot_size - size + 1][loc_sums == size]

        if can_use_locs.numel() == 0:
            return None

        start = can_use_locs[0].item()
        page_ids = self.indexes[start:start + size]
        self.mem_state[page_ids] = 0
        for idx in page_ids.tolist():
            self.page_table[idx] = PageEntry(1, kind, layer, request_id, True, self.time_counter)

        return page_ids, start, start + size

    def free(self, page_ids: torch.Tensor):
        self.mem_state[page_ids] = 1
        for idx in page_ids.tolist():
            self.page_table[idx] = PageEntry(status=0)
            self.cpu_storage.pop(idx, None)

    def get_buffer(self, kind: PageType, layer: Optional[int] = None) -> torch.Tensor:
        page_ids = [i for i, entry in enumerate(self.page_table)
                    if entry.status == 1 and entry.data_type == kind and entry.layer == layer]
        if not page_ids:
            return None
        return self.retrieve_buffer(torch.tensor(page_ids, dtype=torch.long, device="cuda"))

    def page_out(self, page_ids: torch.Tensor) -> torch.Tensor:
        cpu_tensor = self.pool[page_ids].detach().cpu().clone()
        for idx in page_ids.tolist():
            self.page_table[idx].gpu_resident = False
            self.cpu_storage[idx] = cpu_tensor[idx - page_ids[0].item()]
        self.mem_state[page_ids] = 1
        return cpu_tensor

    def page_in(self, page_ids: torch.Tensor):
        for idx in page_ids.tolist():
            if not self.page_table[idx].gpu_resident:
                if torch.sum(self.mem_state).item() == 0:
                    victim = self._find_victim_pages(1)
                    if victim:
                        self._evict_pages(victim)
                self.pool[idx] = self.cpu_storage.pop(idx).to(self.pool.device)
                self.page_table[idx].gpu_resident = True
                self.mem_state[idx] = 0

    def retrieve_buffer(self, page_ids: torch.Tensor) -> torch.Tensor:
        self.page_in(page_ids)
        for idx in page_ids.tolist():
            self.page_table[idx].last_used = self.time_counter
        return self.pool[page_ids]

    def get_kv_cache(self, layer_id: int):
        return self.get_buffer(PageType.KV_CACHE, layer=layer_id)

    def get_activations(self, layer_id: int):
        return self.get_buffer(PageType.ACTIVATION, layer=layer_id)

    def reset(self):
        self.mem_state[:] = 1
        for i in range(self.tot_size):
            self.page_table[i] = PageEntry(status=0)
        self.cpu_storage.clear()
