from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


class FinetuningStore:
    """Reserves KV token slots for FT samples on sglang's 3-level pool.

    Slots are pulled out of the allocator's free_pages via alloc() so that
    available_size reflects reality, and are also tracked in
    self._reserved_indices. The reserved-predicate hook installed on the
    allocator is defense-in-depth against any code path that might
    re-introduce a reserved index into free_pages.
    """

    def __init__(
        self,
        req_to_token_pool: "ReqToTokenPool",
        kv_pool_allocator: "TokenToKVPoolAllocator",
    ):
        self.req_to_token_pool = req_to_token_pool
        self.kv_pool_allocator = kv_pool_allocator
        self._reserved_indices: set[int] = set()
        self.kv_pool_allocator.set_reserved_predicate(self.is_reserved)

    def reserve(self, num_tokens: int) -> List[int]:
        out = self.kv_pool_allocator.alloc(num_tokens)
        if out is None:
            raise RuntimeError(
                f"FinetuningStore.reserve({num_tokens}) failed: "
                f"insufficient free KV slots"
            )
        indices = [int(i) for i in out.tolist()]
        self._reserved_indices.update(indices)
        return indices

    def commit(self, indices: List[int]) -> None:
        for i in indices:
            assert i in self._reserved_indices, f"commit of non-reserved idx {i}"

    def release(self, indices: List[int]) -> None:
        for i in indices:
            self._reserved_indices.discard(i)
        tensor = torch.tensor(
            indices, dtype=torch.int64, device=self.kv_pool_allocator.device
        )
        self.kv_pool_allocator.free(tensor)

    def is_reserved(self, idx: int) -> bool:
        return idx in self._reserved_indices
