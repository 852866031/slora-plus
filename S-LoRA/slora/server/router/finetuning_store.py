import uuid
from bisect import bisect_right, bisect_left
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from slora.server.io_struct import Batch, Req
from slora.server.sampling_params import SamplingParams


def get_finetuning_sampling_params() -> SamplingParams:
    """
    Return a 'dummy' sampling params object suitable for fine-tuning requests.
    By default, no sampling or advanced penalties are used.
    """
    return SamplingParams(
        do_sample=False,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=False,   # training typically doesn't stop on generated EOS
        max_new_tokens=1,
        stop_sequences=[],
    )


class FinetuningManager:
    """
    Efficient fine-tuning sample manager:
    - load() reads samples, tokenizes, constructs Req objects
    - pop_best_under(max_tokens) returns the untrained request with the largest input_len <= max_tokens
      (does NOT mark it trained)
    - confirmed_trained(reqs) marks a list of requests as trained for the current epoch
    - advance_epoch() increments epoch and resets all marks
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,                     # callable: text -> {"input_ids": [...]}
        adapter_dir: str,
        total_epochs: int,
        max_saved_finetuning_tokens: int,
        max_prepare: Optional[int] = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.adapter_dir = adapter_dir
        self.total_epochs = int(total_epochs)
        self.max_prepare = max_prepare
        self.trust_remote_code = trust_remote_code

        # Storage
        self.reqs: List[Req] = []
        self.req_ids: List[str] = []
        self.id2idx: Dict[str, int] = {}

        # Per-epoch training marks
        self.trained: List[bool] = []
        self.current_epoch: int = 0

        # Fast retrieval structures
        # length -> deque of indices for untrained requests of that length
        self.len_buckets: Dict[int, deque] = defaultdict(deque)
        self.sorted_lengths: List[int] = []

        self._bucket_template: Dict[int, Tuple[int, ...]] = {}
        self._sorted_template: List[int] = []

        # Summary
        self.total_tokens_in_memory: int = 0
        self.finetuning_tokens_processed = 0
        self.pending_bwd_tokens = 0
        self.epoch_avg_loss_list = []
        self.loss_list =[]
        self.max_saved_finetuning_tokens = max_saved_finetuning_tokens

    def load(self) -> int:
        loaded = 0
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue

                toks = self.tokenizer(text)
                prompt_ids = toks.get("input_ids", [])
                req_id = uuid.uuid4().hex

                # Build Req (formerly _make_req)
                sp = get_finetuning_sampling_params()
                req = Req(
                    adapter_dir=self.adapter_dir,
                    request_id=req_id,
                    prompt_ids=prompt_ids,
                    sample_params=sp,
                    is_finetuning=True,
                    text=text,
                )

                # Append + index (formerly _append_req)
                idx = len(self.reqs)
                self.reqs.append(req)
                self.req_ids.append(req.request_id)
                self.id2idx[req.request_id] = idx
                self.total_tokens_in_memory += req.input_len

                loaded += 1
                if self.max_prepare is not None and loaded >= self.max_prepare:
                    break
        self._build_templates_from_current()
        self._reset_epoch_structures()
        return loaded

    def _build_templates_from_current(self) -> None:
        """
        Build immutable templates from current self.reqs.
        Called once after load(). If you ever append more reqs later,
        call this again to refresh templates.
        """
        tmp_buckets: Dict[int, List[int]] = defaultdict(list)
        for idx, req in enumerate(self.reqs):
            tmp_buckets[req.input_len].append(idx)

        # Store as tuples to guarantee immutability and low copy cost
        self._bucket_template = {L: tuple(idxs) for L, idxs in tmp_buckets.items()}
        self._sorted_template = sorted(self._bucket_template.keys())
    
    def _reset_epoch_structures(self) -> None:
        """
        Reset per-epoch mutable structures from immutable templates.
        O(U + total_indices) memory copy, no tokenization or scanning of reqs.
        """
        self.trained = [False] * len(self.reqs)
        # Deep-copy lists into deques for O(1) pops at the head
        self.len_buckets = {L: deque(list(idxs)) for L, idxs in self._bucket_template.items()}
        self.sorted_lengths = list(self._sorted_template)

    def advance_epoch(self) -> bool:
        if self.current_epoch >= self.total_epochs:
            return False
        self.current_epoch += 1
        self._reset_epoch_structures()
        return True

    def pop_best_under(
        self,
        max_tokens: int,
        exclude: list = None,
    ) -> Optional["Req"]:
        if not self.sorted_lengths:
            return None
        exclude_ids = [req.request_id for req in exclude] if exclude else []
        pos = bisect_right(self.sorted_lengths, max_tokens) - 1
        while pos >= 0:
            L = self.sorted_lengths[pos]
            dq = self.len_buckets.get(L)
            if not dq:
                pos -= 1
                continue
            for idx in dq:
                if self.trained[idx]:
                    continue
                req_id = self.req_ids[idx]  # O(1) lookup
                if req_id in exclude_ids:
                    continue
                return self.reqs[idx]
            pos -= 1
        return None
    

    def has_next(self) -> bool:
        return bool(self.len_buckets)

    def confirmed_trained(self, reqs: List[Req]) -> int:
        # Group indices by length
        by_len: Dict[int, set] = {}
        for req in reqs:
            idx = self.id2idx.get(req.request_id)
            if idx is None or self.trained[idx]:
                continue
            self.trained[idx] = True
            L = self.reqs[idx].input_len
            by_len.setdefault(L, set()).add(idx)

        marked = 0
        for L, to_remove in by_len.items():
            dq = self.len_buckets.get(L)
            if not dq:
                continue
            # Filter once
            kept = [i for i in dq if i not in to_remove]
            if kept:
                self.len_buckets[L] = deque(kept)
            else:
                del self.len_buckets[L]
                pos = bisect_left(self.sorted_lengths, L)
                if pos < len(self.sorted_lengths) and self.sorted_lengths[pos] == L:
                    del self.sorted_lengths[pos]
            marked += len(to_remove)
        return marked

    def ready_for_bwd(self):
        if self.pending_bwd_tokens >= self.max_saved_finetuning_tokens:
            return True
        elif self.pop_best_under(self.max_saved_finetuning_tokens - self.pending_bwd_tokens) is None:
            return True
        elif self.pending_bwd_tokens > 0 and not self.len_buckets:
            return True
        else:
            return False
    
    def update_finetuning_status_after_fwd(self, batch: Batch):
        self.confirmed_trained(batch.reqs)
        for req in batch.reqs:
            if req.is_finetuning:
                self.pending_bwd_tokens += req.input_len
    
    def update_finetuning_status_after_bwd(self, loss_list, num_processed_tokens):
        self.loss_list.extend(loss_list)
        self.finetuning_tokens_processed += num_processed_tokens
        self.pending_bwd_tokens -= num_processed_tokens
        bar_width = 50
        ratio = self.finetuning_tokens_processed / max(self.total_tokens_in_memory, 1)
        filled_len = int(bar_width * ratio)
        empty_len = bar_width - filled_len
        grey = "*"
        white = " "
        bar = grey * filled_len + white * empty_len
        print(f"Epoch: {self.current_epoch+1}/{self.total_epochs} [{bar}] {ratio:.1%} ", end="", flush=True)
        if self.finetuning_tokens_processed >= self.total_tokens_in_memory:
            self.epoch_avg_loss_list.append(np.mean(self.loss_list))
            print(f" Average Loss: {self.epoch_avg_loss_list[-1]:.6f}")
            self.loss_list = []
            self.advance_epoch()
            if self.current_epoch >= self.total_epochs:
                print("=== Loss List ===")
                for i, loss in enumerate(self.epoch_avg_loss_list):
                    print(f"Backward Epoch {i}: Loss = {loss:.6f}")
                print("=== End of Loss List ===", flush=True)
            else:
                self.finetuning_tokens_processed = 0
        else:
            print()

    def finetuning_is_finished(self):
        return self.current_epoch >= self.total_epochs and self.pending_bwd_tokens == 0