# SPDX-License-Identifier: Apache-2.0
"""Finetuning CORPUS store — faithful port of DeltaServe-vLLM's
deltaserve/finetuning_store.py (only the `dprint` logging shim differs).

NOTE: this is distinct from `deltaserve/finetuning_store.py` in this fork, which
is a KV-slot reservation helper (different role, same vLLM name collision). This
module is the corpus sample store that makes sglang co-serving STORE-DRIVEN —
continuous FT injection from a tokenized corpus, like vLLM — instead of
client-request-tagged. Loads/tokenizes the corpus once, serves length-bucketed
untrained samples, with claim/commit/release for in-flight FT steps + epochs.
Pure Python, no GPU coupling.
"""

import logging
import uuid
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

_log = logging.getLogger(__name__)
def dprint(*a, **kw):  # noqa: E704 - logging shim matching the vLLM original
    _log.info(" ".join(str(x) for x in a))


@dataclass
class FinetuningSample:
    """One tokenized finetuning sample (prefill-only; never decoded)."""

    request_id: str
    prompt_ids: List[int]
    text: str
    adapter: Optional[str] = None

    @property
    def input_len(self) -> int:
        return len(self.prompt_ids)


class FinetuningStore:
    """Length-bucketed store of tokenized finetuning samples (vLLM-faithful)."""

    def __init__(
        self,
        data_path: Optional[str],
        tokenize: Callable[[str], List[int]],
        adapter: Optional[str] = None,
        total_epochs: int = 1,
        max_saved_finetuning_tokens: int = 256,
        max_prepare: Optional[int] = None,
    ) -> None:
        self.data_path = data_path
        self.tokenize = tokenize
        self.adapter = adapter
        self.total_epochs = int(total_epochs)
        self.max_saved_finetuning_tokens = int(max_saved_finetuning_tokens)
        self.max_prepare = max_prepare

        self.samples: List[FinetuningSample] = []
        self.id2idx: Dict[str, int] = {}
        self.trained: List[bool] = []
        self.current_epoch = 0
        self.total_tokens_in_memory = 0

        self.len_buckets: Dict[int, deque] = defaultdict(deque)
        self.sorted_lengths: List[int] = []
        self._claimed: Set[int] = set()
        self._bucket_template: Dict[int, Tuple[int, ...]] = {}
        self._sorted_template: List[int] = []

    # -- loading -----------------------------------------------------------

    def load(self) -> int:
        """Read + tokenize the corpus. Samples with input_len >
        max_saved_finetuning_tokens are dropped (can't fit the buffer)."""
        if self.data_path is None:
            return 0
        loaded = 0
        dropped_lens: List[int] = []
        cap = self.max_saved_finetuning_tokens
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                prompt_ids = list(self.tokenize(text))
                if len(prompt_ids) > cap:
                    dropped_lens.append(len(prompt_ids))
                    continue
                sample = FinetuningSample(
                    request_id=uuid.uuid4().hex,
                    prompt_ids=prompt_ids,
                    text=text,
                    adapter=self.adapter,
                )
                idx = len(self.samples)
                self.samples.append(sample)
                self.id2idx[sample.request_id] = idx
                self.total_tokens_in_memory += sample.input_len
                loaded += 1
                if self.max_prepare is not None and loaded >= self.max_prepare:
                    break
        self._build_templates()
        self._reset_epoch_structures()
        dprint(
            f"[ft-store] loaded {loaded} samples from {self.data_path} | "
            f"{self.total_tokens_in_memory} tokens, "
            f"{len(self._sorted_template)} distinct lengths "
            f"(min={self._sorted_template[0] if self._sorted_template else 0}, "
            f"max={self._sorted_template[-1] if self._sorted_template else 0})"
        )
        if dropped_lens:
            dprint(f"[ft-store] dropped {len(dropped_lens)} oversized samples "
                   f"(> {cap} tokens); lengths {sorted(dropped_lens)}.")
        return loaded

    def _build_templates(self) -> None:
        tmp: Dict[int, List[int]] = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            tmp[sample.input_len].append(idx)
        self._bucket_template = {length: tuple(idxs) for length, idxs in tmp.items()}
        self._sorted_template = sorted(self._bucket_template)

    def _reset_epoch_structures(self) -> None:
        self.trained = [False] * len(self.samples)
        self.len_buckets = {
            length: deque(idxs) for length, idxs in self._bucket_template.items()
        }
        self.sorted_lengths = list(self._sorted_template)
        self._claimed = set()

    # -- selection ---------------------------------------------------------

    def pop_best_under(self, max_tokens: int,
                       exclude: Optional[List[FinetuningSample]] = None
                       ) -> Optional[FinetuningSample]:
        """Largest untrained sample with input_len <= max_tokens (peek)."""
        if not self.sorted_lengths:
            return None
        exclude_ids = {s.request_id for s in exclude} if exclude else set()
        pos = bisect_right(self.sorted_lengths, max_tokens) - 1
        while pos >= 0:
            length = self.sorted_lengths[pos]
            for idx in self.len_buckets.get(length, ()):
                if self.trained[idx]:
                    continue
                if self.samples[idx].request_id in exclude_ids:
                    continue
                return self.samples[idx]
            pos -= 1
        return None

    def pop_next(self, exclude: Optional[List[FinetuningSample]] = None
                 ) -> Optional[FinetuningSample]:
        """Smallest untrained sample (ascending length; peek)."""
        if not self.sorted_lengths:
            return None
        exclude_ids = {s.request_id for s in exclude} if exclude else set()
        for length in self.sorted_lengths:
            dq = self.len_buckets.get(length)
            if not dq:
                continue
            for idx in dq:
                if self.trained[idx]:
                    continue
                if self.samples[idx].request_id in exclude_ids:
                    continue
                return self.samples[idx]
        return None

    # -- marking / epochs --------------------------------------------------

    def claim(self, samples: List[FinetuningSample]) -> int:
        by_len: Dict[int, set] = {}
        for sample in samples:
            idx = self.id2idx.get(sample.request_id)
            if idx is None or self.trained[idx] or idx in self._claimed:
                continue
            self._claimed.add(idx)
            by_len.setdefault(self.samples[idx].input_len, set()).add(idx)
        marked = 0
        for length, to_remove in by_len.items():
            dq = self.len_buckets.get(length)
            if not dq:
                continue
            kept = [i for i in dq if i not in to_remove]
            if kept:
                self.len_buckets[length] = deque(kept)
            else:
                del self.len_buckets[length]
                p = bisect_left(self.sorted_lengths, length)
                if p < len(self.sorted_lengths) and self.sorted_lengths[p] == length:
                    del self.sorted_lengths[p]
            marked += len(to_remove)
        return marked

    def commit_claimed(self, samples: List[FinetuningSample]) -> int:
        marked = 0
        for sample in samples:
            idx = self.id2idx.get(sample.request_id)
            if idx is None or idx not in self._claimed:
                continue
            self._claimed.discard(idx)
            self.trained[idx] = True
            marked += 1
        return marked

    def release_claimed(self, samples: List[FinetuningSample]) -> int:
        by_len: Dict[int, List[int]] = {}
        for sample in samples:
            idx = self.id2idx.get(sample.request_id)
            if idx is None or idx not in self._claimed:
                continue
            self._claimed.discard(idx)
            by_len.setdefault(self.samples[idx].input_len, []).append(idx)
        n = 0
        for length, idxs in by_len.items():
            dq = self.len_buckets.get(length)
            if dq is None:
                self.len_buckets[length] = deque(idxs)
                p = bisect_left(self.sorted_lengths, length)
                if (p == len(self.sorted_lengths)
                        or self.sorted_lengths[p] != length):
                    self.sorted_lengths.insert(p, length)
            else:
                dq.extend(idxs)
            n += len(idxs)
        return n

    def advance_epoch(self) -> bool:
        if self.current_epoch >= self.total_epochs:
            return False
        if self._claimed:
            return False
        self.current_epoch += 1
        self._reset_epoch_structures()
        return True

    def has_next(self) -> bool:
        return bool(self.len_buckets)

    def has_claimed(self) -> bool:
        return bool(self._claimed)

    def __len__(self) -> int:
        return len(self.samples)
