# SPDX-License-Identifier: Apache-2.0
"""Finetuning CORPUS store — faithful port of DeltaServe-vLLM's
deltaserve/finetuning_store.py (only the `dprint` logging shim differs).

微调"语料库"存储 —— 忠实移植自 DeltaServe-vLLM 的 deltaserve/finetuning_store.py
（仅 `dprint` 日志垫片不同）。

NOTE: this is distinct from `deltaserve/finetuning_store.py` in this fork, which
is a KV-slot reservation helper (different role, same vLLM name collision). This
module is the corpus sample store that makes sglang co-serving STORE-DRIVEN —
continuous FT injection from a tokenized corpus, like vLLM — instead of
client-request-tagged. Loads/tokenizes the corpus once, serves length-bucketed
untrained samples, with claim/commit/release for in-flight FT steps + epochs.
Pure Python, no GPU coupling.

中文说明：
注意，本文件不同于本 fork 里的 `deltaserve/finetuning_store.py`（那个是 KV 槽位预留
辅助类，职责不同，只是与 vLLM 撞了同名）。本模块是"语料样本存储"，让 sglang 的协同服务
变成"语料驱动"—— 像 vLLM 一样从一份已分词的语料里持续注入微调，而不是靠客户端请求打标签。
它一次性加载并分词语料，按长度分桶提供"尚未训练"的样本，并通过 claim/commit/release
管理在途的微调步与 epoch。纯 Python，不与 GPU 耦合。
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
    """One tokenized finetuning sample (prefill-only; never decoded).

    中文：一条已分词的微调样本（只做 prefill，永不解码生成）。"""

    request_id: str
    prompt_ids: List[int]
    text: str
    adapter: Optional[str] = None

    @property
    def input_len(self) -> int:
        return len(self.prompt_ids)


class FinetuningStore:
    """Length-bucketed store of tokenized finetuning samples (vLLM-faithful).

    中文：按长度分桶的微调样本存储（与 vLLM 实现一致）。分桶让"在 N 个 token 预算内挑
    最大/最小的未训练样本"成为 O(distinct lengths) 的查找；claim/commit/release 三个
    动作配合调度器管理在途样本与 epoch 推进。"""

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
        max_saved_finetuning_tokens are dropped (can't fit the buffer).

        中文：读取并分词整份语料。input_len 超过 max_saved_finetuning_tokens 的样本会被
        丢弃（放不进激活缓冲）。返回成功载入的样本数。"""
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
        """Largest untrained sample with input_len <= max_tokens (peek).

        中文：在 input_len <= max_tokens 的前提下，返回"最大"的未训练样本（只 peek，不移除；
        实际占用要再调用 claim）。用于贪心地把每个反向批次填到接近 token 预算。"""
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
        """Smallest untrained sample (ascending length; peek).

        中文：返回"最小"的未训练样本（按长度升序；只 peek，不移除）。"""
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
        """中文：把这些样本标记为"在途"——从长度桶里移除，加入 _claimed 集合，于是后续
        pop_* 不会再选中它们。反向真正完成后再调用 commit_claimed 落实为已训练，或失败时
        用 release_claimed 退回。返回实际新标记的数量。"""
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
        """中文：反向成功后把"在途"样本落实为"已训练"（从 _claimed 移到 trained），它们
        在本 epoch 内不再被选中。返回实际落实的数量。"""
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
        """中文：反向失败/丢弃时把"在途"样本退回长度桶（重新可选），并从 _claimed 移除。
        返回实际退回的数量。"""
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
        """中文：本 epoch 的样本都训练完后，推进到下一个 epoch（重置 trained/桶/claimed）。
        若已到 total_epochs 上限、或还有在途样本（_claimed 非空）则不推进，返回 False。"""
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
