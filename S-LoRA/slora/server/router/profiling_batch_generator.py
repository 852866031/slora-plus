import copy
import random
import string
import uuid
from typing import List, Dict, Optional, Tuple

import numpy as np

# External interfaces from your codebase
from ..io_struct import Batch, Req
from ..tokenizer import get_tokenizer
from ..input_params import FinetuneParams
from ..sampling_params import SamplingParams

# ---------------------- Helpers ----------------------

def _infer_sampling_params(max_new_tokens: int) -> SamplingParams:
    return SamplingParams(
        do_sample=False,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=True,
        max_new_tokens=max_new_tokens,
        stop_sequences=[],
    )


def _ft_sampling_params() -> SamplingParams:
    return SamplingParams(
        do_sample=False,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=False,
        max_new_tokens=1,
        stop_sequences=[],
    )


def _random_sentence(num_words: int) -> str:
    words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))) for _ in range(max(1, num_words))]
    return " ".join(words).capitalize() + "."


# ---------------------- Main Class ----------------------
class ProfilingBatchGenerator:

    def __init__(
        self,
        finetune_params: FinetuneParams,
        inference_adapter_dir: str,
        model_weightdir: Optional[str] = None,
        tokenizor_mode: Optional[str] = None,
        trust_remote_code: bool = False,
        max_new_tokens_infer: int = 1,
        rng_seed: int = 42,
    ) -> None:
        self.ft_params = finetune_params
        self.inference_adapter_dir = inference_adapter_dir
        self.max_new_tokens_infer = max_new_tokens_infer
        self.rng = random.Random(rng_seed)
        random.seed(rng_seed)
        np.random.seed(rng_seed)

        try:
            self.tokenizer = get_tokenizer(
                model_weightdir or finetune_params.model_weightdir,
                tokenizor_mode or finetune_params.tokenizor_mode,
                trust_remote_code=trust_remote_code or finetune_params.trust_remote_code,
            )
        except Exception:
            self.tokenizer = get_tokenizer("huggyllama/llama-7b", tokenizor_mode or "auto")

        self.inference_batches: List[Batch] = []
        self.coserving_batches: List[Batch] = []

        # Exact targets
        self._inf_token_targets = [2000, 100, 500, 800, 850, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
        self._coserve_pairs = [(1000, 500), (1500, 100), (800, 100), (900, 100), (2000, 300), (1200, 200), (2000, 500), (3000, 200), (4000, 200)]

        # Pre-tokenized pool used to slice exact-length prompts
        self._token_pool: List[int] = []
        self._pool_cursor: int = 0
        self._POOL_MIN_LEN = 10000  # large enough to cover all batches without frequent wraps

    # ---------------------- Public ----------------------
    def prepare(self) -> None:
        """Populate self.inference_batches and self.coserving_batches with exact totals."""
        self._ensure_token_pool()

        # 1) Build 5 inference-only batches at exact sizes
        for total_tok in self._inf_token_targets:
            b = self._build_inference_batch_exact(total_tok)
            self.inference_batches.append(b)

        # 2) Build 3 co-serving batches at exact (inf, ft) splits
        for n_inf, n_ft in self._coserve_pairs:
            b = self._build_coserve_batch_exact(n_inf, n_ft)
            self.coserving_batches.append(b)

    # ---------------------- Pool & Slicing ----------------------
    def _ensure_token_pool(self) -> None:
        if len(self._token_pool) >= self._POOL_MIN_LEN:
            return
        # Build a long token stream by repeatedly tokenizing sentences and concatenating IDs
        pool: List[int] = []
        while len(pool) < self._POOL_MIN_LEN:
            sent = _random_sentence(self.rng.randint(8, 20))
            ids = self.tokenizer(sent).get("input_ids", [])
            if not ids:
                continue
            pool.extend(ids)
        self._token_pool = pool
        self._pool_cursor = 0

    def _take_slice(self, length: int) -> List[int]:
        """Return a contiguous slice of exactly `length` token IDs from the pool (wraps if needed)."""
        assert length > 0
        pool = self._token_pool
        n = len(pool)
        if self._pool_cursor + length <= n:
            sl = pool[self._pool_cursor : self._pool_cursor + length]
            self._pool_cursor += length
            return sl
        # wrap
        part1 = pool[self._pool_cursor :]
        needed = length - len(part1)
        part2 = pool[:needed]
        self._pool_cursor = needed
        return part1 + part2

    # ---------------------- Builders (Exact) ----------------------
    def _build_inference_batch_exact(self, total_tokens: int) -> Batch:
        # Decide how many requests; spread across 3–8 reqs for variety
        n_reqs = min(12, max(3, total_tokens // 150))  # heuristic
        lengths = self._exact_partition(total_tokens, n_reqs)
        reqs: List[Req] = []
        for L in lengths:
            ids = self._take_slice(L)
            reqs.append(self._new_infer_req_from_ids(ids, max_new_tokens=2))
        return Batch(uuid.uuid4().hex, reqs)

    def _build_coserve_batch_exact(self, n_inf: int, n_ft: int) -> Batch:
        # Partition into several requests (e.g., 4–8 inf reqs, 1–3 ft reqs)
        n_inf_reqs = min(12, max(4, n_inf // 150))
        n_ft_reqs = min(3, max(1, n_ft // 150))
        inf_lengths = self._exact_partition(n_inf, n_inf_reqs)
        ft_lengths = self._exact_partition(n_ft, n_ft_reqs)

        reqs: List[Req] = []
        for L in inf_lengths:
            ids = self._take_slice(L)
            reqs.append(self._new_infer_req_from_ids(ids))
        for L in ft_lengths:
            ids = self._take_slice(L)
            reqs.append(self._new_ft_req_from_ids(ids))

        # Validate strict ratio
        infer_tokens = sum(r.input_len for r in reqs if not r.is_finetuning)
        ft_tokens = sum(r.input_len for r in reqs if r.is_finetuning)
        assert infer_tokens == n_inf and ft_tokens == n_ft, "Exact totals must match"

        return Batch(uuid.uuid4().hex, reqs)

    # ---------------------- Partitioning ----------------------
    def _exact_partition(self, total: int, n_parts: int) -> List[int]:
        """Split `total` into `n_parts` positive integers that sum exactly to `total`."""
        if n_parts <= 1:
            return [total]
        base = total // n_parts
        rem = total % n_parts
        # Distribute the remainder over the first `rem` parts
        parts = [base + 1] * rem + [base] * (n_parts - rem)
        # Shuffle lightly for variety while keeping determinism via RNG
        self.rng.shuffle(parts)
        return parts

    # ---------------------- Request Factories from IDs ----------------------
    def _new_infer_req_from_ids(self, ids: List[int], max_new_tokens = None) -> Req:
        # Decode text only for readability/logging; prompt_ids are authoritative
        try:
            text = self.tokenizer.decode(ids)
        except Exception:
            text = f"<synthetic {len(ids)} tok>"
        return Req(
            adapter_dir=self.inference_adapter_dir,
            request_id=uuid.uuid4().hex,
            prompt_ids=ids,
            sample_params=_infer_sampling_params(self.max_new_tokens_infer if max_new_tokens is None else max_new_tokens),
            is_finetuning=False,
            needs_to_notify_detokenize=True,
            text=text,
        )

    def _new_ft_req_from_ids(self, ids: List[int]) -> Req:
        try:
            text = self.tokenizer.decode(ids)
        except Exception:
            text = f"<synthetic {len(ids)} tok>"
        return Req(
            adapter_dir=self.ft_params.finetuning_lora_path,
            request_id=uuid.uuid4().hex,
            prompt_ids=ids,
            sample_params=_ft_sampling_params(),
            is_finetuning=True,
            needs_to_notify_detokenize=False,
            text=text,
        )


# ---------------------- Optional: pretty summary ----------------------
def summarize_batches(batches: List[Batch]) -> List[Tuple[str, int, int]]:
    out = []
    for b in batches:
        total = sum(r.input_len for r in b.reqs)
        out.append((b.batch_id, total, len(b.reqs)))
    return out
