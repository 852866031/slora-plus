# SPDX-License-Identifier: Apache-2.0
"""Build a sglang `Req` from a FinetuningSample, for store-driven FT injection.

vLLM's ft_injector._make_request builds a vLLM Request (max_tokens=1,
is_finetuning=True, unique id, cache_salt to defeat prefix-cache). This is the
sglang equivalent: a prefill-only Req inserted into the scheduler's
waiting_queue each step, so FT is STORE-DRIVEN like vLLM (not client-tagged).

Opt-in: only used when SGLANG_DS_STORE_DRIVEN=1 (the default request-tagged path
stays untouched).
"""
from __future__ import annotations

import itertools
from typing import Any, List, Optional

_counter = itertools.count()


def make_ft_req(sample, tokenizer, eos_token_ids: Optional[Any] = None):
    """FinetuningSample → prefill-only sglang Req tagged is_finetuning=True.

    max_new_tokens=1 (SFT is forward-only; the prefill captures activations and
    the req retires the same step). Unique rid per injection. The sample is
    stashed on the req so the scheduler can claim/commit/release it in the store.
    """
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    sp = SamplingParams(max_new_tokens=1, temperature=0.0)
    try:
        sp.normalize(tokenizer)
    except Exception:
        pass  # normalize fills stop-token defaults; harmless if it no-ops

    rid = f"__ft__{next(_counter)}_{sample.request_id[:8]}"
    req = Req(
        rid,
        sample.text,
        list(sample.prompt_ids),
        sp,
        eos_token_ids=eos_token_ids,
    )
    req.tokenizer = tokenizer
    req.is_finetuning = True
    req._ft_sample = sample            # for store claim/commit/release
    if sample.adapter:
        req.lora_path = sample.adapter
    return req
