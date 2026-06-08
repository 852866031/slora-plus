# SPDX-License-Identifier: Apache-2.0
"""Build a sglang `Req` from a FinetuningSample, for store-driven FT injection.

从一个 FinetuningSample（语料样本）构造出 sglang 的 `Req`，用于"语料驱动"的微调注入。

vLLM's ft_injector._make_request builds a vLLM Request (max_tokens=1,
is_finetuning=True, unique id, cache_salt to defeat prefix-cache). This is the
sglang equivalent: a prefill-only Req inserted into the scheduler's
waiting_queue each step, so FT is STORE-DRIVEN like vLLM (not client-tagged).

中文说明：
vLLM 的 ft_injector._make_request 会构造一个 vLLM Request（max_tokens=1、
is_finetuning=True、唯一 id、用 cache_salt 来绕过前缀缓存）。本文件是它在 sglang
里的等价实现：每个调度步把一个"只做 prefill"的 Req 塞进调度器的 waiting_queue，
从而让微调像 vLLM 一样由"语料库"驱动（而不是由客户端请求打标签来驱动）。

Opt-in: only used when SGLANG_DS_STORE_DRIVEN=1 (the default request-tagged path
stays untouched).

按需开启：仅当 SGLANG_DS_STORE_DRIVEN=1 时启用；默认的"客户端请求打标签"路径保持不变。
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
