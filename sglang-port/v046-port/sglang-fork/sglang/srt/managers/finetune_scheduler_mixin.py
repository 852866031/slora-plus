# Ported from DeltaServe-vLLM dserve-vllm/vllm/deltaserve/ft_scheduler.py — sglang port (Phase 7).
"""Mixin that injects FT-aware behaviour into the live Scheduler.

Applied at runtime via ``self.__class__`` swap inside ``Scheduler.__init__``
when ``finetune_config.enable_finetuning`` is True, so the base Scheduler's
inference fast path stays byte-identical when FT is disabled.
"""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


class FinetuneSchedulerMixin:
    """FT scheduling hooks. The base ``Scheduler`` is the second base in the
    runtime-synthesised MRO, so every ``super().method(...)`` call here lands
    on the original Scheduler implementation."""

    # ------------------------------------------------------------------
    # Request intake
    # ------------------------------------------------------------------
    def process_input_requests(self, recv_reqs: List[Any]) -> None:
        """Route FT requests to the FT path; pass the rest to the base
        Scheduler's existing intake.

        FT requests are tagged upstream by the FT injector (Phase 8 will
        attach the real marker); for Phase 7 we just split on a duck-typed
        ``is_finetune`` attribute so the contract is testable and a
        non-FT workload behaves exactly as before."""
        if not recv_reqs:
            return super().process_input_requests(recv_reqs)

        ft_reqs = [r for r in recv_reqs if getattr(r, "is_finetune", False)]
        inf_reqs = [r for r in recv_reqs if not getattr(r, "is_finetune", False)]

        if inf_reqs:
            super().process_input_requests(inf_reqs)
        if ft_reqs:
            self._handle_ft_requests(ft_reqs)

    def _handle_ft_requests(self, ft_reqs: List[Any]) -> None:
        """Phase 7 stub. Phase 8 will hand these to the FinetuneInjector to
        enqueue against the FT corpus + activation buffers. We keep them on
        the coordinator side so the mixin contract is exercised in tests."""
        coord = getattr(self, "finetune_coordinator", None)
        if coord is None:
            return
        # Stage requests on the coordinator; Phase 8 ports the injector.
        pending = getattr(coord, "_pending_ft_reqs", None)
        if pending is None:
            coord._pending_ft_reqs = []
            pending = coord._pending_ft_reqs
        pending.extend(ft_reqs)

    # ------------------------------------------------------------------
    # Store-driven SLO-admitted FT injection (vLLM-parity, opt-in).
    #
    # When SGLANG_DS_STORE_DRIVEN=1, FT comes from a corpus store (like vLLM's
    # ft_scheduler), NOT client-tagged requests: each step we SLO-gate-admit
    # samples from the store, build prefill-only Reqs, and push them onto the
    # waiting_queue BEFORE the base batch selection picks them up. Hooked on
    # get_next_batch_to_run, which is live under BOTH event loops (default
    # overlap included). Default (flag off) leaves the base path byte-identical.
    # ------------------------------------------------------------------
    def _ensure_ft_store(self):
        if getattr(self, "_ft_store_inited", False):
            return
        self._ft_store_inited = True
        self._ft_store = None
        import os
        if os.environ.get("SGLANG_DS_STORE_DRIVEN", "0") != "1":
            return
        try:
            from sglang.srt.deltaserve.finetuning_corpus import FinetuningStore
            data_path = os.environ.get("SGLANG_DS_FT_DATA") or getattr(
                getattr(self, "finetune_config", None), "data_path", None)
            if not data_path:
                logger.warning("[DeltaServe] store-driven FT on but no "
                               "SGLANG_DS_FT_DATA / finetune_config.data_path")
                return
            cap = int(getattr(self.finetune_config, "max_saved_finetuning_tokens", 256))
            epochs = int(os.environ.get("SGLANG_DS_FT_EPOCHS", "100"))
            store = FinetuningStore(
                data_path, tokenize=lambda s: self.tokenizer.encode(s),
                total_epochs=epochs, max_saved_finetuning_tokens=cap)
            n = store.load()
            self._ft_store = store
            self._ft_budget = cap
            self._ft_injected = 0
            logger.warning(f"[DeltaServe] store-driven FT: loaded {n} samples "
                           f"from {data_path} (cap={cap}, epochs={epochs})")
        except Exception as e:
            logger.warning(f"[DeltaServe] store-driven FT init failed: {e}")
            self._ft_store = None

    def _ft_in_flight(self) -> bool:
        """True if an FT batch is already waiting/running — throttle to one at a
        time (mirrors vLLM's fill-buffer-then-backward cadence)."""
        if any(getattr(r, "is_finetuning", False) for r in self.waiting_queue):
            return True
        rb = getattr(self, "running_batch", None)
        if rb is not None and getattr(rb, "reqs", None):
            if any(getattr(r, "is_finetuning", False) for r in rb.reqs):
                return True
        return False

    def _admit_and_inject_ft(self):
        store = getattr(self, "_ft_store", None)
        if store is None:
            return
        from sglang.srt.deltaserve.gates import is_finetuning_started
        if not is_finetuning_started():
            return
        if self._ft_in_flight():
            return  # one FT batch in flight at a time
        if not store.has_next() and not store.advance_epoch():
            return  # corpus exhausted across all epochs
        # SLO gate: skip admit when inference decode is already SLO-stressed.
        try:
            from sglang.srt.deltaserve.coserve_slo import get_slo
            slo = get_slo()
            rec = slo._last_decode_dur
            if slo.estimator.is_ready and rec is not None \
                    and rec > slo.defer_frac * slo.max_tbt_slo:
                return
        except Exception:
            pass
        # Greedy pack up to the per-step FT token budget (largest-first).
        admitted, remaining = [], int(getattr(self, "_ft_budget", 256))
        while remaining > 0:
            s = store.pop_best_under(remaining, exclude=admitted)
            if s is None:
                break
            admitted.append(s)
            remaining -= s.input_len
        if not admitted:
            return
        store.claim(admitted)
        from sglang.srt.deltaserve.ft_inject import make_ft_req
        eos = getattr(getattr(self, "model_config", None), "hf_eos_token_id", None)
        for s in admitted:
            try:
                self.waiting_queue.append(make_ft_req(s, self.tokenizer, eos_token_ids=eos))
                self._ft_injected += 1
            except Exception as e:
                logger.warning(f"[DeltaServe] FT inject failed: {e}")
                store.release_claimed([s])

    def get_next_batch_to_run(self, *args, **kwargs):
        self._ensure_ft_store()
        if getattr(self, "_ft_store", None) is not None:
            try:
                self._admit_and_inject_ft()
            except Exception as e:
                logger.warning(f"[DeltaServe] FT admit/inject failed: {e}")
        return super().get_next_batch_to_run(*args, **kwargs)

    # ------------------------------------------------------------------
    # Event loop — wrap prefill with pause/resume signals
    # ------------------------------------------------------------------
    def event_loop_normal(self) -> None:
        """Minimal override of the base event loop. Single pause-site,
        single resume-site around the per-step batch dispatch — the
        coordinator's no-op channel keeps this cheap when the backward
        child isn't connected.

        We deliberately do NOT touch ``event_loop_overlap``; that path
        needs more careful surgery and is out of scope for Phase 7.
        """
        # Reuse the base loop's machinery but interpose pause/resume around
        # each iteration's prefill dispatch. We inline the loop here rather
        # than refactor the base to keep the diff scope to this mixin.
        from sglang.srt import environ as envs  # local import — base does the same

        while True:
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                coord = getattr(self, "finetune_coordinator", None)
                if coord is not None:
                    coord.gpu_pause_backward()
                try:
                    result = self.run_batch(batch)
                    self.process_batch_result(batch, result)
                finally:
                    if coord is not None:
                        coord.gpu_resume_backward()
            else:
                self.on_idle()

            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.invariant_checker.self_check_during_busy()
