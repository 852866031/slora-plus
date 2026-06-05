# Ported from DeltaServe-vLLM dserve-vllm/vllm/deltaserve/ft_scheduler.py — sglang port (Phase 7).
"""Mixin that injects FT-aware behaviour into the live Scheduler.

Applied at runtime via ``self.__class__`` swap inside ``Scheduler.__init__``
when ``finetune_config.enable_finetuning`` is True, so the base Scheduler's
inference fast path stays byte-identical when FT is disabled.
"""

from __future__ import annotations

from typing import Any, List


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
    # Batch selection — additive hook
    # ------------------------------------------------------------------
    # sglang has no `_select_batch`; the closest equivalent called by
    # event_loop_normal is `get_next_batch_to_run`. We override it here and
    # consult coordinator.reserve() before letting FT tokens ride along.
    def _select_batch(self, *args, **kwargs):
        """Additive admission gate. Delegates to the base scheduler's batch
        selection (``get_next_batch_to_run``) and asks the coordinator
        whether any FT tokens may ride along this step."""
        batch = super().get_next_batch_to_run(*args, **kwargs)
        coord = getattr(self, "finetune_coordinator", None)
        if coord is not None and batch is not None:
            # Try to admit one capacity-sized FT slice this step. The actual
            # FT-token attachment to the batch lands in Phase 8 (FinetuneInjector);
            # here we only consult the gate so the test can verify the contract.
            if coord.reserve(coord.per_step_budget):
                coord.note_injection(batch)
        return batch

    def get_next_batch_to_run(self, *args, **kwargs):
        # Route the base call through the FT-aware selector so the event loop
        # picks up admission decisions without us editing event_loop_normal
        # in the base Scheduler.
        return self._select_batch(*args, **kwargs)

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
