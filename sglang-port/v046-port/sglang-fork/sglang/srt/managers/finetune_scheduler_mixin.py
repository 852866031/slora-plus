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
    #
    # 中文：当 SGLANG_DS_STORE_DRIVEN=1 时，微调来自"语料库存储"（与 vLLM 的 ft_scheduler
    # 一致），而不是客户端打标签的请求：每个调度步我们用 SLO 门控从语料里准入若干样本，构造
    # "只做 prefill"的 Req，在 base 的批次选择挑走它们之前先塞进 waiting_queue。挂在
    # get_next_batch_to_run 上 —— 这个方法在两种 event loop（包括默认的 overlap）下都在跑。
    # 默认（关闭该标志）时 base 路径逐字节不变。
    # ------------------------------------------------------------------
    def _ensure_ft_store(self):
        """Lazy-load the FT corpus store on first call. Store-driven FT is now the
        DEFAULT (vLLM-faithful) whenever a corpus is resolvable — from
        ``SGLANG_DS_FT_DATA`` or ``finetune_config.data_path`` (the latter wired
        from ``--finetune-data-path``). Explicit opt-out: ``SGLANG_DS_STORE_DRIVEN=0``.
        With no corpus configured the store stays None and the legacy
        client-request-tagged path is untouched, so existing harnesses still work.

        中文：首次调用时懒加载语料库。store-driven 现在是默认行为（与 vLLM 一致）：只要能解析到
        语料路径（SGLANG_DS_FT_DATA 或 finetune_config.data_path，后者由 --finetune-data-path
        注入）就开启。显式关闭用 SGLANG_DS_STORE_DRIVEN=0。若没有配置语料，store 保持 None，
        原有的"客户端请求打标签"路径保持不变，旧脚本照常工作。只初始化一次。"""
        if getattr(self, "_ft_store_inited", False):
            return
        self._ft_store_inited = True
        self._ft_store = None
        import os
        flag = os.environ.get("SGLANG_DS_STORE_DRIVEN")  # None=auto, "0"=off, "1"=force
        if flag == "0":
            return  # explicit opt-out
        try:
            from sglang.srt.deltaserve.finetuning_corpus import FinetuningStore
            data_path = os.environ.get("SGLANG_DS_FT_DATA") or getattr(
                getattr(self, "finetune_config", None), "data_path", None)
            if not data_path:
                # No corpus → default-off (legacy request-tagged path stays live).
                # Only warn if store-driven was explicitly forced on.
                if flag == "1":
                    logger.warning("[DeltaServe] SGLANG_DS_STORE_DRIVEN=1 but no "
                                   "SGLANG_DS_FT_DATA / finetune_config.data_path "
                                   "(--finetune-data-path); FT will not run.")
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
            self._ft_unspent_prefill = 0   # leaky-bucket credit (Phase D shaper)
            logger.warning(f"[DeltaServe] store-driven FT: loaded {n} samples "
                           f"from {data_path} (cap={cap}, epochs={epochs})")
        except Exception as e:
            logger.warning(f"[DeltaServe] store-driven FT init failed: {e}")
            self._ft_store = None

    def _ft_in_flight(self) -> bool:
        """True if an FT batch is already waiting/running — throttle to one at a
        time (mirrors vLLM's fill-buffer-then-backward cadence).

        中文：若已有微调批次在 waiting/running 队列中，返回 True —— 同一时刻只允许一个
        微调批次（对齐 vLLM "先填缓冲再反向" 的节奏）。"""
        if any(getattr(r, "is_finetuning", False) for r in self.waiting_queue):
            return True
        rb = getattr(self, "running_batch", None)
        if rb is not None and getattr(rb, "reqs", None):
            if any(getattr(r, "is_finetuning", False) for r in rb.reqs):
                return True
        return False

    def _ft_backward_client(self):
        """Reach the parent-side BackwardClient (lives in model_runner) from the
        scheduler. Path differs by worker class: overlap wraps the real worker in
        ``.worker``; the non-overlap worker exposes ``.model_runner`` directly.
        Returns None when the subprocess backward isn't in use (in-process path).

        中文：从调度器一路找到父进程侧的 BackwardClient（它挂在 model_runner 上）。不同的
        worker 实现路径不同：overlap 模式把真正的 worker 包在 ``.worker`` 里；非 overlap
        模式直接暴露 ``.model_runner``。未启用子进程反向（走进程内路径）时返回 None。"""
        tw = getattr(self, "tp_worker", None)
        if tw is None:
            return None
        mr = getattr(tw, "model_runner", None) or getattr(
            getattr(tw, "worker", None), "model_runner", None)
        if mr is None:
            return None
        return getattr(mr, "_ds_bwd_client", None)

    def _admit_and_inject_ft(self):
        """中文：每个调度步的"准入 + 注入"。依次检查：微调门是否已开（is_finetuning_started）、
        是否已有微调批次在途（一次一个）、反向子进程是否繁忙（按反向节奏配速）、SLO 是否吃紧、
        语料是否还有样本。通过后用 pop_best_under 贪心装满到 _ft_budget（token 预算），claim
        这些样本，构造 prefill-only Req 塞进 waiting_queue。"""
        store = getattr(self, "_ft_store", None)
        if store is None:
            return
        from sglang.srt.deltaserve.gates import is_finetuning_started
        if not is_finetuning_started():
            return
        # [forward_interruptible / tier A] OPT-IN inference-first guard. When
        # SGLANG_DS_FT_TIER_A=1, do NOT inject FT while any inference (non-FT)
        # request is waiting to prefill — inference always wins, so the FT prefill
        # never shares/precedes the inference prefill batch (mirrors vLLM's
        # would_step_be_ft_only / "if self.waiting: return False"). Default OFF:
        # under a dense timeline the waiting queue almost always holds inference,
        # so a blanket guard would STARVE FT and regress the proven continuous-fire
        # throughput — and parity is already met without it (the backward is
        # MPS-isolated and FT prefills are tiny). Leave it as a knob for workloads
        # that value TTFT protection over FT throughput. The heavier tier-C
        # mid-forward abort is intentionally not ported (marginal on co-serve;
        # see EXPERIMENTS.md S-store-driven / S-forward-interruptible).
        # 中文：[tier A] 按需开启的"推理优先"门控。SGLANG_DS_FT_TIER_A=1 时，只要还有推理
        # （非微调）请求在等待 prefill，就不注入微调 —— 推理永远优先（对应 vLLM 的
        # would_step_be_ft_only）。默认关闭：密集时间线下等待队列几乎总有推理，一刀切会饿死
        # 微调、回退掉已验证的连续反向吞吐；而且不开它也已达到 parity（反向被 MPS 隔离、微调
        # prefill 很小）。保留为"更看重 TTFT 而非微调吞吐"场景的旋钮。更重的 tier-C 前向中途
        # 中止刻意不移植（协同服务下收益边际，见 EXPERIMENTS.md）。
        import os as _os
        if _os.environ.get("SGLANG_DS_FT_TIER_A", "0") == "1" and \
                any(not getattr(r, "is_finetuning", False) for r in self.waiting_queue):
            return
        if self._ft_in_flight():
            return  # one FT batch in flight at a time
        # Pace injection to the BACKWARD cadence, not the (fast) prefill cadence.
        # The backward subprocess is single-flight: if we keep injecting FT
        # prefills while it's still training the previous batch, those backwards
        # are DROPPED (BackwardClient throttle) and FT silently stalls — exactly
        # the bug the first 8B store-driven run hit (5 fires then nothing).
        # Gate on is_busy() so we admit the next FT batch only after the child
        # consumes the current one (vLLM-faithful fill-then-backward cadence).
        # 中文：把注入节奏对齐到"反向"节奏，而不是（很快的）prefill 节奏。反向子进程同一时刻
        # 只跑一个：若在它还在训练上一批时继续灌入微调 prefill，这些反向会被丢弃（见
        # BackwardClient 的节流），微调就会悄悄停滞 —— 这正是第一次 8B 语料驱动跑出的 bug
        # （只触发 5 次反向后就再无动静）。用 is_busy() 门控：只有子进程消化完当前批次后，
        # 才准入下一批微调（与 vLLM 的"先填缓冲、再反向"节奏一致）。
        bc = self._ft_backward_client()
        if bc is not None and bc.is_busy():
            return
        if not store.has_next() and not store.advance_epoch():
            return  # corpus exhausted across all epochs
        # Phase D: SLO-aware iterative admission (replaces the coarse greedy pack).
        # admit_ft_to_step() predicts the upcoming step's time and admits FT
        # samples one-by-one only while BOTH the TTFT (waiting prefill) and TBT
        # (running decode) SLOs still hold — the dual-SLO controller from vLLM's
        # ft_scheduler.admit_ft_to_step. It already claim()s the admitted samples.
        # 中文：[Phase D] SLO 感知的迭代式准入（取代粗粒度的贪心装包）。admit_ft_to_step()
        # 预测即将执行的这一步的耗时，逐条准入微调样本，且仅在 TTFT（等待中的 prefill）与
        # TBT（运行中的 decode）两个 SLO 都仍满足时才继续 —— 对应 vLLM ft_scheduler 的
        # 双 SLO 控制器。它内部已对准入样本做了 claim()。
        admitted = self.admit_ft_to_step()
        if not admitted:
            return
        from sglang.srt.deltaserve.ft_inject import make_ft_req
        eos = getattr(getattr(self, "model_config", None), "hf_eos_token_id", None)
        for s in admitted:
            try:
                self.waiting_queue.append(make_ft_req(s, self.tokenizer, eos_token_ids=eos))
                self._ft_injected += 1
            except Exception as e:
                logger.warning(f"[DeltaServe] FT inject failed: {e}")
                store.release_claimed([s])

    def _current_step_features(self):
        """Estimate the inference composition of the step about to run (BEFORE FT
        is added): decode part from the running batch (exact), prefill part from
        the waiting-queue head within the prefill token budget. Returns
        (StepFeatures, earliest_waiting_arrival_walltime|None). Port of vLLM
        ft_scheduler._current_step_features (§1.7 dual-SLO surface).

        中文：估计即将执行的这一步在"加入微调之前"的推理构成：decode 部分来自 running_batch
        （精确），prefill 部分来自 waiting_queue 队首、受 prefill token 预算约束。返回
        (StepFeatures, 最早等待请求的到达墙钟时间|None)，供 TTFT 截止时间与 TBT 预测使用。"""
        from sglang.srt.deltaserve.estimator import StepFeatures
        b_d = 0
        k = 0.0
        rb = getattr(self, "running_batch", None)
        if rb is not None and getattr(rb, "reqs", None):
            for req in rb.reqs:
                if getattr(req, "is_finetuning", False):
                    continue
                b_d += 1
                k += float(getattr(req, "seqlen", 0) or 0)   # decode KV context
        budget = int(getattr(self, "max_prefill_tokens", 0) or 8192)
        remaining = max(0, budget - b_d)   # decode consumes ~1 token/req
        t_in = 0.0
        prefill_lens = []
        earliest = None
        for req in self.waiting_queue:
            if getattr(req, "is_finetuning", False):
                continue
            wq = float(getattr(getattr(req, "time_stats", None),
                               "wait_queue_entry_time", 0.0) or 0.0)
            if earliest is None and wq > 0:
                earliest = wq
            if remaining <= 0:
                continue
            prompt_len = len(getattr(req, "origin_input_ids", ()) or ())
            computed = len(getattr(req, "prefix_indices", ()) or ())
            n = min(prompt_len - computed, remaining)
            if n <= 0:
                continue
            prefill_lens.append(int(n))
            t_in += n
            remaining -= n
        feats = StepFeatures(t_in=t_in, p=len(prefill_lens), t_ft=0.0,
                             b_d=b_d, k=k, prefill_lens=prefill_lens or None)
        return feats, earliest

    def admit_ft_to_step(self):
        """Three-regime SLO-aware FT admission (port of vLLM ft_scheduler
        admit_ft_to_step, stages 1–5). Returns the list of admitted+claimed
        FinetuningSamples (caller builds Reqs). Stage-0 preconditions
        (ft_started / backward-not-busy / store-has-work) are checked by the
        caller _admit_and_inject_ft; here we do features → phase gate → baseline
        headroom → iterative EAGER-regime per-sample dual-SLO admission → claim.

        中文：三区间 SLO 感知准入（移植自 vLLM 的 admit_ft_to_step，阶段 1–5）。返回已准入
        并 claim 的样本列表。阶段 0 的前置条件（微调已开/反向不忙/语料有货）由调用方
        _admit_and_inject_ft 检查；这里做：取特征 → 阶段门控 → 基线余量检查 → 用 EAGER
        区间逐条做"双 SLO（TTFT+TBT）"迭代准入 → claim。冷启动（估计器未就绪）时退化为
        受 token 预算约束的无 SLO 准入（与旧贪心行为一致）。"""
        import time as _time
        from sglang.srt.deltaserve.estimator import (
            REGIME_DECODE_ONLY, REGIME_EAGER, REGIME_INF_PREFILL, StepFeatures,
        )
        from sglang.srt.deltaserve.coserve_slo import get_slo
        store = self._ft_store
        slo = get_slo()
        est = slo.estimator
        ft_cfg = getattr(self, "finetune_config", None)

        # ── Stage 1: features for the upcoming step (no FT yet) ──
        feats, earliest_arrival = self._current_step_features()
        is_decode_only = (feats.t_in == 0 and feats.b_d > 0)
        is_idle = (feats.t_in == 0 and feats.b_d == 0)
        has_prefill = (feats.t_in > 0)

        # ── Stage 2: phase gate ──
        phase = getattr(ft_cfg, "coserving_admission_phase", "both")
        if phase == "prefill" and is_decode_only:
            return []
        decode_only_margin = (
            float(getattr(ft_cfg, "decode_only_ft_safety_margin", 0.7))
            if (phase == "both" and is_decode_only) else 1.0)

        # ── Stage 3: baseline-without-FT prediction + headroom check ──
        if not est.is_ready:
            t_baseline = 0.0   # cold-start: no SLO gating, token-cap only
        elif is_decode_only:
            t_baseline = est.predict(feats, regime=REGIME_DECODE_ONLY)
        elif is_idle:
            t_baseline = 0.0
        else:
            t_baseline = est.predict(feats, regime=REGIME_INF_PREFILL)

        now = _time.time()
        queue_wait = 0.0   # sglang overlap depth is shallow; conservative 0
        ttft_deadline = (
            (earliest_arrival + 0.9 * slo.ttft_slo)
            if (has_prefill and earliest_arrival is not None) else None)

        if feats.b_d > 0 and est.is_ready:
            if t_baseline >= slo.max_tbt_slo * decode_only_margin:
                return []   # already over TBT without FT
        if ttft_deadline is not None and est.is_ready:
            if (ttft_deadline - now - queue_wait - t_baseline) <= 0:
                return []   # already over TTFT without FT

        # ── Stage 4: shapers (outer pre-filters) + iterative greedy ──
        _match_factor = float(getattr(ft_cfg, "match_prefill_workload_factor", 0.0))
        _prop_factor = float(getattr(ft_cfg, "ft_tokens_admission_constrain_factor", -1.0))
        max_iterations = None
        leaky_triggered = False
        if _match_factor > 0 and has_prefill:
            _peek = store.pop_next()
            if _peek is None:
                return []
            credit = (self._ft_unspent_prefill + feats.t_in) * _match_factor
            if credit >= _peek.input_len:
                leaky_triggered = True
                max_iterations = 1
            else:
                self._ft_unspent_prefill += int(feats.t_in)
                return []

        buffer_cap = int(getattr(self, "_ft_budget", 256))
        token_cap = buffer_cap
        if not leaky_triggered and _prop_factor != -1 and has_prefill:
            token_cap = min(buffer_cap, int(feats.t_in * _prop_factor))
        if token_cap <= 0:
            return []

        admitted = []
        cur_t_in = float(feats.t_in)
        cur_t_ft = 0.0
        cur_p = int(feats.p)
        cur_prefill_lens = list(feats.prefill_lens) if feats.prefill_lens else []
        while True:
            if max_iterations is not None and len(admitted) >= max_iterations:
                break
            if cur_t_ft >= token_cap:
                break
            if not store.has_next() and not store.advance_epoch():
                break
            candidate = store.pop_next(exclude=admitted)
            if candidate is None:
                break
            if cur_t_ft + candidate.input_len > token_cap:
                break
            new_prefill_lens = cur_prefill_lens + [candidate.input_len]
            hypothetical = StepFeatures(
                t_in=cur_t_in + candidate.input_len, p=cur_p + 1,
                t_ft=cur_t_ft + candidate.input_len, b_d=feats.b_d, k=feats.k,
                prefill_lens=new_prefill_lens)
            if est.is_ready:
                t_with_ft = est.predict(hypothetical, regime=REGIME_EAGER)
                if feats.b_d > 0 and t_with_ft > slo.max_tbt_slo * decode_only_margin:
                    break
                if ttft_deadline is not None and \
                        (ttft_deadline - now - queue_wait - t_with_ft) <= 0:
                    break
            admitted.append(candidate)
            cur_t_in += candidate.input_len
            cur_t_ft += candidate.input_len
            cur_p += 1
            cur_prefill_lens = new_prefill_lens

        # ── Stage 5: commit (claim; caller builds Reqs) ──
        if not admitted:
            return []
        store.claim(admitted)
        return admitted

    def get_next_batch_to_run(self, *args, **kwargs):
        """中文：在 base 调度器选批之前，先（按需）确保语料已加载，再做一次"准入+注入"，
        把语料驱动的微调 Req 放进 waiting_queue，然后照常交给 base 的 get_next_batch_to_run。
        标志关闭时这两步是 no-op，路径与原版一致。"""
        self._ensure_ft_store()
        if getattr(self, "_ft_store", None) is not None:
            try:
                self._admit_and_inject_ft()
            except Exception as e:
                logger.warning(f"[DeltaServe] FT admit/inject failed: {e}")
        return super().get_next_batch_to_run(*args, **kwargs)

    def _commit_finished_ft(self, batch) -> None:
        """Commit any finished store-driven FT samples back to the corpus, so the
        store can advance epochs. claim() at inject removed them from the length
        buckets; commit_claimed() now clears them from the in-flight `_claimed`
        set and marks them trained — without this, `_claimed` grows unbounded and
        `advance_epoch()` (which refuses while anything is claimed) can never fire,
        so the corpus is silently capped at one pass. FT Reqs are prefill-only
        (max_new_tokens=1) so they finish in the prefill result; `_ft_sample` is
        cleared after commit to make this idempotent across any re-entry.

        中文：把"已完成"的语料驱动微调样本提交回语料库，使其能推进 epoch。注入时的 claim()
        把样本从长度桶移除；这里的 commit_claimed() 再把它们从在途集合 `_claimed` 清掉并标记
        为已训练。否则 `_claimed` 会无限增长，而 advance_epoch()（只要还有在途样本就拒绝推进）
        永远无法触发，语料就被悄悄限制在"只过一遍"。微调 Req 是 prefill-only（max_new_tokens=1），
        在 prefill 结果里就 finish；提交后把 `_ft_sample` 置空，保证多次进入也幂等。"""
        store = getattr(self, "_ft_store", None)
        if store is None or batch is None:
            return
        reqs = getattr(batch, "reqs", None)
        if not reqs:
            return
        done = []
        for r in reqs:
            if not getattr(r, "is_finetuning", False):
                continue
            s = getattr(r, "_ft_sample", None)
            if s is not None and r.finished():
                done.append(s)
                r._ft_sample = None   # idempotent: don't re-commit on re-entry
        if done:
            try:
                store.commit_claimed(done)
            except Exception as e:
                logger.warning(f"[DeltaServe] FT commit_claimed failed: {e}")

    def process_batch_result_prefill(self, batch, result, *args, **kwargs):
        """中文：先让 base 处理 prefill 结果（设置 finish、缓存、流式输出），再把本批中
        已完成的微调样本提交回语料库。微调 Req 在这一步 finish，所以提交点选在这里。"""
        out = super().process_batch_result_prefill(batch, result, *args, **kwargs)
        if getattr(self, "_ft_store", None) is not None:
            try:
                self._commit_finished_ft(batch)
            except Exception as e:
                logger.warning(f"[DeltaServe] FT commit hook failed: {e}")
        return out

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
