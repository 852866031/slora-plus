from typing import List

from sglang.srt.deltaserve.finetuning_store_stub import FinetuningStore, FTSample
from sglang.srt.managers.io_struct import GenerateReqInput


class FinetuneInjector:
    """Pulls FT samples from a FinetuningStore and packs them into FT-tagged
    GenerateReqInputs for the scheduler (Phase 3 wiring; Phase 4 will drive
    this from the event loop)."""

    def __init__(
        self,
        corpus: FinetuningStore,
        lora_dir: str,
        lora_name: str = "ft-adapter",
    ):
        self.corpus = corpus
        self.lora_dir = lora_dir
        self.lora_name = lora_name

    def next_batch(self, max_tokens: int) -> List[GenerateReqInput]:
        budget = max_tokens
        picks: List[FTSample] = []
        while budget > 0:
            got = self.corpus.claim(1)
            if not got:
                break
            s = got[0]
            if s.approx_tokens > budget and picks:
                self.corpus.release_claimed([s.rid])
                break
            picks.append(s)
            budget -= s.approx_tokens
        return [
            GenerateReqInput(
                text=s.prompt,
                rid=s.rid,
                lora_path=self.lora_dir,
                is_finetuning=True,
            )
            for s in picks
        ]

    def commit(self, req_ids: List[str]) -> None:
        self.corpus.commit_claimed(req_ids)

    def abort(self, req_ids: List[str]) -> None:
        self.corpus.release_claimed(req_ids)
