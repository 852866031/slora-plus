from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FTSample:
    rid: str
    prompt: str
    target: str
    approx_tokens: int = 64


class FinetuningStore:
    """Phase-3 stub. Phase 5 replaces with the real corpus."""

    def __init__(self, samples: List[FTSample]):
        self._available: List[FTSample] = list(samples)
        self._claimed: Dict[str, FTSample] = {}

    def claim(self, n: int) -> List[FTSample]:
        take = self._available[:n]
        self._available = self._available[n:]
        for s in take:
            self._claimed[s.rid] = s
        return take

    def commit_claimed(self, rids: List[str]) -> None:
        for rid in rids:
            self._claimed.pop(rid, None)

    def release_claimed(self, rids: List[str]) -> None:
        for rid in rids:
            s = self._claimed.pop(rid, None)
            if s is not None:
                self._available.append(s)
