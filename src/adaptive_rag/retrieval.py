"""统一检索接口：vector / bm25 / hybrid（RRF 融合）。"""
from dataclasses import dataclass

import numpy as np

from .config import HYBRID_CANDIDATES, RRF_K
from .indexing import BM25Index, VectorIndex


@dataclass
class Retrieved:
    doc_id: str
    title: str
    text: str
    score: float


class Retriever:
    def __init__(
        self,
        corpus: dict[str, dict],
        vector_index: VectorIndex,
        bm25_index: BM25Index,
        mode: str = "vector",  # vector | bm25 | hybrid
    ):
        self.corpus = corpus
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        if mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError(f"未知检索模式: {mode}")
        self.mode = mode

    def retrieve(self, query: str, k: int = 5) -> list[Retrieved]:
        if self.mode == "vector":
            ids, scores = self.vector_index.search(query, k)
            return self._pack(ids, scores)
        if self.mode == "bm25":
            ids, scores = self.bm25_index.search(query, k)
            return self._pack(ids, scores)
        return self._hybrid(query, k)

    def _hybrid(self, query: str, k: int) -> list[Retrieved]:
        """Reciprocal Rank Fusion：对两路排名取 1/(RRF_K + rank) 之和，鲁棒且免调分数量纲。

        每路先取 HYBRID_CANDIDATES 条再融合截断到 k，避免融合后 top-k 被单路占满。
        """
        depth = max(k, HYBRID_CANDIDATES)
        v_ids, _ = self.vector_index.search(query, depth)
        b_ids, _ = self.bm25_index.search(query, depth)
        rrf: dict[str, float] = {}
        for ids in (v_ids, b_ids):
            for rank, doc_id in enumerate(ids, start=1):
                rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        ranked = sorted(rrf.items(), key=lambda x: -x[1])[:k]
        return self._pack([d for d, _ in ranked], np.array([s for _, s in ranked]))

    def _pack(self, doc_ids: list[str], scores: np.ndarray) -> list[Retrieved]:
        return [
            Retrieved(
                doc_id=doc_id,
                title=self.corpus[doc_id]["title"],
                text=self.corpus[doc_id]["text"],
                score=float(score),
            )
            for doc_id, score in zip(doc_ids, scores)
        ]
