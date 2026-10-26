"""Cross-Encoder 重排：粗排（向量召回）之后对候选段落与问题的相关度做精排。

两阶段检索的标准做法：召回阶段用双塔模型（快但粗），精排阶段用 CrossEncoder
（把问题和段落拼在一起过模型，慢但准）。
"""
from .config import RERANK_CANDIDATES, RERANK_MODEL
from .retrieval import Retrieved


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self._model = None  # 惰性加载

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[Retrieved], top_k: int) -> list[Retrieved]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: -float(x[1]))
        for c, s in ranked:  # 用精排分数覆盖召回分数
            c.score = float(s)
        return [c for c, _ in ranked[:top_k]]


class RerankRetriever:
    """两阶段检索器：向量粗排召回 RERANK_CANDIDATES 条 -> CrossEncoder 精排取 top-k。

    对外暴露与 Retriever 相同的 .retrieve() 接口，评测框架无需区分。
    """

    def __init__(self, base, reranker: Reranker | None = None, candidate_k: int = RERANK_CANDIDATES):
        self.base = base
        self.reranker = reranker or Reranker()
        self.candidate_k = candidate_k

    def retrieve(self, query: str, k: int = 5) -> list[Retrieved]:
        candidates = self.base.retrieve(query, k=self.candidate_k)
        return self.reranker.rerank(query, candidates, top_k=k)
