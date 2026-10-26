"""三种 RAG 策略，统一 ask() 接口，方便评测框架无差别调用。"""
import time

from .config import FINAL_K, TOP_K
from .generation import answer_question, decompose_question, get_client
from .retrieval import Retrieved


class BaseRAG:
    def __init__(self, retriever, client=None, model: str | None = None):
        self.retriever = retriever
        if client is None:
            client, model = get_client()
        self.client = client
        self.model = model

    def ask(self, question: str) -> dict:
        start = time.perf_counter()
        result = self._ask(question)
        result["latency_s"] = time.perf_counter() - start
        return result

    def _ask(self, question: str) -> dict:
        raise NotImplementedError


class NaiveRAG(BaseRAG):
    """基线：单轮向量检索 top-k -> 生成。"""

    def _ask(self, question: str) -> dict:
        retrieved = self.retriever.retrieve(question, k=TOP_K)
        answer = answer_question(question, [r.text for r in retrieved], self.client, self.model)
        return {"answer": answer, "retrieved": retrieved, "route": "direct"}


class HybridRAG(BaseRAG):
    """检索改进：BM25 + 向量 RRF 融合 -> 生成。"""

    def _ask(self, question: str) -> dict:
        retrieved = self.retriever.retrieve(question, k=TOP_K)
        answer = answer_question(question, [r.text for r in retrieved], self.client, self.model)
        return {"answer": answer, "retrieved": retrieved, "route": "direct"}


class AdaptiveRAG(BaseRAG):
    """系统改进：路由器分流——简单题单跳，复杂题分解成子问题多跳检索。"""

    def __init__(self, retriever, router, client=None, model: str | None = None):
        super().__init__(retriever, client, model)
        self.router = router

    def _ask(self, question: str) -> dict:
        route = self.router.classify(question)
        if route == "simple":
            retrieved = self.retriever.retrieve(question, k=TOP_K)
            sub_questions = []
        else:
            # 多跳路径：分解 -> 每个子问题各检索 -> 合并去重
            sub_questions = decompose_question(question, self.client, self.model)
            seen: dict[str, Retrieved] = {}
            for sub in sub_questions:
                for r in self.retriever.retrieve(sub, k=TOP_K):
                    seen.setdefault(r.doc_id, r)
            # 用原始问题对合并后的证据池重排，优先与完整问题最相关的段落
            reranked = self.retriever.retrieve(question, k=FINAL_K)
            by_id = {r.doc_id: r for r in reranked}
            ordered = list(by_id.values()) + [r for d, r in seen.items() if d not in by_id]
            retrieved = ordered[:FINAL_K]

        answer = answer_question(question, [r.text for r in retrieved], self.client, self.model)
        return {
            "answer": answer,
            "retrieved": retrieved,
            "route": route,
            "sub_questions": sub_questions,
        }
