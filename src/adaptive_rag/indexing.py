"""向量索引 + BM25 索引。

语料只有几千段，numpy 矩阵暴力余弦检索即可（毫秒级），
不必引入 Faiss/Chroma 这类重依赖——也方便你看清检索的每一步。
"""
import json
import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi

from .config import EMBED_MODEL, INDEX_DIR

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class VectorIndex:
    """sentence-transformers 编码 + 归一化向量的余弦检索。"""

    def __init__(self, model_name: str = EMBED_MODEL):
        self.model_name = model_name
        self._model = None  # 惰性加载，避免 import 就下载模型
        self.vectors: np.ndarray | None = None
        self.doc_ids: list[str] = []

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(self, corpus: list[dict]) -> None:
        texts = [d["text"] for d in corpus]
        print(f"[index] 编码 {len(texts)} 篇段落（首次运行需下载模型 ~130MB）...")
        vecs = self.model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        self.vectors = np.asarray(vecs, dtype=np.float32)
        self.doc_ids = [d["doc_id"] for d in corpus]

    def search(self, query: str, k: int) -> tuple[list[str], np.ndarray]:
        q = np.asarray(self.model.encode([query], normalize_embeddings=True), dtype=np.float32)[0]
        scores = self.vectors @ q  # 归一化后点积 = 余弦相似度
        top = np.argsort(-scores)[:k]
        return [self.doc_ids[i] for i in top], scores[top]

    def save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(INDEX_DIR / "vectors.npy", self.vectors)
        (INDEX_DIR / "meta.json").write_text(
            json.dumps({"model": self.model_name, "doc_ids": self.doc_ids}), encoding="utf-8"
        )

    @classmethod
    def load(cls) -> "VectorIndex":
        meta = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
        idx = cls(model_name=meta["model"])
        idx.vectors = np.load(INDEX_DIR / "vectors.npy")
        idx.doc_ids = meta["doc_ids"]
        return idx


class BM25Index:
    """稀疏词频检索，与向量检索互补（专有名词匹配更强）。"""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.doc_ids: list[str] = []

    def build(self, corpus: list[dict]) -> None:
        print(f"[index] 构建 BM25 索引...")
        self.doc_ids = [d["doc_id"] for d in corpus]
        tokenized = [_tokenize(d["text"]) for d in tqdm(corpus, desc="tokenize")]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int) -> tuple[list[str], np.ndarray]:
        scores = self.bm25.get_scores(_tokenize(query))
        top = np.argsort(-scores)[:k]
        return [self.doc_ids[i] for i in top], scores[top]

    def save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with (INDEX_DIR / "bm25.pkl").open("wb") as f:
            pickle.dump({"doc_ids": self.doc_ids, "state": self.bm25}, f)

    @classmethod
    def load(cls) -> "BM25Index":
        with (INDEX_DIR / "bm25.pkl").open("rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx.doc_ids = data["doc_ids"]
        idx.bm25 = data["state"]
        return idx
