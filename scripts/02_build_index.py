"""Step 2: 构建向量索引 + BM25 索引（无需 API key）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config import PROCESSED_DIR
from adaptive_rag.indexing import BM25Index, VectorIndex


def load_corpus() -> list[dict]:
    path = PROCESSED_DIR / "corpus.jsonl"
    if not path.exists():
        raise FileNotFoundError("找不到语料库，请先运行 scripts/01_prepare_data.py")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    corpus = load_corpus()

    if (Path("data") / "index" / "vectors.npy").exists():
        print("[index] 向量索引已存在，跳过构建")
        vec = VectorIndex.load()
    else:
        vec = VectorIndex()
        vec.build(corpus)
        vec.save()

    if (Path("data") / "index" / "bm25.pkl").exists():
        print("[index] BM25 索引已存在，跳过构建")
    else:
        bm25 = BM25Index()
        bm25.build(corpus)
        bm25.save()

    print("索引构建完成。下一步（零成本验证检索）:")
    print("  python scripts/03_run_eval.py --strategy naive --retrieval-only")
