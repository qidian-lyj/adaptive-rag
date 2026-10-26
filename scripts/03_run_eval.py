"""Step 3: 跑评测。

用法:
  python scripts/03_run_eval.py --strategy naive --retrieval-only      # 零成本，只测检索
  python scripts/03_run_eval.py --strategy rerank --retrieval-only     # 重排策略检索评测
  python scripts/03_run_eval.py --strategy naive --n 50                # 端到端小规模试水
  python scripts/03_run_eval.py --strategy adaptive --n 50 --router llm  # LLM 路由
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config import PROCESSED_DIR, RESULTS_DIR, TOP_K
from adaptive_rag.evaluation import evaluate, evaluate_retrieval, save_result
from adaptive_rag.generation import get_client
from adaptive_rag.indexing import BM25Index, VectorIndex
from adaptive_rag.pipeline import AdaptiveRAG, HybridRAG, NaiveRAG
from adaptive_rag.rerank import RerankRetriever
from adaptive_rag.retrieval import Retriever
from adaptive_rag.router import LLMRouter, RuleBasedRouter

STRATEGY_MODE = {"naive": "vector", "hybrid": "hybrid", "rerank": "vector"}


def load_eval_set() -> list[dict]:
    path = PROCESSED_DIR / "eval.jsonl"
    if not path.exists():
        raise FileNotFoundError("找不到评测集，请先运行 scripts/01_prepare_data.py")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def build(strategy: str, with_llm: bool, router_kind: str = "rule"):
    corpus = {
        d["doc_id"]: d
        for d in (json.loads(l) for l in (PROCESSED_DIR / "corpus.jsonl").read_text(encoding="utf-8").splitlines())
    }
    base = Retriever(
        corpus,
        VectorIndex.load(),
        BM25Index.load(),
        mode=STRATEGY_MODE.get(strategy, "vector"),
    )
    if not with_llm:
        return RerankRetriever(base) if strategy == "rerank" else base

    client, model = get_client()
    if strategy == "adaptive":
        router = LLMRouter(client, model) if router_kind == "llm" else RuleBasedRouter()
        return AdaptiveRAG(base, router, client, model)
    cls = {"naive": NaiveRAG, "hybrid": HybridRAG, "rerank": NaiveRAG}[strategy]
    retriever = RerankRetriever(base) if strategy == "rerank" else base
    return cls(retriever, client, model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=["naive", "hybrid", "rerank", "adaptive"])
    ap.add_argument("--n", type=int, default=50, help="评测题数")
    ap.add_argument("--retrieval-only", action="store_true", help="只测检索质量，不调用 LLM（零成本）")
    ap.add_argument("--router", choices=["rule", "llm"], default="rule", help="adaptive 策略的路由器")
    ap.add_argument("--tag", default="", help="结果文件名附加标记（如 rule/llm 路由对比实验）")
    args = ap.parse_args()

    records = load_eval_set()
    if args.retrieval_only:
        retriever = build(args.strategy, with_llm=False)
        result = evaluate_retrieval(retriever, records, k=TOP_K)
        result["strategy"] = args.strategy
        save_result({"metrics": result, "details": []}, RESULTS_DIR / f"{args.strategy}_retrieval.json")
        print(result)
        return

    pipeline = build(args.strategy, with_llm=True, router_kind=args.router)
    result = evaluate(pipeline, records, n=args.n)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    suffix = f"_{args.tag}" if args.tag else ""
    save_result(result, RESULTS_DIR / f"{args.strategy}_{args.n}{suffix}.json")


if __name__ == "__main__":
    main()
