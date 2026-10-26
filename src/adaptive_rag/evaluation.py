"""评测：HotpotQA 官方 EM/F1 + 证据召回（Doc Recall）+ 延迟。"""
import json
import re
import string
import time

from tqdm import tqdm

_CITE_RE = re.compile(r"\[\d+\]")  # 生成答案中的引用标注 [1][2]


def strip_citations(text: str) -> str:
    """评测前剥离引用标记，否则 'Yes [1][2]' 无法匹配 gold 'yes'。"""
    return _CITE_RE.sub(" ", text)


# ---------- HotpotQA 官方归一化与指标（照官方实现，保证可比性） ----------
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return " ".join(w for w in text.split() if w not in {"a", "an", "the"})

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    return white_space_fix(remove_articles(s.lower()))


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def doc_recall(retrieved_titles: list[str], gold_titles: list[str]) -> float:
    """证据召回：检索到的黄金段落比例。纯检索指标，不依赖 LLM。"""
    if not gold_titles:
        return 0.0
    hit = set(retrieved_titles) & set(gold_titles)
    return len(hit) / len(gold_titles)


# ---------- 评测循环 ----------
def evaluate_retrieval(retriever, records: list[dict], k: int = 5) -> dict:
    """只测检索质量（零 API 成本），用于 Day 1 验证链路。"""
    recalls = []
    for r in tqdm(records, desc="retrieval eval"):
        hits = retriever.retrieve(r["question"], k=k)
        recalls.append(doc_recall([h.title for h in hits], r["gold_titles"]))
    return {"doc_recall@%d" % k: sum(recalls) / len(recalls), "n": len(records)}


def evaluate(pipeline, records: list[dict], n: int | None = None) -> dict:
    """端到端评测：EM / F1 / Doc Recall / 平均延迟。"""
    subset = records if n is None else records[:n]
    details, ems, f1s, recalls, latencies, routes = [], [], [], [], [], []

    for r in tqdm(subset, desc="eval"):
        try:
            out = pipeline.ask(r["question"])
        except Exception as e:  # 单题失败不拖垮整个评测
            print(f"\n[warn] 题目失败: {e}")
            out = {"answer": "", "retrieved": [], "latency_s": 0.0, "route": "error"}

        pred, retrieved_titles = strip_citations(out["answer"]), [h.title for h in out["retrieved"]]
        em, f1 = exact_match(pred, r["answer"]), f1_score(pred, r["answer"])
        rec = doc_recall(retrieved_titles, r["gold_titles"])

        ems.append(em)
        f1s.append(f1)
        recalls.append(rec)
        latencies.append(out["latency_s"])
        routes.append(out.get("route", ""))
        details.append(
            {
                "question": r["question"],
                "gold": r["answer"],
                "pred": pred,
                "em": em,
                "f1": round(f1, 4),
                "doc_recall": round(rec, 4),
                "route": out.get("route", ""),
                "sub_questions": out.get("sub_questions", []),
            }
        )
        time.sleep(0)  # 占位：如遇限流可改为 time.sleep(0.2)

    n_total = max(len(subset), 1)
    metrics = {
        "strategy": type(pipeline).__name__,
        "n": len(subset),
        "EM": sum(ems) / n_total,
        "F1": sum(f1s) / n_total,
        "doc_recall@5": sum(recalls) / n_total,
        "avg_latency_s": sum(latencies) / n_total,
    }
    route_stats = {rt: routes.count(rt) for rt in set(routes)}
    if route_stats:
        metrics["routes"] = route_stats
    return {"metrics": metrics, "details": details}


def save_result(result: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] 结果已保存: {path}")
