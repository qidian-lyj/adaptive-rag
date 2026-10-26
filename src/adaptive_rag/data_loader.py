"""下载并整理 HotpotQA（distractor 设定），构建语料库与评测集。

设计说明：
HotpotQA 每道题自带 10 个候选段落（2 个含答案 + 8 个干扰项）。
原论文是"阅读理解"任务（10 段全给模型看）；我们把其中出现过的段落去重后
合并成一个开放语料库，评测时模型只能靠检索自己找证据——这才是真正的 RAG 任务。

数据来源（二选一，自动切换）：
1. HuggingFace datasets: hotpotqa/hotpot_qa (distractor, validation)
2. 本地文件: 手动下载 hotpot_dev_distractor_v1.json 放到 data/raw/
"""
import json

from tqdm import tqdm

from .config import N_EVAL_QUESTIONS, PROCESSED_DIR, RAW_DIR


def _load_raw_records(n: int) -> list[dict]:
    """优先读本地 JSON（离线友好），否则从 HuggingFace 下载。"""
    local = RAW_DIR / "hotpot_dev_distractor_v1.json"
    if local.exists():
        print(f"[data] 使用本地文件: {local}")
        records = json.loads(local.read_text(encoding="utf-8"))
        return records[:n]

    print("[data] 从 HuggingFace 下载 hotpotqa/hotpot_qa (distractor, validation)...")
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    ds = ds.select(range(min(n, len(ds))))
    return [dict(r) for r in ds]


def _paragraph_text(sentences: list[str]) -> str:
    return "".join(sentences).strip()


def prepare(n: int = N_EVAL_QUESTIONS) -> None:
    records = _load_raw_records(n)

    # 1) 构建语料库：所有题目的候选段落去重合并（段落即文档，无需再分块）
    corpus: dict[str, dict] = {}
    eval_set = []
    for r in records:
        titles = r["context"]["title"]
        sents = r["context"]["sentences"]
        for title, ss in zip(titles, sents):
            text = _paragraph_text(ss)
            if title not in corpus:  # 同一 Wikipedia 段落跨题复用，按标题去重
                corpus[title] = {"doc_id": title, "title": title, "text": text}

        gold_titles = sorted(set(r["supporting_facts"]["title"]))
        eval_set.append(
            {
                "question": r["question"],
                "answer": r["answer"],
                "gold_titles": gold_titles,
                "type": r.get("type", ""),  # comparison / bridge
                "level": r.get("level", ""),  # easy / medium / hard
            }
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = PROCESSED_DIR / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for doc in corpus.values():
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    eval_path = PROCESSED_DIR / "eval.jsonl"
    with eval_path.open("w", encoding="utf-8") as f:
        for item in eval_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[data] 语料库: {len(corpus)} 篇段落 -> {corpus_path}")
    print(f"[data] 评测集: {len(eval_set)} 题 -> {eval_path}")


if __name__ == "__main__":
    prepare()
