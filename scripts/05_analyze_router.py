"""Step 5: 路由器准确率分析（零成本，不需要 LLM API）。

HotpotQA 自带问题类型标注：comparison（比较题 -> simple）/ bridge（桥接题 -> multi_hop）。
把它当作路由的金标准，统计规则路由器的分类质量——这是自适应策略有效性分析的第一步。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config import PROCESSED_DIR, RESULTS_DIR
from adaptive_rag.router import RuleBasedRouter

GOLD_ROUTE = {"comparison": "simple", "bridge": "multi_hop"}


def main():
    records = [
        json.loads(line)
        for line in (PROCESSED_DIR / "eval.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    router = RuleBasedRouter()

    correct = 0
    confusion = {"simple->simple": 0, "simple->multi_hop": 0, "multi_hop->simple": 0, "multi_hop->multi_hop": 0}
    per_type = {}
    examples_wrong = []

    for r in records:
        gold = GOLD_ROUTE.get(r["type"])
        if gold is None:
            continue
        pred = router.classify(r["question"])
        hit = pred == gold
        correct += hit
        confusion[f"{gold}->{pred}"] += 1
        t = per_type.setdefault(r["type"], {"n": 0, "correct": 0})
        t["n"] += 1
        t["correct"] += hit
        if not hit and len(examples_wrong) < 5:
            examples_wrong.append({"question": r["question"], "gold": gold, "pred": pred, "type": r["type"]})

    n = sum(t["n"] for t in per_type.values())
    result = {
        "router": "RuleBasedRouter",
        "n": n,
        "accuracy": round(correct / max(n, 1), 4),
        "per_type": {k: round(v["correct"] / v["n"], 4) for k, v in per_type.items()},
        "confusion(gold->pred)": confusion,
        "examples_wrong": examples_wrong,
    }

    out = RESULTS_DIR / "router_rule_based.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== 规则路由器 vs 金标准 =====")
    print(f"总体准确率: {result['accuracy']:.1%}  (n={n})")
    for t, acc in result["per_type"].items():
        print(f"  {t}: {acc:.1%}")
    print(f"混淆矩阵 (gold->pred): {confusion}")

    # LLM 路由器：从已有的 adaptive *_llm 端到端结果中读取 route 字段统计（无需再调 API）
    llm_files = sorted(RESULTS_DIR.glob("adaptive_*_llm.json"))
    llm_files = [f for f in llm_files if f.stem.count("_") == 2]  # 排除 _tail 等分段文件
    if llm_files:
        det = json.loads(llm_files[-1].read_text(encoding="utf-8")).get("details") or []
        gold_map = {
            json.loads(line)["question"]: json.loads(line)["type"]
            for line in (PROCESSED_DIR / "eval.jsonl").read_text(encoding="utf-8").splitlines()
        }
        lc = ln = 0
        lper: dict[str, list[int]] = {}
        for d in det:
            qtype = gold_map.get(d.get("question", ""))
            g = GOLD_ROUTE.get(qtype)
            if g is None:
                continue
            hit = d.get("route") == g
            lc += hit
            ln += 1
            t = lper.setdefault(qtype, [0, 0])
            t[0] += hit
            t[1] += 1
        if ln:
            result["llm_router"] = {
                "n": ln,
                "accuracy": round(lc / ln, 4),
                "per_type": {k: round(a / b, 4) for k, (a, b) in lper.items()},
            }
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print("\n===== LLM 路由器 vs 金标准 =====")
            print(f"总体准确率: {result['llm_router']['accuracy']:.1%}  (n={ln})")
            for t, acc in result["llm_router"]["per_type"].items():
                print(f"  {t}: {acc:.1%}")

    print(f"\n[done] 已保存: {out}")


if __name__ == "__main__":
    main()
