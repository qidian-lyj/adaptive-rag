"""Step 7: 合并分段评测结果（如第 1~500 题与第 501~1000 题）为完整结果文件。

用法:
  python scripts/07_merge_results.py --parts results/naive_500.json results/naive_500_tail.json --out results/naive_1000.json

说明: details 直接拼接；EM/F1/召回取加权平均；延迟按题数加权；routes 相加。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True, help="两个及以上的分段结果 JSON")
    ap.add_argument("--out", required=True, help="合并后的输出文件")
    args = ap.parse_args()

    parts = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.parts]
    all_details = []
    total_n = 0
    routes: dict[str, int] = {}
    latency_weighted = 0.0

    for p in parts:
        m = p["metrics"]
        n = m["n"]
        total_n += n
        latency_weighted += m["avg_latency_s"] * n
        for k, v in m.get("routes", {}).items():
            routes[k] = routes.get(k, 0) + v
        all_details.extend(p.get("details") or [])

    if total_n == 0 or len(all_details) != total_n:
        raise SystemExit(f"题目数对不上: metrics n 合计 {total_n}, details 共 {len(all_details)}")

    metrics = {
        "strategy": parts[0]["metrics"].get("strategy", ""),
        "merged_from": [Path(p).name for p in args.parts],
        "n": total_n,
        "EM": round(sum(d["em"] for d in all_details) / total_n, 4),
        "F1": round(sum(d["f1"] for d in all_details) / total_n, 4),
        "doc_recall@5": round(sum(d["doc_recall"] for d in all_details) / total_n, 4),
        "avg_latency_s": round(latency_weighted / total_n, 4),
    }
    if routes:
        metrics["routes"] = routes

    out = {"metrics": metrics, "details": all_details}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge] {args.out}: n={total_n} EM={metrics['EM']} F1={metrics['F1']} "
          f"recall={metrics['doc_recall@5']} latency={metrics['avg_latency_s']}s")


if __name__ == "__main__":
    main()
