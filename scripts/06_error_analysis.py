"""Step 6: 错误分析——把端到端评测的失败 case 归因为「检索失败」或「生成失败」。

这是报告里最出彩的部分之一：不仅报告分数，还解释分数从哪丢的。

用法:
  python scripts/06_error_analysis.py                      # 分析 results/ 下所有端到端结果
  python scripts/06_error_analysis.py --file results/naive_50.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # 避免中文控制台打印失败

from adaptive_rag.config import RESULTS_DIR


def analyze_file(path: Path, max_examples: int = 3) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    details = data.get("details") or []
    if not details:
        return ""
    metrics = data["metrics"]
    strategy = metrics.get("strategy", path.stem)

    wrong = [d for d in details if d["em"] == 0]
    retrieval_miss = [d for d in wrong if d["doc_recall"] < 1.0]   # 证据没找全
    generation_miss = [d for d in wrong if d["doc_recall"] >= 1.0]  # 证据齐了仍答错

    lines = [f"\n## {strategy}（n={metrics.get('n')}）", ""]
    lines.append(f"- 失败题数: {len(wrong)} / {len(details)}")
    lines.append(f"- **检索失败**（gold 段落未全部召回）: {len(retrieval_miss)}")
    lines.append(f"- **生成失败**（证据齐但答错/答偏）: {len(generation_miss)}")
    if details:
        lines.append(f"- 检索失败占失败比例: {len(retrieval_miss) / max(len(wrong), 1):.0%}")
    lines.append("")

    for label, cases in [("检索失败示例", retrieval_miss), ("生成失败示例", generation_miss)]:
        lines.append(f"### {label}")
        for d in cases[:max_examples]:
            lines.append(f"- Q: {d['question']}")
            lines.append(f"  - gold: `{d['gold']}`  pred: `{d['pred']}`  doc_recall: {d['doc_recall']}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="", help="指定单个结果文件；缺省则分析 results/ 下全部端到端结果")
    args = ap.parse_args()

    files = [Path(args.file)] if args.file else sorted(RESULTS_DIR.glob("*_*.json"))
    files = [f for f in files if "retrieval" not in f.stem and f.stem != "comparison"]

    report = ["# 错误分析报告", ""]
    for f in files:
        try:
            report.append(analyze_file(f))
        except Exception as e:
            report.append(f"\n## {f.name}: 解析失败 {e}\n")

    if len(report) <= 2:
        print("还没有端到端评测结果，先运行 03_run_eval.py --strategy naive --n 50")
        return

    out = RESULTS_DIR / "error_analysis.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"[done] 已保存: {out}")


if __name__ == "__main__":
    main()
