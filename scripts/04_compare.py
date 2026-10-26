"""Step 4: 汇总所有实验结果 -> 对比表 + 柱状图。"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config import RESULTS_DIR


def main():
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name == "comparison.csv":
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
        rows.append(metrics)

    if not rows:
        print("results/ 目录下还没有实验结果，先运行 03_run_eval.py")
        return

    df = pd.DataFrame(rows)
    cols = [c for c in ["strategy", "n", "EM", "F1", "doc_recall@5", "avg_latency_s"] if c in df.columns]
    df = df[cols]
    print("\n===== 实验对比 =====")
    print(df.to_string(index=False))
    df.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    print(f"\n[compare] 表格已保存: {RESULTS_DIR / 'comparison.csv'}")

    # 柱状图（英文标签，避免 matplotlib 中文乱码）
    end2end = df[df["EM"].notna()] if "EM" in df.columns else df.iloc[0:0]
    if not end2end.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(end2end))
        width = 0.35
        ax.bar([i - width / 2 for i in x], end2end["EM"], width, label="EM")
        ax.bar([i + width / 2 for i in x], end2end["F1"], width, label="F1")
        ax.set_xticks(list(x))
        ax.set_xticklabels(end2end["strategy"])
        ax.set_ylabel("Score")
        ax.set_title("Adaptive-RAG: Strategy Comparison")
        ax.legend()
        fig.tight_layout()
        out = RESULTS_DIR / "comparison.png"
        fig.savefig(out, dpi=150)
        print(f"[compare] 图表已保存: {out}")


if __name__ == "__main__":
    main()
