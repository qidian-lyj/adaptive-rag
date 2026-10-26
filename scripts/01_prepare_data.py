"""Step 1: 下载 HotpotQA 并构建语料库 + 评测集。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config import N_EVAL_QUESTIONS
from adaptive_rag.data_loader import prepare

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_EVAL_QUESTIONS
    prepare(n)
    print("下一步: python scripts/02_build_index.py")
