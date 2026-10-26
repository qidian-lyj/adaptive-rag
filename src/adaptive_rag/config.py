"""全局配置：路径、模型、检索参数。所有可调参数集中在这里，方便做对比实验。"""
from pathlib import Path

# 路径
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
RESULTS_DIR = ROOT / "results"

# 数据规模
N_EVAL_QUESTIONS = 500  # 评测题量。先用 50 试水，最终实验建议 300~500

# 检索
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 小模型，CPU 也能跑
TOP_K = 5        # 每轮检索返回的段落数
RRF_K = 60       # Reciprocal Rank Fusion 平滑常数（经典取值）
FINAL_K = 8      # 多跳合并证据后的最大段落数

# 生成
MAX_CONTEXTS = 8  # 最多带多少段参考资料进 prompt
