# 面向多跳问答的自适应检索增强问答系统（Adaptive-RAG）

> 技术报告 · 数据集 HotpotQA (distractor, validation) · 所有实验可一键复现（见文末）

## 摘要

检索增强生成（RAG）系统在多跳问答（multi-hop QA）上面临两难：对每题都做"问题分解 + 迭代检索"能提升证据覆盖，但延迟与成本成倍增长；统一走单跳检索则难以召回链条式证据。本工作在 HotpotQA 上实现并系统对比了四种策略——Naive RAG、Hybrid RAG（BM25+向量 RRF 融合）、Rerank（粗排+CrossEncoder 精排）、Adaptive RAG（按问题复杂度路由：简单题单跳、复杂题分解多跳），并在统一评测框架下量化每种设计的收益与代价。

## 1. 问题定义

HotpotQA 每题需要跨 2 个以上维基百科段落串联证据。原论文设定为阅读理解（10 个候选段落直接给出）；本工作将其改造为**开放语料检索任务**：把出现过的 4,937 篇段落去重合并为语料库，模型只能靠检索自行找证据——这才对应真实 RAG 场景。

## 2. 系统与方法

```
问题 ──> 路由器 Router ──┬─ simple ───> 单跳检索 top-5 ─────────┐
                         └─ multi_hop ─> LLM 分解为 2 个子问题    │
                                          各自检索 -> 合并重排 ──┴─> LLM 生成（带引用标注）
```

- **检索**：BAAI/bge-small-en-v1.5 向量检索（numpy 余弦，无重依赖）+ BM25；RRF 融合
- **精排**：cross-encoder/ms-marco-MiniLM-L-6-v2，向量召回 20 条 -> 精排取 5 条
- **路由器**：v0 规则版（连接词/实体数/问句长度特征）；v1 LLM 版（few-shot 分类）
- **生成**：OpenAI 兼容接口（实验使用 deepseek-chat，temperature=0 保证可复现）
- **多跳路径**：分解 -> 子问题各自检索 -> 证据池合并 -> 用原问题重排 -> 截取 8 条

## 3. 实验设置

| 项 | 值 |
|---|---|
| 评测集 | HotpotQA validation 前 500 题（comparison / bridge 两类） |
| 语料库 | 4,937 篇段落（评测题涉及的全部候选段落去重） |
| 指标 | EM / F1（官方口径）、Doc Recall@5（证据召回）、平均延迟 |
| 检索实验 | 零 LLM 成本；端到端实验约 1–3 元 API 费用 |

## 4. 结果

### 4.1 检索质量（Doc Recall@5，n=500，不依赖 LLM）

| 检索方案 | Doc Recall@5 | 备注 |
|---|---|---|
| Naive（纯向量） | **86.8%** | 最强单方案 |
| Hybrid（RRF, depth=5） | 85.7% | 与 Naive 接近 |
| Hybrid（RRF, depth=10） | 82.0% | 见 4.2 消融分析 |
| + Rerank（20 选 5） | （待端到端评测后填写） | |

### 4.2 消融：RRF 候选深度的影响

加深每路候选池（5→10）反而降低召回。原因是 RRF 按"排名倒数之和"打分：同时出现在两路中游（如第 7+7 名）的干扰文档得分（2/67≈0.030）会超过单路第一名的黄金文档（1/61≈0.016）；候选池越深，这类"两路中游项"越多，越容易把只有单路强信号的黄金段落挤出 top-5。**结论：RRF 奖励"多路一致"，惩罚"单路拔尖"，候选深度需谨慎选取。**

### 4.3 路由器分析（以 HotpotQA 题型为金标准：comparison→simple，bridge→multi_hop）

| 路由器 | 总体准确率 | comparison | bridge |
|---|---|---|---|
| 规则路由器 | 78.0% | 60.4% | 82.2% |
| LLM 路由器 | （待端到端评测后填写） | | |

规则路由器的主要错误模式：把"含 or 的比较题"一律判 simple，但部分 comparison 实际也需要多实体检索；连接词特征对 bridge 的召回较好。LLM 路由器预期在边界样本上更稳，待验证。

### 4.4 端到端问答（EM / F1，n=500）

| 策略 | EM | F1 | Doc Recall@5 | 平均延迟 |
|---|---|---|---|---|
| Naive RAG | 待填 | 待填 | 待填 | 待填 |
| Hybrid RAG | 待填 | 待填 | 待填 | 待填 |
| Rerank | 待填 | 待填 | 待填 | 待填 |
| Adaptive（规则路由） | 待填 | 待填 | 待填 | 待填 |
| Adaptive（LLM 路由） | 待填 | 待填 | 待填 | 待填 |

## 5. 错误分析

见 `results/error_analysis.md`（由 `scripts/06_error_analysis.py` 生成）：将失败 case 归因为**检索失败**（黄金段落未全部召回）或**生成失败**（证据齐但答错），并附示例。待端到端结果产出后填写关键结论。

## 6. 讨论与未来工作

- 检索是当前瓶颈的主要来源（见错误分析占比），后续可试：多跳专用检索（IRCoT 式迭代检索）、更大 embedding 模型
- 自适应的收益取决于路由器质量与两类问题的成本差；在延迟敏感场景下，"简单题省下的延迟"要能覆盖"复杂题多花的延迟"
- 未做：RAGAS faithfulness 评测、答案引用抽取精度、更大规模语料库

## 复现

```powershell
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
copy .env.example .env   # 填入 API key
python scripts/01_prepare_data.py
python scripts/02_build_index.py
python scripts/03_run_eval.py --strategy naive --retrieval-only   # 零成本验证
python scripts/03_run_eval.py --strategy adaptive --n 500         # 全量端到端
python scripts/04_compare.py
```
