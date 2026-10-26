"""自适应路由：判断问题走「单跳检索」还是「多跳分解」。

v0 是规则路由（基于 HotpotQA 两类问题的表面特征），刻意留有提升空间——
Week 2 把它升级为 LLM 路由并统计路由准确率，本身就是一个很好的对比实验。
"""
import re


class RuleBasedRouter:
    """规则路由器。

    HotpotQA 有两类多跳问题：
    - comparison: 两个实体比属性（"Who was born first, A or B?"）
    - bridge:     需要跨段落串联线索（"The director of X was born in which city?"）
    """

    _CONNECTIVES = re.compile(
        r"\b(after|before|during|following|which led to|as a result|because of)\b", re.I
    )

    def classify(self, question: str) -> str:
        """返回 'simple' 或 'multi_hop'。"""
        words = question.split()

        # 线索 1：时间/因果连接词，往往意味着需要串联两个事实
        if self._CONNECTIVES.search(question):
            return "multi_hop"

        # 线索 2：含 " or " 的比较问句（comparison）通常两个实体各查一段就够
        if " or " in question.lower():
            return "simple"

        # 线索 3：大写实体词多于 2 个，多半涉及多实体链条
        entities = [w for w in words if w[:1].isupper() and not question.startswith(w)]
        if len(entities) > 2:
            return "multi_hop"

        # 线索 4：问题过长，大概率是复合问句
        if len(words) > 14:
            return "multi_hop"

        return "simple"


class LLMRouter:
    """LLM 路由器：用小模型做 0/2-shot 分类，判断问题是否需要多跳分解。

    对比实验素材：规则路由 vs LLM 路由的「路由准确率」与「下游 EM」。
    """

    _PROMPT = """You route questions to a retrieval system. Classify each question as:
- simple: answerable by retrieving ONE relevant paragraph. Two-entity comparisons (e.g. "Who was born first, A or B?") are simple: each fact can be looked up independently.
- multi_hop: requires CHAINING facts across paragraphs, e.g. "The director of film X was born in which city?" (first find the director of X, then look up the director's birthplace).

Examples:
Q: Who was born first, Thomas Edison or Nikola Tesla? -> simple
Q: The novel whose author died in 1861 was adapted into a film directed by whom? -> multi_hop
Q: Where was the president who signed the law born? -> multi_hop
Q: Which magazine was started first, Arthur's Magazine or First for Women? -> simple

Q: {question}
Answer with exactly one word, simple or multi_hop:"""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def classify(self, question: str) -> str:
        from .generation import _chat

        raw = _chat(self.client, self.model, self._PROMPT.format(question=question)).strip().lower()
        return "multi_hop" if "multi" in raw else "simple"
