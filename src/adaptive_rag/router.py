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
    """LLM 路由器（Week 2 升级点：实现它并与规则路由对比）。"""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def classify(self, question: str) -> str:
        # TODO(Week 2): 用少量 few-shot 让 LLM 输出 simple / multi_hop，
        # 并在评测集上统计路由准确率 + 对下游 EM 的影响。
        raise NotImplementedError("Week 2 待实现")
