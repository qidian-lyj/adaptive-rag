"""OpenAI 兼容 LLM 客户端与 Prompt。支持 DeepSeek / OpenAI / 本地 Ollama。"""
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from .config import MAX_CONTEXTS

load_dotenv()  # 读取项目根目录的 .env


def get_client() -> tuple[OpenAI, str]:
    """返回 (client, model_name)，配置来自环境变量。"""
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    if not api_key:
        raise RuntimeError(
            "缺少 LLM_API_KEY：请复制 .env.example 为 .env 并填写（支持 DeepSeek/OpenAI/Ollama）"
        )
    return OpenAI(api_key=api_key, base_url=base_url), model


def _chat(client: OpenAI, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # 评测任务用贪心解码，保证可复现
    )
    return resp.choices[0].message.content.strip()


def answer_question(question: str, contexts: list[str], client: OpenAI, model: str) -> str:
    """基于给定参考资料回答问题，要求标注引用编号。"""
    refs = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts[:MAX_CONTEXTS]))
    prompt = f"""You are a precise QA assistant. Answer the question using ONLY the references below.

Rules:
1. Do not use any knowledge outside the references. If the answer is not in them, say "I don't know".
2. Keep the answer SHORT - HotpotQA answers are usually short entities (a name, a date, yes/no).
3. After the answer, cite the reference numbers you used, like [1][3].

References:
{refs}

Question: {question}
Answer:"""
    return _chat(client, model, prompt)


def decompose_question(question: str, client: OpenAI, model: str) -> list[str]:
    """把多跳问题拆解成 2 个子问题（Adaptive RAG 的关键步骤）。"""
    prompt = f"""Decompose the following multi-hop question into exactly 2 simpler sub-questions
that can each be answered by retrieving one Wikipedia paragraph.
Output ONLY a JSON array of 2 strings, no other text.

Question: {question}
JSON array:"""
    raw = _chat(client, model, prompt)
    try:
        subs = json.loads(_extract_json_array(raw))
        if isinstance(subs, list) and len(subs) >= 1:
            return [str(s) for s in subs][:2]
    except Exception:
        pass
    return [question]  # 解析失败时退化为原始问题，保证流程不中断


def _extract_json_array(text: str) -> str:
    """从模型输出中鲁棒地提取第一个 JSON 数组。"""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    return m.group(0) if m else "[]"
