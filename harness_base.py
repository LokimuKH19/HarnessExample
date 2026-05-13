"""
harness_base.py — Harness 基类（考生不可修改）
"""


class Harness:
    def __init__(self, call_llm, count_tokens, count_messages_tokens, max_prompt_tokens: int):
        self.call_llm = call_llm                          # 调用 LLM；prompt 超过 max_prompt_tokens 时会被自动截断
        self.count_tokens = count_tokens                  # 计算单段文本的 token 数
        self.count_messages_tokens = count_messages_tokens  # 计算 messages 列表的总 token 数
        self.max_prompt_tokens = max_prompt_tokens
        self.memory: list[tuple[str, str]] = []

    def update(self, text: str, label: str) -> None:
        """接收一条带标签的训练样本，更新内部记忆。"""
        self.memory.append((text, label))

    def predict(self, _text: str) -> str:
        """对文本预测标签，返回标签字符串。子类必须实现。"""
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__
