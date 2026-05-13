"""
llm_client.py
================
LLM 客户端配置文件 —— 修改下方 OPENAI_CONFIG 接入你的 API。

适用于：OpenAI Compatible格式。
"""

import os
import time

# ============================================================
# 本地测试时，你可以修改这里接入你的 API
# ============================================================
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"   # 填入你的 API endpoint
API_KEY  = "sk-cee118558ddd41ab824216e5157ad44d"                     # 填入你的 API key
MODEL    = "qwen3.6-max-preview"           # 填入你的模型名

# ============================================================
# 以下代码无需修改
# ============================================================

OPENAI_CONFIG = {
    "base_url": BASE_URL,
    "api_key":  API_KEY,
    "model":    MODEL,
    "temperature": 1.0,
    "top_p":       1.0,
    "max_tokens":  8192,
}

_client = None


def _init_client():
    global _client
    from openai import OpenAI
    _client = OpenAI(
        base_url=OPENAI_CONFIG["base_url"],
        api_key=OPENAI_CONFIG["api_key"],
    )


def call_llm(messages: list[dict], retries: int = 2) -> str:
    """调用 LLM，输入 OpenAI 格式 messages，返回回复文本。"""
    if _client is None:
        _init_client()

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = _client.chat.completions.create(
                model=OPENAI_CONFIG["model"],
                messages=messages,
                temperature=OPENAI_CONFIG["temperature"],
                top_p=OPENAI_CONFIG["top_p"],
                max_tokens=OPENAI_CONFIG["max_tokens"],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
    raise last_err


_tokenizer = None
_tokenizer_loaded = False
_tokenizer_lock = __import__("threading").Lock()


def _load_tokenizer():
    from transformers import AutoTokenizer
    import os
    _dir = os.path.join(os.path.dirname(__file__), "tokenizer")
    return AutoTokenizer.from_pretrained(_dir, trust_remote_code=True)


def count_tokens(text: str) -> int:
    """计算单段文本的 token 数。"""
    global _tokenizer, _tokenizer_loaded
    if not _tokenizer_loaded:
        with _tokenizer_lock:
            if not _tokenizer_loaded:
                _tokenizer = _load_tokenizer()
                _tokenizer_loaded = True
    if not text:
        return 0
    return len(_tokenizer(text, add_special_tokens=False)["input_ids"])


def count_messages_tokens(messages: list[dict]) -> int:
    """计算 messages 列表的总 token 数（仅计算content，与评分系统一致）。
    可在调用 call_llm 前用于检查是否超出 max_prompt_tokens。
    """
    return count_tokens(" ".join(m.get("content", "") for m in messages))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """将文本截断到至多 max_tokens 个 token（使用真实 tokenizer）。"""
    global _tokenizer, _tokenizer_loaded
    if not _tokenizer_loaded:
        _tokenizer = _load_tokenizer()
        _tokenizer_loaded = True
    if not text:
        return text
    ids = _tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return _tokenizer.decode(ids[:max_tokens])


if __name__ == "__main__":
    print("Testing LLM connection...")
    try:
        result = call_llm([{"role": "user", "content": "Say 'hello' in one word."}])
        print(f"✓ Connected. Response: {result[:100]}")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("请检查 llm_client.py 中的 OPENAI_CONFIG")
