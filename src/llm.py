"""Lớp mỏng quanh API tương thích OpenAI.

Client được tạo lười (lazy) để test/import không cần .env.
Biến môi trường:
  LLM_API_KEY, LLM_BASE_URL, LLM_MODEL        (bắt buộc: KEY, MODEL)
  LLM_JUDGE_MODEL   model chấm điểm (nên KHÁC model sinh, xem README)
  LLM_EMBED_MODEL   nếu đặt -> bật dense retrieval; nếu không chỉ dùng BM25
  LLM_JSON_MODE=1   bật response_format=json_object (không phải provider nào cũng hỗ trợ)
  LLM_TIMEOUT       giây, mặc định 120 (30s cũ quá ngắn cho đầu ra JSON dài)
"""
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    key = os.getenv("LLM_API_KEY")
    if not key:
        raise RuntimeError("Missing LLM_API_KEY in .env")
    return OpenAI(
        api_key=key,
        base_url=os.getenv("LLM_BASE_URL") or None,
        timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        max_retries=3,
    )


def chat(system: str, user: str, temperature: float = 0.2,
         model: str | None = None, json_mode: bool | None = None) -> str:
    model = model or os.getenv("LLM_MODEL")
    if not model:
        raise RuntimeError("Missing LLM_MODEL in .env")
    kwargs = {}
    if json_mode is None:
        json_mode = os.getenv("LLM_JSON_MODE") == "1"
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    r = _client().chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )
    return r.choices[0].message.content or ""


def judge_chat(system: str, user: str) -> str:
    """Gọi model chấm điểm, temperature 0."""
    return chat(system, user, temperature=0.0,
                model=os.getenv("LLM_JUDGE_MODEL") or os.getenv("LLM_MODEL"))


def has_embeddings() -> bool:
    return bool(os.getenv("LLM_EMBED_MODEL"))


def embed(texts: list[str]) -> list[list[float]]:
    model = os.getenv("LLM_EMBED_MODEL")
    if not model:
        raise RuntimeError("Missing LLM_EMBED_MODEL in .env")
    out: list[list[float]] = []
    for i in range(0, len(texts), 64):
        r = _client().embeddings.create(model=model, input=texts[i:i + 64])
        out.extend(d.embedding for d in r.data)
    return out


if __name__ == "__main__":
    print(chat("You are a helpful AI assistant.",
               "Explain what an API is in 3 simple sentences."))
