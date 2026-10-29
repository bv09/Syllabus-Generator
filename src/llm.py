import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL")

if not API_KEY:
    raise ValueError("Missing LLM_API_KEY in .env")

if not MODEL:
    raise ValueError("Missing LLM_MODEL in .env")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL or None,
    timeout=30.0,
)

def chat(system: str, user: str, temperature: float = 0.3) -> str:
    r = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return r.choices[0].message.content or ""

if __name__ == "__main__":
    print("Starting...", flush=True)
    print(f"Model: {MODEL}", flush=True)

    answer = chat(
        system="You are a helpful AI assistant.",
        user="Explain what an API is in 3 simple sentences.",
    )

    print("Response received!", flush=True)
    print(answer)
