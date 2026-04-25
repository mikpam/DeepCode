from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def estimate_tokens(text: str, model: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def build_client() -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai package is required. Install dependencies with pip install -r requirements.txt") from exc
    return OpenAI(api_key=api_key)


def generate_answer(client: Any, model: str, prompt: str, temperature: float = 0) -> str:
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
        )
        text = getattr(response, "output_text", "")
        if text:
            return text.strip()
    except Exception:
        pass

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return completion.choices[0].message.content.strip()


def safe_json_loads(value: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(value)
    except Exception:
        return None
