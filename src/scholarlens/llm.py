
"""Thin LLM wrapper: Gemini, Anthropic API or local Ollama, JSON helpers, on-disk cache."""
from __future__ import annotations

import hashlib
import json
import os
import re

from . import config

_anthropic_client = None
_gemini_client = None


def _cache_file(key: str):
    return config.CACHE_DIR / (
        hashlib.md5(key.encode("utf-8")).hexdigest() + ".txt"
    )


def _anthropic(system: str, user: str, max_tokens: int, temperature):
    """Call the Anthropic Messages API."""
    global _anthropic_client

    import anthropic

    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()

    kw = {
        "model": config.LLM_MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }

    # Some Anthropic models/API versions do not accept temperature.
    if temperature is not None:
        kw["temperature"] = temperature

    try:
        resp = _anthropic_client.messages.create(**kw)
    except (TypeError, anthropic.BadRequestError):
        kw.pop("temperature", None)
        resp = _anthropic_client.messages.create(**kw)

    return "".join(
        block.text
        for block in resp.content
        if getattr(block, "type", "") == "text"
    )


def _gemini(system: str, user: str, max_tokens: int, temperature):
    """Call Google's Gemini API using the google-genai SDK."""
    global _gemini_client

    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it in your terminal environment."
        )

    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=api_key)

    generation_config = {
        "system_instruction": system,
        "max_output_tokens": max_tokens,
    }

    if temperature is not None:
        generation_config["temperature"] = temperature

    response = _gemini_client.models.generate_content(
        model=config.LLM_MODEL,
        contents=user,
        config=types.GenerateContentConfig(**generation_config),
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response. Check the model, API access, "
            "and whether the response was blocked."
        )

    return response.text


def _ollama(system: str, user: str, max_tokens: int, temperature):
    """Call a locally running Ollama server."""
    import requests

    r = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        timeout=300,
        json={
            "model": config.OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": temperature if temperature is not None else 0.0,
                "num_predict": max_tokens,
            },
        },
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def complete(
    system: str,
    user: str,
    max_tokens: int = 2000,
    temperature: float | None = 0.0,
) -> str:
    """Generate text using the configured provider, with optional caching."""
    provider = config.LLM_PROVIDER.strip().lower()
    model = (
        config.OLLAMA_MODEL if provider == "ollama" else config.LLM_MODEL
    )

    key = (
        f"{provider}|{model}|{max_tokens}|{temperature}|{system}|{user}"
    )
    use_cache = config.LLM_CACHE and temperature == 0.0
    cache_path = _cache_file(key)

    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    providers = {
        "gemini": _gemini,
        "anthropic": _anthropic,
        "ollama": _ollama,
    }

    fn = providers.get(provider)
    if fn is None:
        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. "
            "Choose 'gemini', 'anthropic', or 'ollama' in config.py."
        )

    result = fn(system, user, max_tokens, temperature)

    if use_cache:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(result, encoding="utf-8")

    return result


def extract_json(text: str):
    """Parse JSON directly or extract the first balanced object/array."""
    text = re.sub(
        r"^```(?:json)?|```$",
        "",
        text.strip(),
        flags=re.M,
    ).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for i, ch in enumerate(text):
        if ch not in "{[":
            continue

        opening = ch
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escaped = False

        for j in range(i, len(text)):
            char = text[j]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1

                if depth == 0:
                    try:
                        return json.loads(text[i : j + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError("No JSON found in model output")


def complete_json(
    system: str,
    user: str,
    max_tokens: int = 2000,
):
    """Generate and parse JSON, retrying once if the output is invalid."""
    out = complete(system, user, max_tokens)

    try:
        return extract_json(out)
    except ValueError:
        out = complete(
            system,
            user + "\n\nReturn ONLY valid JSON, nothing else.",
            max_tokens,
            temperature=0.0,
        )
        return extract_json(out)
