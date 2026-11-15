"""LLM-backendek: Transformers (alapértelmezett) és Ollama (F12).  Felelős: B"""

from __future__ import annotations

from functools import lru_cache

import config


@lru_cache(maxsize=1)
def _get_transformer_runtime(model_name: str):
    """Lusta modellbetöltés, hogy az import és a gyors tesztek ne töltsenek modellt."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _generate_transformers(system_prompt: str, user_prompt: str) -> str:
    import torch

    tokenizer, model, device = _get_transformer_runtime(config.LLM_MODEL)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    temperature = float(config.LLM_TEMPERATURE)
    generation_args = {
        "max_new_tokens": int(config.LLM_MAX_NEW_TOKENS),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_args.update(do_sample=True, temperature=temperature)
    else:
        generation_args["do_sample"] = False

    with torch.inference_mode():
        output = model.generate(**inputs, **generation_args)
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _generate_ollama(system_prompt: str, user_prompt: str) -> str:
    import requests

    url = config.OLLAMA_URL.rstrip("/") + "/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": float(config.LLM_TEMPERATURE),
            "num_predict": int(config.LLM_MAX_NEW_TOKENS),
        },
    }
    try:
        response = requests.post(url, json=payload, timeout=(10, 180))
        response.raise_for_status()
        body = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Az Ollama-backend nem érhető el ({url}): {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("Az Ollama hibás HTTP-választ adott.") from exc

    try:
        content = body["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Az Ollama válaszából hiányzik a generált szöveg.") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Az Ollama üres választ adott.")
    return content.strip()


def generate(system_prompt: str, user_prompt: str, backend: str | None = None) -> str:
    """Szöveget generál a választott backenddel, és a nyers választ adja vissza."""
    selected = backend or config.LLM_BACKEND
    if selected == "transformers":
        return _generate_transformers(system_prompt, user_prompt)
    if selected == "ollama":
        return _generate_ollama(system_prompt, user_prompt)
    supported = ", ".join(config.LLM_BACKENDS)
    raise ValueError(f"Ismeretlen LLM-backend: {selected!r}. Elérhető: {supported}.")
