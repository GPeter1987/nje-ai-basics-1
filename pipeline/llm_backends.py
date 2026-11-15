"""LLM-backendek: Transformers (alapértelmezett) és Ollama (F12).  Felelős: B"""

from __future__ import annotations

from functools import lru_cache

import config

# Ezzel a szöveggel kezdődik a hibaüzenet, ha az Ollama-szerver nem fut; a felület
# erre ismeri fel, és ad érthető útmutatást.
OLLAMA_UNAVAILABLE = "Az Ollama-backend nem érhető el"


def _dtype_kwargs() -> dict[str, str]:
    """A modell a saját tárolt pontosságában töltődjön (Qwen: bf16, kb. 3 GB RAM).

    Paraméter nélkül a transformers 4.x fp32-ben tölt (kb. 6 GB). A 4.56-tól a
    paraméter neve `dtype`, előtte `torch_dtype`.
    """
    import transformers
    from packaging.version import Version

    name = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    return {name: "auto"}


@lru_cache(maxsize=1)
def _get_transformer_runtime(model_name: str):
    """Lusta modellbetöltés, hogy az import és a gyors tesztek ne töltsenek modellt."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **_dtype_kwargs())
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
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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


def _generate_ollama(system_prompt: str, user_prompt: str, json_schema: dict | None = None) -> str:
    import requests

    url = config.OLLAMA_URL.rstrip("/") + "/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        # JSON-sémánál az Ollama a sémához köti a kimenetet (strukturált kimenet, enumokkal)
        "format": json_schema or "json",
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
        raise RuntimeError(f"{OLLAMA_UNAVAILABLE} ({url}): {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("Az Ollama hibás HTTP-választ adott.") from exc

    try:
        content = body["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Az Ollama válaszából hiányzik a generált szöveg.") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Az Ollama üres választ adott.")
    return content.strip()


def ollama_available(timeout: float = 1.0) -> bool:
    """Gyors ellenőrzés a felületnek: fut-e az Ollama-szerver."""
    import requests

    try:
        response = requests.get(config.OLLAMA_URL.rstrip("/") + "/api/tags", timeout=timeout)
        return response.ok
    except requests.RequestException:
        return False


def generate(
    system_prompt: str,
    user_prompt: str,
    backend: str | None = None,
    json_schema: dict | None = None,
) -> str:
    """Szöveget generál a választott backenddel, és a nyers választ adja vissza.

    json_schema: az Ollama ehhez köti a kimenetet; a Transformers-backend figyelmen kívül
    hagyja (ott a prompt és az analyzer validálása véd).
    """
    selected = backend or config.LLM_BACKEND
    if selected == "transformers":
        return _generate_transformers(system_prompt, user_prompt)
    if selected == "ollama":
        return _generate_ollama(system_prompt, user_prompt, json_schema)
    supported = ", ".join(config.LLM_BACKENDS)
    raise ValueError(f"Ismeretlen LLM-backend: {selected!r}. Elérhető: {supported}.")
