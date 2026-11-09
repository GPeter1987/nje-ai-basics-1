"""
A modellek előletöltése és ellenőrzése.  Felelős: A

A letöltés után a lánc internet nélkül is fut (a modellek a Hugging Face
gyorsítótárába kerülnek: ~/.cache/huggingface).

Mit tölt le:
    1. modell  ViT / Food-101          (nateraw/food)              ~345 MB
    3. modell  all-MiniLM-L6-v2        (sentence-transformers)      ~91 MB
    2. modell  Qwen2.5-1.5B-Instruct   a választott backend szerint:
                 --llm ollama        Ollamán keresztül (qwen2.5:1.5b, Q4)   ~1 GB
                 --llm transformers  Hugging Face-ről (safetensors)        ~3,1 GB
                 --llm both / none

Futtatás (a projekt gyökeréből):
    python scripts/download_models.py                 # LLM a config.LLM_BACKEND szerint
    python scripts/download_models.py --llm ollama
    python scripts/download_models.py --no-verify     # csak letöltés, betöltési próba nélkül
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

# A szkript a scripts/ mappából fut, ezért a projekt gyökerét fel kell venni az útvonalra
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# Csak a szükséges fájlokat töltjük le. Pl. a MiniLM repóban ONNX-, OpenVINO-,
# TensorFlow- és Rust-változat is van – ezekkel együtt ~1 GB lenne a ~91 MB helyett.
HF_ALLOW_PATTERNS: dict[str, list[str]] = {
    config.RECOGNIZER_MODEL: ["config.json", "preprocessor_config.json", "pytorch_model.bin"],
    config.EMBEDDING_MODEL: [
        "*.json",
        "1_Pooling/*",
        "model.safetensors",
        "tokenizer*",
        "vocab.txt",
    ],
    config.LLM_MODEL: ["*.json", "merges.txt", "vocab.json", "model.safetensors"],
}


@dataclass
class StepResult:
    name: str
    ok: bool
    message: str


# --- Hugging Face modellek -------------------------------------------------------


def download_hf_model(repo_id: str) -> str:
    """Letölti a modell szükséges fájljait a HF gyorsítótárba; a helyi mappát adja vissza."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, allow_patterns=HF_ALLOW_PATTERNS.get(repo_id))


def folder_size_mb(path: str | Path) -> float:
    files = [p for p in Path(path).rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files) / (1024 * 1024)


def verify_recognizer() -> str:
    """A ViT betölthető, és a címkéi egyeznek a mi 101 kulcsunkkal."""
    from PIL import Image

    from pipeline import recognizer
    from pipeline.food101_labels import FOOD101_LABELS

    classifier = recognizer._get_classifier()
    labels = set(classifier.model.config.id2label.values())
    if labels != set(FOOD101_LABELS):
        raise RuntimeError("a modell címkéi eltérnek a pipeline/food101_labels.py listától")
    recognizer.recognize(Image.new("RGB", (224, 224)))
    return f"{len(labels)} osztály, címkék rendben"


def verify_embedder() -> str:
    """A MiniLM betölthető, és 384 dimenziós vektort ad."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    vector = model.encode("main course; indian; chicken, onion, tomato; stewed; spicy")
    if vector.shape != (config.EMBEDDING_DIM,):
        raise RuntimeError(f"váratlan vektorméret: {vector.shape}")
    return f"vektor: {vector.shape[0]} dimenzió"


def verify_llm_transformers() -> str:
    """Csak a tokenizert és a chat templatet ellenőrizzük (a 3 GB-os modell betöltése lassú)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], tokenize=False, add_generation_prompt=True
    )
    return f"tokenizer és chat template rendben ({len(prompt)} karakteres prompt)"


# --- Ollama ----------------------------------------------------------------------


def ollama_models(base_url: str) -> list[str]:
    """Az Ollamán elérhető modellek neve. Ha a szerver nem fut, ConnectionError."""
    response = requests.get(f"{base_url}/api/tags", timeout=5)
    response.raise_for_status()
    return [m["name"] for m in response.json().get("models", [])]


def ollama_has_model(available: list[str], model: str) -> bool:
    # a "qwen2.5:1.5b" és a "qwen2.5:1.5b-instruct-q4_K_M" stb. is elfogadható; tag nélkül :latest
    wanted = model if ":" in model else f"{model}:latest"
    return any(name == wanted or name.startswith(wanted + "-") for name in available)


def ollama_pull(base_url: str, model: str) -> None:
    """Letölti a modellt az Ollama REST API-n, folyamatjelzéssel."""
    with requests.post(
        f"{base_url}/api/pull", json={"model": model, "stream": True}, stream=True, timeout=30
    ) as response:
        response.raise_for_status()
        last_status = ""
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if "error" in event:
                raise RuntimeError(event["error"])
            status = event.get("status", "")
            total, done = event.get("total"), event.get("completed")
            if total and done:
                print(f"\r    {status}: {done / total:6.1%}", end="", flush=True)
            elif status != last_status:
                print(f"\n    {status}", end="", flush=True)
            last_status = status
        print()


def verify_llm_ollama(base_url: str, model: str) -> str:
    start = time.perf_counter()
    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
            "options": {"temperature": 0, "num_predict": 5},
        },
        timeout=120,
    )
    response.raise_for_status()
    answer = response.json()["message"]["content"].strip()
    return f"válasz: {answer!r} ({time.perf_counter() - start:.1f} s)"


def setup_ollama(base_url: str, model: str, verify: bool) -> StepResult:
    name = f"2. modell (Ollama: {model})"
    try:
        available = ollama_models(base_url)
    except requests.RequestException:
        return StepResult(
            name,
            False,
            f"az Ollama nem érhető el ({base_url}). Telepítsd (ollama.com), indítsd el, "
            "majd futtasd újra a szkriptet.",
        )

    try:
        if ollama_has_model(available, model):
            message = "már letöltve"
        else:
            print(f"    letöltés: {model} (~1 GB)")
            ollama_pull(base_url, model)
            message = "letöltve"
        if verify:
            message += "; " + verify_llm_ollama(base_url, model)
        return StepResult(name, True, message)
    except Exception as exc:  # noqa: BLE001
        return StepResult(name, False, f"{type(exc).__name__}: {exc}")


# --- Fő program --------------------------------------------------------------------


def run_hf_step(name: str, repo_id: str, verify_func, verify: bool) -> StepResult:
    try:
        print(f"    letöltés: {repo_id}")
        path = download_hf_model(repo_id)
        message = f"{folder_size_mb(path):.0f} MB"
        if verify:
            print("    ellenőrzés: betöltés...")
            message += "; " + verify_func()
        return StepResult(name, True, message)
    except Exception as exc:  # noqa: BLE001
        return StepResult(name, False, f"{type(exc).__name__}: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NutriLens – modellek előletöltése")
    parser.add_argument(
        "--llm",
        choices=["ollama", "transformers", "both", "none"],
        default=config.LLM_BACKEND,
        help=f"a 2. modell (Qwen) honnan fusson (alapból: {config.LLM_BACKEND})",
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="ne próbálja betölteni a modelleket letöltés után"
    )
    parser.add_argument("--ollama-url", default=config.OLLAMA_URL)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Windows-konzolon / átirányított kimenetnél se legyen hiba az ékezetes betűktől
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    verify = not args.no_verify
    results: list[StepResult] = []

    print("[1/3] ViT / Food-101")
    results.append(
        run_hf_step("1. modell (ViT)", config.RECOGNIZER_MODEL, verify_recognizer, verify)
    )

    print("[2/3] all-MiniLM-L6-v2")
    results.append(
        run_hf_step("3. modell (MiniLM)", config.EMBEDDING_MODEL, verify_embedder, verify)
    )

    print(f"[3/3] Qwen2.5-1.5B-Instruct (--llm {args.llm})")
    if args.llm in ("transformers", "both"):
        print("    figyelem: ~3,1 GB")
        results.append(
            run_hf_step(
                "2. modell (Transformers)", config.LLM_MODEL, verify_llm_transformers, verify
            )
        )
    if args.llm in ("ollama", "both"):
        results.append(setup_ollama(args.ollama_url, config.OLLAMA_MODEL, verify))
    if args.llm == "none":
        print("    kihagyva")

    print("\n--- Összesítés ---")
    for result in results:
        mark = "OK  " if result.ok else "HIBA"
        print(f"[{mark}] {result.name}: {result.message}")

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
