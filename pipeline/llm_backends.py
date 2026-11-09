"""
LLM-backendek: Transformers (alapértelmezett) és Ollama (F12).  Felelős: B
"""


def generate(system_prompt: str, user_prompt: str, backend: str | None = None) -> str:
    """Szöveget generál a választott backenddel, és a nyers választ adja vissza."""
    raise NotImplementedError("TODO (B): Transformers- és Ollama-backend")
