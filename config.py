"""
NutriLens – központi beállítások.

Minden modellnév, küszöb és útvonal itt van, hogy a modulok ne tartalmazzanak
"beégetett" értékeket. Ha valamit módosítani kell (pl. másik checkpoint),
elég itt átírni.

Néhány érték környezeti változóval felülírható (pl. NUTRILENS_LLM_BACKEND=ollama).
"""

import os
from pathlib import Path

# --- Útvonalak ---------------------------------------------------------------

# A projekt gyökérkönyvtára (ahol ez a fájl van)
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
DB_PATH = Path(os.getenv("NUTRILENS_DB_PATH", DATA_DIR / "nutrilens.db"))
SEED_CSV_PATH = DATA_DIR / "foods_seed.csv"
SAMPLES_DIR = ROOT_DIR / "samples"

# --- 1. modell: ételfelismerés (ViT, Food-101) ------------------------------

RECOGNIZER_MODEL = "nateraw/food"
RECOGNIZER_TOP_K = 3

# Ez alatt a bizonyosság alatt figyelmeztetünk, és a felhasználó választ (F3)
CONFIDENCE_THRESHOLD = 0.40

# --- 2. modell: strukturált elemzés (Qwen2.5-1.5B-Instruct) -----------------

LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Transformers-backend
OLLAMA_MODEL = "qwen2.5:1.5b"  # Ollama-backend
OLLAMA_URL = os.getenv("NUTRILENS_OLLAMA_URL", "http://localhost:11434")

LLM_BACKENDS = ("transformers", "ollama")
LLM_BACKEND = os.getenv("NUTRILENS_LLM_BACKEND", "transformers")

LLM_TEMPERATURE = 0.2
LLM_MAX_NEW_TOKENS = 400
LLM_MAX_RETRIES = 1  # hibás JSON esetén ennyiszer próbáljuk újra

# --- 3. modell: ajánlás (all-MiniLM-L6-v2) ----------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
RECOMMEND_TOP_K = 5

# Ajánlási módok (F8): belső kulcs -> megjelenített név
RECOMMEND_MODES = {
    "similar": "Hasonló",
    "healthier": "Egészségesebb (kevesebb kcal)",
    "allergen_free": "Allergénmentes",
}
DEFAULT_RECOMMEND_MODE = "similar"

# --- Bemenet és adag ---------------------------------------------------------

ALLOWED_IMAGE_TYPES = ("jpg", "jpeg", "png", "webp")
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB (F1)

DEFAULT_GRAMS = 250  # F4
MIN_GRAMS = 10
MAX_GRAMS = 2000

# --- Tápérték-ellenőrzés -----------------------------------------------------

# 4*fehérje + 4*szénhidrát + 9*zsír ennyivel térhet el a megadott kcal-tól
KCAL_TOLERANCE = 0.15

# --- Összes letöltendő Hugging Face modell (scripts/download_models.py) ------

HF_MODELS = (RECOGNIZER_MODEL, LLM_MODEL, EMBEDDING_MODEL)
