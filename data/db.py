"""SQLite-elérés. A kapcsolatokat minden művelet után lezárjuk. Felelős: C."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import config
from pipeline.schemas import DishAnalysis, Nutrition, Recognition

SCHEMA = """
CREATE TABLE IF NOT EXISTS foods (
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    name_en TEXT NOT NULL, name_hu TEXT NOT NULL,
    category TEXT NOT NULL, cuisine TEXT NOT NULL,
    ingredients TEXT NOT NULL, allergens TEXT NOT NULL, tags TEXT NOT NULL,
    kcal_100g REAL NOT NULL CHECK(kcal_100g >= 0),
    protein_100g REAL NOT NULL CHECK(protein_100g >= 0),
    carbs_100g REAL NOT NULL CHECK(carbs_100g >= 0),
    fat_100g REAL NOT NULL CHECK(fat_100g >= 0),
    description_en TEXT NOT NULL,
    embedding BLOB, embedding_model TEXT,
    in_food101 INTEGER NOT NULL CHECK(in_food101 IN (0, 1)),
    source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analysis_cache (
    dish_key TEXT NOT NULL, json TEXT NOT NULL, model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(dish_key, model)
);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL,
    image_hash TEXT, dish_key TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    grams REAL NOT NULL CHECK(grams > 0),
    kcal INTEGER NOT NULL CHECK(kcal >= 0),
    protein REAL NOT NULL CHECK(protein >= 0),
    carbs REAL NOT NULL CHECK(carbs >= 0),
    fat REAL NOT NULL CHECK(fat >= 0),
    is_estimate INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS history_created_at ON history(created_at);
"""


def get_connection() -> sqlite3.Connection:
    """Kompatibilis az eredeti API-val; a hívó felelős a close()-ért."""
    path = Path(config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connection():
    conn = get_connection()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    with connection() as conn:
        conn.executescript(SCHEMA)


def _require_db() -> None:
    if not Path(config.DB_PATH).is_file():
        raise FileNotFoundError("Nincs adatbázis. Futtasd: python scripts/build_db.py")


def _food(row: sqlite3.Row) -> dict:
    food = dict(row)
    for column in ("ingredients", "allergens", "tags"):
        food[column] = json.loads(food[column])
    return food


def get_food(key: str) -> dict | None:
    _require_db()
    with connection() as conn:
        row = conn.execute("SELECT * FROM foods WHERE key = ?", (key,)).fetchone()
    return _food(row) if row else None


def list_foods() -> list[dict]:
    _require_db()
    with connection() as conn:
        return [_food(row) for row in conn.execute("SELECT * FROM foods ORDER BY key")]


def save_analysis(analysis: DishAnalysis, model: str) -> None:
    """B modulja használhatja; a különböző backendek cache-e nem keveredik."""
    with connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO analysis_cache VALUES (?, ?, ?, ?)",
            (analysis.dish, analysis.model_dump_json(), model, _now()),
        )


def get_analysis(dish_key: str, model: str) -> DishAnalysis | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT json FROM analysis_cache WHERE dish_key = ? AND model = ?",
            (dish_key, model),
        ).fetchone()
    if row is None:
        return None
    analysis = DishAnalysis.model_validate_json(row["json"])
    return analysis.model_copy(update={"source": "cache"})


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_history(recognition: Recognition, nutrition: Nutrition) -> int:
    if recognition.key != nutrition.dish_key:
        raise ValueError("A felismerés és a tápérték ételkulcsa eltér.")
    if not all(math.isfinite(v) for v in (
        nutrition.grams, nutrition.protein, nutrition.carbs, nutrition.fat,
    )):
        raise ValueError("Az előzmény csak véges tápértékeket tartalmazhat.")
    with connection() as conn:
        cursor = conn.execute(
            """INSERT INTO history
            (created_at, image_hash, dish_key, confidence, grams, kcal, protein, carbs, fat,
             is_estimate) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (_now(), recognition.image_hash, recognition.key, recognition.score,
             nutrition.grams, nutrition.kcal, nutrition.protein, nutrition.carbs,
             nutrition.fat, int(nutrition.is_estimate)),
        )
        return cursor.lastrowid


def list_history(limit: int = 200) -> list[dict]:
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("A limit pozitív egész szám legyen.")
    with connection() as conn:
        return [dict(row) for row in conn.execute(
            """SELECT h.*, COALESCE(f.name_hu, h.dish_key) AS name_hu
            FROM history h LEFT JOIN foods f ON f.key = h.dish_key
            ORDER BY h.created_at DESC, h.id DESC LIMIT ?""", (limit,),
        )]


def daily_summary(day: date | None = None) -> dict:
    """Helyi naptári nap szerint; nem csak a UI-ban listázott utolsó 200 sorból."""
    day = day or date.today()
    with connection() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS meals, COALESCE(SUM(kcal), 0) AS kcal,
            COALESCE(SUM(protein), 0) AS protein, COALESCE(SUM(carbs), 0) AS carbs,
            COALESCE(SUM(fat), 0) AS fat FROM history WHERE substr(created_at, 1, 10) = ?""",
            (day.isoformat(),),
        ).fetchone()
    return dict(row)
