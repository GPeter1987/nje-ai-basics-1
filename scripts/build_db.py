"""
foods_seed.csv -> SQLite + előre számolt beágyazások.  Felelős: C

Futtatás:  python scripts/build_db.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data import db  # noqa: E402
from data.terms import EU_ALLERGENS  # noqa: E402
from pipeline.food101_labels import FOOD101_LABELS  # noqa: E402
from pipeline.recommender import encode_texts  # noqa: E402

FIELDS = (
    "key", "name_en", "name_hu", "category", "cuisine", "ingredients", "allergens", "tags",
    "kcal_100g", "protein_100g", "carbs_100g", "fat_100g", "description_en", "in_food101", "source",
)


def load_seed(path: Path) -> list[dict]:
    foods = []
    keys = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Hiányzó CSV-oszlopok.")
        for line, raw in enumerate(reader, 2):
            food = {key: raw[key].strip() for key in FIELDS}
            try:
                if any(not food[key] for key in FIELDS):
                    raise ValueError("üres mező")
                if food["key"] in keys:
                    raise ValueError("ismétlődő ételkulcs")
                for column in ("ingredients", "allergens", "tags"):
                    values = json.loads(food[column])
                    if not isinstance(values, list) or any(
                        not isinstance(v, str) or not v.strip() for v in values
                    ):
                        raise ValueError(f"hibás JSON-lista: {column}")
                    food[column] = list(dict.fromkeys(v.strip().lower() for v in values))
                if not food["ingredients"]:
                    raise ValueError("nincsenek összetevők")
                if set(food["allergens"]) - EU_ALLERGENS:
                    raise ValueError("ismeretlen allergénkulcs")
                for column in ("kcal_100g", "protein_100g", "carbs_100g", "fat_100g"):
                    food[column] = float(food[column])
                    if not math.isfinite(food[column]) or food[column] < 0:
                        raise ValueError(f"nem véges/negatív tápérték: {column}")
                if any(food[c] > 100 for c in ("protein_100g", "carbs_100g", "fat_100g")):
                    raise ValueError("100 g feletti makrotápanyag")
                food["in_food101"] = int(food["in_food101"])
                if food["in_food101"] not in (0, 1):
                    raise ValueError("in_food101 nem 0/1")
                if bool(food["in_food101"]) != (food["key"] in FOOD101_LABELS):
                    raise ValueError("hibás Food-101 jelölés")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CSV {line}. sor ({food['key']}): {exc}") from exc
            macro_kcal = 4 * food["protein_100g"] + 4 * food["carbs_100g"] + 9 * food["fat_100g"]
            if abs(macro_kcal - food["kcal_100g"]) > food["kcal_100g"] * config.KCAL_TOLERANCE:
                warnings.warn(
                    f"{food['key']}: kcal-eltérés >15% ({food['kcal_100g']} / {macro_kcal:.1f}).",
                    stacklevel=2,
                )
            keys.add(food["key"])
            foods.append(food)
    if not set(FOOD101_LABELS).issubset(keys):
        raise ValueError("Hiányzó Food-101 ételek.")
    if sum(not food["in_food101"] for food in foods) < 100:
        raise ValueError("Legalább 100 további ajánlható étel szükséges.")
    return foods


def build_database(seed_path: Path | None = None, *, embeddings: bool = True) -> int:
    """Hibás CSV/modell esetén a korábbi foods tábla és az előzmények megmaradnak."""
    foods = load_seed(seed_path or config.SEED_CSV_PATH)
    vectors = encode_texts([f["description_en"] for f in foods]) if embeddings else None
    db.init_db()
    columns = (*FIELDS, "embedding", "embedding_model")
    with db.connection() as conn:
        conn.execute("DELETE FROM foods")
        for i, food in enumerate(foods):
            values = [json.dumps(food[c], ensure_ascii=False) if isinstance(food[c], list)
                      else food[c] for c in FIELDS]
            values.extend((vectors[i].astype("float32").tobytes(), config.EMBEDDING_MODEL)
                          if vectors is not None else (None, None))
            placeholders = ", ".join("?" for _ in columns)
            conn.execute(
                f"INSERT INTO foods ({', '.join(columns)}) VALUES ({placeholders})", values,
            )
    return len(foods)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, default=config.SEED_CSV_PATH)
    parser.add_argument("--without-embeddings", action="store_true",
                        help="Csak adatok; az ajánlóhoz később teljes újraépítés szükséges.")
    args = parser.parse_args()
    try:
        count = build_database(args.seed, embeddings=not args.without_embeddings)
    except Exception as exc:
        print(f"Adatbázis-építési hiba: {exc}", file=sys.stderr)
        return 1
    print(f"{count} étel betöltve: {config.DB_PATH}")
    if args.without_embeddings:
        print("FIGYELEM: beágyazások nélkül az ajánló még nem használható.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
