"""
3. modell – ajánlás (all-MiniLM-L6-v2).  Felelős: C

Bemenet: DishAnalysis (a 2. modelltől).  Kimenet: top-k Recommendation.
"""

from functools import lru_cache

import numpy as np

import config
from data.db import get_food, list_foods
from data.terms import EU_ALLERGENS, allergen_key, english
from pipeline.schemas import DishAnalysis, Recommendation, RecommendMode


@lru_cache(maxsize=1)
def get_embedder():
    """Lusta betöltés: a DB és a tesztek torch nélkül is használhatók."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")


def build_query_text(analysis: DishAnalysis) -> str:
    """A Qwen strukturált mezői adják a lekérdezést, nem a kép vagy egy kész ajánlás."""
    terms = [analysis.dish.replace("_", " "), analysis.category, analysis.cuisine,
             *analysis.main_ingredients, analysis.cooking_method,
             *analysis.flavor_profile, *analysis.dietary_tags]
    return "; ".join(dict.fromkeys(english(term) for term in terms if english(term)))


def encode_texts(texts: list[str]) -> np.ndarray:
    vectors = np.asarray(get_embedder().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
    ), dtype=np.float32)
    if vectors.shape != (len(texts), config.EMBEDDING_DIM) or not np.isfinite(vectors).all():
        raise ValueError("A MiniLM érvénytelen méretű vagy nem véges vektorokat adott.")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("A MiniLM nullvektort adott.")
    return vectors


def recommend(
    analysis: DishAnalysis, mode: RecommendMode = "similar", k: int = 5
) -> list[Recommendation]:
    """Hasonló / egészségesebb / allergénmentes ételeket ajánl az adatbázisból."""
    if mode not in config.RECOMMEND_MODES:
        raise ValueError(f"Ismeretlen ajánlási mód: {mode}")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("A k pozitív egész szám legyen.")
    original = get_food(analysis.dish)
    estimate = analysis.nutrition_estimate_per_100g
    baseline = original["kcal_100g"] if original else (estimate.kcal if estimate else None)
    if baseline is None:
        raise LookupError("A kcal-különbséghez nincs tápértékadat vagy AI-becslés.")

    excluded = {allergen_key(a) for a in analysis.allergens}
    if mode == "allergen_free":
        if analysis.source == "fallback":
            raise ValueError(
                "Tartalék elemzésből nem állapíthatók meg biztonságosan az allergének."
            )
        unknown = excluded - EU_ALLERGENS
        if unknown:
            raise ValueError(f"Ismeretlen allergén; a szűrés nem biztonságos: {', '.join(unknown)}")

    candidates = []
    for food in list_foods():
        if food["key"] == analysis.dish:
            continue
        if mode == "healthier" and food["kcal_100g"] >= baseline:
            continue
        if mode == "allergen_free" and excluded.intersection(
            allergen_key(a) for a in food["allergens"]
        ):
            continue
        candidates.append(food)
    if not candidates:
        return []

    vectors = []
    for food in candidates:
        if food["embedding_model"] != config.EMBEDDING_MODEL or food["embedding"] is None:
            raise ValueError("Hiányzó/elavult beágyazások. Futtasd: python scripts/build_db.py")
        vector = np.frombuffer(food["embedding"], dtype=np.float32)
        if (vector.shape != (config.EMBEDDING_DIM,) or not np.isfinite(vector).all()
                or np.linalg.norm(vector) == 0):
            raise ValueError(f"Sérült beágyazás: {food['key']}. Építsd újra az adatbázist.")
        vectors.append(vector)

    matrix = np.stack(vectors)
    query = encode_texts([build_query_text(analysis)])[0]
    scores = np.clip((matrix @ query) / (
        np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    ), -1, 1)
    ranking = sorted(
        range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]["key"])
    )
    analysis_items = {english(t) for t in (
        analysis.main_ingredients + analysis.flavor_profile + analysis.dietary_tags
    )} - {""}
    results = []
    for i in ranking[:k]:
        food = candidates[i]
        shared = sorted(analysis_items.intersection(food["ingredients"] + food["tags"]))
        diff = round(food["kcal_100g"] - baseline, 1)
        reason = (
            "Közös összetevők/címkék: " + ", ".join(shared)
            if shared else "Szemantikai hasonlóság."
        )
        reason += f" Energiaeltérés: {diff:+g} kcal/100 g."
        if mode == "allergen_free":
            reason += " Az elemzésben jelzett allergének nélkül az adatbázis szerint."
        results.append(Recommendation(
            key=food["key"], name_en=food["name_en"], name_hu=food["name_hu"],
            similarity=float(scores[i]), kcal_100g=food["kcal_100g"],
            kcal_diff_100g=diff, shared_items=shared, reason=reason,
        ))
    return results
