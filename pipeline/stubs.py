"""
Ideiglenes helyettesítők (stubok) a még el nem készült modulokhoz.  Felelős: A

Amíg B (analyzer) és C (nutrition, recommender) modulja NotImplementedError-t
dob, az orchestrator ezeket használja, hogy a lánc végig futtatható és
tesztelhető legyen. A pipeline-napló ilyenkor "ideiglenes helyettesítő"
megjegyzést kap, így látszik, hogy nem valódi eredményről van szó.

Amint egy modul elkészül, az orchestrator automatikusan a valódit hívja,
ezt a fájlt nem kell módosítani.
"""

from pipeline.food101_labels import FOOD101_LABELS, label_to_name_en, label_to_name_hu
from pipeline.schemas import DishAnalysis, Nutrition, Recognition, Recommendation, RecommendMode

STUB_NOTE = "ideiglenes helyettesítő – a modul még nincs kész"


def minimal_analysis(recognition: Recognition, source: str = "fallback") -> DishAnalysis:
    """Minimális elemzés pusztán az ételnévből (nincs LLM).

    Ezt használjuk stubként, és tartalék-válaszként is, ha az elemzés hibára fut.
    """
    return DishAnalysis(
        dish=recognition.key,
        category="ismeretlen",
        cuisine="ismeretlen",
        main_ingredients=[recognition.name_en.lower()],
        cooking_method="ismeretlen",
        allergens=[],
        flavor_profile=[],
        dietary_tags=[],
        description=f"{recognition.name_en} (részletes elemzés nem érhető el).",
        source=source,
    )


def stub_analyze(recognition: Recognition) -> DishAnalysis:
    return minimal_analysis(recognition, source="fallback")


def stub_compute_nutrition(dish_key: str, grams: float) -> Nutrition:
    """Kitalált, átlagos értékek (200 kcal / 100 g) – csak a lánc kipróbálásához."""
    factor = grams / 100
    return Nutrition(
        dish_key=dish_key,
        grams=grams,
        kcal=round(200 * factor),
        protein=round(8 * factor, 1),
        carbs=round(25 * factor, 1),
        fat=round(7 * factor, 1),
        is_estimate=True,
    )


def stub_recommend(
    analysis: DishAnalysis, mode: RecommendMode = "similar", k: int = 5
) -> list[Recommendation]:
    """Az ábécében a felismert étel utáni k Food-101 étel, kitalált pontszámmal."""
    keys = [key for key in FOOD101_LABELS if key != analysis.dish]
    start = next((i for i, key in enumerate(keys) if key > analysis.dish), 0)
    picked = (keys[start:] + keys[:start])[:k]
    return [
        Recommendation(
            key=key,
            name_en=label_to_name_en(key),
            name_hu=label_to_name_hu(key),
            similarity=round(0.9 - 0.1 * i, 2),
            kcal_100g=200.0,
            kcal_diff_100g=0.0,
            shared_items=[],
            reason=STUB_NOTE,
        )
        for i, key in enumerate(picked)
    ]
