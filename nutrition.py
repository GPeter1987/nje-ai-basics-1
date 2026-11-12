"""
Grammalapú tápérték-számítás.  Felelős: C

    érték(adag) = érték(100 g) * gramm / 100
"""

import math

import config
from data.db import get_food
from pipeline.schemas import Nutrition


def compute_nutrition(dish_key: str, grams: float) -> Nutrition:
    """Kiszámolja a tápértéket a megadott grammra az adatbázis 100 g-os értékeiből."""
    if not math.isfinite(grams) or not config.MIN_GRAMS <= grams <= config.MAX_GRAMS:
        raise ValueError(f"Az adag {config.MIN_GRAMS}–{config.MAX_GRAMS} g között lehet.")
    food = get_food(dish_key)
    if food is None:
        raise LookupError(f"Nincs tápérték az adatbázisban: {dish_key}")
    factor = grams / 100
    return Nutrition(
        dish_key=dish_key, grams=grams,
        kcal=round(food["kcal_100g"] * factor),
        protein=round(food["protein_100g"] * factor, 1),
        carbs=round(food["carbs_100g"] * factor, 1),
        fat=round(food["fat_100g"] * factor, 1),
    )
