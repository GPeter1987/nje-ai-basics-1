"""
Grammalapú tápérték-számítás.  Felelős: C

    érték(adag) = érték(100 g) * gramm / 100
"""

from pipeline.schemas import Nutrition


def compute_nutrition(dish_key: str, grams: float) -> Nutrition:
    """Kiszámolja a tápértéket a megadott grammra az adatbázis 100 g-os értékeiből."""
    raise NotImplementedError("TODO (C): DB-lekérdezés és arányosítás")
