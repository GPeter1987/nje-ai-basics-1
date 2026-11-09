"""
2. modell – strukturált elemzés (Qwen2.5-1.5B-Instruct).  Felelős: B

Bemenet: Recognition (az 1. modelltől).  Kimenet: validált DishAnalysis.
Hibás JSON esetén 1 újrapróbálkozás, utána tartalék-válasz az adatbázisból.
"""

from pipeline.schemas import DishAnalysis, Recognition


def analyze(recognition: Recognition) -> DishAnalysis:
    """Strukturált elemzést készít a felismert ételről."""
    raise NotImplementedError("TODO (B): prompt, LLM-hívás, JSON-validálás")
