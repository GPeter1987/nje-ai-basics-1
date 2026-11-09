"""
3. modell – ajánlás (all-MiniLM-L6-v2).  Felelős: C

Bemenet: DishAnalysis (a 2. modelltől).  Kimenet: top-k Recommendation.
"""

from pipeline.schemas import DishAnalysis, Recommendation, RecommendMode


def recommend(
    analysis: DishAnalysis, mode: RecommendMode = "similar", k: int = 5
) -> list[Recommendation]:
    """Hasonló / egészségesebb / allergénmentes ételeket ajánl az adatbázisból."""
    raise NotImplementedError("TODO (C): beágyazás, koszinusz-keresés, szűrés")
