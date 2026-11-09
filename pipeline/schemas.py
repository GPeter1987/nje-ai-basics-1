"""
Közös adatmodellek (Pydantic) – ez a csapat "szerződése".

Minden modul ezeket az osztályokat adja vissza és fogadja, így az egyes
lépések egymástól függetlenül fejleszthetők és tesztelhetők:

    recognize(image)                    -> Recognition       (A, 1. modell)
    analyze(recognition)                -> DishAnalysis      (B, 2. modell)
    recommend(analysis, mode, k=5)      -> list[Recommendation]  (C, 3. modell)
    compute_nutrition(dish_key, grams)  -> Nutrition         (C)

FONTOS: a mezőket csak közös döntéssel változtatjuk (lásd specifikáció 7. fejezet).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Az ajánlási módok belső kulcsai (a megjelenített neveket a config.RECOMMEND_MODES adja)
RecommendMode = Literal["similar", "healthier", "allergen_free"]


def _clean_str_list(values: list[str]) -> list[str]:
    """Szóközök levágása, kisbetűsítés, üres elemek és ismétlődések eltávolítása.

    Az LLM néha "Tej", " tej" vagy duplikált elemeket ad – így ezek egységesek
    lesznek, és az ajánló szűrője (allergének, közös összetevők) megbízhatóan működik.
    """
    cleaned: list[str] = []
    for value in values:
        item = str(value).strip().lower()
        if item and item not in cleaned:
            cleaned.append(item)
    return cleaned


# --- 1. modell: ételfelismerés ----------------------------------------------


class Candidate(BaseModel):
    """Egy felismerési jelölt (a top-3 lista egy eleme)."""

    key: str = Field(description="Food-101 címke, egyben adatbázis-kulcs, pl. 'chicken_curry'")
    name_en: str = Field(description="Emberi név angolul, pl. 'Chicken curry'")
    name_hu: str | None = Field(default=None, description="Magyar név, ha ismert (F13)")
    score: float = Field(ge=0.0, le=1.0, description="Valószínűség 0–1 között")


class Recognition(BaseModel):
    """Az 1. modell (ViT) kimenete, a 2. modell bemenete."""

    key: str = Field(description="A kiválasztott étel kulcsa (top-1 vagy kézi javítás)")
    name_en: str
    name_hu: str | None = None
    score: float = Field(ge=0.0, le=1.0, description="A kiválasztott étel bizonyossága")
    top3: list[Candidate] = Field(min_length=1, max_length=3)
    is_confident: bool = Field(description="score >= küszöb (F3)")
    manually_selected: bool = Field(
        default=False, description="A felhasználó javította-e a felismerést (F3)"
    )
    image_hash: str | None = Field(default=None, description="A kép SHA-256 hash-e (F10)")

    @model_validator(mode="after")
    def _top3_sorted(self) -> Recognition:
        scores = [c.score for c in self.top3]
        if scores != sorted(scores, reverse=True):
            raise ValueError("a top3 listának csökkenő bizonyosság szerint kell rendezettnek lennie")
        return self


# --- 2. modell: strukturált elemzés ------------------------------------------


class NutritionEstimate(BaseModel):
    """Az LLM tartalék tápérték-becslése 100 g-ra (csak ha az étel nincs a DB-ben)."""

    kcal: float = Field(ge=0)
    protein: float = Field(ge=0)
    carbs: float = Field(ge=0)
    fat: float = Field(ge=0)


class DishAnalysis(BaseModel):
    """A 2. modell (Qwen) validált JSON-kimenete, a 3. modell bemenete.

    A mezők megegyeznek a specifikáció 4.2 fejezetének JSON-sémájával,
    kiegészítve a `source` metaadattal (nem az LLM tölti ki).
    """

    model_config = ConfigDict(extra="ignore")  # az LLM által hozzáírt extra mezőket eldobjuk

    dish: str = Field(description="Az étel kulcsa, pl. 'chicken_curry'")
    category: str = Field(description="pl. főétel, leves, desszert")
    cuisine: str = Field(description="pl. indiai, olasz")
    main_ingredients: list[str] = Field(min_length=1)
    cooking_method: str
    allergens: list[str] = Field(default_factory=list, description="EU 14 fő allergénje közül")
    flavor_profile: list[str] = Field(default_factory=list)
    dietary_tags: list[str] = Field(default_factory=list)
    description: str
    nutrition_estimate_per_100g: NutritionEstimate | None = None

    # Honnan származik az elemzés (a pipeline-nézetben és a UI-ban jelezzük)
    source: Literal["llm", "cache", "fallback"] = "llm"

    @field_validator("main_ingredients", "allergens", "flavor_profile", "dietary_tags")
    @classmethod
    def _normalize_lists(cls, values: list[str]) -> list[str]:
        return _clean_str_list(values)

    @field_validator("dish", "category", "cuisine", "cooking_method", "description")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("a mező nem lehet üres")
        return value


# --- Tápérték ---------------------------------------------------------------


class Nutrition(BaseModel):
    """Tápérték a megadott grammra (az adatbázis 100 g-os értékeiből számolva)."""

    dish_key: str
    grams: float = Field(gt=0)
    kcal: int = Field(ge=0, description="Egészre kerekítve")
    protein: float = Field(ge=0, description="1 tizedesre kerekítve")
    carbs: float = Field(ge=0)
    fat: float = Field(ge=0)
    is_estimate: bool = Field(
        default=False, description="True, ha LLM-becslés ('AI-becslés' címke a UI-ban)"
    )


# --- 3. modell: ajánlás ------------------------------------------------------


class Recommendation(BaseModel):
    """Egy ajánlott alternatíva (F7)."""

    key: str
    name_en: str
    name_hu: str | None = None
    similarity: float = Field(ge=-1.0, le=1.0, description="Koszinusz-hasonlóság")
    kcal_100g: float = Field(ge=0)
    kcal_diff_100g: float = Field(description="Ajánlott − felismert étel kcal/100 g (negatív = kevesebb)")
    shared_items: list[str] = Field(
        default_factory=list, description="Közös összetevők/címkék az indokláshoz"
    )
    reason: str = Field(description="Rövid, ember által olvasható indoklás")


# --- Lánc (orchestrator) ------------------------------------------------------


class StepLog(BaseModel):
    """Egy lánclépés naplója a "Pipeline" fülhöz (F9)."""

    name: str = Field(description="pl. '1. Ételfelismerés (ViT)'")
    input: Any = None
    output: Any = None
    duration_ms: float = Field(ge=0)
    ok: bool = True
    error: str | None = None


class PipelineResult(BaseModel):
    """A teljes lánc eredménye – ezt kapja meg a Streamlit-felület."""

    recognition: Recognition | None = None
    analysis: DishAnalysis | None = None
    nutrition: Nutrition | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    steps: list[StepLog] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Minden lépés hiba nélkül lefutott-e."""
        return all(step.ok for step in self.steps)

    @property
    def total_ms(self) -> float:
        return sum(step.duration_ms for step in self.steps)
