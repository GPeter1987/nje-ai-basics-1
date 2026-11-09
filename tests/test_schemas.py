"""A közös Pydantic modellek tesztjei."""

import json

import pytest
from pydantic import ValidationError

from pipeline.schemas import (
    Candidate,
    DishAnalysis,
    Nutrition,
    PipelineResult,
    Recognition,
    StepLog,
)

# A specifikáció 4.2 fejezetének példa-JSON-ja
SPEC_EXAMPLE_JSON = """
{
  "dish": "chicken_curry",
  "category": "főétel",
  "cuisine": "indiai",
  "main_ingredients": ["csirkehús", "hagyma", "paradicsom", "curry fűszerkeverék", "kókusztej"],
  "cooking_method": "párolt",
  "allergens": ["tej"],
  "flavor_profile": ["fűszeres", "krémes"],
  "dietary_tags": ["gluténmentes"],
  "description": "Fűszeres, krémes, szószos csirkeétel.",
  "nutrition_estimate_per_100g": null
}
"""


def make_recognition(**overrides) -> Recognition:
    data = {
        "key": "chicken_curry",
        "name_en": "Chicken curry",
        "score": 0.87,
        "top3": [
            Candidate(key="chicken_curry", name_en="Chicken curry", score=0.87),
            Candidate(key="butter_chicken", name_en="Butter chicken", score=0.08),
            Candidate(key="ramen", name_en="Ramen", score=0.02),
        ],
        "is_confident": True,
    }
    data.update(overrides)
    return Recognition(**data)


def test_recognition_valid():
    rec = make_recognition()
    assert rec.top3[0].key == rec.key
    assert rec.manually_selected is False


def test_recognition_rejects_unsorted_top3():
    with pytest.raises(ValidationError):
        make_recognition(
            top3=[
                Candidate(key="ramen", name_en="Ramen", score=0.02),
                Candidate(key="chicken_curry", name_en="Chicken curry", score=0.87),
            ]
        )


def test_recognition_rejects_score_out_of_range():
    with pytest.raises(ValidationError):
        make_recognition(score=1.5)


def test_dish_analysis_accepts_spec_example():
    analysis = DishAnalysis.model_validate(json.loads(SPEC_EXAMPLE_JSON))
    assert analysis.dish == "chicken_curry"
    assert analysis.allergens == ["tej"]
    assert analysis.source == "llm"
    assert analysis.nutrition_estimate_per_100g is None


def test_dish_analysis_normalizes_lists_and_ignores_extra():
    data = json.loads(SPEC_EXAMPLE_JSON)
    data["allergens"] = [" Tej", "tej", "", "GLUTÉN"]
    data["extra_field_from_llm"] = "valami"
    analysis = DishAnalysis.model_validate(data)
    assert analysis.allergens == ["tej", "glutén"]
    assert not hasattr(analysis, "extra_field_from_llm")


@pytest.mark.parametrize("missing", ["dish", "category", "main_ingredients", "description"])
def test_dish_analysis_requires_fields(missing):
    data = json.loads(SPEC_EXAMPLE_JSON)
    del data[missing]
    with pytest.raises(ValidationError):
        DishAnalysis.model_validate(data)


def test_dish_analysis_rejects_blank_string():
    data = json.loads(SPEC_EXAMPLE_JSON)
    data["category"] = "   "
    with pytest.raises(ValidationError):
        DishAnalysis.model_validate(data)


def test_nutrition_rejects_negative_values():
    with pytest.raises(ValidationError):
        Nutrition(dish_key="x", grams=250, kcal=-1, protein=1, carbs=1, fat=1)


def test_pipeline_result_ok_and_total_time():
    result = PipelineResult(
        steps=[
            StepLog(name="1", duration_ms=100),
            StepLog(name="2", duration_ms=50, ok=False, error="hiba"),
        ]
    )
    assert result.ok is False
    assert result.total_ms == 150
    # JSON-ná alakítható (a pipeline-nézet és a napló miatt)
    assert json.loads(result.model_dump_json())["steps"][1]["error"] == "hiba"
