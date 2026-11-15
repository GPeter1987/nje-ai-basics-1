"""F13: az angol elemzés magyar megjelenítése (szótárral, nem AI-fordítással)."""

import pytest

from data.terms import EU_ALLERGENS
from pipeline import hungarian
from pipeline.schemas import DishAnalysis


def make_analysis(**overrides):
    data = {
        "dish": "pizza",
        "category": "main course",
        "cuisine": "Italian",
        "main_ingredients": ["wheat flour", "tomatoes", "mozzarella", "olive oil", "basil"],
        "cooking_method": "baked",
        "allergens": ["gluten", "milk"],
        "flavor_profile": ["savory", "crispy"],
        "dietary_tags": ["vegetarian"],
        "description": "A flatbread with tomato and cheese.",
    }
    data.update(overrides)
    return DishAnalysis(**data)


def test_analysis_fields_shown_in_hungarian():
    hu = hungarian.analysis_hu(make_analysis())
    assert hu["category"] == "főétel"
    assert hu["cuisine"] == "olasz"
    assert hu["main_ingredients"] == [
        "búzaliszt",
        "paradicsom",
        "mozzarella",
        "olívaolaj",
        "bazsalikom",
    ]
    assert hu["cooking_method"] == "sütőben sült"
    assert hu["allergens"] == ["glutén", "tej"]
    assert hu["flavor_profile"] == ["sós-fűszeres", "ropogós"]
    assert hu["dietary_tags"] == ["vegetáriánus"]
    assert hu["summary"] == (
        "Olasz főétel. Jellemző összetevők: búzaliszt, paradicsom, mozzarella, olívaolaj, "
        "bazsalikom. Elkészítés: sütőben sült."
    )


def test_every_closed_vocabulary_value_has_hungarian_name():
    from pipeline import analyzer

    assert set(hungarian.ALLERGENS_HU) == EU_ALLERGENS
    assert set(analyzer.CATEGORIES) <= set(hungarian.CATEGORIES_HU)
    assert set(analyzer.DIETARY_TAGS) <= set(hungarian.DIETARY_TAGS_HU)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("cooked, steamed, or grilled", "főzött, párolt vagy grillezett"),
        ("boiled and stir-fried", "főtt és pirított"),
        ("chicken or shrimp", "csirke vagy garnélarák"),
        ("Tomatoes", "paradicsom"),  # többes szám, nagybetű
    ],
)
def test_phrases_translated(text, expected):
    assert hungarian.phrase_hu(text) == expected


def test_unknown_terms_stay_english_and_summary_has_no_mixed_sentence():
    analysis = make_analysis(
        cuisine="Ethiopian",
        main_ingredients=["injera", "lentils", "berbere spice"],
        cooking_method="boiled, simmered, and cooked in a curry sauce",
    )
    hu = hungarian.analysis_hu(analysis)
    assert hu["cuisine"] == "Ethiopian"  # nem találunk ki fordítást
    assert hu["cooking_method"] == "boiled, simmered, and cooked in a curry sauce"
    assert hu["main_ingredients"] == ["injera", "lencse", "berbere spice"]
    # az összegzésbe csak magyar rész kerül
    assert hu["summary"] == "Főétel. Jellemző összetevők: lencse."


def test_duplicates_after_translation_removed():
    hu = hungarian.analysis_hu(make_analysis(main_ingredients=["onion", "onions", "red onion"]))
    assert hu["main_ingredients"] == ["hagyma", "lilahagyma"]


def test_hungarian_input_passes_through():
    """A C-tesztek magyar értékű elemzéseit változatlanul hagyja."""
    hu = hungarian.analysis_hu(make_analysis(category="főétel", main_ingredients=["csirkehús"]))
    assert hu["category"] == "főétel"
    assert hu["main_ingredients"] == ["csirkehús"]
