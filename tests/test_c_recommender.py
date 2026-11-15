import numpy as np
import pytest

import config
from data import db
from pipeline import recommender
from pipeline.schemas import NutritionEstimate


def test_hungarian_analysis_builds_english_query(dish_analysis):
    text = recommender.build_query_text(dish_analysis)
    assert "chicken" in text and "indian" in text and "tomato" in text and "spicy" in text
    assert "csirke" not in text and "főétel" not in text
    assert text.isascii()
    changed = dish_analysis.model_copy(update={"main_ingredients": ["tofu"]})
    assert recommender.build_query_text(changed) != text


def test_similar_top5_excludes_original_and_explains(c_database, dish_analysis):
    results = recommender.recommend(dish_analysis)
    assert len(results) == 5
    assert len({r.key for r in results}) == 5
    assert all(r.key != dish_analysis.dish for r in results)
    assert all(-1 <= r.similarity <= 1 and r.reason for r in results)
    assert [r.similarity for r in results] == sorted([r.similarity for r in results], reverse=True)
    baseline = db.get_food(dish_analysis.dish)["kcal_100g"]
    for result in results:
        assert result.kcal_diff_100g == pytest.approx(round(result.kcal_100g - baseline, 1))


def test_cosine_ranks_unnormalized_vectors_correctly(c_database, dish_analysis):
    opposite = np.zeros(384, dtype=np.float32)
    opposite[0] = -10
    target = np.zeros(384, dtype=np.float32)
    target[0] = 25
    with db.connection() as conn:
        conn.execute("UPDATE foods SET embedding = ?", (opposite.tobytes(),))
        conn.execute(
            "UPDATE foods SET embedding = ? WHERE key = 'chickpea_curry'", (target.tobytes(),)
        )
    results = recommender.recommend(dish_analysis, k=2)
    assert results[0].key == "chickpea_curry"
    assert results[0].similarity == pytest.approx(1)
    assert results[1].similarity == pytest.approx(-1)
    assert "tomato" in results[0].shared_items


def test_healthier_strictly_less_kcal(c_database, dish_analysis):
    results = recommender.recommend(dish_analysis, mode="healthier")
    assert len(results) == 5
    assert all(r.kcal_100g < db.get_food(dish_analysis.dish)["kcal_100g"] for r in results)


def test_allergens_from_analysis_drive_filter(c_database, dish_analysis):
    analysis = dish_analysis.model_copy(update={"allergens": ["Tej", "glutén", "szója", "tojás"]})
    results = recommender.recommend(analysis, mode="allergen_free", k=200)
    assert results
    for item in results:
        assert not {"milk", "gluten", "soybeans", "eggs"}.intersection(
            db.get_food(item.key)["allergens"]
        )


@pytest.mark.parametrize("updates", [
    {"source": "fallback"}, {"allergens": ["ismeretlen allergén"]},
])
def test_uncertain_allergens_fail_closed(c_database, dish_analysis, updates):
    with pytest.raises(ValueError):
        recommender.recommend(dish_analysis.model_copy(update=updates), mode="allergen_free")


def test_short_or_empty_results_do_not_relax_filter(c_database, dish_analysis):
    with db.connection() as conn:
        conn.execute("UPDATE foods SET kcal_100g = 1 WHERE key = ?", (dish_analysis.dish,))
    assert recommender.recommend(dish_analysis, mode="healthier") == []


@pytest.mark.parametrize("column,value", [
    ("embedding", None), ("embedding", b"bad"),
    ("embedding", np.zeros(384, dtype=np.float32).tobytes()),
    ("embedding", np.full(384, np.nan, dtype=np.float32).tobytes()),
    ("embedding_model", "different-model"),
])
def test_invalid_vectors_are_not_silently_used(c_database, dish_analysis, column, value):
    with db.connection() as conn:
        conn.execute(f"UPDATE foods SET {column} = ? WHERE key = 'apple_pie'", (value,))
    with pytest.raises(ValueError):
        recommender.recommend(dish_analysis)


@pytest.mark.parametrize("mode,k", [
    ("random", 5), ("similar", 0), ("similar", 2.5), ("similar", True),
])
def test_invalid_settings_rejected(c_database, dish_analysis, mode, k):
    with pytest.raises(ValueError):
        recommender.recommend(dish_analysis, mode=mode, k=k)


def test_unknown_dish_requires_nutrition_reference(c_database, dish_analysis):
    analysis = dish_analysis.model_copy(update={"dish": "unknown_food"})
    with pytest.raises(LookupError):
        recommender.recommend(analysis)
    analysis.nutrition_estimate_per_100g = NutritionEstimate(kcal=200, protein=10, carbs=10, fat=10)
    assert len(recommender.recommend(analysis)) == 5


def test_bad_query_vector_rejected(monkeypatch):
    class BadEmbedder:
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), config.EMBEDDING_DIM))

    monkeypatch.setattr(recommender, "get_embedder", lambda: BadEmbedder())
    with pytest.raises(ValueError, match="nullvektor"):
        recommender.encode_texts(["test"])
