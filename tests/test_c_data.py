import csv
import json
from datetime import date, timedelta

import numpy as np
import pytest

import config
from data import db
from nutrition import compute_nutrition
from pipeline import recognizer
from pipeline.food101_labels import FOOD101_LABELS
from pipeline.schemas import Nutrition
from scripts import build_db, generate_seed


def test_seed_coverage_sources_and_embeddings(c_database):
    foods = db.list_foods()
    assert len(foods) == 201
    assert {f["key"] for f in foods if f["in_food101"]} == set(FOOD101_LABELS)
    assert sum(not f["in_food101"] for f in foods) == 100
    for food in foods:
        assert food["name_hu"] and food["description_en"]
        assert all(ingredient.isascii() for ingredient in food["ingredients"])
        assert "USDA SR28 recipe estimate; NDB " in food["source"]
        assert len(food["embedding"]) == config.EMBEDDING_DIM * 4
        assert food["embedding_model"] == config.EMBEDDING_MODEL
        assert len(np.frombuffer(food["embedding"], dtype=np.float32)) == 384


def test_seed_is_reproducible(tmp_path, monkeypatch):
    expected = config.SEED_CSV_PATH.read_bytes()
    monkeypatch.setattr(config, "SEED_CSV_PATH", tmp_path / "foods.csv")
    assert generate_seed.generate() == 201
    assert config.SEED_CSV_PATH.read_bytes() == expected


def test_usda_recipe_calculation_matches_seed():
    ingredients = json.loads((config.DATA_DIR / "usda_ingredients.json").read_text("utf-8"))
    foods = build_db.load_seed(config.SEED_CSV_PATH)
    for food in foods:
        parts = food["source"].split("; ")[1:]
        for nutrient in ("kcal", "protein", "carbs", "fat"):
            actual = 0
            for part in parts:
                _, ndb, rest = part.split(" ", 2)
                name, weight = rest.rsplit(" ", 1)
                assert ingredients[name]["ndb"] == ndb
                actual += ingredients[name][nutrient] * float(weight[:-1]) / 100
            assert food[f"{nutrient}_100g"] == pytest.approx(actual, abs=0.00051)
        if "vegan" in food["tags"]:
            assert not {"milk", "eggs"}.intersection(food["allergens"])


def test_nutrition_scaling_and_rounding(c_database):
    food = db.get_food("chicken_curry")
    result = compute_nutrition("chicken_curry", 250)
    assert result.kcal == round(food["kcal_100g"] * 2.5)
    assert result.protein == round(food["protein_100g"] * 2.5, 1)
    assert result.carbs == round(food["carbs_100g"] * 2.5, 1)
    assert result.fat == round(food["fat_100g"] * 2.5, 1)
    assert not result.is_estimate


@pytest.mark.parametrize("grams", [0, -1, 9, 2001, float("nan"), float("inf")])
def test_invalid_grams_rejected(c_database, grams):
    with pytest.raises(ValueError, match="adag"):
        compute_nutrition("chicken_curry", grams)


@pytest.mark.parametrize("grams", [10, 2000])
def test_boundary_grams_accepted(c_database, grams):
    assert compute_nutrition("chicken_curry", grams).grams == grams


def test_missing_food_and_sql_injection(c_database):
    assert db.get_food("' OR 1=1 --") is None
    with pytest.raises(LookupError):
        compute_nutrition("no_food", 250)


def test_missing_db_does_not_create_empty_file(tmp_path, monkeypatch):
    path = tmp_path / "missing.db"
    monkeypatch.setattr(config, "DB_PATH", path)
    with pytest.raises(FileNotFoundError, match="build_db"):
        db.get_food("pizza")
    assert not path.exists()


def test_cache_is_validated_and_model_specific(c_database, dish_analysis):
    db.save_analysis(dish_analysis, "qwen:test")
    cached = db.get_analysis(dish_analysis.dish, "qwen:test")
    assert cached.source == "cache"
    assert cached.main_ingredients == dish_analysis.main_ingredients
    assert db.get_analysis(dish_analysis.dish, "other-model") is None
    db.save_analysis(dish_analysis.model_copy(update={"description": "Új leírás"}), "qwen:test")
    assert db.get_analysis(dish_analysis.dish, "qwen:test").description == "Új leírás"


def test_history_survives_rebuild_and_sums_only_selected_day(c_database):
    recognition = recognizer.Recognition(
        key="chicken_curry", name_en="Chicken curry", score=0.8, is_confident=True,
        image_hash="test-hash",
        top3=[recognizer.Candidate(key="chicken_curry", name_en="Chicken curry", score=0.8)],
    )
    nutrition = compute_nutrition("chicken_curry", 250)
    first_id = db.save_history(recognition, nutrition)
    second_id = db.save_history(recognition, nutrition)
    yesterday = date.today() - timedelta(days=1)
    with db.connection() as conn:
        conn.execute("UPDATE history SET created_at = ? WHERE id = ?",
                     (yesterday.isoformat() + "T12:00:00+01:00", first_id))
    build_db.build_database()
    history = db.list_history()
    assert len(history) == 2
    assert history[0]["id"] == second_id
    assert history[0]["image_hash"] == "test-hash"
    assert db.daily_summary()["kcal"] == nutrition.kcal
    assert db.daily_summary(yesterday)["protein"] == nutrition.protein
    assert db.daily_summary(date(2000, 1, 1))["meals"] == 0


def test_history_supports_ai_estimate_without_food_row(c_database):
    recognition = recognizer.Recognition(
        key="unknown_food", name_en="Unknown food", score=0.2, is_confident=False,
        top3=[recognizer.Candidate(key="pizza", name_en="Pizza", score=0.2)],
    )
    nutrition = Nutrition(dish_key="unknown_food", grams=100, kcal=100,
                          protein=10, carbs=10, fat=2, is_estimate=True)
    db.save_history(recognition, nutrition)
    assert db.list_history()[0]["is_estimate"] == 1
    assert db.list_history()[0]["name_hu"] == "unknown_food"


def rewrite_csv(tmp_path, change):
    with config.SEED_CSV_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    change(rows)
    path = tmp_path / "bad.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("column,value", [
    ("kcal_100g", "nan"), ("fat_100g", "-1"), ("protein_100g", "101"),
    ("allergens", '["unknown"]'), ("ingredients", '{}'), ("ingredients", '[]'),
    ("in_food101", "7"), ("source", ""),
])
def test_bad_csv_does_not_replace_existing_database(c_database, tmp_path, column, value):
    before = db.get_food("pizza")
    path = rewrite_csv(tmp_path, lambda rows: rows[0].__setitem__(column, value))
    with pytest.raises(ValueError, match="CSV"):
        build_db.build_database(path)
    assert db.get_food("pizza") == before


def test_kcal_check_warns_but_keeps_usda_value(tmp_path):
    path = rewrite_csv(tmp_path, lambda rows: rows[0].__setitem__("kcal_100g", "1000"))
    with pytest.warns(UserWarning, match="kcal"):
        foods = build_db.load_seed(path)
    assert foods[0]["kcal_100g"] == 1000


def test_without_embeddings_is_explicit(c_database):
    build_db.build_database(embeddings=False)
    assert db.get_food("pizza")["embedding"] is None


def test_model_error_preserves_previous_foods(c_database, monkeypatch):
    def broken(texts):
        raise RuntimeError("model unavailable")

    before = db.get_food("pizza")
    monkeypatch.setattr(build_db, "encode_texts", broken)
    with pytest.raises(RuntimeError):
        build_db.build_database()
    assert db.get_food("pizza") == before
