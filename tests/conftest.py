"""C tesztjeinek saját, ideiglenes adatbázisa és letöltés nélküli modellje."""

import numpy as np
import pytest

import config
from pipeline.schemas import DishAnalysis
from scripts import build_db


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), config.EMBEDDING_DIM), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors


@pytest.fixture
def c_database(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("pipeline.recommender.get_embedder", lambda: FakeEmbedder())
    build_db.build_database()
    return config.DB_PATH


@pytest.fixture
def dish_analysis():
    return DishAnalysis(
        dish="chicken_curry", category="főétel", cuisine="indiai",
        main_ingredients=["csirkehús", "paradicsom", "hagyma", "curry fűszerkeverék"],
        cooking_method="párolt", allergens=["tej"], flavor_profile=["fűszeres", "krémes"],
        dietary_tags=["gluténmentes"], description="Fűszeres csirkeétel.",
    )
