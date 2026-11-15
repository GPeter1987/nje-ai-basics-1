"""Valódi C-modulok integrációja az A orchestratorával, hálózat nélkül."""

import pytest
from PIL import Image

from pipeline import analyzer, orchestrator, recognizer, recommender
from pipeline.schemas import PipelineResult
from scripts.build_db import build_database


def test_c_modules_in_full_chain(c_database, dish_analysis, monkeypatch):
    monkeypatch.setattr(recognizer, "_predict", lambda image: [
        ("chicken_curry", 0.8), ("ramen", 0.1), ("pho", 0.05),
    ])
    monkeypatch.setattr(analyzer, "analyze", lambda recognition: dish_analysis)
    result = orchestrator.run_pipeline(Image.new("RGB", (32, 32)), grams=300, mode="healthier")
    assert result.ok
    assert result.nutrition.grams == 300
    assert not result.nutrition.is_estimate
    assert len(result.recommendations) == 5
    assert all(step.note is None for step in result.steps)
    PipelineResult.model_validate_json(result.model_dump_json())


@pytest.mark.slow
def test_real_minilm(tmp_path, monkeypatch, dish_analysis):
    import config

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "real.db")
    build_database()
    results = recommender.recommend(dish_analysis)
    assert len(results) == 5
    assert all(item.similarity > 0 for item in results)
