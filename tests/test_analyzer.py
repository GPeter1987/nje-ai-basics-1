import json

import pytest

import config
from pipeline import analyzer, llm_backends
from pipeline.schemas import DishAnalysis, Recognition


def make_recognition(**overrides):
    data = {
        "key": "chicken_curry",
        "name_en": "Chicken curry",
        "score": 0.35,
        "is_confident": False,
        "top3": [
            {"key": "chicken_curry", "name_en": "Chicken curry", "score": 0.35},
            {"key": "pho", "name_en": "Pho", "score": 0.30},
            {"key": "ramen", "name_en": "Ramen", "score": 0.20},
        ],
    }
    data.update(overrides)
    return Recognition(**data)


def valid_response(**overrides):
    result = {
        "dish": "chicken_curry",
        "category": "main course",
        "cuisine": "Indian",
        "main_ingredients": ["chicken", "tomato", "onion"],
        "cooking_method": "stewed",
        "allergens": ["milk"],
        "flavor_profile": ["spicy", "creamy"],
        "dietary_tags": [],
        "description": "A spiced chicken dish with a creamy sauce.",
        "nutrition_estimate_per_100g": None,
    }
    result.update(overrides)
    return json.dumps(result)


def test_analyze_validates_json_and_passes_recognition(monkeypatch):
    calls = []

    def generate(system_prompt, user_prompt, backend=None):
        calls.append((system_prompt, user_prompt, backend))
        return "```json\n" + valid_response() + "\n```"

    monkeypatch.setattr(llm_backends, "generate", generate)
    monkeypatch.setattr(analyzer.db, "get_analysis", lambda *args: None, raising=False)
    monkeypatch.setattr(analyzer.db, "save_analysis", lambda *args: None, raising=False)

    result = analyzer.analyze(make_recognition())

    assert isinstance(result, DishAnalysis)
    assert result.source == "llm"
    assert result.dish == "chicken_curry"
    assert len(calls) == 1
    assert '"recognition_needs_review": true' in calls[0][1]
    assert '"key": "pho"' in calls[0][1]


def test_analyze_normalizes_comma_separated_list_fields(monkeypatch):
    response = valid_response(
        main_ingredients="chicken, tomato; onion",
        allergens="milk, eggs",
        flavor_profile="spicy, aromatic, rich",
        dietary_tags="none",
    )
    monkeypatch.setattr(llm_backends, "generate", lambda *args, **kwargs: response)
    monkeypatch.setattr(analyzer.db, "get_analysis", lambda *args: None, raising=False)
    monkeypatch.setattr(analyzer.db, "save_analysis", lambda *args: None, raising=False)

    result = analyzer.analyze(make_recognition())

    assert result.main_ingredients == ["chicken", "tomato", "onion"]
    assert result.allergens == ["milk", "eggs"]
    assert result.flavor_profile == ["spicy", "aromatic", "rich"]
    assert result.dietary_tags == []


def test_analyze_retries_invalid_response(monkeypatch):
    prompts = []
    responses = iter(["not json", valid_response()])
    monkeypatch.setattr(
        llm_backends, "generate", lambda system, user, backend=None: prompts.append(user) or next(responses)
    )
    monkeypatch.setattr(analyzer.db, "get_analysis", lambda *args: None, raising=False)
    monkeypatch.setattr(analyzer.db, "save_analysis", lambda *args: None, raising=False)
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 1)

    result = analyzer.analyze(make_recognition())

    assert result.source == "llm"
    assert len(prompts) == 2
    assert "previous response was invalid" in prompts[1]


def test_analyze_uses_database_fallback_after_retries(monkeypatch):
    monkeypatch.setattr(llm_backends, "generate", lambda *args, **kwargs: "not json")
    monkeypatch.setattr(analyzer.db, "get_analysis", lambda *args: None, raising=False)
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 0)
    monkeypatch.setattr(
        analyzer.db,
        "get_food",
        lambda key: {
            "category": "main course",
            "cuisine": "Indian",
            "ingredients": ["chicken", "tomato"],
            "allergens": ["milk"],
            "tags": ["stewed", "spicy"],
            "description_en": "Database recipe description.",
        },
        raising=False,
    )

    result = analyzer.analyze(make_recognition())

    assert result.source == "fallback"
    assert result.main_ingredients == ["chicken", "tomato"]
    assert result.allergens == ["milk"]


def test_analyze_raises_when_invalid_and_no_database_fallback(monkeypatch):
    monkeypatch.setattr(llm_backends, "generate", lambda *args, **kwargs: "not json")
    monkeypatch.setattr(analyzer.db, "get_analysis", lambda *args: None, raising=False)
    monkeypatch.setattr(analyzer.db, "get_food", lambda key: None, raising=False)
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 0)

    with pytest.raises(analyzer.AnalysisResponseError, match="adatbázis-tartalék sincs"):
        analyzer.analyze(make_recognition())


def test_analyze_returns_cache_without_calling_model(monkeypatch):
    cached = DishAnalysis(
        dish="chicken_curry",
        category="main course",
        cuisine="Indian",
        main_ingredients=["chicken"],
        cooking_method="stewed",
        description="Cached analysis.",
        source="cache",
    )
    monkeypatch.setattr(analyzer.db, "get_analysis", lambda *args: cached, raising=False)
    monkeypatch.setattr(
        llm_backends, "generate", lambda *args, **kwargs: pytest.fail("cache should be used")
    )

    result = analyzer.analyze(make_recognition())

    assert result.source == "cache"


@pytest.mark.parametrize("backend", ["transformers", "ollama"])
def test_generate_routes_to_selected_backend(monkeypatch, backend):
    called = []
    monkeypatch.setattr(llm_backends, "_generate_transformers", lambda *args: called.append("hf") or "a")
    monkeypatch.setattr(llm_backends, "_generate_ollama", lambda *args: called.append("ollama") or "b")

    result = llm_backends.generate("system", "user", backend=backend)

    assert called == ["hf" if backend == "transformers" else "ollama"]
    assert result in {"a", "b"}


def test_generate_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Ismeretlen LLM-backend"):
        llm_backends.generate("system", "user", backend="unknown")


def test_ollama_request_uses_chat_json_and_config(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "{\"ok\": true}"}}

    def post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr("requests.post", post)
    result = llm_backends._generate_ollama("system", "user")

    assert result == '{"ok": true}'
    assert captured["url"].endswith("/api/chat")
    assert captured["json"]["format"] == "json"
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert captured["json"]["options"]["temperature"] == config.LLM_TEMPERATURE
