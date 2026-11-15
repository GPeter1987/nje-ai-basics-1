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

    def generate(system_prompt, user_prompt, backend=None, json_schema=None):
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


@pytest.mark.parametrize(
    "dish_value", ["Chicken curry", "chicken-curry", "CHICKEN_CURRY", "", None]
)
def test_display_name_or_missing_dish_accepted_without_retry(monkeypatch, dish_value):
    """A modell a nevet írja a kulcs helyett / kihagyja: nincs drága újrapróba."""
    calls = []
    response = json.loads(valid_response())
    response["dish"] = dish_value
    monkeypatch.setattr(
        llm_backends, "generate", lambda *a, **k: calls.append(1) or json.dumps(response)
    )
    monkeypatch.setattr(analyzer.db, "get_analysis", lambda *args: None, raising=False)
    monkeypatch.setattr(analyzer.db, "save_analysis", lambda *args: None, raising=False)

    result = analyzer.analyze(make_recognition())
    assert result.dish == "chicken_curry"
    assert result.source == "llm"
    assert len(calls) == 1


def test_different_dish_still_rejected():
    """Ha a válasz egy másik ételről szól, az továbbra is hiba (újrapróba / tartalék)."""
    with pytest.raises(analyzer.AnalysisResponseError, match="más ételkulcsot"):
        analyzer._validate_response(valid_response(dish="pho"), "chicken_curry")


def test_analyze_normalizes_comma_separated_list_fields(monkeypatch):
    response = valid_response(
        main_ingredients="chicken, cream; egg noodles",
        allergens="peanuts, fish",  # a modell allergénlistáját nem használjuk
        flavor_profile="spicy, aromatic, rich",
        dietary_tags="none",
    )
    monkeypatch.setattr(llm_backends, "generate", lambda *args, **kwargs: response)
    monkeypatch.setattr(analyzer.db, "get_analysis", lambda *args: None, raising=False)
    monkeypatch.setattr(analyzer.db, "save_analysis", lambda *args: None, raising=False)

    result = analyzer.analyze(make_recognition())

    assert result.main_ingredients == ["chicken", "cream", "egg noodles"]
    assert result.allergens == ["eggs", "gluten", "milk"]  # az összetevőkből
    assert result.flavor_profile == ["spicy", "aromatic", "rich"]
    assert result.dietary_tags == []


def test_analyze_retries_invalid_response(monkeypatch):
    prompts = []
    responses = iter(["not json", valid_response()])
    monkeypatch.setattr(
        llm_backends,
        "generate",
        lambda system, user, **kwargs: prompts.append(user) or next(responses),
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
    monkeypatch.setattr(
        llm_backends, "_generate_transformers", lambda *args: called.append("hf") or "a"
    )
    monkeypatch.setattr(
        llm_backends, "_generate_ollama", lambda *args: called.append("ollama") or "b"
    )

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
            return {"message": {"content": '{"ok": true}'}}

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


@pytest.mark.parametrize("backend", ["transformers", "ollama"])
def test_analyze_uses_selected_backend_for_generation_and_cache(monkeypatch, backend):
    """F12: a felületen választott backend jut el az LLM-hívásig és a cache-kulcsig."""
    generated, cache_keys = [], []
    monkeypatch.setattr(
        llm_backends,
        "generate",
        lambda system, user, backend=None, **kwargs: generated.append(backend) or valid_response(),
    )
    monkeypatch.setattr(
        analyzer.db, "get_analysis", lambda dish, key: cache_keys.append(key), raising=False
    )
    monkeypatch.setattr(analyzer.db, "save_analysis", lambda *args: None, raising=False)

    analyzer.analyze(make_recognition(), backend=backend)

    assert generated == [backend]
    model = config.LLM_MODEL if backend == "transformers" else config.OLLAMA_MODEL
    assert cache_keys == [f"{backend}:{model}:{analyzer.PROMPT_VERSION}"]


def test_analyze_defaults_to_configured_backend(monkeypatch):
    generated = []
    monkeypatch.setattr(config, "LLM_BACKEND", "transformers")
    monkeypatch.setattr(
        llm_backends,
        "generate",
        lambda system, user, backend=None, **kwargs: generated.append(backend) or valid_response(),
    )
    monkeypatch.setattr(analyzer.db, "get_analysis", lambda *args: None, raising=False)
    monkeypatch.setattr(analyzer.db, "save_analysis", lambda *args: None, raising=False)

    analyzer.analyze(make_recognition())

    assert generated == ["transformers"]


def test_analyze_rejects_unknown_backend_before_calling_llm(monkeypatch):
    monkeypatch.setattr(llm_backends, "generate", lambda *a, **k: pytest.fail("no LLM call"))
    with pytest.raises(ValueError, match="Ismeretlen LLM-backend"):
        analyzer.analyze(make_recognition(), backend="gpt")


@pytest.mark.parametrize("version,name", [("4.44.0", "torch_dtype"), ("5.0.0", "dtype")])
def test_transformers_model_loads_in_stored_precision(monkeypatch, version, name):
    """A Qwen bf16-ban töltődjön (kb. 3 GB), ne fp32-ben (kb. 6 GB) – transformers 4.x-en sem."""
    import transformers

    monkeypatch.setattr(transformers, "__version__", version)
    assert llm_backends._dtype_kwargs() == {name: "auto"}


def test_ollama_down_gives_recognizable_error(monkeypatch):
    import requests

    def refuse(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("requests.post", refuse)
    monkeypatch.setattr("requests.get", refuse)
    with pytest.raises(RuntimeError, match=llm_backends.OLLAMA_UNAVAILABLE):
        llm_backends._generate_ollama("system", "user")
    assert llm_backends.ollama_available() is False


# --- v2 prompt: sablonmásolás és ellentmondó címkék ---------------------------------


def test_llm_allergen_list_is_replaced_by_ingredient_rules():
    """Regresszió (v1): a modell pizzához 11 allergént, szinte mindenhez "peanuts"-ot adott.
    A v2-ben az allergének a modell összetevőiből, szabályokkal készülnek."""
    raw = valid_response(
        main_ingredients=["wheat flour", "tomato sauce", "mozzarella", "olive oil"],
        allergens=["peanuts", "fish", "celery", "lupin", "sesame", "mustard", "eggs"],
    )
    assert analyzer._validate_response(raw, "chicken_curry").allergens == ["gluten", "milk"]


@pytest.mark.parametrize(
    "ingredients,expected",
    [
        (["wheat flour", "mozzarella", "tomato"], ["gluten", "milk"]),
        (["sushi rice", "raw tuna", "nori", "soy sauce"], ["fish", "gluten", "soybeans"]),
        (["chickpeas", "tahini", "lemon", "garlic"], ["sesame"]),
        (["clams", "potatoes", "cream", "celery"], ["celery", "milk", "molluscs"]),
        (
            ["rice noodles", "shrimp", "peanuts", "egg", "fish sauce"],
            ["crustaceans", "eggs", "fish", "peanuts"],
        ),
        (
            ["beef patty", "bun", "cheddar", "dijon mustard", "mayonnaise"],
            ["eggs", "gluten", "milk", "mustard"],
        ),
        (["potatoes", "salt", "vegetable oil"], []),
    ],
)
def test_allergens_from_ingredients(ingredients, expected):
    assert analyzer.allergens_from_ingredients(ingredients) == expected


@pytest.mark.parametrize(
    "ingredient",
    [
        "coconut milk",
        "rice noodles",
        "corn tortilla",
        "eggplant",
        "nutmeg",
        "butternut squash",
        "coconut",
        "champignon",
    ],
)
def test_allergen_rule_exceptions(ingredient):
    assert analyzer.allergens_from_ingredients([ingredient]) == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("main course", "main course"),
        ("Main Course", "main course"),
        ("main course, side dish, or other", "main course"),
        ("a hearty soup", "soup"),
        ("street food", "other"),
    ],
)
def test_category_normalized_to_allowed_values(raw, expected):
    result = analyzer._validate_response(valid_response(category=raw), "chicken_curry")
    assert result.category == expected


@pytest.mark.parametrize(
    "ingredients,tags,expected",
    [
        # pizza: glutén + tej mellett nem lehet gluténmentes / tejmentes, de vegetáriánus igen
        (
            ["wheat flour", "mozzarella", "tomato"],
            ["gluten-free", "milk-free", "vegetarian", "vegan"],
            ["vegetarian"],
        ),
        # csirkés curry: se nem vegán, se nem vegetáriánus
        (["chicken", "tomato", "onion"], ["vegan", "vegetarian", "gluten-free"], ["gluten-free"]),
        # hal: nem vegetáriánus
        (["rice", "nori", "raw tuna"], ["vegetarian", "dairy-free"], ["dairy-free"]),
        # "x-free" címke, ha x allergén
        (["almonds", "sugar"], ["nut-free", "tree-nuts-free"], []),
        # a "coconut" nem "cod", a "champignon" nem "ham"
        (["coconut milk", "champignon", "rice"], ["vegan"], ["vegan"]),
    ],
)
def test_contradicting_dietary_tags_dropped(ingredients, tags, expected):
    raw = valid_response(main_ingredients=ingredients, dietary_tags=tags)
    assert analyzer._validate_response(raw, "chicken_curry").dietary_tags == expected


def test_prompt_v2_has_few_shot_examples_and_no_template_lists():
    prompt = analyzer.SYSTEM_PROMPT
    assert analyzer.PROMPT_VERSION == "dish-analysis-v2"
    for example in analyzer.EXAMPLES:
        assert json.dumps(example, ensure_ascii=False) in prompt
        DishAnalysis.model_validate(example)  # a példák megfelelnek a közös sémának
    # a v1 hibája: a JSON-sablonban felsorolt opciókat másolta a modell
    assert '"category": "main course, soup' not in prompt
    assert "allergens" not in prompt


def test_response_schema_is_sent_to_ollama(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": valid_response()}}

    monkeypatch.setattr("requests.post", lambda url, **kw: captured.update(kw) or Response())
    monkeypatch.setattr(analyzer.db, "get_analysis", lambda *args: None, raising=False)
    monkeypatch.setattr(analyzer.db, "save_analysis", lambda *args: None, raising=False)

    analyzer.analyze(make_recognition(), backend="ollama")

    schema = captured["json"]["format"]
    assert schema["properties"]["dish"]["enum"] == ["chicken_curry"]  # csak a felismert kulcs
    assert analyzer.RESPONSE_SCHEMA["properties"]["dish"] == {"type": "string"}  # nem módosult
    assert "allergens" not in schema["properties"]  # az összetevőkből képezzük
    assert schema["properties"]["category"]["enum"] == list(analyzer.CATEGORIES)
    assert set(schema["required"]) <= set(DishAnalysis.model_fields)


def test_accented_dish_name_matches_key():
    """Regresszió (v2 mérés): a modell a pho-hoz "Phở"-t írt, ami nem egyezett a kulccsal."""
    assert analyzer._validate_response(valid_response(dish="Phở"), "pho").dish == "pho"
