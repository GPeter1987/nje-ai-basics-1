"""Az orchestrator (a teljes lánc) tesztjei.

Az 1. modellt ál-modellel helyettesítjük, B és C moduljait pedig vagy a
stubok, vagy monkeypatch-elt függvények adják. A `slow` teszt a valódi
modellekkel fut a samples/ képein (integrációs teszt).
"""

import io

import pytest
from PIL import Image

import config
import nutrition as nutrition_module
from pipeline import analyzer, orchestrator, recognizer, recommender, stubs
from pipeline.orchestrator import (
    STEP_ANALYZE,
    STEP_NUTRITION,
    STEP_RECOGNIZE,
    STEP_RECOMMEND,
    STEP_SELECT,
    continue_pipeline,
    recognize_image,
    run_pipeline,
)
from pipeline.schemas import DishAnalysis, NutritionEstimate, PipelineResult


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 60, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def isolate_c_modules(request, monkeypatch):
    """A stub-ág tesztje ne függjön a valódi C adatbázistól vagy modellletöltéstől.

    A valódi C-modulok integrációját a test_c_pipeline.py ellenőrzi.
    """
    if request.node.get_closest_marker("slow"):
        return

    def pending(*args, **kwargs):
        raise NotImplementedError("A tesztben szándékosan helyettesítő fut.")

    monkeypatch.setattr(nutrition_module, "compute_nutrition", pending)
    monkeypatch.setattr(recommender, "recommend", pending)


@pytest.fixture
def fake_vit(monkeypatch):
    """Az 1. modell helyett: 87% chicken_curry."""
    monkeypatch.setattr(
        recognizer,
        "_predict",
        lambda image: [("chicken_curry", 0.87), ("ramen", 0.08), ("pho", 0.03)],
    )


@pytest.fixture
def unsure_vit(monkeypatch):
    """Az 1. modell helyett: bizonytalan felismerés (35%)."""
    monkeypatch.setattr(
        recognizer,
        "_predict",
        lambda image: [("ramen", 0.35), ("pho", 0.30), ("miso_soup", 0.20)],
    )


def step_names(result: PipelineResult) -> list[str]:
    return [s.name for s in result.steps]


# --- A teljes lánc stubokkal --------------------------------------------------


def test_full_chain_with_stubs(fake_vit):
    """B és C modulja még nincs kész -> stubok futnak, a lánc végigér."""
    result = run_pipeline(png_bytes(), grams=250, mode="similar")

    assert step_names(result) == [STEP_RECOGNIZE, STEP_ANALYZE, STEP_NUTRITION, STEP_RECOMMEND]
    assert result.ok
    assert result.recognition.key == "chicken_curry"
    assert result.recognition.image_hash is not None
    assert result.analysis.dish == "chicken_curry"
    assert result.nutrition.grams == 250
    assert len(result.recommendations) == config.RECOMMEND_TOP_K
    assert all(r.key != "chicken_curry" for r in result.recommendations)
    # a napló jelzi, hogy helyettesítő futott
    assert all(s.note == stubs.STUB_NOTE for s in result.steps[1:])


def test_chain_passes_outputs_forward(fake_vit, monkeypatch):
    """Minden modell az előző kimenetét kapja (ez a lánc lényege)."""
    seen = {}

    def fake_analyze(recognition):
        seen["analyze_input"] = recognition
        return stubs.minimal_analysis(recognition, source="llm")

    def fake_recommend(analysis, mode, k):
        seen["recommend_input"] = (analysis, mode, k)
        return []

    monkeypatch.setattr(analyzer, "analyze", fake_analyze)
    monkeypatch.setattr(recommender, "recommend", fake_recommend)

    result = run_pipeline(png_bytes(), mode="healthier")

    assert seen["analyze_input"] == result.recognition
    assert seen["recommend_input"] == (result.analysis, "healthier", config.RECOMMEND_TOP_K)
    # a pipeline-napló ugyanezt mutatja
    analyze_log = result.steps[1]
    assert analyze_log.input["key"] == "chicken_curry"
    assert analyze_log.output["dish"] == "chicken_curry"
    assert analyze_log.note is None


def test_step_logs_have_timing_and_json_output(fake_vit):
    result = run_pipeline(png_bytes())
    for step in result.steps:
        assert step.duration_ms >= 0
        assert step.output is not None
    PipelineResult.model_validate_json(result.model_dump_json())  # JSON-ba menthető


# --- Hibakezelés ---------------------------------------------------------------


def test_invalid_image_does_not_crash():
    result = run_pipeline(b"ez nem kep")
    assert step_names(result) == [STEP_RECOGNIZE]
    assert result.ok is False
    assert "nem olvasható" in result.steps[0].error
    assert result.recognition is None


def test_analyzer_error_uses_fallback(fake_vit, monkeypatch):
    def broken(recognition):
        raise RuntimeError("az LLM nem érhető el")

    monkeypatch.setattr(analyzer, "analyze", broken)
    result = run_pipeline(png_bytes())

    step = result.steps[1]
    assert step.ok is False
    assert "az LLM nem érhető el" in step.error
    assert step.note == "tartalék-érték"
    assert result.analysis.source == "fallback"
    # a lánc folytatódik a tartalék elemzéssel
    assert result.steps[-1].name == STEP_RECOMMEND


def test_missing_nutrition_uses_ai_estimate(fake_vit, monkeypatch):
    def analysis_with_estimate(recognition):
        analysis = stubs.minimal_analysis(recognition, source="llm")
        return analysis.model_copy(
            update={
                "nutrition_estimate_per_100g": NutritionEstimate(
                    kcal=150, protein=10, carbs=8, fat=9
                )
            }
        )

    def not_in_db(dish_key, grams):
        raise KeyError(dish_key)

    monkeypatch.setattr(analyzer, "analyze", analysis_with_estimate)
    monkeypatch.setattr(nutrition_module, "compute_nutrition", not_in_db)

    result = run_pipeline(png_bytes(), grams=250)

    assert result.nutrition.is_estimate is True
    assert result.nutrition.kcal == 375  # 150 kcal/100 g * 250 g (a specifikáció példája)
    assert result.nutrition.protein == 25.0
    assert result.steps[2].note == "tartalék-érték"


def test_missing_nutrition_without_estimate(fake_vit, monkeypatch):
    monkeypatch.setattr(analyzer, "analyze", lambda r: stubs.minimal_analysis(r, source="llm"))
    monkeypatch.setattr(
        nutrition_module,
        "compute_nutrition",
        lambda key, grams: (_ for _ in ()).throw(KeyError(key)),
    )
    result = run_pipeline(png_bytes())
    assert result.nutrition is None
    assert result.steps[2].ok is False


def test_recommender_error_gives_empty_list(fake_vit, monkeypatch):
    def broken(analysis, mode, k):
        raise ValueError("nincs adatbázis")

    monkeypatch.setattr(recommender, "recommend", broken)
    result = run_pipeline(png_bytes())
    assert result.recommendations == []
    assert result.steps[-1].ok is False


@pytest.mark.parametrize("grams", [5, 2001])
def test_invalid_grams_rejected(grams):
    with pytest.raises(ValueError, match="adag"):
        run_pipeline(png_bytes(), grams=grams)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mód"):
        run_pipeline(png_bytes(), mode="random")


# --- Alacsony bizonyosság és kézi javítás (F3) -----------------------------


def test_low_confidence_continues_with_needs_review(unsure_vit):
    result = run_pipeline(png_bytes())
    assert result.recognition.needs_review is True
    assert result.analysis is not None


def test_selected_key_is_applied_and_logged(unsure_vit):
    result = run_pipeline(png_bytes(), selected_key="pho")
    assert step_names(result)[:2] == [STEP_RECOGNIZE, STEP_SELECT]
    assert result.recognition.key == "pho"
    assert result.recognition.manually_selected is True
    assert result.analysis.dish == "pho"


def test_two_phase_flow_for_ui(unsure_vit):
    recognition, steps = recognize_image(png_bytes(), threshold=0.4)
    assert recognition.needs_review
    recognition = orchestrator.apply_selection(recognition, "miso_soup", steps)
    result = continue_pipeline(recognition, grams=300, mode="allergen_free", steps=steps)

    assert step_names(result)[:2] == [STEP_RECOGNIZE, STEP_SELECT]
    assert result.recognition.key == "miso_soup"
    assert result.nutrition.grams == 300
    assert result.mode == "allergen_free"


def test_accepts_pil_image_directly(fake_vit):
    result = run_pipeline(Image.new("RGB", (16, 16)))
    assert result.recognition.key == "chicken_curry"
    assert result.recognition.image_hash is None


# --- Integrációs teszt valódi modellekkel ---------------------------------


@pytest.mark.slow
def test_integration_on_samples():
    """A samples/ képein a teljes lánc hiba nélkül lefut, a kimenet megfelel a sémának."""
    pytest.importorskip("transformers")
    samples = [
        p
        for p in config.SAMPLES_DIR.iterdir()
        if p.suffix.lower().lstrip(".") in config.ALLOWED_IMAGE_TYPES
    ]
    if not samples:
        pytest.skip("Nincs tesztkép a samples/ mappában.")
    for path in samples:
        result = run_pipeline(path)
        assert result.ok, [s.error for s in result.steps if not s.ok]
        assert isinstance(result.analysis, DishAnalysis)
        PipelineResult.model_validate_json(result.model_dump_json())
