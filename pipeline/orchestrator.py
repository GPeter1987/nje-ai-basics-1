"""
A lánc összefűzése.  Felelős: A

    kép ──► 1. ViT ──► Recognition ──► 2. Qwen ──► DishAnalysis ──► 3. MiniLM ──► ajánlások
                            │                            │
                            └──► tápérték (DB) ◄─────────┘ (AI-becslés, ha nincs a DB-ben)

Minden lépésről napló készül (StepLog): mit kapott, mit adott tovább, mennyi
ideig tartott, és volt-e hiba. Ez jelenik meg a "Pipeline" fülön (F9).

Hibakezelés: egyik lépés hibája sem omlasztja össze az appot. Hiba esetén a
napló érthető üzenetet kap, és tartalék-érték megy tovább, ha van ilyen.
Ha egy modul még nincs kész (NotImplementedError), ideiglenes helyettesítő fut
(lásd pipeline/stubs.py).

Két felhasználási mód:
  1. Egyben:     run_pipeline(feltoltott_fajl, grams=250, mode="similar")
  2. Két lépésben, a Streamlit-felülethez, ahol alacsony bizonyosságnál a
     felhasználó választ a két lépés között (F3):
         recognition, steps = recognize_image(feltoltott_fajl, threshold=0.4)
         ... a felhasználó választ: recognition = apply_selection(recognition, "ramen", steps)
         result = continue_pipeline(recognition, grams=250, mode="similar", steps=steps)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from PIL import Image

import config
import nutrition as nutrition_module
from pipeline import analyzer, recognizer, recommender, stubs
from pipeline.schemas import (
    DishAnalysis,
    Nutrition,
    PipelineResult,
    Recognition,
    RecommendMode,
    StepLog,
)

logger = logging.getLogger(__name__)

# A lépések neve a pipeline-nézetben
STEP_RECOGNIZE = "1. Ételfelismerés (ViT, Food-101)"
STEP_SELECT = "1b. Kézi javítás"
STEP_ANALYZE = "2. Elemzés (Qwen2.5-1.5B)"
STEP_NUTRITION = "Tápérték-számítás (adatbázis)"
STEP_RECOMMEND = "3. Ajánlás (all-MiniLM-L6-v2)"

ImageSource = bytes | str | Path | BinaryIO | Image.Image


# --- Általános lépés-futtató ----------------------------------------------------


def _to_log(value: Any) -> Any:
    """Pydantic modellek (és listáik) JSON-barát alakra a naplóhoz."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_to_log(v) for v in value]
    return value


def _run_step(
    name: str,
    func: Callable[[], Any],
    *,
    log_input: Any = None,
    stub: Callable[[], Any] | None = None,
    fallback: Callable[[], Any] | None = None,
) -> tuple[Any, StepLog]:
    """Lefuttat egy lépést időméréssel és hibakezeléssel.

    func:     a valódi lépés (paraméter nélküli függvény, pl. lambda)
    stub:     ezt hívjuk, ha a modul még nincs kész (NotImplementedError)
    fallback: ezt hívjuk, ha a lépés hibára futott (tartalék-érték)
    """
    note = None
    error = None
    start = time.perf_counter()
    try:
        result = func()
    except NotImplementedError:
        if stub is None:
            raise
        result = stub()
        note = stubs.STUB_NOTE
    except Exception as exc:  # noqa: BLE001 – szándékosan minden hibát elkapunk
        logger.exception("Hiba a(z) '%s' lépésben", name)
        error = f"{type(exc).__name__}: {exc}"
        result = None
        if fallback is not None:
            try:
                result = fallback()
                note = "tartalék-érték"
            except Exception:  # noqa: BLE001
                logger.exception("A tartalék-érték sem állítható elő ('%s')", name)
    duration_ms = (time.perf_counter() - start) * 1000

    step = StepLog(
        name=name,
        input=_to_log(log_input),
        output=_to_log(result),
        duration_ms=round(duration_ms, 1),
        ok=error is None,
        error=error,
        note=note,
    )
    return result, step


# --- 1. lépés: felismerés ---------------------------------------------------------


def recognize_image(
    source: ImageSource, threshold: float | None = None
) -> tuple[Recognition | None, list[StepLog]]:
    """Kép betöltése és felismerése. Hiba esetén (None, napló) a visszatérési érték."""
    if threshold is None:
        threshold = config.CONFIDENCE_THRESHOLD

    log_input: dict[str, Any] = {"threshold": threshold}

    def step() -> Recognition:
        if isinstance(source, Image.Image):
            image, image_hash = source.convert("RGB"), None
        else:
            image, image_hash = recognizer.load_image(source)
        log_input.update({"image_size": list(image.size), "image_hash": image_hash})
        return recognizer.recognize(image, threshold=threshold, image_hash=image_hash)

    recognition, log = _run_step(STEP_RECOGNIZE, step, log_input=log_input)
    return recognition, [log]


def apply_selection(recognition: Recognition, key: str, steps: list[StepLog]) -> Recognition:
    """A felhasználó kézi választása (F3); a naplóba is bekerül."""
    selected, log = _run_step(
        STEP_SELECT,
        lambda: recognizer.select_candidate(recognition, key),
        log_input={"from": recognition.key, "to": key},
    )
    steps.append(log)
    return selected if selected is not None else recognition


# --- 2–3. lépés: elemzés, tápérték, ajánlás -------------------------------------------


def _nutrition_from_estimate(
    analysis: DishAnalysis | None, dish_key: str, grams: float
) -> Nutrition:
    """Tartalék: az LLM 100 g-os becslése a megadott grammra ("AI-becslés")."""
    if analysis is None or analysis.nutrition_estimate_per_100g is None:
        raise LookupError(f"Nincs tápérték-adat és AI-becslés sem: {dish_key}")
    est = analysis.nutrition_estimate_per_100g
    factor = grams / 100
    return Nutrition(
        dish_key=dish_key,
        grams=grams,
        kcal=round(est.kcal * factor),
        protein=round(est.protein * factor, 1),
        carbs=round(est.carbs * factor, 1),
        fat=round(est.fat * factor, 1),
        is_estimate=True,
    )


def continue_pipeline(
    recognition: Recognition,
    grams: float = config.DEFAULT_GRAMS,
    mode: RecommendMode = config.DEFAULT_RECOMMEND_MODE,
    k: int = config.RECOMMEND_TOP_K,
    steps: list[StepLog] | None = None,
) -> PipelineResult:
    """A lánc folytatása a (megerősített) felismeréstől."""
    _validate_inputs(grams, mode)
    steps = list(steps or [])

    # 2. modell: az 1. modell kimenete a bemenete
    analysis, log = _run_step(
        STEP_ANALYZE,
        lambda: analyzer.analyze(recognition),
        log_input=recognition,
        stub=lambda: stubs.stub_analyze(recognition),
        fallback=lambda: stubs.minimal_analysis(recognition),
    )
    steps.append(log)

    # Tápérték: az 1. modell ételkulcsa + adatbázis; ha nincs a DB-ben, az LLM becslése
    nutrition, log = _run_step(
        STEP_NUTRITION,
        lambda: nutrition_module.compute_nutrition(recognition.key, grams),
        log_input={"dish_key": recognition.key, "grams": grams},
        stub=lambda: stubs.stub_compute_nutrition(recognition.key, grams),
        fallback=lambda: _nutrition_from_estimate(analysis, recognition.key, grams),
    )
    steps.append(log)

    # 3. modell: a 2. modell kimenete a bemenete (szűrők is onnan jönnek)
    recommendations = []
    if analysis is not None:
        recommendations, log = _run_step(
            STEP_RECOMMEND,
            lambda: recommender.recommend(analysis, mode=mode, k=k),
            log_input={"analysis": _to_log(analysis), "mode": mode, "k": k},
            stub=lambda: stubs.stub_recommend(analysis, mode=mode, k=k),
            fallback=list,
        )
        steps.append(log)

    return PipelineResult(
        grams=grams,
        mode=mode,
        recognition=recognition,
        analysis=analysis,
        nutrition=nutrition,
        recommendations=recommendations or [],
        steps=steps,
    )


def run_pipeline(
    source: ImageSource,
    grams: float = config.DEFAULT_GRAMS,
    mode: RecommendMode = config.DEFAULT_RECOMMEND_MODE,
    threshold: float | None = None,
    selected_key: str | None = None,
    k: int = config.RECOMMEND_TOP_K,
) -> PipelineResult:
    """A teljes lánc egy képre.

    selected_key: ha a felhasználó már kiválasztotta a helyes ételt (F3).
    Alacsony bizonyosságnál választás nélkül is továbbmegy: ilyenkor a 2. modell
    a top-3 jelöltet is megkapja (recognition.needs_review == True).
    """
    _validate_inputs(grams, mode)

    recognition, steps = recognize_image(source, threshold=threshold)
    if recognition is None:
        # Felismerés nélkül nincs mit továbbadni – a napló tartalmazza a hibát
        return PipelineResult(grams=grams, mode=mode, steps=steps)

    if selected_key is not None and selected_key != recognition.key:
        recognition = apply_selection(recognition, selected_key, steps)

    return continue_pipeline(recognition, grams=grams, mode=mode, k=k, steps=steps)


def _validate_inputs(grams: float, mode: str) -> None:
    if not config.MIN_GRAMS <= grams <= config.MAX_GRAMS:
        raise ValueError(
            f"Az adag {config.MIN_GRAMS} és {config.MAX_GRAMS} g között lehet (kapott: {grams})."
        )
    if mode not in config.RECOMMEND_MODES:
        raise ValueError(f"Ismeretlen ajánlási mód: {mode}")
