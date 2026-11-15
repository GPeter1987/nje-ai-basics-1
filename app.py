"""
Streamlit belépési pont.  Felelős: C

Futtatás:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from data import db
from pipeline import orchestrator, recognizer, recommender
from pipeline.schemas import Nutrition, PipelineResult


@st.cache_resource(show_spinner="MiniLM ajánlómodell betöltése…")
def load_recommender():
    return recommender.get_embedder()


def nutrition_cards(values: Nutrition | dict) -> None:
    data = values.model_dump() if isinstance(values, Nutrition) else values
    for column, (key, label, unit) in zip(st.columns(4), [
        ("kcal", "Energia", "kcal"), ("protein", "Fehérje", "g"),
        ("carbs", "Szénhidrát", "g"), ("fat", "Zsír", "g"),
    ], strict=True):
        column.metric(label, f"{data[key]:g} {unit}")


def render_result(result: PipelineResult, database_ready: bool) -> None:
    for step in result.steps:
        if step.error:
            st.warning(f"{step.name}: {step.error}")
    if result.nutrition:
        nutrition = result.nutrition
        st.subheader(f"Tápérték · {nutrition.grams:g} g")
        nutrition_cards(nutrition)
        if nutrition.is_estimate:
            st.warning("AI-becslés – nincs adatbázisban ellenőrizhető tápérték.")
        else:
            st.caption(
                "USDA-alapanyagokból számolt mintarecept; a tényleges recept és adag eltérhet."
            )
        with st.expander("Makrotápanyagok energiamegoszlása és adatforrás"):
            st.bar_chart(pd.DataFrame({"kcal": {
                "Fehérje": nutrition.protein * 4,
                "Szénhidrát": nutrition.carbs * 4,
                "Zsír": nutrition.fat * 9,
            }}))
            if database_ready:
                food = db.get_food(nutrition.dish_key)
                if food:
                    st.write(food["source"])
            st.caption("A diagram a 4/4/9 kcal/g közelítést használja.")

    if result.analysis:
        analysis = result.analysis
        st.subheader("AI-elemzés")
        source_names = {"llm": "Valódi LLM", "cache": "Gyorsítótár", "fallback": "Tartalék elemzés"}
        st.caption(source_names[analysis.source])
        if analysis.source == "fallback":
            st.warning(
                "Nem valódi Qwen-elemzés. A B modul még hiányzik, vagy az elemzés hibára futott. "
                "Az összetevők és allergének ebből nem állapíthatók meg megbízhatóan."
            )
        st.write(analysis.description)
        left, right = st.columns(2)
        with left:
            st.write(f"**Kategória:** {analysis.category} · **Konyha:** {analysis.cuisine}")
            st.write("**Összetevők:** " + ", ".join(analysis.main_ingredients))
            st.write(f"**Elkészítés:** {analysis.cooking_method}")
        with right:
            st.write(
                "**Allergének:** " + (", ".join(analysis.allergens) or "Nincs jelzett allergén")
            )
            st.write("**Ízprofil:** " + (", ".join(analysis.flavor_profile) or "–"))
            st.write("**Diétás címkék:** " + (", ".join(analysis.dietary_tags) or "–"))

    st.subheader("Ajánlott alternatívák")
    st.caption(
        "A hasonlósági pontszám nem felismerési valószínűség. "
        "A kcal-eltérés mindig 100 g-ra szól."
    )
    if result.mode == "allergen_free":
        st.warning(
            "A szűrés csak az elemzésben jelzett allergéneket és a mintarecept összetevőit veszi "
            "figyelembe. Nem allergénmentességi garancia; a csomagolást és "
            "keresztszennyeződést ellenőrizd."
        )
    if not result.recommendations:
        st.info(
            "Nincs megjeleníthető ajánlás. Nézd meg a Pipeline fület "
            "és a fenti figyelmeztetéseket."
        )
    elif len(result.recommendations) < config.RECOMMEND_TOP_K:
        st.info(f"A szűrőknek csak {len(result.recommendations)} étel felel meg.")
    for item in result.recommendations:
        with st.container(border=True):
            name, similarity, kcal = st.columns([3, 1, 1])
            name.markdown(f"**{item.name_hu or item.name_en}**")
            name.caption(item.name_en)
            similarity.metric("Hasonlóság", f"{item.similarity:.1%}")
            kcal.metric("kcal / 100 g", f"{item.kcal_100g:g}",
                        delta=f"{item.kcal_diff_100g:+g}", delta_color="inverse")
            st.write(item.reason)

    if result.nutrition and result.recognition and database_ready:
        if st.button("Mentés az előzményekbe", disabled=st.session_state.get("saved", False)):
            try:
                db.save_history(result.recognition, result.nutrition)
            except Exception as exc:
                st.error(f"Az előzmény nem menthető: {exc}")
            else:
                st.session_state.saved = True
                st.rerun()
        if st.session_state.get("saved"):
            st.success("Az étkezés mentve. Új elemzéssel rögzíthetsz újabb étkezést.")


def render_analysis(grams: int, mode: str, threshold: float, database_ready: bool) -> None:
    uploaded = st.file_uploader("Ételfotó feltöltése", type=list(config.ALLOWED_IMAGE_TYPES))
    st.caption("JPG, PNG vagy WEBP · legfeljebb 10 MB · a felismerő csak 101 ételtípust ismer.")
    content = uploaded.getvalue() if uploaded else None
    fingerprint = (hashlib.sha256(content).hexdigest(), threshold) if content else None
    if fingerprint != st.session_state.get("image_fingerprint"):
        st.session_state.image_fingerprint = fingerprint
        for key in ("recognition", "steps", "result", "result_settings", "selected_dish", "saved"):
            st.session_state.pop(key, None)
    if not content:
        st.info("Tölts fel egy ételfotót, majd indítsd el a felismerést.")
        return
    try:
        image, _ = recognizer.load_image(content)
    except recognizer.ImageValidationError as exc:
        st.error(str(exc))
        return
    st.image(image, caption=uploaded.name, width=480)
    if st.button("1. Étel felismerése", type="primary"):
        with st.spinner("ViT ételfelismerés… Az első modellbetöltés hosszabb lehet."):
            recognition, steps = orchestrator.recognize_image(content, threshold)
        st.session_state.recognition = recognition
        st.session_state.steps = steps
        st.session_state.pop("result", None)
        st.session_state.pop("saved", None)
        st.session_state.pop("selected_dish", None)
    recognition = st.session_state.get("recognition")
    if recognition is None:
        for step in st.session_state.get("steps", []):
            if step.error:
                st.error(step.error)
        return

    st.subheader(recognition.name_hu or recognition.name_en)
    st.metric("A modell bizonyossága", f"{recognition.score:.1%}")
    st.progress(recognition.score)
    st.bar_chart(pd.DataFrame({"Bizonyosság (%)": {
        c.name_hu or c.name_en: c.score * 100 for c in recognition.top3
    }}))
    if recognition.needs_review:
        st.warning("Alacsony bizonyosság! Válaszd ki vagy erősítsd meg az ételt a folytatás előtt.")
    choices = dict(recognizer.dish_choices())
    keys = list(dict.fromkeys([c.key for c in recognition.top3] + list(choices)))
    selected = st.selectbox("Nem ez? Válaszd ki vagy erősítsd meg:", keys,
                            format_func=choices.__getitem__, key="selected_dish")
    settings = (grams, mode, selected)
    if settings != st.session_state.get("result_settings"):
        st.session_state.pop("result", None)
        st.session_state.pop("saved", None)
    if st.button("2. Elemzés és ajánlások", type="primary", disabled=not database_ready):
        steps = list(st.session_state.steps)
        confirmed = recognition
        if recognition.needs_review or selected != recognition.key:
            confirmed = orchestrator.apply_selection(recognition, selected, steps)
        with st.spinner("Elemzés, tápértékszámítás és MiniLM-ajánlás…"):
            # Modellbetöltési hibát az orchestrator is kezel és naplóz.
            try:
                load_recommender()
            except Exception as exc:
                st.warning(f"A MiniLM nem tölthető be: {exc}")
            result = orchestrator.continue_pipeline(confirmed, grams=grams, mode=mode, steps=steps)
        st.session_state.result = result
        st.session_state.result_settings = settings
        st.session_state.saved = False
    if st.session_state.get("result"):
        render_result(st.session_state.result, database_ready)


def render_pipeline() -> None:
    result = st.session_state.get("result")
    steps = result.steps if result else st.session_state.get("steps", [])
    if not steps:
        st.info("A felismerés után itt látható, mit kapott és adott tovább minden modell.")
    if result:
        st.metric("Összes feldolgozási idő", f"{result.total_ms:.0f} ms")
    for step in steps:
        with st.expander(f"{'✓' if step.ok else '⚠'} {step.name} · {step.duration_ms:.1f} ms"):
            if step.note:
                st.warning(step.note)
            if step.error:
                st.error(step.error)
            st.write("**Bemenet**")
            st.json(step.input)
            st.write("**Kimenet**")
            st.json(step.output)
    st.caption(
        "A napló a közös orchestrator be- és kimeneteit mutatja; "
        "a nyers LLM-szöveget a B/A modul tudja naplózni."
    )


def render_history(database_ready: bool) -> None:
    if not database_ready:
        st.info("Az előzményekhez előbb építsd fel az adatbázist.")
        return
    try:
        day = st.date_input("Napi összesítő dátuma")
        summary = db.daily_summary(day)
        st.subheader(f"Napi összesítő · {summary['meals']} étkezés")
        nutrition_cards(summary)
        st.bar_chart(pd.DataFrame({"gramm": {
            "Fehérje": summary["protein"], "Szénhidrát": summary["carbs"], "Zsír": summary["fat"],
        }}))
        history = db.list_history()
        if history:
            frame = pd.DataFrame(history)[[
                "created_at", "name_hu", "grams", "kcal", "protein", "carbs", "fat", "is_estimate",
            ]].rename(columns={
                "created_at": "Időpont", "name_hu": "Étel", "grams": "Adag (g)",
                "kcal": "Energia (kcal)", "protein": "Fehérje (g)", "carbs": "Szénhidrát (g)",
                "fat": "Zsír (g)", "is_estimate": "AI-becslés",
            })
            frame["AI-becslés"] = frame["AI-becslés"].astype(bool)
            st.dataframe(frame, hide_index=True)
            st.caption("Az utolsó 200 étkezés. A napi összeg az összes mentett étkezésből készül.")
        else:
            st.info("Még nincs mentett étkezés.")
    except Exception as exc:
        st.error(f"Az előzmények nem tölthetők be: {exc}")


def main() -> None:
    st.set_page_config(page_title="NutriLens", page_icon="🥗", layout="wide")
    st.title("🥗 NutriLens")
    st.markdown("**Egy fotó. Három AI-modell. Tudatosabb ételválasztás.**")
    st.caption("Egyetemi bemutató · lokális modellek · magyar felület")
    with st.sidebar:
        st.header("Beállítások")
        grams = st.number_input("Adag (g)", min_value=config.MIN_GRAMS, max_value=config.MAX_GRAMS,
                                value=config.DEFAULT_GRAMS, step=10)
        mode = st.selectbox("Ajánlási mód", list(config.RECOMMEND_MODES),
                            format_func=config.RECOMMEND_MODES.__getitem__)
        threshold = st.slider("Bizonyossági küszöb (%)", 0, 100,
                              round(config.CONFIDENCE_THRESHOLD * 100)) / 100
        st.caption(f"LLM-backend: {config.LLM_BACKEND}. A backendváltás a B modul feladata.")
        st.divider()
        st.warning(
            "Az alkalmazás nem orvosi vagy dietetikai tanácsadás. A tápértékek receptfüggők."
        )
    database_ready = False
    try:
        if Path(config.DB_PATH).is_file():
            database_ready = bool(db.list_foods())
        if not database_ready:
            st.warning("Az adatbázis még nem áll készen. Futtasd: python scripts/build_db.py")
    except Exception as exc:
        st.error(f"Adatbázishiba: {exc}. Futtasd: python scripts/build_db.py")
    analysis_tab, pipeline_tab, history_tab = st.tabs(["Elemzés", "Pipeline", "Előzmények"])
    with analysis_tab:
        render_analysis(grams, mode, threshold, database_ready)
    with pipeline_tab:
        render_pipeline()
    with history_tab:
        render_history(database_ready)


if __name__ == "__main__":
    main()
