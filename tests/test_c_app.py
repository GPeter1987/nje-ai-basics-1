import io
from types import SimpleNamespace

import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

import config
from data import db
from pipeline import analyzer, llm_backends, recognizer

# Az első futás (torch/transformers import) lassú gépen jóval több 3 s-nál
APP_TIMEOUT = 60


def test_empty_app_without_database(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "missing.db")
    app = AppTest.from_file(config.ROOT_DIR / "app.py", default_timeout=APP_TIMEOUT).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Elemzés", "Pipeline", "Előzmények"]
    assert app.number_input[0].value == 250
    assert any("build_db" in warning.value for warning in app.warning)
    assert not config.DB_PATH.exists()


def test_app_recognition_analysis_history_and_invalidation(c_database, dish_analysis, monkeypatch):
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    uploaded = SimpleNamespace(name="test.png", getvalue=lambda: image.getvalue())
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    monkeypatch.setattr(
        recognizer,
        "_predict",
        lambda image: [
            ("chicken_curry", 0.35),
            ("ramen", 0.3),
            ("pho", 0.2),
        ],
    )
    # A valódi Qwen helyett (az LLM-et a test_analyzer.py teszteli)
    monkeypatch.setattr(
        analyzer,
        "analyze",
        lambda rec, backend=None: dish_analysis.model_copy(update={"dish": rec.key}),
    )
    app = AppTest.from_file(config.ROOT_DIR / "app.py", default_timeout=APP_TIMEOUT).run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Alacsony bizonyosság" in w.value for w in app.warning)
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state["result"].recognition.manually_selected
    assert len(app.session_state["result"].recommendations) == 5
    next(b for b in app.button if b.label == "Mentés az előzményekbe").click().run()
    assert not app.exception
    assert len(db.list_history()) == 1
    assert next(b for b in app.button if b.label == "Mentés az előzményekbe").disabled
    app.number_input[0].set_value(300).run()
    assert not app.exception
    assert "result" not in app.session_state
    assert len(db.list_history()) == 1


def test_sidebar_backend_switch_warns_when_ollama_is_down(tmp_path, monkeypatch):
    """F12: a backend az oldalsávban váltható; leállt Ollamánál érthető útmutatás jelenik meg."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "missing.db")
    # a NUTRILENS_LLM_BACKEND környezeti változótól függetlenül
    monkeypatch.setattr(config, "LLM_BACKEND", "transformers")
    monkeypatch.setattr(llm_backends, "ollama_available", lambda timeout=1.0: False)
    app = AppTest.from_file(config.ROOT_DIR / "app.py", default_timeout=APP_TIMEOUT).run()
    backend_box = next(box for box in app.sidebar.selectbox if box.label.startswith("LLM-backend"))
    assert backend_box.value == "transformers"
    assert not any("Ollama nem fut" in error.value for error in app.error)
    backend_box.set_value("ollama").run()
    assert not app.exception
    assert any("Ollama nem fut" in error.value for error in app.error)
