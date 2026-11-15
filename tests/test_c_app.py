import io
from types import SimpleNamespace

import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

import config
from data import db
from pipeline import recognizer


def test_empty_app_without_database(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "missing.db")
    app = AppTest.from_file(config.ROOT_DIR / "app.py").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Elemzés", "Pipeline", "Előzmények"]
    assert app.number_input[0].value == 250
    assert any("build_db" in warning.value for warning in app.warning)
    assert not config.DB_PATH.exists()


def test_app_recognition_analysis_history_and_invalidation(c_database, monkeypatch):
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    uploaded = SimpleNamespace(name="test.png", getvalue=lambda: image.getvalue())
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    monkeypatch.setattr(recognizer, "_predict", lambda image: [
        ("chicken_curry", 0.35), ("ramen", 0.3), ("pho", 0.2),
    ])
    app = AppTest.from_file(config.ROOT_DIR / "app.py").run()
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
