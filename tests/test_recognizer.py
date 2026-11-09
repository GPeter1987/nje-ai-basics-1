"""Az 1. modell (recognizer.py) tesztjei.

A legtöbb teszt a modellt egy "ál-modellel" helyettesíti (monkeypatch), így
torch és letöltés nélkül, gyorsan futnak. A valódi ViT-et a `slow` jelölésű
tesztek használják (kihagyás: pytest -m "not slow").
"""

import io

import pytest
from PIL import Image

import config
from pipeline import recognizer
from pipeline.food101_labels import FOOD101_HU, FOOD101_LABELS, label_to_name_en
from pipeline.recognizer import (
    ImageValidationError,
    dish_choices,
    load_image,
    recognize,
    select_candidate,
)


def image_bytes(fmt: str = "PNG", size=(64, 48), mode="RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color=(200, 100, 50) if mode == "RGB" else 128).save(buffer, format=fmt)
    return buffer.getvalue()


def fake_predictions(*pairs):
    """Ál-modell: a megadott (címke, valószínűség) párokat adja vissza."""
    return lambda image: list(pairs)


# --- Címkék ------------------------------------------------------------------


def test_label_map_has_101_classes():
    assert len(FOOD101_LABELS) == 101
    assert all(FOOD101_HU[label] for label in FOOD101_LABELS)


@pytest.mark.parametrize(
    "label, expected",
    [("chicken_curry", "Chicken curry"), ("pizza", "Pizza"), ("cup_cakes", "Cupcakes")],
)
def test_label_to_name_en(label, expected):
    assert label_to_name_en(label) == expected


# --- load_image (F1) ---------------------------------------------------------


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_load_image_accepts_allowed_formats(fmt):
    image, image_hash = load_image(image_bytes(fmt))
    assert image.mode == "RGB"
    assert image.size == (64, 48)
    assert len(image_hash) == 64


def test_load_image_converts_to_rgb():
    image, _ = load_image(image_bytes("PNG", mode="L"))
    assert image.mode == "RGB"


def test_load_image_same_bytes_same_hash():
    data = image_bytes()
    assert load_image(data)[1] == load_image(io.BytesIO(data))[1]


def test_load_image_rejects_gif():
    with pytest.raises(ImageValidationError, match="Nem támogatott"):
        load_image(image_bytes("GIF", mode="L"))


def test_load_image_rejects_garbage():
    with pytest.raises(ImageValidationError, match="nem olvasható"):
        load_image(b"ez nem kep")


def test_load_image_rejects_empty():
    with pytest.raises(ImageValidationError, match="üres"):
        load_image(b"")


def test_load_image_rejects_too_large(monkeypatch):
    monkeypatch.setattr(config, "MAX_IMAGE_BYTES", 100)
    with pytest.raises(ImageValidationError, match="túl nagy"):
        load_image(image_bytes())


def test_load_image_missing_file(tmp_path):
    with pytest.raises(ImageValidationError, match="nem található"):
        load_image(tmp_path / "nincs.jpg")


def test_load_image_from_path(tmp_path):
    path = tmp_path / "kep.jpg"
    path.write_bytes(image_bytes("JPEG"))
    image, _ = load_image(path)
    assert image.mode == "RGB"


# --- recognize (F2, F3) ------------------------------------------------------


def test_recognize_confident(monkeypatch):
    monkeypatch.setattr(
        recognizer,
        "_predict",
        fake_predictions(("ramen", 0.05), ("chicken_curry", 0.87), ("pho", 0.03), ("sushi", 0.01)),
    )
    rec = recognize(Image.new("RGB", (10, 10)), image_hash="abc")

    assert rec.key == "chicken_curry"
    assert rec.name_hu == "Csirkecurry"
    assert rec.score == pytest.approx(0.87)
    assert [c.key for c in rec.top3] == ["chicken_curry", "ramen", "pho"]
    assert rec.is_confident is True
    assert rec.needs_review is False
    assert rec.image_hash == "abc"


def test_recognize_low_confidence_needs_review(monkeypatch):
    monkeypatch.setattr(
        recognizer,
        "_predict",
        fake_predictions(("ramen", 0.35), ("pho", 0.30), ("miso_soup", 0.20)),
    )
    rec = recognize(Image.new("RGB", (10, 10)))
    assert rec.is_confident is False
    assert rec.needs_review is True


def test_recognize_custom_threshold(monkeypatch):
    monkeypatch.setattr(recognizer, "_predict", fake_predictions(("ramen", 0.35), ("pho", 0.30)))
    assert recognize(Image.new("RGB", (10, 10)), threshold=0.30).is_confident is True


def test_recognize_threshold_is_inclusive(monkeypatch):
    monkeypatch.setattr(recognizer, "_predict", fake_predictions(("ramen", 0.40), ("pho", 0.30)))
    assert recognize(Image.new("RGB", (10, 10))).is_confident is True


def test_select_candidate_from_top3(monkeypatch):
    monkeypatch.setattr(
        recognizer,
        "_predict",
        fake_predictions(("ramen", 0.35), ("pho", 0.30), ("miso_soup", 0.20)),
    )
    rec = select_candidate(recognize(Image.new("RGB", (10, 10))), "pho")

    assert rec.key == "pho"
    assert rec.score == pytest.approx(0.30)
    assert rec.manually_selected is True
    assert rec.needs_review is False
    assert [c.key for c in rec.top3] == ["ramen", "pho", "miso_soup"]  # a modell válasza megmarad


def test_select_candidate_from_full_list(monkeypatch):
    monkeypatch.setattr(recognizer, "_predict", fake_predictions(("ramen", 0.35), ("pho", 0.30)))
    rec = select_candidate(recognize(Image.new("RGB", (10, 10))), "lasagna")
    assert rec.key == "lasagna"
    assert rec.score == 0.0
    assert rec.manually_selected is True


def test_select_candidate_unknown_key(monkeypatch):
    monkeypatch.setattr(recognizer, "_predict", fake_predictions(("ramen", 0.9)))
    with pytest.raises(ValueError):
        select_candidate(recognize(Image.new("RGB", (10, 10))), "gulyas")


def test_dish_choices_sorted_and_complete():
    choices = dish_choices()
    assert len(choices) == 101
    assert ("chicken_curry", "Csirkecurry (Chicken curry)") in choices
    assert ("pizza", "Pizza") in choices
    keys = [display for _, display in choices]
    assert keys == sorted(keys, key=recognizer._sort_key)


# --- Valódi modell (lassú) ---------------------------------------------------


@pytest.mark.slow
def test_real_model_labels_match_our_map():
    """A modell címkéi pontosan a mi 101 kulcsunk – különben az adatbázis-kulcsok elcsúsznak."""
    pytest.importorskip("transformers")
    classifier = recognizer._get_classifier()
    assert set(classifier.model.config.id2label.values()) == set(FOOD101_LABELS)


@pytest.mark.slow
def test_real_model_on_samples():
    """A samples/ mappa képein lefut a felismerés, és sémának megfelelő eredményt ad."""
    pytest.importorskip("transformers")
    samples = [
        p
        for p in config.SAMPLES_DIR.iterdir()
        if p.suffix.lower().lstrip(".") in config.ALLOWED_IMAGE_TYPES
    ]
    if not samples:
        pytest.skip("Nincs tesztkép a samples/ mappában.")
    for path in samples:
        image, image_hash = load_image(path)
        rec = recognize(image, image_hash=image_hash)
        assert rec.key in FOOD101_LABELS
        assert len(rec.top3) == 3
