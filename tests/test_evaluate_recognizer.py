"""A scripts/evaluate_recognizer.py tesztjei – ál-modellel, generált képekkel."""

import importlib.util
import json
import sys

import pytest
from PIL import Image

import config
from pipeline import recognizer

_spec = importlib.util.spec_from_file_location(
    "evaluate_recognizer", config.ROOT_DIR / "scripts" / "evaluate_recognizer.py"
)
ev = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_recognizer"] = ev
_spec.loader.exec_module(ev)

# Az ál-modell a kép színe alapján "ismer fel" – így tudjuk, mit kell kapnunk
COLOR_PREDICTIONS = {
    (255, 0, 0): [("pizza", 0.90), ("lasagna", 0.05), ("ravioli", 0.02)],  # biztos, helyes
    (0, 255, 0): [("ramen", 0.35), ("pho", 0.30), ("miso_soup", 0.20)],  # bizonytalan
    (0, 0, 255): [("sushi", 0.70), ("sashimi", 0.20), ("gyoza", 0.05)],  # biztos, hibás
    (9, 9, 9): [("hamburger", 0.20), ("steak", 0.15), ("tacos", 0.10)],  # egyéb étel
}


def predict_by_color(image):
    """A legközelebbi színhez tartozó választ adja (a JPEG kicsit elmossa a színeket)."""
    pixel = image.getpixel((4, 4))
    color = min(
        COLOR_PREDICTIONS, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, pixel, strict=True))
    )
    return COLOR_PREDICTIONS[color]


@pytest.fixture
def fake_vit(monkeypatch):
    monkeypatch.setattr(recognizer, "_predict", predict_by_color)
    monkeypatch.setattr(recognizer, "_get_classifier", lambda: None)


def save_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


@pytest.fixture
def folder_dataset(tmp_path):
    """Almappás címkézés."""
    save_image(tmp_path / "pizza" / "a.jpg", (255, 0, 0))
    save_image(tmp_path / "pho" / "b.png", (0, 255, 0))
    save_image(tmp_path / "sashimi" / "c.jpg", (0, 0, 255))
    save_image(tmp_path / "egyeb" / "gulyas.jpg", (9, 9, 9))
    (tmp_path / "pizza" / "jegyzet.txt").write_text("nem kép")
    return tmp_path


# --- Címkék -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("chicken_curry", "chicken_curry"),
        ("Chicken curry", "chicken_curry"),
        ("Csirkecurry", "chicken_curry"),
        ("  PIZZA ", "pizza"),
        ("Görög saláta", "greek_salad"),
        ("gorog-salata", "greek_salad"),
        ("gulyás", None),
        ("egyeb", None),
    ],
)
def test_resolve_label(raw, expected):
    assert ev.resolve_label(raw) == expected


# --- Minták beolvasása -----------------------------------------------------------------


def test_collect_samples_from_folders(folder_dataset):
    samples = ev.collect_samples(folder_dataset)
    assert len(samples) == 4  # a .txt kimarad
    assert {s.true_key for s in samples} == {"pizza", "pho", "sashimi", None}


def test_collect_samples_from_csv(tmp_path):
    save_image(tmp_path / "1.jpg", (255, 0, 0))
    save_image(tmp_path / "2.jpg", (9, 9, 9))
    (tmp_path / "labels.csv").write_text(
        "file,label\n1.jpg,Pizza\n2.jpg,gulyás\n", encoding="utf-8"
    )
    samples = ev.collect_samples(tmp_path)
    assert [(s.path.name, s.true_key) for s in samples] == [("1.jpg", "pizza"), ("2.jpg", None)]


def test_collect_samples_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        ev.collect_samples(tmp_path / "nincs")


# --- Mérőszámok --------------------------------------------------------------------------


def test_summary_metrics(fake_vit, folder_dataset):
    rows = ev.evaluate_samples(ev.collect_samples(folder_dataset), threshold=0.4)
    s = ev.summarize(rows, 0.4)

    assert s["food101_images"] == 3 and s["other_images"] == 1
    # pizza helyes; pho: ramen-t mond (top-3-ban benne van); sashimi: sushi-t mond (top-3-ban)
    assert s["top1_accuracy_pct"] == pytest.approx(33.3)
    assert s["top3_accuracy_pct"] == 100.0
    assert s["flagged_pct_food101"] == pytest.approx(33.3)  # csak a 35%-os
    assert s["accuracy_when_not_flagged_pct"] == 50.0  # pizza jó, sushi rossz
    assert s["confident_but_wrong"] == 1
    assert s["flagged_pct_other"] == 100.0  # a gulyásnál rákérdez
    assert s["mean_confidence_correct"] == 0.9


def test_threshold_table(fake_vit, folder_dataset):
    rows = ev.evaluate_samples(ev.collect_samples(folder_dataset), threshold=0.4)
    table = {t["threshold"]: t for t in ev.threshold_table(rows)}
    assert table[0.2]["flagged_pct_food101"] == 0.0
    assert table[0.8]["flagged_pct_food101"] == pytest.approx(66.7)
    assert table[0.8]["accuracy_when_not_flagged_pct"] == 100.0


def test_broken_image_is_reported_not_fatal(fake_vit, tmp_path):
    save_image(tmp_path / "pizza" / "jo.jpg", (255, 0, 0))
    (tmp_path / "pizza" / "rossz.jpg").write_bytes(b"nem kep")
    rows = ev.evaluate_samples(ev.collect_samples(tmp_path), threshold=0.4)
    s = ev.summarize(rows, 0.4)
    assert s["images_failed"] == 1
    assert s["top1_accuracy_pct"] == 100.0


# --- Teljes futás ---------------------------------------------------------------------------


def test_main_writes_outputs(fake_vit, folder_dataset, tmp_path):
    out = tmp_path / "out"
    assert ev.main(["--images", str(folder_dataset), "--out", str(out)]) == 0

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["top3_accuracy_pct"] == 100.0
    assert len(summary["thresholds"]) == len(ev.THRESHOLDS)

    csv_text = (out / "results.csv").read_text(encoding="utf-8-sig")
    assert "pizza|lasagna|ravioli" in csv_text

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Top-1 pontosság | 33.3%" in report
    assert "## Hibás felismerések" in report


def test_main_empty_folder(tmp_path):
    assert ev.main(["--images", str(tmp_path)]) == 1
