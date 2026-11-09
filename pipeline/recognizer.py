"""
1. modell – ételfelismerés (ViT, Food-101).  Felelős: A

Bemenet:  a feltöltött kép (PIL Image, RGB).
Kimenet:  Recognition – a top-1 étel, a bizonyossága és a top-3 jelölt.

Fontos korlát: a modell csak a Food-101 101 osztályát ismeri, ismeretlen
ételnél is a "legközelebbit" adja. Ezért van a bizonyossági küszöb (F3):
ez alatt a felhasználó választja ki a helyes ételt (select_candidate).

Használat:
    image, image_hash = load_image(feltoltott_bajtok)       # F1: ellenőrzés
    recognition = recognize(image, image_hash=image_hash)   # F2: top-1 + top-3
    if recognition.needs_review:                            # F3: alacsony bizonyosság
        recognition = select_candidate(recognition, "ramen")
"""

from __future__ import annotations

import hashlib
import io
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageOps, UnidentifiedImageError

import config
from pipeline.food101_labels import FOOD101_LABELS, label_to_name_en, label_to_name_hu
from pipeline.schemas import Candidate, Recognition

# A PIL által visszaadott formátumnevek, amelyeket elfogadunk (F1)
_ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(ValueError):
    """A feltöltött fájl nem használható képként (rossz formátum, túl nagy, sérült)."""


# --- Kép betöltése és ellenőrzése (F1) ---------------------------------------


def load_image(source: bytes | str | Path | BinaryIO) -> tuple[Image.Image, str]:
    """Beolvassa és ellenőrzi a képet, majd RGB-re konvertálja.

    A `source` lehet bájtsor, fájlútvonal vagy megnyitott fájl
    (pl. a Streamlit `st.file_uploader` által adott objektum).

    Visszaad: (RGB kép, a fájl SHA-256 hash-e). A hash az előzményekhez kell (F10).
    Hiba esetén ImageValidationError-t dob, érthető magyar üzenettel.
    """
    data = _read_bytes(source)

    if not data:
        raise ImageValidationError("A fájl üres.")
    if len(data) > config.MAX_IMAGE_BYTES:
        size_mb = len(data) / (1024 * 1024)
        max_mb = config.MAX_IMAGE_BYTES / (1024 * 1024)
        raise ImageValidationError(
            f"A kép túl nagy ({size_mb:.1f} MB). Legfeljebb {max_mb:.0f} MB lehet."
        )

    try:
        image = Image.open(io.BytesIO(data))
        image_format = image.format
        image.load()  # itt derül ki, ha a fájl sérült
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("A fájl nem olvasható képként, vagy sérült.") from exc

    if image_format not in _ALLOWED_PIL_FORMATS:
        allowed = ", ".join(t.upper() for t in config.ALLOWED_IMAGE_TYPES)
        raise ImageValidationError(
            f"Nem támogatott képformátum ({image_format}). Elfogadott: {allowed}."
        )

    # A telefonos fotók gyakran csak EXIF-ben tárolják az elforgatást
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    image_hash = hashlib.sha256(data).hexdigest()
    return image, image_hash


def _read_bytes(source: bytes | str | Path | BinaryIO) -> bytes:
    if isinstance(source, bytes | bytearray):
        return bytes(source)
    if isinstance(source, str | Path):
        path = Path(source)
        if not path.is_file():
            raise ImageValidationError(f"A fájl nem található: {path}")
        return path.read_bytes()
    # fájlszerű objektum (pl. Streamlit UploadedFile)
    if hasattr(source, "seek"):
        source.seek(0)
    return source.read()


# --- A modell ----------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_classifier():
    """Egyszer tölti be a ViT-modellt, utána a memóriában tartja.

    A transformers importja itt van (és nem a fájl elején), így a modul
    többi része – és a tesztek – torch nélkül is használhatók.
    """
    from transformers import pipeline

    return pipeline("image-classification", model=config.RECOGNIZER_MODEL, device=-1)  # -1 = CPU


def _predict(image: Image.Image) -> list[tuple[str, float]]:
    """A modell nyers kimenete: (címke, valószínűség) párok, csökkenő sorrendben."""
    classifier = _get_classifier()
    # A pipeline maga végzi az előfeldolgozást (224x224-re méretezés, normalizálás)
    predictions = classifier(image, top_k=len(FOOD101_LABELS))
    return [(p["label"], float(p["score"])) for p in predictions]


# --- Felismerés (F2) ---------------------------------------------------------


def _make_candidate(label: str, score: float) -> Candidate:
    return Candidate(
        key=label,
        name_en=label_to_name_en(label),
        name_hu=label_to_name_hu(label),
        score=round(min(max(score, 0.0), 1.0), 4),
    )


def recognize(
    image: Image.Image,
    threshold: float | None = None,
    image_hash: str | None = None,
) -> Recognition:
    """Felismeri az ételt a képen.

    threshold:  bizonyossági küszöb 0–1 között (alapból config.CONFIDENCE_THRESHOLD);
                a UI csúszkájáról érkezik.
    image_hash: a load_image által számolt hash, változatlanul továbbadjuk.
    """
    if threshold is None:
        threshold = config.CONFIDENCE_THRESHOLD

    if image.mode != "RGB":
        image = image.convert("RGB")

    predictions = _predict(image)
    if not predictions:
        raise RuntimeError("A felismerő modell nem adott eredményt.")

    predictions.sort(key=lambda p: p[1], reverse=True)
    top3 = [
        _make_candidate(label, score) for label, score in predictions[: config.RECOGNIZER_TOP_K]
    ]
    best = top3[0]

    return Recognition(
        key=best.key,
        name_en=best.name_en,
        name_hu=best.name_hu,
        score=best.score,
        top3=top3,
        is_confident=best.score >= threshold,
        image_hash=image_hash,
    )


# --- Kézi javítás (F3) -------------------------------------------------------


def select_candidate(recognition: Recognition, key: str) -> Recognition:
    """A felhasználó által választott ételre cseréli a felismerést.

    A `key` lehet a top-3 egyike, vagy bármely Food-101 étel a listából.
    A top-3 megmarad (a pipeline-nézet így mutatja, mit mondott a modell).
    Ha a választott étel nincs a top-3-ban, a bizonyossága 0, mert a modell
    erre az ételre nem adott érdemi valószínűséget.
    """
    if key not in FOOD101_LABELS:
        raise ValueError(f"Ismeretlen étel: {key}")

    chosen = next((c for c in recognition.top3 if c.key == key), None)
    if chosen is None:
        chosen = _make_candidate(key, 0.0)

    return recognition.model_copy(
        update={
            "key": chosen.key,
            "name_en": chosen.name_en,
            "name_hu": chosen.name_hu,
            "score": chosen.score,
            "manually_selected": True,
        }
    )


def dish_choices() -> list[tuple[str, str]]:
    """Az összes felismerhető étel a "Nem ez? Válaszd ki:" listához (F3).

    Visszaad: (kulcs, megjelenített név) párok, magyar név szerint rendezve,
    pl. ("chicken_curry", "Csirkecurry (Chicken curry)").
    """
    choices = []
    for label in FOOD101_LABELS:
        name_en = label_to_name_en(label)
        name_hu = label_to_name_hu(label) or name_en
        display = name_hu if name_hu.lower() == name_en.lower() else f"{name_hu} ({name_en})"
        choices.append((label, display))
    return sorted(choices, key=lambda c: _sort_key(c[1]))


def _sort_key(text: str) -> str:
    """Ékezetek nélküli, kisbetűs kulcs, hogy pl. az 'Á' ne a lista végére kerüljön."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))
