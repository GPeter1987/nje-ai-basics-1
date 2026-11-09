"""
Az 1. modell (ViT / Food-101) pontosságmérése saját képeken.  Felelős: A

Mér: top-1 és top-3 pontosság, bizonyosság a helyes és a hibás találatoknál,
a küszöb (F3) hatása, futási idő. A Food-101-en kívüli ételeknél (pl. gulyás)
azt méri, hányszor jelez helyesen a küszöb ("nem biztos, válassz").

A képek címkézése (kétféle lehet):

  a) almappák – a mappa neve a címke:
         eval_images/
             chicken_curry/  kep1.jpg kep2.jpg
             pizza/          ...
             egyeb/          gulyas.jpg      <- Food-101-en kívüli étel

  b) labels.csv a mappában (fejléc: file,label):
         file,label
         kep1.jpg,chicken_curry
         kep2.jpg,Pizza
         kep3.jpg,Csirkecurry
         gulyas.jpg,egyeb

A címke lehet Food-101 kulcs, angol vagy magyar név. Ami egyikkel sem egyezik
(pl. "egyeb", "gulyas"), azt Food-101-en kívüli ételnek vesszük – a szkript
kiírja ezeket, hogy az elgépelések kiderüljenek.

Futtatás:
    python scripts/evaluate_recognizer.py --images eval_images
    python scripts/evaluate_recognizer.py --images samples --threshold 0.5

Kimenet (eval_results/<időbélyeg>/): results.csv (képenként), summary.json, report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline import recognizer  # noqa: E402
from pipeline.food101_labels import (  # noqa: E402
    FOOD101_HU,
    FOOD101_LABELS,
    label_to_name_en,
)

DEFAULT_OUTPUT_DIR = config.ROOT_DIR / "eval_results"
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
IMAGE_SUFFIXES = {f".{t}" for t in config.ALLOWED_IMAGE_TYPES}


# --- Címkék ----------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Kisbetű, ékezetek nélkül, a szóköz/kötőjel/aláhúzás egységesen '_'."""
    decomposed = unicodedata.normalize("NFD", text.strip().lower())
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    for sep in (" ", "-"):
        plain = plain.replace(sep, "_")
    return plain


def _build_label_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for key in FOOD101_LABELS:
        for variant in (key, label_to_name_en(key), FOOD101_HU[key]):
            lookup[_normalize(variant)] = key
    return lookup


_LABEL_LOOKUP = _build_label_lookup()


def resolve_label(raw: str) -> str | None:
    """Címke -> Food-101 kulcs, vagy None, ha Food-101-en kívüli étel."""
    return _LABEL_LOOKUP.get(_normalize(raw))


# --- Minták beolvasása ---------------------------------------------------------------


@dataclass
class Sample:
    path: Path
    raw_label: str
    true_key: str | None  # None = Food-101-en kívüli étel


def collect_samples(folder: Path) -> list[Sample]:
    """A képek és címkéik: labels.csv-ből, ha van, különben az almappák nevéből."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"A mappa nem létezik: {folder}")

    labels_csv = folder / "labels.csv"
    pairs: list[tuple[Path, str]] = []
    if labels_csv.is_file():
        with labels_csv.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("file") and row.get("label"):
                    pairs.append((folder / row["file"].strip(), row["label"].strip()))
    else:
        for subdir in sorted(p for p in folder.iterdir() if p.is_dir()):
            for image in sorted(subdir.iterdir()):
                if image.suffix.lower() in IMAGE_SUFFIXES:
                    pairs.append((image, subdir.name))

    return [Sample(path, raw, resolve_label(raw)) for path, raw in pairs]


# --- Kiértékelés ---------------------------------------------------------------------


@dataclass
class Row:
    file: str
    label: str
    true_key: str | None
    in_food101: bool
    pred_key: str | None = None
    pred_score: float | None = None
    top3: list[str] = field(default_factory=list)
    top1_correct: bool | None = None
    top3_correct: bool | None = None
    flagged: bool | None = None  # a küszöb alatt volt -> a UI rákérdezne (F3)
    duration_ms: float | None = None
    error: str | None = None


def evaluate_samples(samples: list[Sample], threshold: float) -> list[Row]:
    rows = []
    for i, sample in enumerate(samples, start=1):
        row = Row(
            file=str(sample.path),
            label=sample.raw_label,
            true_key=sample.true_key,
            in_food101=sample.true_key is not None,
        )
        try:
            image, _ = recognizer.load_image(sample.path)
            start = time.perf_counter()
            rec = recognizer.recognize(image, threshold=threshold)
            row.duration_ms = round((time.perf_counter() - start) * 1000, 1)
            row.pred_key, row.pred_score = rec.key, rec.score
            row.top3 = [c.key for c in rec.top3]
            row.flagged = not rec.is_confident
            if row.in_food101:
                row.top1_correct = rec.key == sample.true_key
                row.top3_correct = sample.true_key in row.top3
        except Exception as exc:  # noqa: BLE001 – egy rossz kép ne állítsa le a mérést
            row.error = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        mark = "HIBA" if row.error else ("OK " if row.top1_correct else "   ")
        print(f"  [{i:3}/{len(samples)}] {mark} {Path(row.file).name}: {row.pred_key}")
    return rows


def _pct(part: int, total: int) -> float | None:
    return round(100 * part / total, 1) if total else None


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 3) if values else None


def summarize(rows: list[Row], threshold: float) -> dict:
    ok_rows = [r for r in rows if r.error is None]
    inside = [r for r in ok_rows if r.in_food101]
    outside = [r for r in ok_rows if not r.in_food101]
    correct = [r for r in inside if r.top1_correct]
    wrong = [r for r in inside if not r.top1_correct]
    accepted = [r for r in inside if not r.flagged]

    return {
        "threshold": threshold,
        "images_total": len(rows),
        "images_failed": len(rows) - len(ok_rows),
        "food101_images": len(inside),
        "other_images": len(outside),
        "top1_accuracy_pct": _pct(len(correct), len(inside)),
        "top3_accuracy_pct": _pct(sum(bool(r.top3_correct) for r in inside), len(inside)),
        "mean_confidence_correct": _mean([r.pred_score for r in correct]),
        "mean_confidence_wrong": _mean([r.pred_score for r in wrong]),
        "flagged_pct_food101": _pct(sum(bool(r.flagged) for r in inside), len(inside)),
        "accuracy_when_not_flagged_pct": _pct(
            sum(bool(r.top1_correct) for r in accepted), len(accepted)
        ),
        # a legrosszabb eset: hibás, mégis a küszöb fölött -> a UI nem figyelmeztetne
        "confident_but_wrong": len([r for r in wrong if not r.flagged]),
        "flagged_pct_other": _pct(sum(bool(r.flagged) for r in outside), len(outside)),
        "mean_duration_ms": _mean([r.duration_ms for r in ok_rows if r.duration_ms is not None]),
    }


def threshold_table(rows: list[Row], thresholds=THRESHOLDS) -> list[dict]:
    """Mi történne más küszöbbel? (a küszöb megválasztásának indoklásához)"""
    ok_rows = [r for r in rows if r.error is None]
    inside = [r for r in ok_rows if r.in_food101]
    outside = [r for r in ok_rows if not r.in_food101]
    table = []
    for t in thresholds:
        accepted = [r for r in inside if r.pred_score >= t]
        table.append(
            {
                "threshold": t,
                "flagged_pct_food101": _pct(len(inside) - len(accepted), len(inside)),
                "accuracy_when_not_flagged_pct": _pct(
                    sum(bool(r.top1_correct) for r in accepted), len(accepted)
                ),
                "flagged_pct_other": _pct(sum(r.pred_score < t for r in outside), len(outside)),
            }
        )
    return table


# --- Kimenet ---------------------------------------------------------------------------


def _fmt(value, suffix: str = "") -> str:
    return "–" if value is None else f"{value}{suffix}"


def build_report(summary: dict, table: list[dict], rows: list[Row], images_dir: Path) -> str:
    s = summary
    lines = [
        "# Ételfelismerés (ViT / Food-101) – pontosságmérés",
        "",
        f"- Időpont: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Modell: `{config.RECOGNIZER_MODEL}`",
        f"- Képek: `{images_dir}` – {s['images_total']} db "
        f"({s['food101_images']} Food-101, {s['other_images']} egyéb, "
        f"{s['images_failed']} hibás fájl)",
        f"- Bizonyossági küszöb: {s['threshold']:.0%}",
        "",
        "## Eredmények",
        "",
        "| Mérőszám | Érték |",
        "| --- | --- |",
        f"| Top-1 pontosság | {_fmt(s['top1_accuracy_pct'], '%')} |",
        f"| Top-3 pontosság | {_fmt(s['top3_accuracy_pct'], '%')} |",
        f"| Átlagos bizonyosság – helyes találat | {_fmt(s['mean_confidence_correct'])} |",
        f"| Átlagos bizonyosság – hibás találat | {_fmt(s['mean_confidence_wrong'])} |",
        f"| Küszöb alatti (rákérdez) – Food-101 képek | {_fmt(s['flagged_pct_food101'], '%')} |",
        f"| Pontosság, ha a küszöb fölött van | {_fmt(s['accuracy_when_not_flagged_pct'], '%')} |",
        f"| Hibás, mégis küszöb fölötti | {s['confident_but_wrong']} db |",
        f"| Küszöb alatti – Food-101-en kívüli ételek | {_fmt(s['flagged_pct_other'], '%')} |",
        f"| Átlagos futási idő képenként | {_fmt(s['mean_duration_ms'], ' ms')} |",
        "",
        "## A küszöb hatása",
        "",
        "| Küszöb | Rákérdez (Food-101) | Pontosság a küszöb fölött | Rákérdez (egyéb étel) |",
        "| --- | --- | --- | --- |",
    ]
    for t in table:
        lines.append(
            f"| {t['threshold']:.0%} | {_fmt(t['flagged_pct_food101'], '%')} | "
            f"{_fmt(t['accuracy_when_not_flagged_pct'], '%')} | "
            f"{_fmt(t['flagged_pct_other'], '%')} |"
        )

    wrong = [r for r in rows if r.in_food101 and r.error is None and not r.top1_correct]
    if wrong:
        lines += [
            "",
            "## Hibás felismerések",
            "",
            "| Kép | Valódi | Felismert | Bizonyosság | Top-3-ban? |",
        ]
        lines.append("| --- | --- | --- | --- | --- |")
        for r in wrong:
            lines.append(
                f"| {Path(r.file).name} | {r.true_key} | {r.pred_key} | {r.pred_score:.0%} | "
                f"{'igen' if r.top3_correct else 'nem'} |"
            )

    failed = [r for r in rows if r.error]
    if failed:
        lines += ["", "## Nem feldolgozható fájlok", ""]
        lines += [f"- {Path(r.file).name}: {r.error}" for r in failed]

    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, rows: list[Row], summary: dict, table: list[dict], report: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            data["top3"] = "|".join(row.top3)
            writer.writerow(data)
    (out_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "thresholds": table}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")


# --- Fő program --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViT / Food-101 pontosságmérés")
    parser.add_argument("--images", type=Path, default=config.SAMPLES_DIR, help="a képek mappája")
    parser.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD)
    parser.add_argument("--out", type=Path, default=None, help="kimeneti mappa")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)

    samples = collect_samples(args.images)
    if not samples:
        print(f"Nincs címkézett kép itt: {args.images} (almappák vagy labels.csv kell).")
        return 1

    unknown = sorted({s.raw_label for s in samples if s.true_key is None})
    print(
        f"{len(samples)} kép, {len(samples) - sum(s.true_key is None for s in samples)} Food-101."
    )
    if unknown:
        print(f"Food-101-en kívüliként kezelt címkék (elgépelés?): {', '.join(unknown)}")

    print("Modell betöltése...")
    recognizer._get_classifier()
    rows = evaluate_samples(samples, args.threshold)

    summary = summarize(rows, args.threshold)
    table = threshold_table(rows)
    report = build_report(summary, table, rows, args.images)

    out_dir = args.out or DEFAULT_OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    write_outputs(out_dir, rows, summary, table, report)

    print()
    print(report)
    print(f"Mentve: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
