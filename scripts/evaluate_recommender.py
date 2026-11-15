"""10 étel top-5 ajánlásának exportja kézi relevanciaértékeléshez.

python scripts/evaluate_recommender.py
python scripts/evaluate_recommender.py --ratings eval_results/recommender/ratings.csv

A bemenetek referenciaelemzések a seedből, nem valódi Qwen-válaszok.
A relevant oszlopot ember tölti ki (1=releváns, 0=nem releváns). Nem gyártunk mérési eredményt.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.db import get_food  # noqa: E402
from pipeline.recommender import recommend  # noqa: E402
from pipeline.schemas import DishAnalysis  # noqa: E402

EVALUATION_DISHES = (
    "chicken_curry", "pizza", "sushi", "hamburger", "caesar_salad", "ramen",
    "spaghetti_bolognese", "grilled_salmon", "apple_pie", "falafel",
)


def export(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    target = output / "ratings.csv"
    if target.exists():
        raise FileExistsError(f"Nem írjuk felül a korábbi értékelést: {target}")
    rows, inputs = [], []
    for key in EVALUATION_DISHES:
        food = get_food(key)
        if food is None:
            raise LookupError(key)
        analysis = DishAnalysis(
            dish=key, category=food["category"], cuisine=food["cuisine"],
            main_ingredients=food["ingredients"], cooking_method=food["tags"][0],
            allergens=food["allergens"], flavor_profile=food["tags"][1:],
            dietary_tags=[], description=food["description_en"], source="fallback",
        )
        inputs.append(analysis.model_dump())
        for rank, item in enumerate(recommend(analysis), 1):
            rows.append({
                "dish": key, "rank": rank, "recommendation": item.key,
                "name_hu": item.name_hu, "similarity": round(item.similarity, 4),
                "reason": item.reason, "relevant": "", "note": "",
            })
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "reference_analyses.json").write_text(
        json.dumps(inputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return len(rows)


def summarize(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50 or {r["dish"] for r in rows} != set(EVALUATION_DISHES):
        raise ValueError("10 referenciaételhez összesen 50 ajánlás szükséges.")
    per_dish = {}
    for dish in EVALUATION_DISHES:
        group = [r for r in rows if r["dish"] == dish]
        if len(group) != 5 or {r["rank"] for r in group} != {"1", "2", "3", "4", "5"}:
            raise ValueError(f"Hibás top-5: {dish}")
        if any(r["relevant"] not in ("", "0", "1") for r in group):
            raise ValueError("A relevant mező csak üres, 0 vagy 1 lehet.")
        completed = [int(r["relevant"]) for r in group if r["relevant"]]
        per_dish[dish] = {
            "rated": len(completed),
            "precision_at_5": sum(completed) / 5 if len(completed) == 5 else None,
        }
    complete = all(item["rated"] == 5 for item in per_dish.values())
    return {
        "rated": sum(item["rated"] for item in per_dish.values()), "total": 50,
        "mean_precision_at_5": sum(item["precision_at_5"] for item in per_dish.values()) / 10
        if complete else None,
        "per_dish": per_dish,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("eval_results/recommender"))
    parser.add_argument("--ratings", type=Path)
    args = parser.parse_args()
    try:
        if args.ratings:
            print(json.dumps(summarize(args.ratings), ensure_ascii=False, indent=2))
        else:
            print(f"{export(args.output)} ajánlás exportálva: {args.output / 'ratings.csv'}")
    except Exception as exc:
        print(f"Kiértékelési hiba: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
