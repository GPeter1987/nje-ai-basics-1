"""Az USDA SR28 négy tápanyagának kinyerése ellenőrizhető helyi pillanatképbe.

Egyszeri adat-előkészítés, az alkalmazás futtatásához nem szükséges.
python scripts/prepare_usda.py --directory <FOOD_DES.txt és NUT_DATA.txt mappája>
Forrás: https://github.com/alyssaq/usda-sqlite/tree/master/data (USDA SR28)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Belső angol összetevő, USDA NDB-szám, EU-allergének.
INGREDIENTS = [
    ("chicken", "05064", []), ("beef", "23563", []),
    ("raw beef", "23557", []), ("pork", "10022", []), ("duck", "05142", []),
    ("turkey", "05220", []), ("salmon", "15237", ["fish"]),
    ("raw salmon", "15236", ["fish"]), ("tuna", "15121", ["fish"]),
    ("raw tuna", "15117", ["fish"]), ("cod", "15016", ["fish"]),
    ("shrimp", "15271", ["crustaceans"]), ("lobster", "15148", ["crustaceans"]),
    ("crab", "15137", ["crustaceans"]), ("squid", "15175", ["molluscs"]),
    ("clam", "15159", ["molluscs"]), ("mussels", "15165", ["molluscs"]),
    ("oysters", "15167", ["molluscs"]), ("scallops", "15172", ["molluscs"]),
    ("snail", "90560", ["molluscs"]), ("octopus", "15230", ["molluscs"]),
    ("goose liver", "05150", []), ("bacon", "10860", []),
    ("sausage", "07022", []), ("cheese", "01009", ["milk"]),
    ("ricotta", "01037", ["milk"]), ("mozzarella", "01026", ["milk"]),
    ("feta", "01019", ["milk"]), ("eggs", "01129", ["eggs"]),
    ("egg white", "01124", ["eggs"]), ("milk", "01077", ["milk"]),
    ("cream", "01053", ["milk"]), ("yogurt", "01116", ["milk"]),
    ("rice", "20045", []), ("brown rice", "20037", []),
    ("pasta", "20121", ["gluten"]), ("rice noodles", "20134", []),
    ("bread", "18069", ["gluten", "milk"]),
    ("wheat flour", "20081", ["gluten"]), ("tortilla", "18364", ["gluten"]),
    ("corn tortilla", "18363", []), ("potato", "11365", []),
    ("tomato", "11529", []), ("onion", "11282", []), ("carrot", "11124", []),
    ("broccoli", "11091", []), ("spinach", "11457", []),
    ("mushroom", "11260", []), ("beet", "11081", []),
    ("cucumber", "11205", []), ("cabbage", "11109", []), ("lettuce", "11251", []),
    ("chickpea", "16057", []), ("lentil", "16070", []), ("beans", "16033", []),
    ("tofu", "16426", ["soybeans"]), ("soybeans", "11451", ["soybeans"]),
    ("olive oil", "04053", []), ("butter", "01145", ["milk"]),
    ("sugar", "19335", []), ("apple", "09003", []), ("banana", "09040", []),
    ("strawberry", "09316", []), ("avocado", "09037", []),
    ("coconut milk", "12117", []), ("almonds", "12061", ["tree nuts"]),
    ("walnuts", "12155", ["tree nuts"]), ("peanuts", "16390", ["peanuts"]),
    ("sesame", "12023", ["sesame"]), ("chocolate", "19904", []),
    ("cocoa", "19165", []), ("oats", "08120", ["gluten"]),
    ("quinoa", "20137", []), ("cornmeal", "20020", []),
    ("peas", "11305", []), ("pepper", "11821", []), ("water", "14411", []),
    ("curry powder", "02015", []), ("cinnamon", "02010", []),
    ("coffee", "14180", []), ("garlic", "11215", []), ("lemon", "09150", []),
    ("seaweed", "11669", []), ("mustard", "02046", ["mustard"]),
    ("soy sauce", "16123", ["soybeans", "gluten"]), ("celery", "11143", ["celery"]),
]


def prepare(directory: Path, output: Path) -> None:
    def rows(name):
        with (directory / name).open(encoding="latin-1", newline="") as handle:
            yield from csv.reader(handle, delimiter="^", quotechar="~")

    descriptions = {r[0]: r[2] for r in rows("FOOD_DES.txt")}
    selected = {code for _, code, _ in INGREDIENTS}
    nutrients = {code: {} for code in selected}
    columns = {"208": "kcal", "203": "protein", "205": "carbs", "204": "fat"}
    for row in rows("NUT_DATA.txt"):
        if row[0] in selected and row[1] in columns:
            nutrients[row[0]][columns[row[1]]] = float(row[2])
    result = {}
    for name, code, allergens in INGREDIENTS:
        if set(nutrients[code]) != set(columns.values()):
            raise ValueError(f"Hiányos USDA-rekord: {code}")
        result[name] = {
            "ndb": code, "description": descriptions[code], "allergens": allergens,
            **nutrients[code],
        }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]
                        / "data" / "usda_ingredients.json")
    args = parser.parse_args()
    prepare(args.directory, args.output)
