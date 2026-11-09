# NutriLens – AI ételfelismerő és ajánló

Egyetemi projekt (Mesterséges intelligencia alapjai). A rendszer egy ételfotóból felismeri az ételt, strukturált elemzést készít róla, kiszámolja a tápértéket a megadott grammra, és hasonló vagy egészségesebb alternatívákat ajánl egy saját adatbázisból.

## A lánc

Három, egymás eredményére épülő, előre tanított modell (mind ingyenes, lokálisan, CPU-n is fut):

| Lépés | Modell | Bemenet | Kimenet |
| --- | --- | --- | --- |
| 1. Ételfelismerés | `nateraw/food` (ViT, Food-101) | kép | ételnév + bizonyosság, top-3 |
| 2. Elemzés | `Qwen/Qwen2.5-1.5B-Instruct` | ételnév + bizonyosság | validált JSON |
| 3. Ajánlás | `sentence-transformers/all-MiniLM-L6-v2` | a JSON-ból épített szöveg | top-5 hasonló étel |

A tápértéket nem az LLM adja: az SQLite-adatbázis 100 g-os értékeiből számoljuk.

## Telepítés és futtatás

```bash
pip install -r requirements.txt
python scripts/build_db.py
streamlit run app.py
```

Opcionálisan a modellek előre letölthetők (utána offline is működik):

```bash
python scripts/download_models.py
```

## Tesztek

```bash
pytest
```

## Felismerési pontosság mérése

A képeket almappákba kell tenni (a mappa neve az étel címkéje), vagy mellé egy `labels.csv` fájlt (`file,label`). A Food-101-en kívüli ételek mehetnek pl. egy `egyeb/` mappába.

```bash
python scripts/evaluate_recognizer.py --images eval_images
```

Az eredmény az `eval_results/<időbélyeg>/` mappába kerül: `report.md`, `results.csv`, `summary.json`.

## Projektstruktúra

```
app.py                  Streamlit belépési pont (C)
config.py               modellnevek, küszöbök, útvonalak (A)
nutrition.py            grammalapú tápérték-számítás (C)
pipeline/
  schemas.py            közös Pydantic modellek (A+B)
  recognizer.py         1. modell – ViT (A)
  analyzer.py           2. modell – Qwen, promptok, JSON-validálás (B)
  llm_backends.py       Transformers / Ollama (B)
  recommender.py        3. modell – MiniLM (C)
  orchestrator.py       a lánc összefűzése, időmérés, hibakezelés (A)
data/
  foods_seed.csv        tápérték- és leírásadatok (C)
  db.py                 SQLite-elérés (C)
scripts/
  build_db.py           CSV -> SQLite + beágyazások (C)
  download_models.py    modellek előletöltése (A)
tests/                  pytest, modulonként
samples/                tesztképek a demóhoz
```

## Közös interfész

```python
recognize(image) -> Recognition
analyze(recognition) -> DishAnalysis
recommend(analysis, mode, k=5) -> list[Recommendation]
compute_nutrition(dish_key, grams) -> Nutrition
```

Az adatmodellek a [pipeline/schemas.py](pipeline/schemas.py) fájlban vannak. Ezeket csak közös döntéssel változtatjuk.

## Csapat

| Szerep | Név | Modell |
| --- | --- | --- |
| A – Vision és integráció | Gaál Péter | ViT / Food-101 |
| B – LLM-elemzés | _név_ | Qwen2.5-1.5B |
| C – Ajánló, adat és UI | _név_ | MiniLM |
