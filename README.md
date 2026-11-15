# NutriLens – AI ételfelismerő és ajánló

Egyetemi projekt (Mesterséges intelligencia alapjai). A rendszer egy ételfotóból felismeri az ételt, strukturált elemzést készít róla, kiszámolja a tápértéket a megadott grammra, és hasonló, egészségesebb vagy allergénmentes alternatívákat ajánl egy saját adatbázisból.

## A lánc

Három, egymás eredményére épülő, előre tanított modell. Mind ingyenes, és lokálisan, CPU-n is fut:

| Lépés             | Modell                                   | Bemenet                         | Kimenet                                                                    |
| ----------------- | ---------------------------------------- | ------------------------------- | -------------------------------------------------------------------------- |
| 1. Ételfelismerés | `nateraw/food` (ViT, Food-101)           | kép                             | ételnév + bizonyosság, top-3                                               |
| 2. Elemzés        | `Qwen/Qwen2.5-1.5B-Instruct`             | ételnév + bizonyosság (+ top-3) | validált JSON: kategória, konyha, összetevők, allergének, ízprofil, leírás |
| 3. Ajánlás        | `sentence-transformers/all-MiniLM-L6-v2` | a JSON-ból épített szöveg       | top-5 hasonló étel, indoklással                                            |

A tápértéket nem az LLM adja: az SQLite-adatbázis 100 g-os értékeiből számoljuk. Ha a bizonyosság a küszöb alatt van, a felület megerősítést kér, és a felhasználó választhat a jelöltek közül (F3).

## Telepítés és futtatás

Python 3.11 vagy újabb kell.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_db.py
streamlit run app.py
```

A modellek előre is letölthetők; utána offline is működik (`HF_HUB_OFFLINE=1`):

```bash
python scripts/download_models.py               # ViT + MiniLM + Qwen (Transformers)
python scripts/download_models.py --llm ollama  # a Qwen helyett az Ollama-modellt húzza le
```

### LLM-backend (2. modell)

Az elemzés két úton futhat; a felület oldalsávjában váltható (F12):

- **Transformers** (alapértelmezett): a `Qwen/Qwen2.5-1.5B-Instruct` a Hugging Face-ről, bf16-ban, kb. 3 GB RAM. Az első elemzés a modellbetöltés miatt 1-2 percig tart.
- **Ollama**: a `qwen2.5:1.5b` egy helyben futó Ollama-szerveren. Kevesebb memóriát használ és gyorsabb CPU-n (egy elemzés kb. 14 s egy i5-8250U-n). Előtte: `ollama pull qwen2.5:1.5b`, és az Ollamának futnia kell. Ha nem fut, a felület ezt jelzi.

Az alapértelmezés környezeti változóval is átállítható: `NUTRILENS_LLM_BACKEND=ollama`.

Egy étel elemzése gyorsítótárba kerül. Ugyanaz az étel másodszorra azonnal megjelenik, de a két backend és a promptverziók gyorsítótára nem keveredik.

### Az elemzés pontossága (prompt v2)

A Qwen 1.5B saját allergénlistája megbízhatatlan volt: kitalált allergéneket adott, pl. szinte minden ételhez földimogyorót. Az összetevőlistája viszont jó. Ezért a modell az összetevőket adja, és az allergéneket ezekből szabályokkal képezzük (`pipeline/analyzer.py`, `INGREDIENT_ALLERGEN_RULES`). A prompt few-shot példákat tartalmaz, az Ollama pedig JSON-sémához kötve generál. Az összetevőknek ellentmondó diétás címkéket (pl. „vegán” csirkés curry) kiszűrjük.

Mérés 20 ételen, Ollama-backenddel, referenciaként az adatbázis allergénjeivel:

|           | érvényes JSON | pontos allergénegyezés | pontosság | fedés | átlagidő |
| --------- | ------------- | ---------------------- | --------- | ----- | -------- |
| v1 prompt | 20/20         | 0/20                   | 26%       | 35%   | 20,3 s   |
| v2 prompt | 19/20         | 10/19                  | 70%       | 80%   | 13,7 s   |

Az egy érvénytelen válasz („Phở” a „pho” kulcs helyett) azóta javítva. Az ajánló allergénszűrője ettől függetlenül az adatbázis allergénjeit is kizárja.

### Magyar megjelenítés (F13)

Az elemzés angol marad, mert a MiniLM angol szöveggel dolgozik. A felület ellenőrizhető angol → magyar szótárral jeleníti meg (`pipeline/hungarian.py`), nem AI-fordítással: a Qwen 1.5B magyarul használhatatlan szöveget ad. Amit a szótár nem ismer, az angolul marad. Az eredeti angol elemzés a felületen lenyitható panelben látható.

## Felismerési pontosság mérése

A képeket almappákba kell tenni (a mappa neve az étel címkéje), vagy mellé egy `labels.csv` fájlt (`file,label`). A Food-101-en kívüli ételek mehetnek pl. egy `egyeb/` mappába.

Az `eval_images/` mappában 60 szabad licencű kép van a Wikimedia Commonsról: 54 Food-101 étel és 6 magyar étel, ami nincs a Food-101-ben. A `samples/` mappában további 6 kép van a demóhoz. Ezek nem a modell tanítóképei. Szerző, licenc és forrás: `eval_images/ATTRIBUTION.csv`, `samples/ATTRIBUTION.csv`. A CC BY / CC BY-SA képek továbbadásakor a szerzőt fel kell tüntetni.

```bash
python scripts/evaluate_recognizer.py --images eval_images
```

Az eredmény az `eval_results/<időbélyeg>/` mappába kerül: `report.md`, `results.csv`, `summary.json`.

Eredmény az `eval_images/` képein: top-1 92,6%, top-3 98,1%, kb. 0,2–0,35 s képenként (CPU).

**Bizonyossági küszöb: 50%** (a specifikáció 40%-ot javasol). A mérés szerint 50%-nál a Food-101 ételeknél ugyanannyiszor kér megerősítést (1,9%), mint 40%-nál. A Food-101-en kívüli ételeket (pl. lángos, gulyás) viszont 16,7% helyett 50%-ban jelzi bizonytalannak, ilyenkor a felhasználó választhat. A felületen a küszöb csúszkával állítható.

| Küszöb | Rákérdez (Food-101) | Pontosság a küszöb fölött | Rákérdez (egyéb étel) |
| ------ | ------------------- | ------------------------- | --------------------- |
| 40%    | 1,9%                | 94,3%                     | 16,7%                 |
| 50%    | 1,9%                | 94,3%                     | 50,0%                 |
| 60%    | 3,7%                | 96,2%                     | 50,0%                 |

## Tesztek

```bash
pytest -m "not slow"     # gyors tesztek: ál-modellekkel, ideiglenes adatbázissal, internet nélkül
pytest -m slow           # valódi modellekkel, a samples/ képein (integrációs teszt)
ruff check . && ruff format --check .   # külön telepítendő: pip install ruff
```

A lassú tesztek Ollamával gyorsabbak (PowerShell: `$env:NUTRILENS_LLM_BACKEND="ollama"; pytest -m slow`).

Az ajánló részletes leírása, adatforrásai és kézi értékelése: [Ajánló, adatbázis és felület](docs/C_modul.md).

## Korlátok

- A felismerő csak a Food-101 101 ételét ismeri. Más ételre is a legközelebbit mondja, ezért van a küszöb és a kézi választás.
- A Qwen 1.5B kis modell: az összetevőlista néha általános („sauce”, „protein”) vagy hiányos, ilyenkor az allergénlista is az. **Az alkalmazás nem allergén-, orvosi vagy dietetikai tanácsadás.**
- Az adatbázis USDA-alapanyagokból számolt oktatási mintarecepteket tartalmaz, nem mért készételadatokat.

## Projektstruktúra

```
app.py                    Streamlit belépési pont: Elemzés / Pipeline / Előzmények (C)
config.py                 modellnevek, küszöbök, útvonalak (A)
nutrition.py              grammalapú tápérték-számítás (C)
pipeline/
  schemas.py              közös Pydantic modellek (A+B)
  recognizer.py           1. modell – ViT, top-3, kézi választás (A)
  food101_labels.py       a 101 étel kulcsa és magyar neve (A)
  analyzer.py             2. modell – Qwen, prompt, JSON-validálás, allergénszabályok (B)
  llm_backends.py         Transformers / Ollama (B)
  hungarian.py            az elemzés magyar megjelenítése (B)
  recommender.py          3. modell – MiniLM, szűrők, indoklás (C)
  orchestrator.py         a lánc összefűzése, időmérés, hibakezelés (A)
  stubs.py                helyettesítők a még hiányzó modulokhoz (A)
data/
  foods_seed.csv          tápérték- és leírásadatok (C)
  terms.py                magyar–angol szótár az ajánlóhoz (C)
  db.py                   SQLite-elérés (C)
scripts/
  build_db.py             CSV -> SQLite + beágyazások (C)
  download_models.py      modellek előletöltése (A)
  evaluate_recognizer.py  felismerési pontosság mérése (A)
  evaluate_recommender.py ajánlások kézi értékelése (C)
docs/                     részletes modulleírás
tests/                    pytest, modulonként
eval_images/              60 kiértékelő kép, forrásokkal
samples/                  tesztképek a demóhoz
```

## Közös interfész

```python
recognize(image) -> Recognition
analyze(recognition, backend=None) -> DishAnalysis
recommend(analysis, mode, k=5) -> list[Recommendation]
compute_nutrition(dish_key, grams) -> Nutrition
```

Az adatmodellek a [pipeline/schemas.py](pipeline/schemas.py) fájlban vannak. Ezeket csak közös döntéssel változtatjuk.

## Csapat

| Szerep                   | Név            | Modell         |
| ------------------------ | -------------- | -------------- |
| A – Vision és integráció | Gaál Péter     | ViT / Food-101 |
| B – LLM-elemzés          | Nagy Zsolt     | Qwen2.5-1.5B   |
| C – Ajánló, adat és UI   | Szarvas Attila | MiniLM         |
