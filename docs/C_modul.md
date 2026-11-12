# C – Ajánló, adatbázis és felület

## Indítás Windows PowerShellből

```powershell
cd C:\Projects\AI\nje-ai-basics-1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/build_db.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Új gépen előbb `python -m venv .venv`. Python 3.11 vagy újabb szükséges.
A modellek első használatkor a Hugging Face gyorsítótárába töltődnek le.
Utána az adatbázis és a MiniLM helyi fájlokból működik; az ellenőrzött offline
futtatáshoz állítsd a `HF_HUB_OFFLINE=1` környezeti változót.
Csak a MiniLM szükséges a C-részhez; a fotó felismerése a ViT-modellt is letölti.

## Elkészült követelmények

- **F4:** 250 g alapadag, 10–2000 g határok.
- **F6:** adatbázis-alapú arányosítás, egész kcal és egytizedes makrók.
- **F7–F8:** MiniLM, koszinusz top-5, saját étel kizárása, szigorúan kevesebb
  kcal/100 g mód, a Qwen-elemzés allergénjeiből épített kizárás, indoklás.
- **F10–F11:** kifejezett gombnyomásra mentés, előzménytábla és dátum szerinti
  napi összeg. A diagram a makrókat mutatja; a napösszeg nem a 200 soros UI-limitből számol.
- Magyar Streamlit-felület: Elemzés / Pipeline / Előzmények, kártyák,
  diagramok, kézi megerősítés az A modul meglévő API-ján keresztül.
- 201 étel, mind a 101 Food-101-kulcs és 100 további recept; angol keresési
  leírás, magyar megjelenítési név, SQLite-ban 384 darab float32/BLOB vektor.

## Adatok és források – fontos korlát

A `data/foods_seed.csv` **oktatási mintarecepteket** tartalmaz, nem 201 készétel
laboratóriumban mért vagy hitelesített USDA-rekordját. A mennyiségek saját,
egyszerűsített receptarányok. Az alapanyagok négy tápértékét valódi USDA SR28
rekordokból vettük át. Minden sor forrásmezőjében szerepel az alapanyag
**NDB-azonosítója és a 100 g-os receptben használt grammja**.

- USDA-adatok tükre: https://github.com/alyssaq/usda-sqlite/tree/master/data
- Eredeti kiadás: USDA National Nutrient Database for Standard Reference, Release 28.
- USDA/FoodData Central adatletöltés: https://fdc.nal.usda.gov/download-datasets/
- A projektben használt részhalmaz: `data/usda_ingredients.json` (NDB, eredeti
  angol megnevezés, kcal/fehérje/szénhidrát/zsír 100 g-ra).
- A receptarányok: `scripts/generate_seed.py`. Újragenerálás:
  `python scripts/generate_seed.py`; ehhez nincs hálózat vagy LLM.
- A USDA-pillanatkép újrakinyerése: `python scripts/prepare_usda.py --directory
  <FOOD_DES.txt és NUT_DATA.txt mappája>`.

A tálalt összetevők súlyozott átlaga adja a készétel tápértékét. A készétel
főzési vesztesége, szószai, termékspecifikus összetétele nem pontosan modellezett.
Vannak közelítő alapanyag-helyettesítések is (például csirkemell a csirkeételekhez,
darált marha a marhaételekhez). Leadás előtt a mintarecepteket érdemes kézzel
ellenőrizni/pontosítani, vagy készételre vonatkozó hiteles forrásra cserélni.

Az allergének a mintareceptek összetevőiből, kézzel megadott EU14-kulcsokból
származnak, **nem a USDA tápértéktáblájából**. A vegán/vegetáriánus és
glutén-/tejmentes címkék kizárólag erre a mintareceptre érvényesek.
Keresztszennyeződést és eltérő gyártói összetételt nem garantálnak.

A 4/4/9 kcal-ellenőrzés a banánfagylalt és a párolt brokkoli esetén >15%-os
eltérést jelez. Az eredeti USDA-kcal értéket nem módosítjuk a teszt kedvéért:
a rost és az élelmiszerspecifikus Atwater-faktorok miatt a két számítás eltérhet.
Az eltéréseket az építő figyelmeztetésként kiírja, minden tápérték forrása megmarad.

## Kapcsolódás A és B munkájához

A közös `pipeline/schemas.py`, `config.py`, `recognizer.py`, `orchestrator.py`,
`analyzer.py` és `llm_backends.py` változatlan maradt. A C implementációi
ugyanazt az interfészt használják:

```python
recommend(analysis, mode="similar", k=5)  # list[Recommendation]
compute_nutrition(dish_key, grams)       # Nutrition
```

**B elemzője jelenleg NotImplementedError-t dob.** Az A orchestratora ilyenkor
jelölt tartalék elemzéssel folytatja a láncot. A felület ezt feltűnően jelzi;
ez nem bizonyít valódi hárommodelles működést. Allergénmentes módot tartalék
elemzésből szándékosan nem engedünk. A backend kiválasztásához a B-modul API-ja
szükséges; ezért a UI jelenleg csak kiírja a konfigurált backendet.

B használhatja a következő kész adatbázis-függvényeket:

```python
from data.db import get_food, get_analysis, save_analysis

food = get_food("chicken_curry")           # dict, a JSON-listák már listák; nincs találat: None
cached = get_analysis("chicken_curry", "qwen:model/backend/prompt-version")
save_analysis(analysis, "qwen:model/backend/prompt-version")
```

A cache kulcsa az étel + modellazonosító, a visszaadott elemzés `source="cache"`.
Promptváltozáskor a modellkulcsot is verziózni kell. A tartalék elemzéshez B a
`foods` kategória/konyha/összetevő/allergén/leírás mezőit használhatja.

Az allergénlista angol kulcsokat tartalmaz: `gluten`, `crustaceans`, `eggs`,
`fish`, `peanuts`, `soybeans`, `milk`, `tree nuts`, `celery`, `mustard`, `sesame`,
`sulphites`, `lupin`, `molluscs`. A gyakori magyar neveket és szinonimákat a
`data/terms.py` egységesíti. Ismeretlen allergénnél a szűrés hibával leáll,
nem enged át bizonytalan ételt. Kevesebb mint öt találatnál nem lazít a szűrőn.

A MiniLM keresési szövege a **kapott DishAnalysis strukturált mezőiből** készül.
A magyar szavakat ellenőrizhető szótár fordítja angolra. Ez nem általános fordító:
új magyar összetevőkhöz a szótárt bővíteni kell; ismeretlen ékezetes mezők
kimaradnak. Legjobb integráció: B angol kanonikus mezőket ad, a UI fordítva jeleníthet meg.

## Adatbázis és hibakezelés

- Három tábla: `foods`, `analysis_cache`, `history`.
- A foods vektoraihoz `embedding_model` tartozik; eltérő modell vagy sérült,
  hiányzó, nem véges/nulla vektor esetén újraépítést kérünk.
- `build_db.py` előbb validál és beágyaz, utána egy tranzakcióban cseréli a
  foods táblát. Az előzmények és cache nem törlődnek. Nem kell DB-fájlt commitolni.
- Modell nélküli fejlesztés: `python scripts/build_db.py --without-embeddings`.
  Ebben az esetben a tápérték és az előzmények működnek, **az ajánló nem**;
  később a teljes építést kell lefuttatni.
- A `Nutrition.is_estimate` kizárólag LLM-tartalékbecslést jelent a közös
  szerződés szerint. Az adatbázis mintarecept-eredetét a felület külön jelzi.
- A felület az adag/mód/kép/kézi választás változásakor érvényteleníti az előző
  eredményt; egy eredményt újrafuttatás nélkül nem ment kétszer automatikusan.

## Tesztelés és kézi értékelés

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not slow"
.\.venv\Scripts\python.exe -m pytest tests/test_c_pipeline.py -m slow
.\.venv\Scripts\python.exe scripts/evaluate_recommender.py
```

A gyors tesztek ideiglenes DB-vel és ál-beágyazóval futnak, internet nélkül.
Van teszt a szűrőkre, koszinusz-rangsorra, kerekítésre, forrásokra, tranzakcióra,
előzményekre, cache-re és a Streamlit feltöltés–megerősítés–mentés folyamatára.
Az eredeti orchestrator-stubtesztek explicit szimulálják a még hiányzó
C-modulokat; már nem függnek attól, hogy létezik-e a valódi adatbázis.

Az export a tíz referenciaételhez 50 ajánlást és a bemeneti referenciaelemzéseket
írja az `eval_results/recommender/` mappába. A `ratings.csv` **relevant** mezőjét
ember tölti ki: 1 vagy 0, opcionális indoklással a note mezőben. Összesítés:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_recommender.py --ratings eval_results/recommender/ratings.csv
```

A precision@5 csak teljesen kitöltött ételnél, az átlag csak mind az 50 értékelés
után jelenik meg. **A kézi relevanciaértékelés még elvégzendő**, az export nem
igazolja azt, és a három AI-modell teljes láncának kiértékelését sem helyettesíti.

### Elvégzett ellenőrzések (2026. november 12.)

- Python 3.13 / Windows: **129 gyors teszt sikeres**.
- Valódi MiniLM-mel DB-építés és top-5 keresés: **1 lassú teszt sikeres**,
  külön `HF_HUB_OFFLINE=1` módban is.
- Ruff és `git diff --check`: sikeres (Git csak Windows-sorvég figyelmeztetést adott).
- Élő Streamlit böngészőpróba Edge-ben: három fül és napi összesítő megjelent,
  Streamlit-kivétel nélkül. A szervert a próba után leállítottuk.
- Az 50 ajánlás értékelőlapja elkészült; emberi minősítést nem írtunk bele.
- A teljes ViT–Qwen–MiniLM láncot valódi Qwennel nem teszteltük, mert B modulja hiányzik.
