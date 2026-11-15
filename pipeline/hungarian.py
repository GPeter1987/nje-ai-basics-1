"""Magyar megjelenítés (F13).  Felelős: B

A 2. modell elemzése angol marad, mert a MiniLM (3. modell) angol szöveggel dolgozik.
A felületen ezt egy ellenőrizhető angol -> magyar szótárral jelenítjük meg, nem újabb
AI-fordítással: a Qwen 1.5B magyarul használhatatlan szöveget ad (kipróbáltuk).
Amit a szótár nem ismer, az angolul marad, így nem jelenik meg kitalált fordítás.
"""

from __future__ import annotations

import re

from pipeline.schemas import DishAnalysis

# A szótártáblák kézzel tördeltek (soronként több pár), ezt a formázó ne bontsa szét.
# fmt: off
CATEGORIES_HU = {
    "main course": "főétel", "soup": "leves", "salad": "saláta", "dessert": "desszert",
    "side dish": "köret", "appetizer": "előétel", "breakfast": "reggeli",
    "sandwich": "szendvics", "other": "egyéb",
}

ALLERGENS_HU = {
    "gluten": "glutén", "crustaceans": "rákfélék", "eggs": "tojás", "fish": "hal",
    "peanuts": "földimogyoró", "soybeans": "szója", "milk": "tej", "tree nuts": "diófélék",
    "celery": "zeller", "mustard": "mustár", "sesame": "szezám",
    "sulphites": "kén-dioxid és szulfitok", "lupin": "csillagfürt", "molluscs": "puhatestűek",
}

DIETARY_TAGS_HU = {
    "vegetarian": "vegetáriánus", "vegan": "vegán", "gluten-free": "gluténmentes",
    "dairy-free": "tejmentes", "milk-free": "tejmentes", "lactose-free": "laktózmentes",
    "nut-free": "diómentes",
}

CUISINES_HU = {
    "american": "amerikai", "belgian": "belga", "british": "brit", "canadian": "kanadai",
    "chinese": "kínai", "french": "francia", "greek": "görög", "hungarian": "magyar",
    "indian": "indiai", "international": "nemzetközi", "italian": "olasz",
    "japanese": "japán", "korean": "koreai", "mediterranean": "mediterrán",
    "mexican": "mexikói", "middle eastern": "közel-keleti", "peruvian": "perui",
    "spanish": "spanyol", "thai": "thai", "turkish": "török", "vietnamese": "vietnámi",
    "german": "német", "austrian": "osztrák", "russian": "orosz", "polish": "lengyel",
    "lebanese": "libanoni", "moroccan": "marokkói", "indonesian": "indonéz",
    "malaysian": "maláj", "filipino": "filippínó", "brazilian": "brazil",
    "caribbean": "karibi", "cajun": "cajun", "swiss": "svájci", "asian": "ázsiai",
    "european": "európai", "southern": "amerikai déli", "tex-mex": "tex-mex",
    "italian-american": "olasz-amerikai", "eastern european": "kelet-európai",
    "scandinavian": "skandináv", "portuguese": "portugál", "egyptian": "egyiptomi",
    "israeli": "izraeli", "persian": "perzsa", "unknown": "ismeretlen",
}

COOKING_METHODS_HU = {
    "baked": "sütőben sült", "boiled": "főtt", "chilled": "hűtött", "cooked": "főzött",
    "fried": "sült", "deep-fried": "bő olajban sült", "deep fried": "bő olajban sült",
    "pan-fried": "serpenyőben sült", "fry": "sült", "frying": "sütés",
    "stir-fried": "pirított", "stir fried": "pirított", "sautéed": "pirított",
    "sauteed": "pirított", "grilled": "grillezett", "roasted": "sült", "raw": "nyers",
    "poached": "buggyantott", "steamed": "párolt", "stewed": "párolt",
    "simmered": "lassú tűzön főtt", "braised": "párolt", "seared": "hirtelen sütött",
    "toasted": "pirított", "smoked": "füstölt", "frozen": "fagyasztott",
    "pureed": "pürésített", "blended": "turmixolt", "mashed": "tört",
    "marinated": "pácolt", "slow-cooked": "lassan főtt", "slow cooked": "lassan főtt",
    "breaded": "panírozott", "glazed": "mázas", "whipped": "felvert",
    "assembled": "összeállított", "mixed": "kevert", "layered": "rétegezett",
    "served hot": "melegen tálalva", "served cold": "hidegen tálalva",
    "baking": "sütés", "boiling": "főzés", "grilling": "grillezés", "steaming": "párolás",
    "roasting": "sütés", "simmering": "lassú főzés",
}

FLAVORS_HU = {
    "savory": "sós-fűszeres", "savoury": "sós-fűszeres", "sweet": "édes",
    "spicy": "fűszeres", "hot": "csípős", "creamy": "krémes", "sour": "savanyú",
    "salty": "sós", "crispy": "ropogós", "crunchy": "roppanós", "smoky": "füstös",
    "fresh": "friss", "fruity": "gyümölcsös", "umami": "umami", "soft": "puha",
    "rich": "gazdag", "tangy": "pikáns", "meaty": "húsos", "cheesy": "sajtos",
    "buttery": "vajas", "herbaceous": "fűszernövényes", "herby": "fűszernövényes",
    "refreshing": "üdítő", "seasoned": "fűszerezett", "vinegary": "ecetes",
    "bitter": "keserű", "light": "könnyű", "nutty": "diós", "garlicky": "fokhagymás",
    "aromatic": "aromás", "mild": "enyhe", "tender": "omlós", "juicy": "szaftos",
    "earthy": "földes", "zesty": "citrusos", "citrusy": "citrusos", "tart": "fanyar",
    "chocolatey": "csokoládés", "sweet and sour": "édes-savanyú", "fluffy": "foszlós",
    "hearty": "laktató", "warm": "meleg", "cold": "hideg", "chewy": "rágós",
    "tomato-based": "paradicsomos", "noodle-based": "tésztás", "rice-based": "rizses",
    "flavorful": "ízletes", "delicate": "finom", "slow": "lassan", "stir": "kevert",
    # a DB címkéi közül
    "vegetarian": "vegetáriánus", "vegan": "vegán", "gluten-free": "gluténmentes",
    "dairy-free": "tejmentes",
}

INGREDIENTS_HU = {
    "almonds": "mandula", "apple": "alma", "avocado": "avokádó", "bacon": "bacon",
    "banana": "banán", "beans": "bab", "beef": "marhahús", "beet": "cékla",
    "bread": "kenyér", "broccoli": "brokkoli", "brown rice": "barna rizs",
    "butter": "vaj", "cabbage": "káposzta", "carrot": "sárgarépa", "celery": "zeller",
    "cheese": "sajt", "chicken": "csirke", "chickpea": "csicseriborsó",
    "chocolate": "csokoládé", "cinnamon": "fahéj", "clam": "kagyló", "cocoa": "kakaó",
    "coconut milk": "kókusztej", "cod": "tőkehal", "coffee": "kávé",
    "corn tortilla": "kukoricatortilla", "cornmeal": "kukoricadara", "crab": "rák",
    "cream": "tejszín", "cucumber": "uborka", "curry powder": "curryfűszer",
    "duck": "kacsa", "egg white": "tojásfehérje", "egg": "tojás", "eggs": "tojás",
    "feta": "feta sajt", "garlic": "fokhagyma", "goose liver": "libamáj",
    "lemon": "citrom", "lentil": "lencse", "lettuce": "salátalevél", "lobster": "homár",
    "milk": "tej", "mozzarella": "mozzarella", "mushroom": "gomba", "mussels": "kagyló",
    "mustard": "mustár", "oats": "zab", "octopus": "polip", "olive oil": "olívaolaj",
    "onion": "hagyma", "oysters": "osztriga", "pasta": "tészta", "peanuts": "földimogyoró",
    "peas": "zöldborsó", "pepper": "bors", "pork": "sertéshús", "potato": "burgonya",
    "quinoa": "quinoa", "raw beef": "nyers marhahús", "raw salmon": "nyers lazac",
    "raw tuna": "nyers tonhal", "raw fish": "nyers hal", "rice": "rizs",
    "rice noodles": "rizstészta", "ricotta": "ricotta", "salmon": "lazac",
    "sausage": "kolbász", "scallops": "fésűkagyló", "seaweed": "tengeri alga",
    "sesame": "szezám", "shrimp": "garnélarák", "snail": "csiga", "soy sauce": "szójaszósz",
    "soybeans": "szójabab", "spinach": "spenót", "squid": "tintahal",
    "strawberry": "eper", "sugar": "cukor", "tofu": "tofu", "tomato": "paradicsom",
    "tortilla": "tortilla", "tuna": "tonhal", "turkey": "pulyka", "walnuts": "dió",
    "water": "víz", "wheat flour": "búzaliszt", "flour": "liszt", "yogurt": "joghurt",
    "baking powder": "sütőpor", "basil": "bazsalikom", "bay leaves": "babérlevél",
    "beef patty": "marhahúspogácsa", "bell pepper": "kaliforniai paprika",
    "bell peppers": "kaliforniai paprika", "broth": "alaplé", "bun": "zsemle",
    "carrots": "sárgarépa", "chili peppers": "csilipaprika", "chili": "csili",
    "chorizo": "chorizo", "clams": "kagyló", "coconut": "kókusz",
    "coconut oil": "kókuszolaj", "coriander": "koriander", "corn": "kukorica",
    "corn tortillas": "kukoricatortilla", "cream cheese": "krémsajt",
    "croutons": "kenyérkocka", "cucumbers": "uborka", "cumin": "kömény",
    "dashi": "dashi alaplé", "dashi stock": "dashi alaplé", "dijon mustard": "dijoni mustár",
    "dill": "kapor", "dough": "tészta", "duck fat": "kacsazsír", "egg whites": "tojásfehérje",
    "egg yolk": "tojássárgája", "feta cheese": "feta sajt", "filo dough": "filotészta",
    "fish": "hal", "fish sauce": "halszósz", "garam masala": "garam masala",
    "garbanzo beans": "csicseriborsó", "ginger": "gyömbér", "green beans": "zöldbab",
    "ground beef": "darált marhahús", "gruyere cheese": "gruyère sajt", "guacamole": "guacamole",
    "ham": "sonka", "kale": "kelkáposzta", "ketchup": "ketchup", "kombu": "kombu alga",
    "lime": "lime", "lime juice": "limelé", "lime leaves": "kaffir-lime levél",
    "mayonnaise": "majonéz", "miso": "miszó", "miso paste": "miszópaszta",
    "mushrooms": "gomba", "noodles": "tészta", "nori": "nori alga", "nuts": "diófélék",
    "oil": "olaj", "olives": "olajbogyó", "kalamata olives": "kalamata olajbogyó",
    "onions": "hagyma", "red onion": "lilahagyma", "red onions": "lilahagyma",
    "parmesan": "parmezán", "parmesan cheese": "parmezán", "pickles": "savanyúság",
    "pistachios": "pisztácia", "potatoes": "burgonya", "red wine vinegar": "vörösborecet",
    "vinegar": "ecet", "ricotta cheese": "ricotta", "romaine lettuce": "római saláta",
    "saffron": "sáfrány", "salsa": "salsa", "salt": "só", "sauce": "szósz",
    "seafood": "tenger gyümölcsei", "seasoning": "fűszerek", "seasonings": "fűszerek",
    "sesame oil": "szezámolaj", "sesame seeds": "szezámmag", "shallots": "salottahagyma",
    "shredded cheese": "reszelt sajt", "sweetened condensed milk": "édesített sűrített tej",
    "syrup": "szirup", "maple syrup": "juharszirup", "tahini": "tahini", "tamarind": "tamarind",
    "thyme": "kakukkfű", "tomato sauce": "paradicsomszósz", "tomatoes": "paradicsom",
    "turmeric": "kurkuma", "vanilla extract": "vaníliakivonat", "vanilla": "vanília",
    "vegetables": "zöldségek", "vinegared rice": "ecetes rizs", "sushi rice": "szusirizs",
    "wasabi": "wasabi", "yeast": "élesztő", "zucchini": "cukkini", "eggplant": "padlizsán",
    "spaghetti": "spagetti", "pancetta": "pancetta", "bean sprouts": "babcsíra",
    "beef broth": "marhahúsleves-alaplé", "chicken broth": "csirkealaplé",
    "breadcrumbs": "zsemlemorzsa", "batter": "tésztabunda", "parsley": "petrezselyem",
    "oregano": "oregánó", "rosemary": "rozmaring", "cilantro": "friss koriander",
    "mint": "menta", "honey": "méz", "lemon juice": "citromlé", "lamb": "bárányhús",
    "veal": "borjúhús", "prosciutto": "prosciutto", "pepperoni": "pepperoni",
    "salami": "szalámi", "anchovies": "szardella", "sour cream": "tejföl",
    "paprika": "pirospaprika", "black pepper": "fekete bors", "white wine": "fehérbor",
    "red wine": "vörösbor", "wine": "bor", "chocolate chips": "csokoládédarabok",
    "cocoa powder": "kakaópor", "whipped cream": "tejszínhab", "heavy cream": "habtejszín",
    "mascarpone": "mascarpone", "ladyfingers": "babapiskóta", "espresso": "eszpresszó",
    "puff pastry": "leveles tészta", "pastry": "tészta", "pie crust": "pitetészta",
    "strawberries": "eper", "blueberries": "áfonya", "raisins": "mazsola",
    "walnut": "dió", "pecans": "pekándió", "cashews": "kesudió", "hazelnuts": "mogyoró",
    "peanut sauce": "mogyorószósz", "tortilla chips": "tortilla chips",
    "sour cream sauce": "tejfölös szósz", "gravy": "barna mártás",
    "cheddar": "cheddar", "cheddar cheese": "cheddar", "swiss cheese": "ementáli sajt",
    "goat cheese": "kecskesajt", "cream sauce": "tejszínes szósz",
    "protein": "fehérjeforrás", "meat": "hús", "salt and vinegar": "só és ecet",
}

# fmt: on

_ALL_TERMS = {
    **INGREDIENTS_HU,
    **FLAVORS_HU,
    **COOKING_METHODS_HU,
    **CUISINES_HU,
    **DIETARY_TAGS_HU,
    **ALLERGENS_HU,
    **CATEGORIES_HU,
}
_CONNECTORS = {"and": "és", "or": "vagy", "then": "majd"}


def term_hu(term: str, table: dict[str, str] | None = None) -> str:
    """Egy kifejezés magyarul; ismeretlen kifejezést változatlanul (angolul) hagy."""
    key = term.strip().casefold()
    lookup = table if table is not None else _ALL_TERMS
    if key in lookup:
        return lookup[key]
    # egyszerű többes szám: "tomatoes" -> "tomato", "olives" -> "olive"
    for suffix in ("es", "s"):
        if key.endswith(suffix) and key[: -len(suffix)] in lookup:
            return lookup[key[: -len(suffix)]]
    return term.strip()


def phrase_hu(text: str, table: dict[str, str] | None = None) -> str:
    """Rövid felsorolás ("boiled and simmered") szavanként; ha egy része ismeretlen,
    az eredeti angol szöveg marad (nem keverünk félig lefordított mondatot)."""
    table = table if table is not None else _ALL_TERMS
    whole = term_hu(text, table)
    if whole != text.strip():
        return whole
    parts = [p for p in re.split(r"\s*(,|\band\b|\bor\b|\bthen\b)\s*", text.strip()) if p]
    result = []
    for part in parts:
        if part == ",":
            result.append(",")
        elif part.casefold() in _CONNECTORS:
            result.append(_CONNECTORS[part.casefold()])
        else:
            translated = term_hu(part, table)
            if translated == part.strip():
                return text.strip()
            result.append(translated)
    text_hu = " ".join(result).replace(" ,", ",")
    # magyarul nincs vessző a "vagy" / "és" előtt: "főtt, párolt, vagy sült" -> "... vagy sült"
    return re.sub(r",\s+(és|vagy|majd)\b", r" \1", text_hu)


def list_hu(items: list[str], table: dict[str, str] | None = None) -> list[str]:
    """Lista fordítása; a fordítás után egyező elemek (pl. 'onion', 'onions') egyszer."""
    return list(dict.fromkeys(phrase_hu(item, table) for item in items))


def allergen_hu(allergen: str) -> str:
    return term_hu(allergen, ALLERGENS_HU)


def _is_hungarian(text: str, translated: set[str]) -> bool:
    """Csupa szótári magyar szóból áll-e (pl. 'főtt és lassú tűzön főtt')."""
    parts = [p for p in re.split(r"\s*(?:,|\bés\b|\bvagy\b|\bmajd\b)\s*", text) if p]
    return bool(parts) and all(part in translated for part in parts)


def analysis_hu(analysis: DishAnalysis) -> dict[str, object]:
    """Az elemzés mezői magyarul a felülethez, és egy sablonból épített magyar összegzés."""
    category = term_hu(analysis.category, CATEGORIES_HU)
    cuisine = term_hu(analysis.cuisine, CUISINES_HU)
    method = phrase_hu(analysis.cooking_method)
    ingredients = list_hu(analysis.main_ingredients)

    # Az összegzésbe csak lefordított rész kerül, hogy ne legyen félig angol mondat.
    translated = set(_ALL_TERMS.values())
    summary = []
    if cuisine in translated and cuisine not in ("ismeretlen", category):
        summary.append(f"{cuisine} {category}.")
    else:
        summary.append(f"{category}.")
    known = [item for item in ingredients if _is_hungarian(item, translated)]
    if known:
        summary.append("Jellemző összetevők: " + ", ".join(known[:6]) + ".")
    if _is_hungarian(method, translated):
        summary.append(f"Elkészítés: {method}.")
    text = " ".join(summary)
    return {
        "category": category,
        "cuisine": cuisine,
        "main_ingredients": ingredients,
        "cooking_method": method,
        "allergens": list_hu(analysis.allergens, ALLERGENS_HU),
        "flavor_profile": list_hu(analysis.flavor_profile),
        "dietary_tags": list_hu(analysis.dietary_tags, DIETARY_TAGS_HU),
        "summary": text[:1].upper() + text[1:],
    }
