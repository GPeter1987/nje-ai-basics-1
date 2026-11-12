"""Ellenőrizhető magyar–angol szótár a MiniLM-hez, nem újabb AI-modell.

Az adatbázis összetevői, allergénjei és címkéi angol, kanonikus kulcsok.
Ismeretlen magyar kifejezést nem próbálunk kitalált fordítással helyettesíteni.
"""

TERMS = {
    "főétel": "main course", "leves": "soup", "saláta": "salad",
    "desszert": "dessert", "köret": "side dish", "előétel": "appetizer",
    "reggeli": "breakfast", "szendvics": "sandwich", "ismeretlen": "",
    "indiai": "indian", "olasz": "italian", "magyar": "hungarian",
    "francia": "french", "amerikai": "american", "mexikói": "mexican",
    "japán": "japanese", "kínai": "chinese", "thai": "thai",
    "vietnámi": "vietnamese", "koreai": "korean", "görög": "greek",
    "mediterrán": "mediterranean", "nemzetközi": "international",
    "párolt": "steamed", "főtt": "boiled", "sült": "baked",
    "grillezett": "grilled", "rántott": "breaded fried", "nyers": "raw",
    "pirított": "stir fried", "bő olajban sült": "deep fried",
    "fűszeres": "spicy", "csípős": "hot", "krémes": "creamy",
    "édes": "sweet", "sós": "savory", "savanyú": "sour",
    "ropogós": "crispy", "füstös": "smoky", "friss": "fresh",
    "gyümölcsös": "fruity", "umami": "umami",
    "vegán": "vegan", "vegetáriánus": "vegetarian",
    "gluténmentes": "gluten-free", "tejmentes": "dairy-free",
    "csirkehús": "chicken", "csirke": "chicken", "marhahús": "beef",
    "sertéshús": "pork", "kacsahús": "duck", "hal": "fish",
    "lazac": "salmon", "tonhal": "tuna", "rák": "crustaceans",
    "garnélarák": "shrimp", "homár": "lobster", "kagyló": "molluscs",
    "tintahal": "squid", "tojás": "eggs", "tojásfehérje": "egg white",
    "tej": "milk", "tejtermék": "milk", "tejszín": "cream",
    "sajt": "cheese", "joghurt": "yogurt", "vaj": "butter",
    "kókusztej": "coconut milk", "hagyma": "onion", "vöröshagyma": "onion",
    "fokhagyma": "garlic", "paradicsom": "tomato", "paprika": "pepper",
    "burgonya": "potato", "krumpli": "potato", "sárgarépa": "carrot",
    "répa": "carrot", "brokkoli": "broccoli", "spenót": "spinach",
    "gomba": "mushroom", "cékla": "beet", "uborka": "cucumber",
    "káposzta": "cabbage", "saláta levél": "lettuce", "salátalevél": "lettuce",
    "rizs": "rice", "barna rizs": "brown rice", "tészta": "pasta",
    "rizstészta": "rice noodles", "kenyér": "bread", "liszt": "wheat flour",
    "búzaliszt": "wheat flour", "zab": "oats", "zabpehely": "oats",
    "kukorica": "corn", "kukoricadara": "cornmeal", "quinoa": "quinoa",
    "csicseriborsó": "chickpea", "lencse": "lentil", "bab": "beans",
    "zöldborsó": "peas", "szója": "soybeans", "tofu": "tofu",
    "alma": "apple", "banán": "banana", "eper": "strawberry",
    "avokádó": "avocado", "citrom": "lemon", "olívaolaj": "olive oil",
    "olaj": "oil", "cukor": "sugar", "méz": "honey",
    "kakaó": "cocoa", "csokoládé": "chocolate", "kávé": "coffee",
    "curry fűszerkeverék": "curry powder", "curry": "curry powder",
    "szezámmag": "sesame", "szezám": "sesame", "földimogyoró": "peanuts",
    "dió": "walnuts", "mandula": "almonds", "diófélék": "tree nuts",
    "glutén": "gluten", "búza": "gluten", "mustár": "mustard",
    "zeller": "celery", "csillagfürt": "lupin", "szulfitok": "sulphites",
    "puhatestűek": "molluscs", "rákfélék": "crustaceans",
}

ALLERGEN_ALIASES = {
    "wheat": "gluten", "wheat flour": "gluten", "barley": "gluten",
    "rye": "gluten", "oats": "gluten", "egg": "eggs", "dairy": "milk",
    "soy": "soybeans", "soya": "soybeans", "nuts": "tree nuts",
    "walnuts": "tree nuts", "almonds": "tree nuts", "shrimp": "crustaceans",
    "crab": "crustaceans", "lobster": "crustaceans", "squid": "molluscs",
    "mussels": "molluscs", "oysters": "molluscs", "sulfites": "sulphites",
}

EU_ALLERGENS = frozenset({
    "gluten", "crustaceans", "eggs", "fish", "peanuts", "soybeans", "milk",
    "tree nuts", "celery", "mustard", "sesame", "sulphites", "lupin", "molluscs",
})


def english(term: str) -> str:
    term = term.strip().lower()
    if term in TERMS:
        return TERMS[term]
    # Az ismeretlen, ékezetes kifejezéseket kihagyjuk, nem adjuk angolként a modellnek.
    return term if term.isascii() else ""


def allergen_key(term: str) -> str:
    translated = english(term)
    return ALLERGEN_ALIASES.get(translated, translated or term.strip().lower())
