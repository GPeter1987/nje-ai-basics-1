"""
A Food-101 adathalmaz 101 osztálya és a megjelenítendő nevek.

A kulcsok pontosan megegyeznek a `nateraw/food` modell címkéivel
(config.json -> id2label), és ugyanezek a kulcsok szerepelnek az adatbázis
`foods.key` oszlopában is.

A magyar nevek a felületen jelennek meg (F13). Ahol nincs bevett magyar név,
ott a nemzetközileg ismert nevet használjuk (pl. "Sushi", "Ramen").
"""

# Kulcs -> magyar név (a modell címkesorrendjében)
FOOD101_HU: dict[str, str] = {
    "apple_pie": "Almás pite",
    "baby_back_ribs": "Sertésoldalas (baby back ribs)",
    "baklava": "Baklava",
    "beef_carpaccio": "Marhacarpaccio",
    "beef_tartare": "Marhatatár",
    "beet_salad": "Céklasaláta",
    "beignets": "Beignet (francia fánk)",
    "bibimbap": "Bibimbap",
    "bread_pudding": "Kenyérpuding",
    "breakfast_burrito": "Reggeli burrito",
    "bruschetta": "Bruschetta",
    "caesar_salad": "Cézársaláta",
    "cannoli": "Cannoli",
    "caprese_salad": "Caprese saláta",
    "carrot_cake": "Répatorta",
    "ceviche": "Ceviche",
    "cheese_plate": "Sajttál",
    "cheesecake": "Sajttorta",
    "chicken_curry": "Csirkecurry",
    "chicken_quesadilla": "Csirkés quesadilla",
    "chicken_wings": "Csirkeszárny",
    "chocolate_cake": "Csokoládétorta",
    "chocolate_mousse": "Csokoládéhab",
    "churros": "Churros",
    "clam_chowder": "Kagylóleves (clam chowder)",
    "club_sandwich": "Klubszendvics",
    "crab_cakes": "Rákfasírt",
    "creme_brulee": "Crème brûlée",
    "croque_madame": "Croque madame",
    "cup_cakes": "Cupcake",
    "deviled_eggs": "Töltött tojás",
    "donuts": "Fánk (donut)",
    "dumplings": "Gombóc (dumpling)",
    "edamame": "Edamame",
    "eggs_benedict": "Benedek-tojás",
    "escargots": "Csiga (escargot)",
    "falafel": "Falafel",
    "filet_mignon": "Bélszínérme (filet mignon)",
    "fish_and_chips": "Fish and chips",
    "foie_gras": "Libamáj",
    "french_fries": "Hasábburgonya",
    "french_onion_soup": "Francia hagymaleves",
    "french_toast": "Bundás kenyér",
    "fried_calamari": "Rántott tintahal",
    "fried_rice": "Pirított rizs",
    "frozen_yogurt": "Fagyasztott joghurt",
    "garlic_bread": "Fokhagymás kenyér",
    "gnocchi": "Gnocchi",
    "greek_salad": "Görög saláta",
    "grilled_cheese_sandwich": "Sajtos melegszendvics",
    "grilled_salmon": "Grillezett lazac",
    "guacamole": "Guacamole",
    "gyoza": "Gyoza",
    "hamburger": "Hamburger",
    "hot_and_sour_soup": "Csípős-savanyú leves",
    "hot_dog": "Hot dog",
    "huevos_rancheros": "Huevos rancheros",
    "hummus": "Humusz",
    "ice_cream": "Fagylalt",
    "lasagna": "Lasagne",
    "lobster_bisque": "Homárkrémleves",
    "lobster_roll_sandwich": "Homáros szendvics",
    "macaroni_and_cheese": "Sajtos makaróni",
    "macarons": "Macaron",
    "miso_soup": "Miso leves",
    "mussels": "Fekete kagyló",
    "nachos": "Nachos",
    "omelette": "Omlett",
    "onion_rings": "Hagymakarika",
    "oysters": "Osztriga",
    "pad_thai": "Pad thai",
    "paella": "Paella",
    "pancakes": "Amerikai palacsinta",
    "panna_cotta": "Panna cotta",
    "peking_duck": "Pekingi kacsa",
    "pho": "Pho leves",
    "pizza": "Pizza",
    "pork_chop": "Sertéskaraj",
    "poutine": "Poutine",
    "prime_rib": "Marhaborda (prime rib)",
    "pulled_pork_sandwich": "Tépett sertéses szendvics",
    "ramen": "Ramen",
    "ravioli": "Ravioli",
    "red_velvet_cake": "Red velvet torta",
    "risotto": "Rizottó",
    "samosa": "Szamosza",
    "sashimi": "Sashimi",
    "scallops": "Fésűkagyló",
    "seaweed_salad": "Tengerihínár-saláta",
    "shrimp_and_grits": "Garnélarák kukoricadarával",
    "spaghetti_bolognese": "Bolognai spagetti",
    "spaghetti_carbonara": "Carbonara spagetti",
    "spring_rolls": "Tavaszi tekercs",
    "steak": "Steak",
    "strawberry_shortcake": "Epres piskóta (shortcake)",
    "sushi": "Sushi",
    "tacos": "Taco",
    "takoyaki": "Takoyaki",
    "tiramisu": "Tiramisu",
    "tuna_tartare": "Tonhaltatár",
    "waffles": "Gofri",
}

# A 101 kulcs, a modell címkesorrendjében
FOOD101_LABELS: tuple[str, ...] = tuple(FOOD101_HU)

# Néhány angol név, ahol a puszta átalakítás ("cup_cakes" -> "Cup cakes") furcsa lenne
_EN_OVERRIDES: dict[str, str] = {
    "cup_cakes": "Cupcakes",
    "creme_brulee": "Crème brûlée",
    "pho": "Phở",
}


def label_to_name_en(label: str) -> str:
    """Címke -> angol megjelenítési név, pl. 'chicken_curry' -> 'Chicken curry'."""
    if label in _EN_OVERRIDES:
        return _EN_OVERRIDES[label]
    return label.replace("_", " ").strip().capitalize()


def label_to_name_hu(label: str) -> str | None:
    """Címke -> magyar név, vagy None, ha nem Food-101 címke."""
    return FOOD101_HU.get(label)
