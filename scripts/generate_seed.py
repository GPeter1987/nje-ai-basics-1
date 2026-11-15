"""201 különböző mintarecept -> reprodukálható foods_seed.csv, USDA-alapértékekből.

A receptek oktatási közelítések, nem hitelesített receptúrák vagy mért készételek.
A súlyok a tálalt adag összetevőarányai; a főtt alapanyagok vizét már az USDA tartalmazza.
"""

# A receptkatalógusban szándékosan egy sor egy recept: könnyebb ellenőrizni az arányokat.
# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pipeline.food101_labels import FOOD101_HU, label_to_name_en  # noqa: E402

# kulcs | kategória | konyha | elkészítés/íz | összetevő:súly (g)
FOOD101_RECIPES = """
apple_pie|dessert|american|baked sweet|apple:50,wheat flour:25,butter:15,sugar:10
baby_back_ribs|main course|american|grilled smoky|pork:85,tomato:8,sugar:5,mustard:2
baklava|dessert|turkish|baked sweet|wheat flour:30,walnuts:25,butter:20,sugar:25
beef_carpaccio|appetizer|italian|raw savory|raw beef:85,olive oil:5,lemon:5,cheese:5
beef_tartare|appetizer|french|raw savory|raw beef:85,eggs:5,onion:5,mustard:5
beet_salad|salad|international|fresh sour|beet:80,onion:8,lemon:7,olive oil:5
beignets|dessert|french|fried sweet|wheat flour:40,milk:20,eggs:15,sugar:15,olive oil:10
bibimbap|main course|korean|stir fried spicy|rice:40,beef:20,carrot:10,spinach:10,eggs:15,sesame:5
bread_pudding|dessert|british|baked sweet|bread:35,milk:35,eggs:15,sugar:10,butter:5
breakfast_burrito|breakfast|mexican|grilled savory|tortilla:30,eggs:30,beans:15,tomato:15,cheese:10
bruschetta|appetizer|italian|toasted fresh|bread:50,tomato:40,olive oil:8,garlic:2
caesar_salad|salad|american|fresh creamy|lettuce:45,chicken:25,bread:10,cheese:10,yogurt:8,mustard:2
cannoli|dessert|italian|fried sweet|wheat flour:25,ricotta:40,sugar:20,butter:10,chocolate:5
caprese_salad|salad|italian|raw fresh|tomato:55,mozzarella:35,olive oil:10
carrot_cake|dessert|american|baked sweet|carrot:30,wheat flour:25,eggs:15,sugar:15,olive oil:10,walnuts:5
ceviche|appetizer|peruvian|raw sour|raw tuna:60,lemon:20,onion:10,tomato:10
cheese_plate|appetizer|french|savory creamy|cheese:70,ricotta:20,walnuts:10
cheesecake|dessert|american|baked creamy|ricotta:45,eggs:15,sugar:15,bread:15,butter:10
chicken_curry|main course|indian|stewed spicy creamy|chicken:40,tomato:20,onion:15,coconut milk:20,curry powder:3,olive oil:2
chicken_quesadilla|main course|mexican|grilled savory|tortilla:35,chicken:30,cheese:20,pepper:15
chicken_wings|main course|american|grilled spicy|chicken:80,olive oil:10,tomato:8,curry powder:2
chocolate_cake|dessert|international|baked sweet|wheat flour:30,eggs:20,sugar:20,butter:15,cocoa:10,milk:5
chocolate_mousse|dessert|french|chilled creamy sweet|chocolate:30,cream:45,eggs:15,sugar:10
churros|dessert|spanish|fried sweet crispy|wheat flour:40,water:20,sugar:20,olive oil:18,cinnamon:2
clam_chowder|soup|american|boiled creamy|clam:25,potato:25,milk:25,onion:10,water:10,butter:5
club_sandwich|sandwich|american|toasted savory|bread:35,chicken:25,bacon:10,lettuce:15,tomato:15
crab_cakes|main course|american|fried savory|crab:60,bread:15,eggs:15,olive oil:8,mustard:2
creme_brulee|dessert|french|baked sweet creamy|cream:55,eggs:25,sugar:20
croque_madame|sandwich|french|grilled savory|bread:30,pork:20,cheese:20,eggs:25,butter:5
cup_cakes|dessert|american|baked sweet|wheat flour:30,sugar:25,butter:20,eggs:20,milk:5
deviled_eggs|appetizer|international|boiled creamy|eggs:85,yogurt:10,mustard:5
donuts|dessert|american|fried sweet|wheat flour:35,milk:20,sugar:20,eggs:10,olive oil:15
dumplings|main course|chinese|steamed savory|wheat flour:30,pork:30,cabbage:25,water:10,soy sauce:5
edamame|appetizer|japanese|boiled savory|soybeans:100
eggs_benedict|breakfast|american|poached creamy|eggs:40,bread:25,pork:15,butter:15,lemon:5
escargots|appetizer|french|baked savory|snail:70,butter:20,garlic:10
falafel|main course|middle eastern|fried crispy|chickpea:75,onion:10,wheat flour:5,olive oil:10
filet_mignon|main course|french|grilled savory|beef:95,butter:5
fish_and_chips|main course|british|fried crispy|cod:40,potato:35,wheat flour:15,olive oil:10
foie_gras|appetizer|french|seared creamy|goose liver:80,butter:20
french_fries|side dish|international|fried crispy|potato:85,olive oil:15
french_onion_soup|soup|french|boiled savory|onion:45,water:30,bread:10,cheese:10,butter:5
french_toast|breakfast|french|fried sweet|bread:45,eggs:25,milk:15,butter:10,sugar:5
fried_calamari|appetizer|mediterranean|fried crispy|squid:65,wheat flour:20,olive oil:15
fried_rice|main course|chinese|stir fried savory|rice:55,eggs:15,peas:10,carrot:10,olive oil:5,soy sauce:5
frozen_yogurt|dessert|american|frozen sweet|yogurt:80,sugar:15,strawberry:5
garlic_bread|side dish|italian|toasted savory|bread:80,butter:15,garlic:5
gnocchi|main course|italian|boiled soft|potato:65,wheat flour:25,eggs:10
greek_salad|salad|greek|raw fresh|tomato:35,cucumber:30,feta:20,onion:5,olive oil:10
grilled_cheese_sandwich|sandwich|american|grilled creamy|bread:55,cheese:35,butter:10
grilled_salmon|main course|international|grilled savory|salmon:95,lemon:5
guacamole|appetizer|mexican|raw creamy|avocado:80,tomato:10,onion:5,lemon:5
gyoza|main course|japanese|fried savory|wheat flour:30,pork:30,cabbage:25,olive oil:10,soy sauce:5
hamburger|sandwich|american|grilled savory|bread:30,beef:40,cheese:10,tomato:10,lettuce:8,mustard:2
hot_and_sour_soup|soup|chinese|boiled spicy sour|water:45,tofu:20,mushroom:15,eggs:10,soy sauce:5,lemon:5
hot_dog|sandwich|american|grilled savory|bread:45,sausage:45,mustard:5,tomato:5
huevos_rancheros|breakfast|mexican|fried spicy|eggs:35,corn tortilla:25,beans:20,tomato:15,olive oil:5
hummus|appetizer|middle eastern|pureed creamy|chickpea:70,sesame:10,olive oil:10,lemon:8,garlic:2
ice_cream|dessert|international|frozen sweet creamy|milk:45,cream:30,sugar:20,eggs:5
lasagna|main course|italian|baked savory|pasta:30,beef:25,tomato:25,cheese:15,milk:5
lobster_bisque|soup|french|boiled creamy|lobster:30,cream:20,tomato:20,onion:10,water:15,butter:5
lobster_roll_sandwich|sandwich|american|toasted creamy|lobster:50,bread:35,butter:10,celery:5
macaroni_and_cheese|main course|american|boiled creamy|pasta:55,cheese:25,milk:15,butter:5
macarons|dessert|french|baked sweet|almonds:35,sugar:40,egg white:20,butter:5
miso_soup|soup|japanese|boiled savory|water:65,tofu:20,soybeans:5,seaweed:5,soy sauce:5
mussels|main course|french|steamed savory|mussels:85,tomato:10,garlic:3,olive oil:2
nachos|appetizer|mexican|baked crispy|corn tortilla:45,cheese:25,beans:15,tomato:10,olive oil:5
omelette|breakfast|french|fried savory|eggs:80,milk:10,butter:10
onion_rings|side dish|american|fried crispy|onion:50,wheat flour:25,eggs:10,olive oil:15
oysters|appetizer|french|raw fresh|oysters:95,lemon:5
pad_thai|main course|thai|stir fried sweet sour|rice noodles:45,shrimp:20,eggs:15,peanuts:10,lemon:5,olive oil:5
paella|main course|spanish|stewed savory|rice:45,chicken:20,shrimp:15,peas:10,tomato:5,olive oil:5
pancakes|breakfast|american|fried sweet|wheat flour:35,milk:30,eggs:20,sugar:10,butter:5
panna_cotta|dessert|italian|chilled creamy sweet|cream:60,milk:25,sugar:15
peking_duck|main course|chinese|roasted sweet savory|duck:65,tortilla:20,cucumber:10,sugar:5
pho|soup|vietnamese|boiled savory|water:45,rice noodles:25,beef:20,onion:8,soy sauce:2
pizza|main course|italian|baked savory|wheat flour:35,water:15,tomato:25,mozzarella:20,olive oil:5
pork_chop|main course|international|grilled savory|pork:95,olive oil:5
poutine|main course|canadian|fried creamy|potato:55,cheese:25,water:10,olive oil:8,wheat flour:2
prime_rib|main course|american|roasted savory|beef:90,butter:10
pulled_pork_sandwich|sandwich|american|slow cooked smoky|pork:45,bread:35,tomato:10,onion:5,sugar:5
ramen|soup|japanese|boiled savory|water:35,pasta:30,pork:15,eggs:10,mushroom:5,soy sauce:5
ravioli|main course|italian|boiled creamy|pasta:45,ricotta:30,spinach:15,tomato:10
red_velvet_cake|dessert|american|baked sweet|wheat flour:30,sugar:25,butter:15,eggs:15,milk:10,cocoa:5
risotto|main course|italian|stewed creamy|rice:55,mushroom:20,cheese:10,onion:10,butter:5
samosa|appetizer|indian|fried spicy|wheat flour:30,potato:35,peas:15,olive oil:15,curry powder:5
sashimi|main course|japanese|raw fresh|raw salmon:50,raw tuna:50
scallops|main course|french|seared savory|scallops:90,butter:8,lemon:2
seaweed_salad|salad|japanese|raw fresh|seaweed:80,sesame:8,soy sauce:7,lemon:5
shrimp_and_grits|main course|american|boiled creamy|shrimp:35,cornmeal:20,water:25,cheese:15,butter:5
spaghetti_bolognese|main course|italian|boiled savory|pasta:50,beef:25,tomato:20,onion:5
spaghetti_carbonara|main course|italian|boiled creamy|pasta:55,eggs:20,bacon:15,cheese:10
spring_rolls|appetizer|chinese|fried crispy|wheat flour:25,cabbage:35,carrot:20,shrimp:10,olive oil:10
steak|main course|american|grilled savory|beef:95,olive oil:5
strawberry_shortcake|dessert|american|baked sweet fruity|strawberry:35,wheat flour:20,cream:20,sugar:15,butter:10
sushi|main course|japanese|raw savory|rice:60,raw salmon:25,cucumber:10,seaweed:5
tacos|main course|mexican|grilled spicy|corn tortilla:30,beef:30,tomato:20,lettuce:10,cheese:10
takoyaki|appetizer|japanese|fried savory|wheat flour:30,octopus:25,eggs:20,water:20,olive oil:5
tiramisu|dessert|italian|chilled sweet creamy|ricotta:35,bread:20,eggs:15,sugar:15,coffee:10,cocoa:5
tuna_tartare|appetizer|japanese|raw fresh|raw tuna:75,avocado:15,lemon:5,soy sauce:5
waffles|breakfast|belgian|baked sweet crispy|wheat flour:35,milk:25,eggs:20,butter:15,sugar:5
"""

# További 100, önálló összetételű alternatíva magyar névvel.
EXTRA_RECIPES = """
chickpea_curry|Csicseriborsó-curry|main course|indian|stewed spicy|chickpea:50,tomato:25,onion:10,coconut milk:10,curry powder:3,olive oil:2
lentil_curry|Lencsecurry|main course|indian|stewed spicy|lentil:50,tomato:25,onion:10,spinach:10,curry powder:3,olive oil:2
tofu_curry|Tofucurry|main course|indian|stewed creamy|tofu:40,broccoli:25,coconut milk:20,onion:10,curry powder:3,olive oil:2
vegetable_curry|Zöldségcurry|main course|indian|stewed spicy|broccoli:30,carrot:25,peas:20,tomato:20,curry powder:3,olive oil:2
turkey_curry|Pulykacurry|main course|indian|stewed spicy|turkey:45,tomato:25,onion:15,yogurt:10,curry powder:3,olive oil:2
spinach_chicken|Spenótos csirke|main course|indian|stewed creamy|chicken:45,spinach:35,yogurt:15,curry powder:3,olive oil:2
red_lentil_soup|Vöröslencse-leves|soup|indian|boiled spicy|lentil:35,tomato:20,carrot:10,water:30,curry powder:3,olive oil:2
chicken_rice_bowl|Csirkés rizstál|main course|international|grilled savory|chicken:40,brown rice:35,broccoli:20,olive oil:5
turkey_quinoa_bowl|Pulykás quinoatál|main course|international|grilled savory|turkey:40,quinoa:35,spinach:20,olive oil:5
tofu_rice_bowl|Tofus rizstál|main course|japanese|grilled savory|tofu:35,brown rice:35,carrot:20,soy sauce:5,sesame:5
baked_falafel|Sütőben sült falafel|main course|middle eastern|baked crispy|chickpea:80,onion:10,wheat flour:5,olive oil:5
lentil_patties|Lencsefasírt|main course|international|baked savory|lentil:65,carrot:15,oats:10,eggs:5,olive oil:5
bean_burger|Babburger|sandwich|american|grilled savory|beans:45,bread:30,lettuce:10,tomato:10,olive oil:5
chickpea_salad|Csicseriborsó-saláta|salad|mediterranean|raw fresh|chickpea:50,tomato:20,cucumber:20,lemon:5,olive oil:5
lentil_salad|Lencsesaláta|salad|mediterranean|fresh sour|lentil:55,tomato:20,onion:10,lemon:10,olive oil:5
quinoa_salad|Quinoasaláta|salad|mediterranean|fresh savory|quinoa:45,cucumber:25,tomato:20,lemon:5,olive oil:5
chicken_salad|Csirkesaláta|salad|international|grilled fresh|chicken:40,lettuce:30,tomato:20,yogurt:5,olive oil:5
tuna_salad|Tonhalsaláta|salad|mediterranean|fresh savory|tuna:40,lettuce:25,tomato:20,beans:10,olive oil:5
salmon_salad|Lazacsaláta|salad|mediterranean|grilled fresh|salmon:35,lettuce:25,cucumber:20,avocado:15,lemon:5
avocado_salad|Avokádósaláta|salad|mexican|raw creamy|avocado:35,tomato:30,cucumber:25,lemon:10
cucumber_yogurt_salad|Joghurtos uborkasaláta|salad|greek|raw creamy|cucumber:70,yogurt:25,garlic:5
carrot_apple_salad|Répás almasaláta|salad|international|raw sweet|carrot:45,apple:45,lemon:10
spinach_walnut_salad|Diós spenótsaláta|salad|mediterranean|raw fresh|spinach:55,apple:25,walnuts:15,olive oil:5
beet_lentil_salad|Céklás lencsesaláta|salad|international|fresh sour|beet:45,lentil:40,onion:5,lemon:5,olive oil:5
cabbage_salad|Káposztasaláta|salad|hungarian|raw sour|cabbage:75,carrot:15,lemon:5,olive oil:5
tomato_soup|Paradicsomleves|soup|hungarian|boiled savory|tomato:60,water:25,onion:10,olive oil:5
broccoli_soup|Brokkolikrémleves|soup|international|boiled creamy|broccoli:50,potato:15,milk:15,water:15,olive oil:5
mushroom_soup|Gombaleves|soup|hungarian|boiled savory|mushroom:40,carrot:15,onion:10,water:30,olive oil:5
carrot_soup|Sárgarépa-krémleves|soup|international|boiled creamy|carrot:55,onion:10,coconut milk:10,water:20,olive oil:5
pea_soup|Zöldborsóleves|soup|hungarian|boiled savory|peas:45,carrot:15,water:30,onion:5,olive oil:5
bean_soup|Bableves|soup|hungarian|boiled savory|beans:40,carrot:15,celery:10,onion:10,water:20,olive oil:5
chicken_vegetable_soup|Csirkés zöldségleves|soup|hungarian|boiled savory|chicken:25,carrot:20,celery:10,peas:10,water:35
tofu_noodle_soup|Tofus tésztaleves|soup|japanese|boiled savory|tofu:25,rice noodles:25,mushroom:15,water:30,soy sauce:5
spinach_soup|Spenótkrémleves|soup|international|boiled creamy|spinach:45,potato:20,yogurt:15,water:15,olive oil:5
lentil_stew|Lencsefőzelék|main course|hungarian|stewed savory|lentil:65,onion:15,yogurt:10,mustard:5,olive oil:5
bean_stew|Babfőzelék|main course|hungarian|stewed savory|beans:65,tomato:15,onion:15,olive oil:5
pea_stew|Zöldborsófőzelék|main course|hungarian|stewed creamy|peas:65,milk:20,wheat flour:10,butter:5
potato_stew|Burgonyafőzelék|main course|hungarian|stewed creamy|potato:65,yogurt:20,onion:10,olive oil:5
stuffed_peppers|Töltött paprika|main course|hungarian|stewed savory|pepper:30,turkey:25,rice:20,tomato:20,olive oil:5
stuffed_cabbage|Töltött káposzta|main course|hungarian|stewed savory|cabbage:40,pork:25,rice:20,tomato:10,olive oil:5
vegetarian_stuffed_peppers|Lencsével töltött paprika|main course|hungarian|baked savory|pepper:35,lentil:30,brown rice:20,tomato:10,olive oil:5
chicken_paprika|Paprikás csirke|main course|hungarian|stewed creamy|chicken:45,pepper:20,onion:15,yogurt:15,olive oil:5
turkey_meatballs|Pulykafasírt|main course|international|baked savory|turkey:60,oats:15,eggs:10,onion:10,olive oil:5
baked_cod|Sült tőkehal|main course|mediterranean|baked fresh|cod:80,tomato:10,lemon:5,olive oil:5
lemon_chicken|Citromos csirke|main course|mediterranean|grilled fresh|chicken:75,lemon:15,garlic:5,olive oil:5
grilled_turkey|Grillezett pulykamell|main course|international|grilled savory|turkey:90,lemon:5,olive oil:5
salmon_broccoli|Lazac brokkolival|main course|international|steamed savory|salmon:50,broccoli:40,lemon:5,olive oil:5
shrimp_quinoa|Garnéla quinoával|main course|mediterranean|grilled savory|shrimp:40,quinoa:40,tomato:10,lemon:5,olive oil:5
chicken_stir_fry|Pirított csirkés zöldség|main course|chinese|stir fried savory|chicken:35,broccoli:25,carrot:20,pepper:10,soy sauce:5,olive oil:5
tofu_stir_fry|Pirított tofus zöldség|main course|chinese|stir fried savory|tofu:35,broccoli:25,carrot:20,mushroom:10,soy sauce:5,olive oil:5
mushroom_rice|Gombás rizs|main course|international|stewed savory|brown rice:50,mushroom:30,onion:15,olive oil:5
vegetable_risotto|Zöldséges rizottó|main course|italian|stewed creamy|rice:45,peas:20,carrot:15,cheese:10,onion:5,butter:5
quinoa_risotto|Quinoarizottó|main course|italian|stewed creamy|quinoa:50,mushroom:25,spinach:15,cheese:5,olive oil:5
tomato_pasta|Paradicsomos tészta|main course|italian|boiled savory|pasta:55,tomato:35,garlic:5,olive oil:5
spinach_pasta|Spenótos tészta|main course|italian|boiled creamy|pasta:50,spinach:30,ricotta:15,olive oil:5
tuna_pasta|Tonhalas tészta|main course|italian|boiled savory|pasta:45,tuna:30,tomato:20,olive oil:5
chicken_pasta|Csirkés tészta|main course|italian|boiled creamy|pasta:45,chicken:30,broccoli:15,yogurt:5,olive oil:5
lentil_bolognese|Lencsés bolognai|main course|italian|stewed savory|pasta:40,lentil:30,tomato:20,onion:5,olive oil:5
vegetable_lasagna|Zöldséges lasagne|main course|italian|baked savory|pasta:30,spinach:25,tomato:20,ricotta:20,olive oil:5
rice_noodle_salad|Rizstészta-saláta|salad|vietnamese|fresh sour|rice noodles:45,cucumber:20,carrot:20,lemon:10,sesame:5
chicken_wrap|Csirkés wrap|sandwich|international|grilled fresh|tortilla:30,chicken:35,lettuce:15,tomato:15,yogurt:5
tofu_wrap|Tofus wrap|sandwich|international|grilled fresh|tortilla:30,tofu:35,lettuce:15,tomato:15,avocado:5
tuna_sandwich|Tonhalas szendvics|sandwich|international|fresh savory|bread:40,tuna:35,lettuce:10,tomato:10,yogurt:5
turkey_sandwich|Pulykás szendvics|sandwich|international|fresh savory|bread:40,turkey:35,lettuce:15,tomato:10
avocado_toast|Avokádós pirítós|breakfast|international|toasted creamy|bread:50,avocado:40,lemon:10
egg_salad_sandwich|Tojáskrémes szendvics|sandwich|international|creamy savory|bread:40,eggs:35,yogurt:15,lettuce:5,mustard:5
hummus_wrap|Humuszos wrap|sandwich|middle eastern|fresh creamy|tortilla:30,chickpea:35,sesame:5,cucumber:15,tomato:10,lemon:5
bean_tacos|Babos taco|main course|mexican|grilled spicy|corn tortilla:30,beans:35,tomato:20,lettuce:10,avocado:5
fish_tacos|Halas taco|main course|mexican|grilled fresh|corn tortilla:30,cod:35,cabbage:20,tomato:10,lemon:5
chicken_tacos|Csirkés taco|main course|mexican|grilled savory|corn tortilla:30,chicken:35,tomato:20,lettuce:10,yogurt:5
vegetable_omelette|Zöldséges omlett|breakfast|international|fried savory|eggs:60,pepper:15,spinach:15,olive oil:10
mushroom_omelette|Gombás omlett|breakfast|french|fried savory|eggs:60,mushroom:25,onion:10,olive oil:5
spinach_egg_white|Spenótos tojásfehérje|breakfast|international|fried savory|egg white:65,spinach:25,onion:5,olive oil:5
tofu_scramble|Tofurántotta|breakfast|international|fried savory|tofu:65,tomato:15,onion:10,olive oil:5,curry powder:5
oat_porridge|Zabkása|breakfast|international|boiled creamy|oats:20,milk:70,banana:10
apple_oat_porridge|Almás zabkása|breakfast|international|boiled sweet|oats:20,water:45,apple:30,walnuts:4,cinnamon:1
banana_oat_pancakes|Banános zabpalacsinta|breakfast|international|fried sweet|banana:40,oats:30,eggs:25,olive oil:5
yogurt_fruit_bowl|Gyümölcsös joghurttál|breakfast|international|fresh sweet|yogurt:60,strawberry:25,oats:10,almonds:5
quinoa_breakfast|Quinoás reggelitál|breakfast|international|boiled sweet|quinoa:40,milk:35,banana:20,almonds:5
overnight_oats|Éjszakai zabkása|breakfast|international|chilled creamy|oats:20,yogurt:45,milk:20,strawberry:15
baked_apple|Sült alma|dessert|international|baked sweet|apple:90,walnuts:5,sugar:4,cinnamon:1
fruit_salad|Gyümölcssaláta|dessert|international|raw fresh|apple:35,banana:30,strawberry:30,lemon:5
strawberry_yogurt|Epres joghurt|dessert|international|chilled fruity|yogurt:70,strawberry:30
banana_ice_cream|Banánfagylalt|dessert|international|frozen sweet|banana:90,cocoa:5,peanuts:5
apple_oat_crumble|Almás zabmorzsa|dessert|british|baked sweet|apple:60,oats:20,butter:10,sugar:10
ricotta_berry_cup|Epres ricottakrém|dessert|italian|chilled creamy|ricotta:55,strawberry:35,sugar:10
chocolate_tofu_mousse|Csokoládés tofuhab|dessert|international|chilled creamy|tofu:60,chocolate:25,banana:15
oat_cookies|Zabkeksz|dessert|international|baked sweet|oats:40,banana:35,peanuts:10,olive oil:10,cocoa:5
rice_pudding|Tejberizs|dessert|hungarian|boiled creamy|rice:45,milk:40,sugar:10,cinnamon:1,butter:4
coconut_rice_pudding|Kókuszos rizspuding|dessert|thai|boiled creamy|rice:45,coconut milk:40,banana:15
baked_potato|Sült burgonya|side dish|international|baked savory|potato:95,olive oil:5
potato_broccoli_mash|Brokkolis burgonyapüré|side dish|international|boiled creamy|potato:50,broccoli:35,milk:10,butter:5
roasted_carrot|Sült sárgarépa|side dish|international|roasted sweet|carrot:90,olive oil:8,curry powder:2
steamed_broccoli|Párolt brokkoli|side dish|international|steamed fresh|broccoli:95,lemon:5
brown_rice_peas|Borsós barna rizs|side dish|international|boiled savory|brown rice:65,peas:30,olive oil:5
quinoa_vegetables|Zöldséges quinoa|side dish|international|steamed fresh|quinoa:50,carrot:20,broccoli:25,olive oil:5
roasted_chickpeas|Sült csicseriborsó|appetizer|middle eastern|roasted crispy|chickpea:90,olive oil:7,curry powder:3
white_bean_dip|Babkrém|appetizer|mediterranean|pureed creamy|beans:80,lemon:10,olive oil:7,garlic:3
pea_hummus|Zöldborsós humusz|appetizer|middle eastern|pureed creamy|peas:45,chickpea:35,sesame:10,lemon:5,olive oil:5
avocado_egg_bowl|Avokádós tojástál|breakfast|international|fresh creamy|eggs:45,avocado:35,tomato:15,lemon:5
"""


def generate() -> int:
    ingredients = json.loads((config.DATA_DIR / "usda_ingredients.json").read_text(encoding="utf-8"))
    recipes = []
    for line in FOOD101_RECIPES.strip().splitlines():
        key, category, cuisine, method, recipe = line.split("|")
        recipes.append((key, FOOD101_HU[key], category, cuisine, method, recipe))
    recipes.extend(tuple(line.split("|")) for line in EXTRA_RECIPES.strip().splitlines())
    foods = []
    for key, name_hu, category, cuisine, method, recipe in recipes:
        parts = [(name, float(weight)) for item in recipe.split(",")
                 for name, weight in [item.rsplit(":", 1)]]
        total = sum(weight for _, weight in parts)
        if total != 100:
            raise ValueError(f"{key}: nem 100 g-os recept: {total}")
        allergens = sorted({a for name, _ in parts for a in ingredients[name]["allergens"]})
        tags = method.split()
        animal = {"chicken", "beef", "raw beef", "pork", "duck", "turkey", "salmon",
                  "raw salmon", "tuna", "raw tuna", "cod", "shrimp", "lobster", "crab",
                  "squid", "clam", "mussels", "oysters", "scallops", "snail", "octopus",
                  "goose liver", "bacon", "sausage"}
        names = [name for name, _ in parts]
        if not animal.intersection(names):
            tags.append("vegetarian")
            if not {"eggs", "milk"}.intersection(allergens):
                tags.append("vegan")
        if "gluten" not in allergens:
            tags.append("gluten-free")
        if "milk" not in allergens:
            tags.append("dairy-free")
        name_en = label_to_name_en(key)
        source = "USDA SR28 recipe estimate; " + "; ".join(
            f"NDB {ingredients[name]['ndb']} {name} {weight:g}g" for name, weight in parts
        )
        food = {
            "key": key, "name_en": name_en, "name_hu": name_hu,
            "category": category, "cuisine": cuisine,
            "ingredients": json.dumps(names, ensure_ascii=False),
            "allergens": json.dumps(allergens), "tags": json.dumps(tags),
            **{f"{nutrient}_100g": round(sum(ingredients[name][nutrient] * weight / 100
               for name, weight in parts), 3) for nutrient in ("kcal", "protein", "carbs", "fat")},
            "description_en": f"{name_en}; {category}; {cuisine}; {', '.join(names)}; {', '.join(tags)}.",
            "in_food101": int(key in FOOD101_HU), "source": source,
        }
        foods.append(food)
    if len(foods) != 201 or len({f["key"] for f in foods}) != 201:
        raise ValueError(f"201 különböző étel szükséges, jelenleg {len(foods)}.")
    with config.SEED_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(foods[0]))
        writer.writeheader()
        writer.writerows(foods)
    return len(foods)


if __name__ == "__main__":
    print(f"{generate()} recept kiírva: {config.SEED_CSV_PATH}")
