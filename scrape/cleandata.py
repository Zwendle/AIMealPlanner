import pandas as pd
import json
import re
from rapidfuzz import process, fuzz
import os

## HELPERS FUNCTIONS ##
## SCRAPER FUNCTIONS ##
## if JSON is null, ignore. otherwise, convert JSON data to dictionary
def json_ignore_null(x):
    if x is None:
        return {}
    else:
        try:
            return json.loads(str(x))
        except:
            return {}

## WHOLE FOODS FUNCTIONS ##
## clean number of servings Whole Foods column
def clean_num_servings_wf(x):
    varied_keywords = ["varied", "varies", 'null']

    ## varied or null serving
    if isinstance(x, str) and any(keyword.lower() in x.lower() for keyword in varied_keywords):
        return None
    ## serving number is approximated
    elif isinstance(x, str) and re.search(r'\d+', x):
        match = re.search(r'\d+', x)
        return int(match.group())
    ## null data
    elif x is None:
        return 1
    ## number
    else:
        return x

# remove whole foods branding from the name column
def clean_wf_branding(x):
    if not isinstance(x, str):
        return x

    # remove branding prefixes
    x = re.sub(r"^365 by whole foods market \s*", "", x, flags=re.IGNORECASE)
    x = re.sub(r"^whole foods market \s*", "", x, flags=re.IGNORECASE)
    x = re.sub(r"^365 everyday value \s*", "", x, flags=re.IGNORECASE)

    return x.strip()

## TRADER JOE'S FUNCTIONS ##
## clean category Trader Joe's column
def clean_category_tj(x):
    if isinstance(x, str) and 'Meat & Seafood' == x:
        return 'meat-seafood'
    elif isinstance(x, str) and 'Dairy & Eggs' == x:
        return 'dairy-eggs'
    else:
        return x.lower()

## get number from nutritional data
def get_number(x):
    if isinstance(x, str) and re.search(r'\d+', x):
        match = re.search(r'\d+', x)
        return float(match.group())
    return None

## get character from nutritional data
def get_char(x):
    if isinstance(x, str):
        match = re.search(r"[A-Za-z]", x)
        if match:
            return match.group()
    return None

## SHARED DATA CLEANING DEFINITIONS ##
# common units
units = r"(fl oz|oz|lb|lbs|fz|fl|g|kg|ml|ct|count|pack|pk|pt)"
# common patterns
patterns = [
    fr"\((\d+(?:\.\d+)?)\s*{units}\)",
    fr",\s*(\d+(?:\.\d+)?)\s*{units}\b",
    fr"\s+(\d+(?:\.\d+)?)\s*{units}$",
    r"(\d+/\d+)\s*(count|ct)",
    fr"(\d+(?:\.\d+)?)-{units}",
    fr"\s+(\d+)\s*(pack|pk|ct|count)\b",
    r"(\d+(?:\.\d+)?)\s*(fl\s*oz)",
    r"(\d+(?:\.\d+)?)\s*(ounce|pound)s?\b",
]
# volumes -> grams
DENSITY = {
    "cup": 240,
    "tbsp": 15,
    "tsp": 5,
    "fl oz": 30,
}
# category mapping
category_mapping = {
    'produce': 'produce',
    'dairy-eggs': 'dairy-eggs',
    'meat': 'meat-seafood',
    'meat-seafood': 'meat-seafood',
    'seafood': 'meat-seafood',
    'pantry-essentials': 'pantry',
    'pantry': 'pantry',
    'breads-rolls-bakery': 'bakery',
    'bakery': 'bakery',
    'desserts': 'desserts',
    'frozen-foods': 'frozen',
    'snacks-chips-salsas-dips': 'snacks'
}

## SHARED DATA CLEANING FUNCTIONS ##
## remove commas from product name
def remove_commas(name):
    if pd.isna(name):
        return name
    return str(name).replace(',', '')

## get serving size from the name column
def get_serving_size(product_name):

    if pd.isna(product_name):
        return None

    product_name = str(product_name).strip()

    ## extract pattern
    for pattern in patterns:
        match = re.search(pattern, product_name, re.IGNORECASE)
        if match:
            num = match.group(1)
            unit = match.group(2) if match.lastindex and match.lastindex >= 2 else ""

            # units
            unit = (
                unit.lower()
                .replace("ounces", "oz")
                .replace("ounce", "oz")
                .replace("pounds", "lb")
                .replace("pound", "lb")
                .replace(" ", "")
            )

            return f"{num} {unit}".strip()

    return None

# remove serving size from the name column
def clean_serving_size(name):

    if not isinstance(name, str):
        return name

    name_clean = name

    # remove patterns and extra spaces
    for pattern in patterns:
        name_clean = re.sub(pattern, "", name_clean, flags=re.IGNORECASE)
    name_clean = re.sub(r"\s+", " ", name_clean).strip()

    return name_clean

# converts serving_size to grams
def serving_size_to_grams(serving):

    if not isinstance(serving, str):
        return None

    s = serving.lower().strip()

    # grams
    m = re.search(r"(\d+(?:\.\d+)?)\s*g\b", s)
    if m:
        return float(m.group(1))

    # oz -> grams
    m = re.search(r"(\d+(?:\.\d+)?)\s*(oz|ounce|ounces|oz\.)\b", s)
    if m:
        return float(m.group(1)) * 28.3495

    # lbs -> grams
    m = re.search(r"(\d+(?:\.\d+)?)\s*(lb|lbs|pound|pounds)\b", s)
    if m:
        return float(m.group(1)) * 453.592

    # fl oz -> grams
    m = re.search(r"(\d+(?:\.\d+)?)\s*(fl\s*oz|fl\.?\s*oz|fl\.?|fl|fz\.?|fz)\b", s)
    if m:
        return float(m.group(1)) * DENSITY["fl oz"]

    # cups -> grams
    m = re.search(r"(\d+(?:\.\d+)?)\s*(cup|cups)\b", s)
    if m:
        return float(m.group(1)) * DENSITY["cup"]

    # tbsp -> grams
    m = re.search(r"(\d+(?:\.\d+)?)\s*(tbsp|tablespoon|tablespoons)\b", s)
    if m:
        return float(m.group(1)) * DENSITY["tbsp"]

    # tsp -> grams
    m = re.search(r"(\d+(?:\.\d+)?)\s*(tsp|teaspoon|teaspoons)\b", s)
    if m:
        return float(m.group(1)) * DENSITY["tsp"]

    # pints -> grams
    m = re.search(r"(\d+(?:\.\d+)?)\s*pt\b", s)
    if m:
        return float(m.group(1)) * 16 * DENSITY["fl oz"]

    return None

# map categories so that they are consistent
def map_category(ingredients):
    ingredients['category'] = ingredients['category'].map(category_mapping)
    return ingredients

def remove_creamers(ingredients):
    ingredients = ingredients[~ingredients['name_clean'].str.contains(r'\bcreamer\b', case=False, na=False)]
    return ingredients

## KAGGLE FUNCTIONS ##
## find the best ingredient match in choices for ingredient
def ingredient_fuzzy_match(query, choices, score_cutoff=70):
    if not query or len(query) < 3:
        return None, 0

    result = process.extractOne(query, choices, scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff)

    if result:
        match, score, idx = result
        return match, score
    return None, 0

## WHOLE FOODS CLEANING ##
# load whole foods scraped data, clean data, add relevant columns
def clean_whole_foods():
    ## load scraped Whole Foods data
    wf = pd.read_csv(WF_PATH)

    ## extract JSON data
    wf["nutrition_json"] = wf["nutrition_json"].apply(json_ignore_null)

    ## flatten JSON data into columns
    wf_json = pd.json_normalize(wf["nutrition_json"])

    ## build cleaned whole foods dataframe
    wf_cleaned = pd.DataFrame({
        'name': wf['name'],
        'name_clean': wf['name'].str.lower().str.strip().apply(clean_serving_size).apply(remove_commas).apply(clean_wf_branding),
        'serving_size': wf_json['serving_size'],
        'num_servings': wf_json['servings_per_container'].apply(clean_num_servings_wf),
        'category': wf['category'],
        'store': 'whole foods',
        'price': wf['price'].astype(float),
        'carbs': wf_json['carbohydrates.amount'].astype(float),
        'carbs_unit': wf_json['carbohydrates.unit'],
        'fat': wf_json['total_fat.amount'].astype(float),
        'fat_unit': wf_json['total_fat.unit'],
        'protein': wf_json['protein.amount'].astype(float),
        'protein_unit': wf_json['protein.unit'],
        'sugar': wf_json['sugars.amount'].astype(float),
        'sugar_unit': wf_json['sugars.unit'],
        'calories': wf_json['calories'].astype(float),
        'is_meat': wf['category'].isin(['meat','seafood'])
    })

    # remove duplicates based on name
    wf_cleaned = wf_cleaned.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)

    # remove supplements and beverages
    categories = ['supplements', 'beverages']
    wf_cleaned = wf_cleaned[~wf_cleaned['category'].isin(categories)]

    # remove creamers
    wf_cleaned = remove_creamers(wf_cleaned)

    # get serving size from name if serving_size is null
    wf_cleaned = wf_cleaned.copy()
    mask = wf_cleaned['serving_size'].isna()
    wf_cleaned.loc[mask, 'serving_size'] = (wf_cleaned.loc[mask, 'name'].apply(get_serving_size))

    # add column with serving size in grams
    wf_cleaned['serving_size_grams'] = wf_cleaned['serving_size'].apply(serving_size_to_grams).round(2)

    # add cost per serving column
    wf_cleaned['cost_per_serving'] = wf_cleaned['price'].div(wf_cleaned['num_servings']).round(2)

    # upload cleaned Whole Foods csv
    wf_cleaned.to_csv(WF_PATH_CLEAN,  index=False)

    return wf_cleaned

## TRADER JOE'S CLEANING ##
# load trader joe's scraped data, clean data, add relevant columns
def clean_trader_joes():
    ## load scraped Trader Joe's data
    tj = pd.read_csv(TJ_PATH)

    ## extract JSON data
    tj["nutrition_json"] = tj["nutrition_json"].apply(json_ignore_null)

    ## flatten JSON data into columns
    tj_json = pd.json_normalize(tj["nutrition_json"])

    ## build cleaned trader joe's dataframe
    tj_cleaned = pd.DataFrame({
        'name': tj['name'],
        'name_clean': tj['name'].str.lower().str.strip().apply(clean_serving_size).apply(remove_commas),
        'serving_size': tj_json['serving_size'],
        'num_servings': tj['serves'],
        'category': tj['category'].apply(clean_category_tj),
        'store': 'trader joes',
        'price': tj['price'].astype(float),
        'carbs': tj_json['total_carbohydrate'].apply(get_number),
        'carbs_unit': tj_json['total_carbohydrate'].apply(get_char),
        'fat': tj_json['total_fat'].apply(get_number),
        'fat_unit': tj_json['total_fat'].apply(get_char),
        'protein': tj_json['protein'].apply(get_number),
        'protein_unit': tj_json['protein'].apply(get_char),
        'sugar': tj_json['sugars'].apply(get_number),
        'sugar_unit': tj_json['sugars'].apply(get_char),
        'calories': tj_json['calories'].astype(float)
    })

    # remove creamers
    tj_cleaned = remove_creamers(tj_cleaned)

    ## get serving size from name if serving_size is null
    tj_cleaned = tj_cleaned.copy()
    mask = tj_cleaned['serving_size'].isna()
    tj_cleaned.loc[mask, 'serving_size'] = (tj_cleaned.loc[mask, 'name'].apply(get_serving_size))

    # add column with serving size in grams
    tj_cleaned['serving_size_grams'] = tj_cleaned['serving_size'].apply(serving_size_to_grams).round(2)

    ## add ismeat column
    tj_cleaned['is_meat'] = tj_cleaned['category'] == 'meat-seafood'

    ## add cost per serving column
    tj_cleaned['cost_per_serving'] = tj_cleaned['price'].div(tj_cleaned['num_servings']).round(2)

    ## upload cleaned Trader Joe's csv
    tj_cleaned.to_csv(TJ_PATH_CLEAN, index=False)

    return tj_cleaned

## LOADING KAGGLE DATA ##
# load Kaggle data, relevant columns and clean name column
def load_kaggle():
    # get product nutritional data from Kaggle
    kaggle_ingredients = pd.read_csv(KAGGLE_PATH, sep="\t", low_memory=False)
    kaggle_ingredients = kaggle_ingredients[[
        "product_name",
        "carbohydrates_100g",
        "fat_100g",
        "proteins_100g",
        "sugars_100g",
        "energy_100g"
    ]].dropna(subset = ["product_name"])

    ## normalize the names
    kaggle_ingredients["kaggle_name_clean"] = kaggle_ingredients["product_name"].str.lower().str.strip()

    ## remove duplicates
    kaggle_ingredients = (kaggle_ingredients.sort_values("product_name")
                          .drop_duplicates(subset=["product_name"], keep="first"))

    ## get kaggle ingredient names
    kaggle_names = kaggle_ingredients["kaggle_name_clean"].tolist()

    return kaggle_ingredients, kaggle_names

## FUZZY MATCH ##
## fuzzy match ingredients and kaggle ingredients, add matching name and score to ingredients df
def kaggle_fuzzy_match(ingredients_merged, kaggle_names):
    ## for each name in kaggle ingredients, find a matching one in our ingredients
    results = []
    for name in ingredients_merged["name_clean"]:
        _name, score = ingredient_fuzzy_match(name, kaggle_names)
        score = round(score, 2)
        results.append((_name, score))

    ## append matching name (kaggle) and score to ingredients_merged
    ingredients_merged["kaggle_name"] = [r[0] for r in results]
    ingredients_merged["match_score"] = [r[1] for r in results]

    ## join kaggle data to ingredients_merged on name columns
    ingredients_joined = ingredients_merged.merge(
        kaggle_ingredients,
        left_on = "kaggle_name",
        right_on = "kaggle_name_clean",
        how = "left"
    )

    ## remove duplicates
    ingredients_joined = ingredients_joined.drop_duplicates(subset=["name"])

    return ingredients_joined

## NUTRITION CONVERSION ##
# converts the kaggle nutrition info (in 100g) to correct units in ingredient df
def convert_nutrition_info(ingredients_joined):
    ## DATA DEFINITIONS ##
    # kaggle column names to ingredient nutrition name column
    nutrition_conversion = {
        "energy_100g": "calories",
        "proteins_100g": "protein",
        "fat_100g": "fat",
        "carbohydrates_100g": "carbs",
        "sugars_100g": "sugars"
    }

    # map new kaggle nutrition columns to existing nutrition columns
    fill_map = {
        "calories": "calories_kaggle",
        "protein": "protein_kaggle",
        "fat": "fat_kaggle",
        "carbs": "carbs_kaggle",
        "sugar": "sugars_kaggle"
    }

    ## TRANSFORMATION ##
    # create nutrition columns based on kaggle data
    for kaggle_col, ingredient_col in nutrition_conversion.items():
        ingredients_joined[ingredient_col + "_kaggle"] = (
            ingredients_joined[kaggle_col] * ingredients_joined["serving_size_grams"] / 100
        ).round(2)

    # boolean if the match score is better than 95, the kaggle name is specific, and the kaggle name and the ingredient
    # name is of similar length
    score = ingredients_joined["match_score"] >= 95
    ing_len = ingredients_joined["name_clean"].str.len().clip(lower=1)
    kag_len  = ingredients_joined["kaggle_name"].str.len().clip(lower=1)
    len_ratio = (ing_len.where(ing_len < kag_len, kag_len) / ing_len.where(ing_len >= kag_len, kag_len))
    sim_len = len_ratio >= 0.75

    good_match = score & sim_len

    ingredients_joined["good_match"] = good_match

    # populate existing nutrition columns if they are null
    for ingredient_col, kaggle_col in fill_map.items():
        # mask to fill in null and good match columns
        mask = good_match & ingredients_joined[ingredient_col].isna()

        # populate nutrition column
        ingredients_joined.loc[mask, ingredient_col] = ingredients_joined.loc[mask, kaggle_col]

        # skip units if calories column
        if ingredient_col == "calories":
            continue

        # populate nutrition unit column
        unit_col = ingredient_col + "_unit"
        ingredients_joined.loc[mask & ingredients_joined[unit_col].isna(), unit_col] = "g"

    return ingredients_joined

if __name__ == "__main__":
    ## GET FILE PATHS ##
    ## current file path
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    ## root dir path
    ROOT_DIR = os.path.dirname(CURRENT_DIR)

    # whole foods scraped data path
    WF_PATH = os.path.join(CURRENT_DIR, "whole-foods", "wholefoods_inventory_store_10031.csv")
    # whole foods cleaned data path
    WF_PATH_CLEAN = os.path.join(ROOT_DIR, "data", "whole_foods_cleaned.csv")
    # trader joe's scraped data path
    TJ_PATH = os.path.join(CURRENT_DIR, "trader-joes", "trader_joes_nov_18.csv")
    # trader joe's cleaned data path
    TJ_PATH_CLEAN = os.path.join(ROOT_DIR, "data", "trader_joes_cleaned.csv")
    # kaggle file path
    KAGGLE_PATH = os.path.join(ROOT_DIR, "data", "kaggle", "kaggle_ingredients.tsv")
    # ingredients staged data path
    STG_ING_PATH = os.path.join(ROOT_DIR, "data", "stg_ingredients.csv")
    # ingredients staged data path
    ING_PATH = os.path.join(ROOT_DIR, "data", "ingredients.csv")

    print("FILE PATHS LOADED")

    ## DATA CLEANING ##
    ## clean whole foods data
    wf_cleaned = clean_whole_foods()
    print("WHOLE FOODS DATA CLEANED")

    ## clean trader joe's data
    tj_cleaned = clean_trader_joes()
    print("TRADER JOE'S DATA CLEANED")

    ## merge tj and wf data
    ingredients_merged = pd.concat([wf_cleaned, tj_cleaned], ignore_index=True)
    print("WHOLE FOODS AND TRADER JOE'S DATA MERGED")

    ## map category columns
    ingredients_merged = map_category(ingredients_merged)
    print("CATEGORY COLUMNS MAPPED")

    ## get kaggle data
    kaggle_ingredients, kaggle_names = load_kaggle()
    print("LOADED KAGGLE DATA")

    ## fuzzy match kaggle data with ingredients
    ingredients_joined = kaggle_fuzzy_match(ingredients_merged, kaggle_names)
    print("FUZZY MATCHED DATA")

    ## convert kaggle nutrition information into correct units for ingredients
    ingredients_joined = convert_nutrition_info(ingredients_joined)
    print("CONVERTED NUTRITION DATA")

    # upload the staging table
    ingredients_joined.to_csv(STG_ING_PATH, index=False)

    # drop columns that are unnecessary for meal planner
    ingredients = ingredients_joined.drop(columns=["kaggle_name", "match_score", "product_name", "kaggle_name_clean",
                                                   "calories_kaggle", "protein_kaggle", "fat_kaggle", "carbs_kaggle",
                                                   "sugars_kaggle", "energy_100g", "proteins_100g", "fat_100g",
                                                   "carbohydrates_100g", "sugars_100g", "good_match"])

    # upload the final table
    ingredients.to_csv(ING_PATH, index=False)
    print("UPLOADED INGREDIENT.CSV")