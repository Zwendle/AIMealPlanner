import pandas as pd
import json
import re

## HELPERS FUNCTIONS ##
## if JSON is null, ignore. otherwise, convert JSON data to dictionary
def json_ignore_null(x):
    if x is None:
        return {}
    else:
        try:
            return json.loads(str(x))
        except:
            return {}

## clean number of servings Whole Foods JSON column
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

## clean category Trader Joe's column
def clean_category(x):
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


## WHOLE FOODS ##
## load scraped Whole Foods data
wf = pd.read_csv('sources/whole-foods/wholefoods_inventory_store_10031.csv')

## extract JSON data
wf["nutrition_json"] = wf["nutrition_json"].apply(json_ignore_null)

## flatten JSON data into columns
wf_json = pd.json_normalize(wf["nutrition_json"])

## build cleaned whole foods dataframe
wf_cleaned = pd.DataFrame({
    'name': wf['name'],
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

## add cost per serving column
wf_cleaned['cost_per_serving'] = wf_cleaned['price'].div(wf_cleaned['num_servings']).round(2)

## upload cleaned Whole Foods csv
wf_cleaned.to_csv("data/whole_foods_cleaned.csv", index=False)

## TRADER JOES ##
## load scraped Trader Joe's data
tj = pd.read_csv('trader_joes_nov_18.csv')

## extract JSON data
tj["nutrition_json"] = tj["nutrition_json"].apply(json_ignore_null)

## flatten JSON data into columns
tj_json = pd.json_normalize(tj["nutrition_json"])

## build cleaned trader joe's dataframe
tj_cleaned = pd.DataFrame({
    'name': tj['name'],
    'serving_size': tj_json['serving_size'],
    'num_servings': tj['serves'],
    'category': tj['category'].apply(clean_category),
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

## add ismeat column
tj_cleaned['is_meat'] = tj_cleaned['category'] == 'meat-seafood'

## add cost per serving column
tj_cleaned['cost_per_serving'] = tj_cleaned['price'].div(tj_cleaned['num_servings']).round(2)

## upload cleaned Trader Joe's csv
tj_cleaned.to_csv("data/trader_joes_cleaned.csv", index=False)

