import re
import math
import pandas as pd
import numpy as np

# unit conversion to grams
UNIT_TO_GRAMS = {
    'g': 1.0,
    'gram': 1.0,
    'grams': 1.0,
    'kg': 1000.0,
    'kilogram': 1000.0,
    'lb': 453.59237,
    'lbs': 453.59237,
    'pound': 453.59237,
    'oz': 28.3495231,
    'fl oz': 29.5735,   # ml ~ grams for water-like liquids approximated later
    'l': 1000.0,
    'ml': 1.0
}

# sensible fallback serving sizes (grams) by broad category
# tweak these if you want different defaults
CATEGORY_DEFAULT_SERVING_G = {
    'produce': 100,    # approximate 1 cup / 1 medium veg
    'meat': 85,        # ~3 oz cooked
    'dairy': 30,       # portion of cheese, yogurt serving might be 30-150, but 30 as base
    'frozen': 120,
    'bakery': 50,
    'pantry': 40,
    'beverage': 240,   # ml->grams approximation for drink serving 8 oz
    'seafood': 85,
    'snack': 30,
    'default': 100
}

# helper numeric parser for messy cells (keeps decimals)
def parse_number(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    s = str(value).strip().lower()
    if s == '' or s in ('nan','none','n/a'):
        return None
    # find first numeric group including decimal
    m = re.search(r'-?\d+(\.\d+)?', s)
    return float(m.group()) if m else None

def extract_weight_and_unit(text: str):
    """
    Try to parse weight/volume like:
      "16 oz", "3 lb", "1.5 kg", "48 FZ", "32 oz bag", "1 gallon", "12 oz (pack of 2)"
    Returns (value: float, unit: str) or (None, None)
    """
    if not isinstance(text, str):
        return None, None
    txt = text.lower()
    # common format: '3 lb', '16 oz', '1.5kg', '32 fl oz'
    patterns = [
        r'(\d+(\.\d+)?)\s*(fl oz|fl\. oz|f z|fz|fl ozs)',
        r'(\d+(\.\d+)?)\s*(oz|ounce|ounces)\b',
        r'(\d+(\.\d+)?)\s*(lb|lbs|pound|pounds)\b',
        r'(\d+(\.\d+)?)\s*(kg|kilogram|kilograms)\b',
        r'(\d+(\.\d+)?)\s*(g|gram|grams)\b',
        r'(\d+(\.\d+)?)\s*(l|liter|litre|liters|litres)\b',
    ]
    for pat in patterns:
        m = re.search(pat, txt)
        if m:
            # first capturing group is numeric
            val = float(m.group(1))
            unit = m.group(3) if len(m.groups()) >= 3 and m.group(3) else m.group(1)
            # normalized unit
            if 'fl' in unit:
                unit_norm = 'fl oz'
            elif unit in ('oz', 'ounce', 'ounces'):
                unit_norm = 'oz'
            elif unit in ('lb','lbs','pound','pounds'):
                unit_norm = 'lb'
            elif unit in ('kg','kilogram','kilograms'):
                unit_norm = 'kg'
            elif unit in ('g','gram','grams'):
                unit_norm = 'g'
            elif unit in ('l','liter','litre','liters','litres'):
                unit_norm = 'l'
            else:
                unit_norm = unit
            return val, unit_norm
    return None, None

def extract_pack_count(text: str):
    """
    detect things like 'pack of 6', '6-pack', '2 x 8 oz', '2 ct', '12 ct', '12 count'
    Return integer count or None
    """
    if not isinstance(text, str):
        return None
    txt = text.lower()
    # explicit 'pack of N' or 'N-pack' or 'N x' or 'Nx' or 'pack of N'
    m = re.search(r'pack of\s*(\d+)', txt)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)\s*[-x×]\s*pack\b', txt)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)\s*pack\b', txt)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)\s*(ct|count|count.)\b', txt)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)\s*-\s*pack\b', txt)
    if m:
        return int(m.group(1))
    # patterns like '2 x 8 oz' => first number is multiplicity
    m = re.search(r'(\d+)\s*[x×]\s*\d+\s*(oz|g|kg|lb|fl oz)\b', txt)
    if m:
        return int(m.group(1))
    return None

def is_probably_non_food(row):
    """
    Heuristic to flag vitamins, supplements, cleaning products, etc.
    We'll mark as non-food if store/category indicate vitamins or words in name include 'vitamin', 'supplement', 'ct', 'tablet', 'caplet', etc.
    """
    name = str(row.get('name','')).lower()
    category = str(row.get('category','')).lower()
    if any(k in name for k in ['vitamin', 'supplement', 'ct', 'tablet', 'caplet', 'pill', 'scalp', 'cleanser', 'shampoo', 'lotion', 'toothpaste']):
        return True
    if 'supplement' in category or 'vitamin' in category:
        return True
    return False

def normalize_ingredients(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    # ensure important columns exist
    for col in ['serving_size', 'serving_size_grams', 'num_servings', 'price',
                'calories', 'protein', 'carbs', 'fat', 'cost_per_serving', 'category', 'name']:
        if col not in df.columns:
            df[col] = np.nan

    # normalize price column name guess variants
    if 'price' not in df.columns and 'cost' in df.columns:
        df['price'] = df['cost']

    # precompute textual fields
    def infer_row(row):
        res = {}
        name = str(row.get('name', '') or row.get('name_clean','') or '')
        res['name'] = name
        res['is_food'] = not is_probably_non_food(row)

        # 1) explicit serving_size_grams
        sgrams = parse_number(row.get('serving_size_grams'))
        res['serving_size_grams'] = float(sgrams) if sgrams else None

        # 2) serving_size text
        serving_size_text = row.get('serving_size') or ''
        if isinstance(serving_size_text, str) and serving_size_text.strip() != '':
            val, unit = extract_weight_and_unit(serving_size_text)
            if val and unit in UNIT_TO_GRAMS:
                res['serving_size_grams'] = val * UNIT_TO_GRAMS[unit]
            else:
                mcup = re.search(r'(\d+(\.\d+)?)\s*cup', serving_size_text.lower())
                if mcup:
                    cups = float(mcup.group(1))
                    cat = str(row.get('category','')).lower()
                    base = 240 if 'beverage' in cat or 'drink' in name.lower() else 100
                    res['serving_size_grams'] = cups * base
                mapple = re.search(r'apple|banana|medium', serving_size_text.lower())
                if mapple and res['serving_size_grams'] is None:
                    res['serving_size_grams'] = 150

        # 3) container weight in product name
        if res['serving_size_grams'] is None:
            nval, nunit = extract_weight_and_unit(name)
            if nval and nunit in UNIT_TO_GRAMS:
                res['container_grams'] = nval * UNIT_TO_GRAMS[nunit]
            else:
                res['container_grams'] = None

        # 4) price + pack count
        price = parse_number(row.get('price') or row.get('cost_per_serving') or row.get('cost') or row.get('price_per_unit'))
        res['price'] = float(price) if price is not None else None
        pack_count = extract_pack_count(name) or extract_pack_count(str(row.get('serving_size') or '')) or parse_number(row.get('num_servings'))
        res['pack_count'] = int(pack_count) if pack_count else None

        # 5) raw macros
        res['raw_cal'] = parse_number(row.get('calories'))
        res['raw_protein'] = parse_number(row.get('protein'))
        res['raw_carbs'] = parse_number(row.get('carbs'))
        res['raw_fat'] = parse_number(row.get('fat'))

        # 6) category
        res['category'] = str(row.get('category') or '').lower()

        return res

    inferred = df.apply(infer_row, axis=1).tolist()

    # COMPUTE normalized values per row
    out = []
    for i, row in df.iterrows():
        info = inferred[i]
        out_row = {}

        out_row['name'] = info['name']
        out_row['is_food'] = info['is_food']

        # final serving_size_grams
        sgrams = info.get('serving_size_grams')
        container_grams = info.get('container_grams')
        pack_count = info.get('pack_count')
        category = info.get('category') or 'default'

        if sgrams is not None:
            serving_size_grams = float(sgrams)
        elif container_grams is not None and pack_count:
            serving_size_grams = float(container_grams) / float(pack_count)
        elif container_grams is not None:
            serving_size_grams = float(container_grams) / (4.0 if container_grams > 400 else 2.0)
        else:
            serving_size_grams = float(CATEGORY_DEFAULT_SERVING_G.get(category.split()[0], CATEGORY_DEFAULT_SERVING_G['default']))

        # num_servings
        if pack_count:
            num_servings = float(pack_count)
        elif container_grams and serving_size_grams:
            num_servings = max(1.0, float(container_grams) / float(serving_size_grams))
        else:
            ns = parse_number(df.at[i, 'num_servings']) if 'num_servings' in df.columns else None
            num_servings = float(ns) if ns else 1.0

        price = info.get('price')
        cost_per_serving = float(price) / num_servings if price is not None else np.nan

        # macros
        raw_cal = info['raw_cal']
        raw_protein = info['raw_protein']
        raw_carbs = info['raw_carbs']
        raw_fat = info['raw_fat']

        cal_per_serving = None
        prot_per_serving = None
        carbs_per_serving = None
        fat_per_serving = None

        if raw_cal is not None:
            if raw_cal > 1000 and container_grams and raw_cal > 3 * (serving_size_grams * 2):
                cal_per_serving = raw_cal / num_servings
            else:
                cal_per_serving = raw_cal

        if raw_protein is not None:
            prot_per_serving = raw_protein / num_servings if raw_protein > 100 else raw_protein
        if raw_carbs is not None:
            carbs_per_serving = raw_carbs / num_servings if raw_carbs > 100 else raw_carbs
        if raw_fat is not None:
            fat_per_serving = raw_fat / num_servings if raw_fat > 100 else raw_fat

        if (cal_per_serving is None) and raw_cal and container_grams:
            try:
                cal_per_serving = (raw_cal / container_grams) * serving_size_grams
            except:
                pass

        out_row['serving_size_grams'] = serving_size_grams
        out_row['num_servings'] = num_servings
        out_row['cost_per_serving'] = cost_per_serving
        out_row['cal_per_serving'] = cal_per_serving or np.nan
        out_row['protein_per_serving'] = prot_per_serving or np.nan
        out_row['carbs_per_serving'] = carbs_per_serving or np.nan
        out_row['fat_per_serving'] = fat_per_serving or np.nan
        out_row['category'] = category
        out_row['price'] = price

        out.append(out_row)

    norm = pd.DataFrame(out, index=df.index)

    # MERGE
    merged = pd.concat([
        df,
        norm[['serving_size_grams','num_servings','cost_per_serving',
              'cal_per_serving','protein_per_serving','carbs_per_serving','fat_per_serving','is_food']]
    ], axis=1)

    # ---------------------------------------------------------
    # 🔥 THE REAL FIX — REMOVE DUPLICATE COLUMNS
    # ---------------------------------------------------------
    merged = merged.loc[:, ~merged.columns.duplicated()]

    # ---------------------------------------------------------
    # CLEAN NUMERICS
    # ---------------------------------------------------------
    numeric_cols = [
        'serving_size_grams','num_servings','cost_per_serving',
        'cal_per_serving','protein_per_serving','carbs_per_serving','fat_per_serving'
    ]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')

    merged['is_food'] = merged['is_food'].fillna(True)

    return merged
