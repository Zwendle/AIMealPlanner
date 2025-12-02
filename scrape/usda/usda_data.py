import pandas as pd
import requests
import time
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
API_KEY = 'JyrOkj0OLrRpY9sh62DChPY02rUasaiRFs2BfpuP'  # <--- PASTE YOUR KEY HERE
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRAPE_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(SCRAPE_DIR)
INPUT_FILE = os.path.join(ROOT_DIR, 'data', 'ingredients.csv')
OUTPUT_FILE = os.path.join(ROOT_DIR, 'data', 'ingredients_enriched.csv')

# USDA Nutrient IDs
NUTRIENT_IDS = {
    'Energy': 1008,          # kcal
    'Protein': 1003,         # g
    'Total lipid (fat)': 1004, # g
    'Carbohydrate, by difference': 1005, # g
    'Sugars, total including NLEA': 2000 # g
}

def get_usda_data(query):
    """
    Returns full nutrition info including units and serving size.
    """
    base_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        'api_key': API_KEY,
        'query': query,
        'pageSize': 1,
        'dataType': ['Foundation', 'SR Legacy'] 
    }

    try:
        r = requests.get(base_url, params=params)
        data = r.json()
        
        if not data.get('foods'):
            return None
            
        food = data['foods'][0]
        nutrients = food.get('foodNutrients', [])
        
        result = {
            'calories': 0, 
            'protein': 0, 'protein_unit': 'g',
            'carbs': 0, 'carbs_unit': 'g',
            'fat': 0, 'fat_unit': 'g',
            'sugar': 0, 'sugar_unit': 'g',
            'serving_size_grams': 100.0
        }
        
        for n in nutrients:
            val = n.get('value', 0)
            unit = n.get('unitName', 'g').lower()
            nid = n.get('nutrientId')
            
            if nid == NUTRIENT_IDS['Energy']:
                result['calories'] = val
            elif nid == NUTRIENT_IDS['Protein']:
                result['protein'] = val
                result['protein_unit'] = unit
            elif nid == NUTRIENT_IDS['Carbohydrate, by difference']:
                result['carbs'] = val
                result['carbs_unit'] = unit
            elif nid == NUTRIENT_IDS['Total lipid (fat)']:
                result['fat'] = val
                result['fat_unit'] = unit
            elif nid == NUTRIENT_IDS['Sugars, total including NLEA']:
                result['sugar'] = val
                result['sugar_unit'] = unit
                
        return result

    except Exception as e:
        print(f"Error fetching {query}: {e}")
        return None

def is_empty(val):
    """Helper to check if a value is missing (NaN, None, or 0)"""
    if pd.isna(val) or val is None:
        return True
    if isinstance(val, (int, float)) and val == 0:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # We want to check EVERY row, because some might have Cost but missing Sugar,
    # others might have Calories but missing Units.
    # However, to save API calls, we only search if at least ONE important thing is missing.
    # Important things: Calories, Protein, Carbs, Fat, or Cost
    
    mask = (
        df['calories'].isna() | (df['calories'] == 0) |
        df['protein'].isna() |
        df['carbs'].isna() |
        df['fat'].isna() | 
        df['cost_per_serving'].isna() | (df['cost_per_serving'] == 0)
    )
    
    rows_to_check = df[mask]
    print(f"Found {len(rows_to_check)} rows with at least one missing field.")
    
    counter = 0
    
    for idx, row in rows_to_check.iterrows():
        term = str(row['name_clean']).split(',')[0]
        needs_nutrition = (is_empty(row['calories']) or is_empty(row['protein']) or is_empty(row['carbs'])
                           or is_empty(row['fat']))
        
        data = None
        if needs_nutrition:
            print(f"[{counter+1}] Fetching USDA for: {term}...", end=" ")
            data = get_usda_data(term)
        else:
            print(f"[{counter+1}] Skipping USDA (Nutrition exists): {term}...", end=" ")

        if data:
            print("✅ Merging data.")
            
            # --- SAFE MERGE LOGIC ---
            # Only overwrite if the current value is Empty/Zero
            
            if is_empty(row['calories']): df.at[idx, 'calories'] = data['calories']
            
            if is_empty(row['protein']): 
                df.at[idx, 'protein'] = data['protein']
                df.at[idx, 'protein_unit'] = data['protein_unit'] # Always take unit if taking value
            
            if is_empty(row['carbs']):
                df.at[idx, 'carbs'] = data['carbs']
                df.at[idx, 'carbs_unit'] = data['carbs_unit']
                
            if is_empty(row['fat']):
                df.at[idx, 'fat'] = data['fat']
                df.at[idx, 'fat_unit'] = data['fat_unit']
                
            if is_empty(row['sugar']):
                df.at[idx, 'sugar'] = data['sugar']
                df.at[idx, 'sugar_unit'] = data['sugar_unit']
                
            if is_empty(row['serving_size_grams']):
                 df.at[idx, 'serving_size_grams'] = data['serving_size_grams']

        else:
            print("⏭️ No action.")

        # --- COST LOGIC (No API call needed if we have price) ---
        # Only calculate cost if it is strictly missing
        if is_empty(row['cost_per_serving']):
            price = float(row.get('price', 0)) if pd.notna(row.get('price')) else 0
            
            if price > 0:
                # If we have num_servings, use it
                servings = row.get('num_servings')
                if is_empty(servings):
                    servings = 4.0 # Estimate 4 servings per pack
                    df.at[idx, 'num_servings'] = servings
                
                # Calculate
                df.at[idx, 'cost_per_serving'] = price / float(servings)
                # print(f"   -> Calculated cost: ${price/float(servings):.2f}")

        counter += 1
        # Add sleep only if we actually hit the API
        if needs_nutrition:
            time.sleep(0.5) 
        
        if counter % 20 == 0:
            df.to_csv(OUTPUT_FILE, index=False)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone! Safely enriched data saved to {OUTPUT_FILE}")