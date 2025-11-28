import pandas as pd
import numpy as np


## NOTE: This function was written with the assistance of LLMS.

# Consolidates data between the original data (which provides nutritional data by serving size) and the enriched USDA data
# USDA data provides more complete nutritional info but is per 100g by default. Thus, this script was used to combine both.
def consolidate_nutrition(original_file, enriched_file, output_file):
    print(f"Loading {original_file} and {enriched_file}...")
    df_orig = pd.read_csv(original_file)
    df_rich = pd.read_csv(enriched_file)

    print(f"Original Columns: {len(df_orig.columns)}")
    
    # 1. ADD MISSING COLUMNS
    # If enriched has 'sugar' or 'fiber' but original doesn't, add them to original
    for col in df_rich.columns:
        if col not in df_orig.columns:
            df_orig[col] = np.nan # Create empty column
            print(f"Added missing column from enriched data: {col}")

    rich_map = df_rich.drop_duplicates(subset=['name_clean']).set_index('name_clean').to_dict('index')

    math_cols = ['calories', 'protein', 'carbs', 'fat', 'sugar', 'fiber']

    count_calc = 0
    count_filled = 0

    print("Consolidating data...")

    for idx, row in df_orig.iterrows():
        name = row['name_clean']
        
        # Skip if no enriched data available
        if name not in rich_map:
            continue

        rich_row = rich_map[name]

        # ---------------------------------------------------------
        # STEP 1: Fix Serving Size Grams (Crucial for the Math)
        # ---------------------------------------------------------
        grams = row.get('serving_size_grams')
        if pd.isna(grams) or grams == 0:
            grams = rich_row.get('serving_size_grams', 100.0)
            df_orig.at[idx, 'serving_size_grams'] = grams
            
            # Also fill the text description if missing
            if pd.isna(row.get('serving_size')):
                df_orig.at[idx, 'serving_size'] = rich_row.get('serving_size')

        # Calculate Ratio (Enriched is per 100g)
        ratio = float(grams) / 100.0

        # ---------------------------------------------------------
        # STEP 2: Update Nutrition (With Math)
        # ---------------------------------------------------------
        row_touched = False
        
        for col in df_orig.columns:
            # Skip non-data columns or identifiers
            if col in ['name', 'name_clean', 'serving_size', 'serving_size_grams', 'num_servings', 'cost_per_serving']:
                continue

            # Get values
            orig_val = row.get(col)
            rich_val = rich_row.get(col)

            # ONLY Update if Original is Missing/Zero AND Rich has data
            if (pd.isna(orig_val) or orig_val == 0) and pd.notna(rich_val):
                
                # A. If it's a Number that needs Scaling (Calories, Protein, etc.)
                if col in math_cols:
                    # Apply the Ratio Math
                    final_val = float(rich_val) * ratio
                    df_orig.at[idx, col] = round(final_val, 2)
                    row_touched = True
                
                # B. If it's a Unit or Text (e.g. 'carbs_unit', 'category', 'store')
                # Just copy it directly
                else:
                    df_orig.at[idx, col] = rich_val
                    count_filled += 1

        if row_touched:
            count_calc += 1

    # Save preserving ALL columns
    df_orig.to_csv(output_file, index=False)
    
    print("-" * 40)
    print(f"Final Columns: {len(df_orig.columns)} (Preserved All)")
    print(f"Rows with scaled nutrition: {count_calc}")
    print(f"Saved to: {output_file}")
    print("-" * 40)

if __name__ == "__main__":
    consolidate_nutrition(
        'ingredients.csv', 
        'ingredients_enriched.csv', 
        'ingredients_final.csv'
    )