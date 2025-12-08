import pandas as pd

def filter_ingredients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply consistent filtering to ingredients dataframe.
    Used across training, evaluation, and inference.
    """
    required_cols = ["carbs", "fat", "protein", "calories", "cost_per_serving"]
    
    df = df.dropna(subset=required_cols)
    for col in required_cols:
        df = df[df[col].astype(str).str.strip() != ""]
    
    return df