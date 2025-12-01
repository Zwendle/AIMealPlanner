import pandas as pd
import numpy as np
import json
import os
from src.eval.onboarding.user_prompt import UserOnboarding
from src.eval.evaluation import MealPlanEvaluator
from src.training.training import train, generate_meal, load_model, model_exists, calculate_reward, parse_number
from src.eval.onboarding.structs import DietaryGoal
from src.utils import filter_ingredients

def main():
    # 1. Load Data
    print("Loading data...")
    try:
        ingredients_df = pd.read_csv("data/ingredients_final.csv")
        ingredients_df = filter_ingredients(ingredients_df)
        # No longer filtering by nutrition_json
    except FileNotFoundError:
        print("Error: data/ingredients_final.csv not found.")
        return

    # 2. Onboarding
    prompter = UserOnboarding("data/ingredients_final.csv")
    user_constraints, numpy_state = prompter.run()
    num_to_pick = user_constraints.num_meals
    ingredients = user_constraints.pantry

    # 3. Model Management
    print("\n" + "=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)
    
    model_path = 'data/Q_table.pickle'
    if model_exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)
    else:
        print("No existing model found. Training new model...")
        train(ingredients_df)
        model = load_model(model_path)

    # 4. Target Mapping (per-meal)
    target_calories = 2000 / 2
    target_protein = 75 / 2
    target_carbs = 250 / 2
    target_fat = 70 / 2

    if DietaryGoal.HIGH_PROTEIN == user_constraints.dietary_goal:
        target_protein = 120 / 2

    if DietaryGoal.KETO == user_constraints.dietary_goal:
        target_carbs = 30 / 2       # 15g per meal
        target_fat = 150 / 2        # 75g per meal
        target_protein = max(target_protein, 100 / 2)  # at least 50g per meal

    if (
        DietaryGoal.LOW_CARB == user_constraints.dietary_goal
        and DietaryGoal.KETO != user_constraints.dietary_goal
    ):
        target_carbs = 100 / 2
        
    training_goal = {
        "target_calories": target_calories,
        "target_protein": target_protein,
        "target_carbs": target_carbs,
        "target_fat": target_fat,
        "vegetarian_diet": DietaryGoal.VEGETARIAN == user_constraints.dietary_goal,
        "target_price": user_constraints.budget_max / 14,
        "pantry": [item.name for item in user_constraints.pantry]
    }

    # 5. Generation (Week Plan)
    print("\n" + "=" * 60)
    print("GENERATING WEEKLY MEAL PLAN")
    print("=" * 60)
        
    plan_data = []
    
    history_file = 'data/meal_history.json'
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
            print(f"Loaded history of {len(history)} past ingredients.")

    # Initialize pantry tracking: start with user's pantry items
    # Create mapping from pantry item names to name_clean for ingredient matching
    pantry_to_name_clean = {}
    for item in user_constraints.pantry:
        matching_row = ingredients_df[
            (ingredients_df['name'] == item.name) | 
            (ingredients_df['name_clean'] == item.name.lower())
        ]
        if not matching_row.empty:
            name_clean = matching_row.iloc[0]['name_clean']
            pantry_to_name_clean[item.name] = name_clean
    
    history = {}
    for item in user_constraints.pantry:
        matching_row = ingredients_df[
            (ingredients_df['name'] == item.name) | 
            (ingredients_df['name_clean'] == item.name.lower())
        ]

        if not matching_row.empty:
            name_clean = matching_row.iloc[0]['name_clean']
            history[name_clean] = item.servings_available
        else:
            print(f"Warning: Pantry item '{item.name}' not found in filtered dataset, skipping")

    for day in range(num_to_pick):
        raw_meal = generate_meal(ingredients_df, training_goal, model, num_to_pick, ingredients=history)
        ingredient_servings = raw_meal
        ingredient_list = list(ingredient_servings.keys())
        
        cost = 0
        cal = 0
        prot = 0
        carbs = 0
        fat = 0
        
        # Track pantry state before this meal
        leftovers_before = history.copy()
        
        for ing, servings_used in ingredient_servings.items():
            row = ingredients_df[ingredients_df['name_clean'] == ing].iloc[0]
            cost_val = parse_number(row.get('cost_per_serving', 0))
            if not np.isnan(cost_val):
                cost += cost_val
            cal += parse_number(row.get('calories', 0))
            prot += parse_number(row.get('protein', 0))
            carbs += parse_number(row.get('carbs', 0))
            fat += parse_number(row.get('fat', 0))
            
            # subtract used servings
            if ing in history:
                history[ing] = max(0, history[ing] - 1.0)
            else:
                history[ing] = ingredients_df.loc[ingredients_df['name_clean'] == ing, 'num_servings'].values[0] - 1
                            
        # calculate leftovers after this meal, remove ingredients with zero servings
        print(history)
        history = {k: v for k, v in history.items() if v > 0}
        leftovers_after = history.copy()
        
        plan_data.append({
            "day": day + 1,
            "ingredients": ingredient_list,
            "ingredient_servings": ingredient_servings,
            "leftovers_before": leftovers_before,
            "leftovers_after": leftovers_after,
            "cost": cost,
            "calories": cal,
            "protein": prot,
            "carbs": carbs,
            "fat": fat
        })
    meal_plan_df = pd.DataFrame(plan_data)
    # 6. Evaluation
    evaluator = MealPlanEvaluator(model, ingredients_df)
    scores = evaluator.evaluate(user_constraints, meal_plan_df)

    # 7. Detailed Reporting
    print("\n" + "=" * 60)
    print("MEAL PLAN EVALUATION REPORT")
    print("=" * 60)
    print(f"Total Score: {scores['total_score']:.2f} / 1.00")
    print(f"Base Score:  {scores['base_score']:.2f}")
    print("-" * 30)
    print(f"Utilization Score (50.0%): {scores['utilization_score']:.2f}")
    print(f"Nutrition Score   (37.5%): {scores['nutrition_score']:.2f}")
    print(f"Cost Score        (12.5%): {scores['cost_score']:.2f}")
    print("-" * 30)
    print(f"Variety Penalty   (Mult):  x{scores['variety_score']:.2f}")
    
    print("\n" + "=" * 60)
    print("WEEKLY MEAL PLAN DETAILS")
    print("=" * 60)
    
    total_weekly_cost = meal_plan_df['cost'].sum()
    
    for _, row in meal_plan_df.iterrows():
        print(f"\nMeal {row['day']}")
        print(f"Ingredients: {', '.join(row['ingredients'])}")
        print(f"Nutrition: {row['calories']:.0f} kcal | P: {row['protein']:.1f}g | C: {row['carbs']:.1f}g | F: {row['fat']:.1f}g")
        print(f"Cost: ${row['cost']:.2f}")
        
        diff = row['calories'] - target_calories/2
        status = "MATCH" if abs(diff) < 100 else ("OVER" if diff > 0 else "UNDER")
        print(f"Target Status: {status} ({diff:+.0f} kcal)")
        
        # Show ingredient servings and leftovers
        servings_dict = row.get('ingredient_servings', {})
        leftovers = row.get('leftovers_after', {})
        if servings_dict:
            # Show only ingredients with remaining servings
            remaining = {ing: qty for ing, qty in leftovers.items() if qty > 0}
            if remaining:
                leftovers_str = ', '.join([f'{ing} ({qty:.1f} serving{"s" if qty != 1 else ""})' 
                                          for ing, qty in remaining.items()])
                print(f"Leftovers After: {leftovers_str}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Weekly Cost: ${total_weekly_cost:.2f}")
    print(f"Budget: ${user_constraints.budget_min} - ${user_constraints.budget_max}")
    
    # Show final pantry state (leftovers after all meals)
    print("\n" + "=" * 60)
    print("FINAL PANTRY STATE (After All Meals)")
    print("=" * 60)
    name_clean_to_pantry_name = {v: k for k, v in pantry_to_name_clean.items()}

    # Separate ingredients with leftovers vs. those that need purchase
    leftovers = {ing: qty for ing, qty in history.items() if qty > 0}
    used_up = {ing: qty for ing, qty in history.items() if qty == 0}
        
    if leftovers:
        print(f"\n ~~~Leftovers ({len(leftovers)} items):")
        for ing, qty in sorted(leftovers.items()):
            # Try to get original pantry name for display
            display_name = name_clean_to_pantry_name.get(ing, ing)
            print(f"  - {display_name}: {qty:.1f} serving{'s' if qty != 1 else ''} remaining")
    
    if used_up:
        print(f"\n Used Up ({len(used_up)} items):")
        for ing in sorted(used_up.keys()):
            display_name = name_clean_to_pantry_name.get(ing, ing)
            print(f"  - {display_name}: completely used")
    
    # Check for ingredients in meals that weren't in initial pantry
    all_meal_ingredients = set()
    for _, row in meal_plan_df.iterrows():
        all_meal_ingredients.update(row['ingredients'])
    pantry_name_cleans = set(pantry_to_name_clean.values())
    needs_purchase = all_meal_ingredients - pantry_name_cleans
    if needs_purchase:
        print(f"\n !!!Needs Purchase ({len(needs_purchase)} items):")
        for ing in sorted(needs_purchase):
            # Find total servings needed
            total_servings = 0
            for _, row in meal_plan_df.iterrows():
                servings_dict = row.get('ingredient_servings', {})
                total_servings += servings_dict.get(ing, 0)
            if total_servings > 0:
                print(f"  - {ing}: {total_servings:.1f} serving{'s' if total_servings != 1 else ''} needed")
        
    with open(history_file, 'w') as f:
        json.dump(history, f)
    print("\n✓ Meal history updated for next week's variety optimization.")

if __name__ == "__main__":
    main()
    