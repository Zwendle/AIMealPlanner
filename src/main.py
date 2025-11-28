import pandas as pd
import numpy as np
import json
import os
from src.eval.onboarding.user_prompt import MealPlanningEvaluator
from src.eval.evaluation import MealPlanEvaluator
from src.training.training import train, generate_meal, load_model, model_exists, calculate_reward, parse_number
from src.eval.onboarding.structs import DietaryGoal

def main():
    # 1. Load Data
    print("Loading data...")
    try:
        ingredients_df = pd.read_csv("data/ingredients_final.csv")
        # No longer filtering by nutrition_json
    except FileNotFoundError:
        print("Error: data/ingredients_final.csv not found.")
        return

    # 2. Onboarding
    prompter = MealPlanningEvaluator("data/ingredients_final.csv")
    user_constraints, numpy_state = prompter.run()

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

    # 4. Target Mapping
    target_calories = 2000 # Default
    target_protein = 50
    target_carbs = 275
    target_fat = 78
    
    if DietaryGoal.HIGH_PROTEIN in user_constraints.dietary_goals:
        target_protein = 120
    if DietaryGoal.KETO in user_constraints.dietary_goals:
        target_carbs = 30
        target_fat = 120
        target_protein = 100
    if DietaryGoal.LOW_CARB in user_constraints.dietary_goals:
        target_carbs = 100
        
    training_goal = {
        "target_calories": target_calories,
        "target_protein": target_protein,
        "target_carbs": target_carbs,
        "target_fat": target_fat,
        "vegetarian_diet": DietaryGoal.VEGETARIAN in user_constraints.dietary_goals,
        "target_price": user_constraints.budget_max / 14,
        "pantry": [item.name for item in user_constraints.pantry]
    }

    # 5. Generation (Week Plan)
    print("\n" + "=" * 60)
    print("GENERATING WEEKLY MEAL PLAN")
    print("=" * 60)
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    plan_data = []
    
    history_file = 'data/meal_history.json'
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
            print(f"Loaded history of {len(history)} past meals.")

    for day in days:
        for meal_type in ["Lunch", "Dinner"]:
            ingredients = generate_meal(ingredients_df, training_goal, model)
            
            cost = 0
            cal = 0
            prot = 0
            carbs = 0
            fat = 0
            
            for ing in ingredients:
                row = ingredients_df[ingredients_df['name_clean'] == ing].iloc[0]
                cost_val = parse_number(row.get('cost_per_serving', 0))
                if not np.isnan(cost_val):
                    cost += cost_val
                cal += parse_number(row.get('calories', 0))
                prot += parse_number(row.get('protein', 0))
                carbs += parse_number(row.get('carbs', 0))
                fat += parse_number(row.get('fat', 0))
            
            plan_data.append({
                "day": day,
                "meal": meal_type,
                "ingredients": ingredients,
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
        print(f"\n{row['day']} - {row['meal']}")
        print(f"Ingredients: {', '.join(row['ingredients'])}")
        print(f"Nutrition: {row['calories']:.0f} kcal | P: {row['protein']:.1f}g | C: {row['carbs']:.1f}g | F: {row['fat']:.1f}g")
        print(f"Cost: ${row['cost']:.2f}")
        
        diff = row['calories'] - target_calories/2
        status = "MATCH" if abs(diff) < 100 else ("OVER" if diff > 0 else "UNDER")
        print(f"Target Status: {status} ({diff:+.0f} kcal)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Weekly Cost: ${total_weekly_cost:.2f}")
    print(f"Budget: ${user_constraints.budget_min} - ${user_constraints.budget_max}")
    
    # 8. Update Week-over-Week History
    new_history = [row['ingredients'] for _, row in meal_plan_df.iterrows()]
    history.extend(new_history)
    if len(history) > 56:
        history = history[-56:]
        
    with open(history_file, 'w') as f:
        json.dump(history, f)
    print("\n✓ Meal history updated for next week's variety optimization.")

if __name__ == "__main__":
    main()
