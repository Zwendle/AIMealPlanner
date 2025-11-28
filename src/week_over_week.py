import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

THIS_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = THIS_DIR.parent                     

sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.eval.onboarding.user_prompt import MealPlanningEvaluator
    from src.training.structs import UserConstraints, Store, DietaryGoal, PantryItem
except ImportError:
    MealPlanningEvaluator = None
    UserConstraints = None
    Store = None
    DietaryGoal = None
    PantryItem = None

STATE_PATH = PROJECT_ROOT / "state.json"
GROCERY_CSV = PROJECT_ROOT / "data" / "ingredients.csv"

import pandas as pd

# training code
try:
    from src.training.training import (
        train,
        generate_meal,
        load_model,
        model_exists,
        parse_number,
    )
except ImportError:
    train = None
    generate_meal = None
    load_model = None
    model_exists = None
    parse_number = None

MODEL_PATH = Path("data/Q_table.pickle")

def dict_user_constraints(state_dict: Dict[str, Any]) -> Optional[UserConstraints]:
    if UserConstraints is None:
        return None

    pantry_items: List[PantryItem] = []
    for item in state_dict.get("pantry", []):
        try:
            store = Store[state_dict.get("preferred_store", "ANY").upper().replace(" ", "_")]
        except (KeyError, AttributeError):
            store = Store.ANY 

        pantry_item = PantryItem(
            name=item.get("name", ""),
            category=item.get("category", "unknown"),
            preferred_store=store,
            serving_size_grams=item.get("serving_size_grams", 100.0),
            servings_available=item.get(
                "servings",
                item.get("servings_available", 0.0)
            ),
            cost_per_serving=item.get("cost_per_serving", 0.0),
            nutrients=item.get("nutrients", {}),
            ingredient_idx=item.get("ingredient_idx"),
        )
        pantry_items.append(pantry_item)

    # Dietary goals
    dietary_goals: List[DietaryGoal] = []
    goals_dict = state_dict.get("dietary_goals", {})
    primary_goal = goals_dict.get("primary_goal", "")
    if primary_goal and primary_goal != "unspecified":
        try:
            goal = DietaryGoal[primary_goal.upper().replace(" ", "_")]
            dietary_goals.append(goal)
        except (KeyError, AttributeError):
            pass

    # Store preference
    try:
        store_pref = Store[state_dict.get("preferred_store", "ANY").upper().replace(" ", "_")]
    except (KeyError, AttributeError):
        store_pref = Store.ANY 

    return UserConstraints(
        pantry=pantry_items,
        preferred_store=store_pref,
        dietary_goals=dietary_goals,
        num_meals_min=state_dict.get("meals_to_make", 3),
        num_meals_max=state_dict.get("meals_to_make", 7),
        budget_min=state_dict.get("budget", 50.0),
        budget_max=state_dict.get("budget", 50.0),
    )


def constraints_to_dict(constraints: UserConstraints) -> Dict[str, Any]:
    pantry_list: List[Dict[str, Any]] = []
    for item in constraints.pantry:
        pantry_list.append(
            {
                "name": item.name,
                "servings": item.servings_available,
                "store": item.preferred_store.name,
                "category": item.category,
                "serving_size_grams": item.serving_size_grams,
                "cost_per_serving": item.cost_per_serving,
                "nutrients": item.nutrients,
                "ingredient_idx": item.ingredient_idx,
            }
        )

    dietary_goals_dict: Dict[str, Any] = {}
    if constraints.dietary_goals:
        dietary_goals_dict["primary_goal"] = constraints.dietary_goals[0].name.lower()

    return {
        "pantry": pantry_list,
        "preferred_store": constraints.preferred_store.name,
        "dietary_goals": dietary_goals_dict,
        "meals_to_make": constraints.num_meals_min,
        "budget": constraints.budget_min,
        "history": [],  
    }

def load_state() -> Optional[Dict[str, Any]]:
    if not STATE_PATH.exists():
        return None
    with STATE_PATH.open("r") as f:
        return json.load(f)


def save_state(state: Dict[str, Any]) -> None:
    """Save the user state to disk as JSON."""
    with STATE_PATH.open("w") as f:
        json.dump(state, f, indent=2)
    print(f"\n[info] State saved to {STATE_PATH.resolve()}\n")


def onboard_user() -> Dict[str, Any]:
    if MealPlanningEvaluator is None or not GROCERY_CSV.exists():
        print("=== Onboarding (Week 0) ===")
        preferred_store = input("Preferred store (Trader Joes / Whole Foods): ").strip()
        goal = input("Enter a dietary goal (or leave blank): ").strip().lower() or "unspecified"

        try:
            meals_to_make = int(input("How many meals per week? ") or "7")
        except ValueError:
            meals_to_make = 7

        try:
            budget = float(input("Weekly budget in dollars? ") or "50.0")
        except ValueError:
            budget = 50.0

        pantry: List[Dict[str, Any]] = []
        print("\nEnter ingredients (type 'done' when finished):")
        while True:
            name = input("Ingredient name (or 'done'): ").strip()
            if name.lower() == "done":
                break
            try:
                servings = float(input(f"How many servings of '{name}'? "))
                pantry.append({"name": name, "servings": servings, "store": preferred_store})
            except ValueError:
                continue

        return {
            "pantry": pantry,
            "preferred_store": preferred_store,
            "dietary_goals": {"primary_goal": goal},
            "meals_to_make": meals_to_make,
            "budget": budget,
            "history": [],
        }

    evaluator = MealPlanningEvaluator(str(GROCERY_CSV))
    user_constraints, _ = evaluator.run()
    return constraints_to_dict(user_constraints)


def _ensure_model(ingredients_df: "pd.DataFrame"):
    if model_exists is None or load_model is None or train is None:
        return None

    if model_exists(): 
        print("[info] Loading existing Q-table model...")
        return load_model()
    else:
        print("[info] No model found. Training a new Q-table model...")
        train(ingredients_df)
        return load_model()



def generate_candidate_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    # implements old behavior if q learning does not work 
    if (
        generate_meal is None
        or parse_number is None
        or not GROCERY_CSV.exists()
    ):
        print("[warn] Training modules or data not found; using placeholder plan.")
        pantry = state.get("pantry", [])
        ingredients: List[Dict[str, Any]] = []
        for item in pantry[:5]:
            ingredients.append(
                {
                    "name": item["name"],
                    "servings_used": min(1.0, float(item.get("servings", 0.0))),
                }
            )

        return {
            "ingredients": ingredients,
            "estimated_cost": 30.0,
            "nutrition_summary": {
                "total_calories": 2000,
                "protein_g": 120,
                "carbs_g": 200,
                "fat_g": 70,
            },
            "score_breakdown": {
                "ingredient_utilization_score": 0.5,
                "nutrition_score": 0.5,
                "variety_score": 0.5,
                "cost_score": 0.5,
                "overall_score": 0.5,
            },
        }

    print("[info] Generating candidate plan from Q-learning model...")

    ingredients_df = pd.read_csv(GROCERY_CSV)

    required_cols = ["name_clean", "calories", "protein", "carbs", "fat", "cost_per_serving"]
    ingredients_df = ingredients_df.dropna(subset=required_cols)

    Q_table = _ensure_model(ingredients_df)
    if Q_table is None:
        print("[warn] Could not load/train model; falling back to placeholder plan.")
        pantry = state.get("pantry", [])
        ingredients: List[Dict[str, Any]] = []
        for item in pantry[:5]:
            ingredients.append(
                {
                    "name": item["name"],
                    "servings_used": min(1.0, float(item.get("servings", 0.0))),
                }
            )
        return {
            "ingredients": ingredients,
            "estimated_cost": 30.0,
            "nutrition_summary": {
                "total_calories": 2000,
                "protein_g": 120,
                "carbs_g": 200,
                "fat_g": 70,
            },
            "score_breakdown": {
                "ingredient_utilization_score": 0.5,
                "nutrition_score": 0.5,
                "variety_score": 0.5,
                "cost_score": 0.5,
                "overall_score": 0.5,
            },
        }

    meals_per_week = 14 
    budget = float(state.get("budget", 50.0))

    primary_goal = (
        state.get("dietary_goals", {}).get("primary_goal", "").lower()
    )
    if "keto" in primary_goal:
        cal, prot, carbs, fat = 2000, 100, 30, 120
    elif "low_carb" in primary_goal or "low carb" in primary_goal:
        cal, prot, carbs, fat = 2000, 80, 100, 90
    elif "protein" in primary_goal or "high_protein" in primary_goal:
        cal, prot, carbs, fat = 2000, 120, 200, 70
    else:
        cal, prot, carbs, fat = 2000, 50, 275, 78

    user_pantry = [item["name"] for item in state.get("pantry", [])]
    all_names_clean = ingredients_df["name_clean"].tolist()
    pantry_for_goal: List[str] = []
    for name in user_pantry:
        if name in all_names_clean:
            pantry_for_goal.append(name)

    training_goal = {
        "target_calories": cal // 2,    
        "target_protein": prot // 2,
        "target_carbs": carbs // 2,
        "target_fat": fat // 2,
        "vegetarian_diet": False,       
        "target_price": budget / meals_per_week,
        "pantry": pantry_for_goal,
    }

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly_rows: List[Dict[str, Any]] = []

    for day in days:
        for meal_type in ["Lunch", "Dinner"]:
            ing_names = generate_meal(ingredients_df, training_goal, Q_table)

            cost = 0.0
            cal_total = prot_total = carbs_total = fat_total = 0.0

            for name in ing_names:
                row = ingredients_df[ingredients_df["name_clean"] == name]
                if row.empty:
                    continue
                row = row.iloc[0]
                cost += parse_number(row.get("cost_per_serving", 0))
                cal_total += parse_number(row.get("calories", 0))
                prot_total += parse_number(row.get("protein", 0))
                carbs_total += parse_number(row.get("carbs", 0))
                fat_total += parse_number(row.get("fat", 0))

            weekly_rows.append(
                {
                    "day": day,
                    "meal": meal_type,
                    "ingredients": ing_names,
                    "cost": cost,
                    "calories": cal_total,
                    "protein": prot_total,
                    "carbs": carbs_total,
                    "fat": fat_total,
                }
            )


    total_cost = sum(r["cost"] for r in weekly_rows)
    total_cal = sum(r["calories"] for r in weekly_rows)
    total_prot = sum(r["protein"] for r in weekly_rows)
    total_carbs = sum(r["carbs"] for r in weekly_rows)
    total_fat = sum(r["fat"] for r in weekly_rows)

    servings_by_name: Dict[str, float] = {}
    for r in weekly_rows:
        for name in r["ingredients"]:
            servings_by_name[name] = servings_by_name.get(name, 0.0) + 1.0

    ingredients_summary = [
        {"name": name, "servings_used": used}
        for name, used in servings_by_name.items()
    ]

    # plug in evaluator
    score_breakdown = {
        "ingredient_utilization_score": 0.5,
        "nutrition_score": 0.5,
        "variety_score": 0.5,
        "cost_score": 0.5,
        "overall_score": 0.5,
    }

    return {
        "weekly_meals": weekly_rows, 
        "ingredients": ingredients_summary,
        "estimated_cost": total_cost,
        "nutrition_summary": {
            "total_calories": total_cal,
            "protein_g": total_prot,
            "carbs_g": total_carbs,
            "fat_g": total_fat,
        },
        "score_breakdown": score_breakdown,
    }

def pantry_plan(state: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    pantry = state.get("pantry", [])
    used_by_name: Dict[str, float] = {}

    for ing in plan.get("ingredients", []):
        name = ing["name"]
        used = float(ing.get("servings_used", 0.0))
        used_by_name[name] = used_by_name.get(name, 0.0) + used

    new_pantry: List[Dict[str, Any]] = []
    for item in pantry:
        name = item["name"]
        servings = float(item.get("servings", 0.0))
        used = used_by_name.get(name, 0.0)
        remaining = max(0.0, servings - used)

        if remaining > 0.0:
            new_item = dict(item)
            new_item["servings"] = remaining
            new_pantry.append(new_item)

    history = list(state.get("history", []))
    history.append(plan)

    new_state = dict(state)
    new_state["pantry"] = new_pantry
    new_state["history"] = history

    print("\n[info] Pantry updated based on accepted plan.\n")
    return new_state

def print_pantry_and_goals(state: Dict[str, Any]) -> None:
    """Print a short summary of the current state."""
    print("=== Current State ===")
    print(f"Preferred store: {state.get('preferred_store', '')}")
    print(f"Weekly budget:  ${state.get('budget', 0.0):.2f}")
    print(f"Meals to make:  {state.get('meals_to_make', 0)}")
    print(f"Dietary goals:  {state.get('dietary_goals', {})}")
    print("\nPantry:")
    pantry: List[Dict[str, Any]] = state.get("pantry", [])
    if not pantry:
        print("  (empty)")
    else:
        for item in pantry:
            servings = item.get("servings", item.get("servings_available", 0))
            print(f"  - {item['name']}: {servings} servings")
    print("=====================\n")


def display_plan(plan: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Print the proposed plan and its cost/nutrition/score breakdown."""
    print("\n=== Proposed Weekly Plan ===")

    print("Ingredients to use:")
    for ing in plan.get("ingredients", []):
        print(f"  - {ing['name']} (use {ing.get('servings_used', '?')} servings)")

    est_cost = float(plan.get("estimated_cost", 0.0))
    budget = float(state.get("budget", 0.0))
    print(f"\nEstimated cost: ${est_cost:.2f} (budget: ${budget:.2f})")

    nutrition = plan.get("nutrition_summary", {})
    if nutrition:
        print("\nEstimated nutrition for the week:")
        print(f"  Calories: {nutrition.get('total_calories', '?')}")
        print(f"  Protein:  {nutrition.get('protein_g', '?')} g")
        print(f"  Carbs:    {nutrition.get('carbs_g', '?')} g")
        print(f"  Fat:      {nutrition.get('fat_g', '?')} g")

    scores = plan.get("score_breakdown", {})
    if scores:
        print("\nScore breakdown (0–1, higher is better):")
        print(f"  Ingredient utilization: {scores.get('ingredient_utilization_score', '?')}")
        print(f"  Nutrition:              {scores.get('nutrition_score', '?')}")
        print(f"  Variety:                {scores.get('variety_score', '?')}")
        print(f"  Cost:                   {scores.get('cost_score', '?')}")
        print(f"  Overall:                {scores.get('overall_score', '?')}")

    print("=============================\n")

#weekly loop
def run_week_over_week_loop() -> None:
    state = load_state()
    if state is None:
        print("[info] No existing state found. Running onboarding.\n")
        state = onboard_user()
        save_state(state)

    while True:
        print_pantry_and_goals(state)

        use_planner = input("Generate a new weekly meal plan? (y/n): ").strip().lower()
        if use_planner != "y":
            print("\nExiting- state is saved. See you next time!")
            break

        plan = generate_candidate_plan(state)
        display_plan(plan, state)

        choice = input("Do you want to use this plan? (y/n): ").strip().lower()
        if choice != "y":
            print("\nGot it, this plan will not be applied. Please try again next week.")
            continue

        state = pantry_plan(state, plan)
        save_state(state)

        again = input("Do you want to plan another week right now? (y/n): ").strip().lower()
        if again != "y":
            print("\nAwesome! Pantry has been updated and saved. Goodbye!")
            break


if __name__ == "__main__":
    run_week_over_week_loop()
