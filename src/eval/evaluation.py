import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.eval.onboarding.structs import UserConstraints, PantryItem, DietaryGoal
import json

class MealPlanEvaluator:
    """
    Evaluates a generated meal plan based on:
    1. Ingredient Utilization (40%)
    2. Nutritional Target Achievement (30%)
    3. Meal Variety (20%)
    4. Purchase Cost (10%)
    """

    def __init__(self, model, ingredients_df: pd.DataFrame):
        self.model = model
        self.ingredients_df = ingredients_df

    def evaluate(self, user_constraints: UserConstraints, meal_plan: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate the total score for the meal plan.
        Returns a dictionary with the total score and individual component scores.
        """
        utilization_score = self._calculate_utilization_score(user_constraints, meal_plan)
        nutrition_score = self._calculate_nutrition_score(user_constraints, meal_plan)
        variety_score = self._calculate_variety_score(meal_plan)
        cost_score = self._calculate_cost_score(user_constraints, meal_plan)

        # Base Score (Utilization, Nutrition, Cost)
        # Utilization: 0.45, Nutrition: 0.35, Cost: 0.20
        base_score = (
            0.45 * utilization_score +
            0.35 * nutrition_score +
            0.20 * cost_score
        )

        # Policy: Variety as a Penalty Multiplier
        total_score = base_score * variety_score

        return {
            "total_score": total_score,
            "base_score": base_score,
            "utilization_score": utilization_score,
            "nutrition_score": nutrition_score,
            "variety_score": variety_score,
            "cost_score": cost_score
        }

    def _calculate_utilization_score(self, constraints: UserConstraints, meal_plan: pd.DataFrame) -> float:
        """
        Maximize use of existing pantry items.
        """
        pantry_items = {item.name: item.servings_available for item in constraints.pantry}
        used_pantry_items = 0
        total_pantry_items = len(pantry_items)
        
        if total_pantry_items == 0:
            pantry_score = 1.0 
        else:
            all_plan_ingredients = []
            for ingredients_list in meal_plan['ingredients']:
                print("ingredients: ", ingredients_list)
                if isinstance(ingredients_list, list):
                    all_plan_ingredients.extend(ingredients_list)
                elif isinstance(ingredients_list, str):
                     all_plan_ingredients.append(ingredients_list)

            for pantry_item in pantry_items:
                if any(pantry_item in str(ing) for ing in all_plan_ingredients):
                    used_pantry_items += 1
            
            pantry_score = used_pantry_items / total_pantry_items
        
        return pantry_score

    def _calculate_nutrition_score(self, constraints: UserConstraints, meal_plan: pd.DataFrame) -> float:
        """
        Minimize deviation from macro targets.
        """
        
        # DEFAULT TARGETS (by meal)
        target_calories = 1000
        target_protein = 25
        target_carbs = 135
        target_fat = 39

        if DietaryGoal.HIGH_PROTEIN == constraints.dietary_goal:
            target_protein = 50
        elif DietaryGoal.KETO == constraints.dietary_goal:
            target_carbs = 15
            target_fat = 75
            target_protein = 50
        elif DietaryGoal.LOW_CARB == constraints.dietary_goal:
            target_carbs = 50

        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fat = 0
        
        for _, row in meal_plan.iterrows():
            ingredients = row['ingredients']
            if isinstance(ingredients, str):
                ingredients = [ingredients] 
            
            for ing_name in ingredients:
                match = self.ingredients_df[self.ingredients_df['name'] == ing_name]
                if not match.empty:
                    row_data = match.iloc[0]
                    # Parse values from columns
                    def parse_val(v):
                        if isinstance(v, (int, float)): return v
                        return float(''.join(filter(lambda x: x.isdigit() or x == '.', str(v))) or 0)
                        
                    total_calories += parse_val(row_data.get('calories', 0))
                    total_protein += parse_val(row_data.get('protein', 0))
                    total_carbs += parse_val(row_data.get('carbs', 0))
                    total_fat += parse_val(row_data.get('fat', 0))

        num_days = meal_plan['day'].nunique() if 'day' in meal_plan.columns else 1
        if num_days > 0:
            avg_calories = total_calories / num_days
            avg_protein = total_protein / num_days
            avg_carbs = total_carbs / num_days
            avg_fat = total_fat / num_days
        else:
            return 0.0

        def deviation(actual, target):
            if target == 0: return 1.0
            return np.exp(-abs(actual - target) / target)

        cal_score = deviation(avg_calories, target_calories)
        prot_score = deviation(avg_protein, target_protein)
        carb_score = deviation(avg_carbs, target_carbs)
        fat_score = deviation(avg_fat, target_fat)

        return (cal_score + prot_score + carb_score + fat_score) / 4.0

    def _calculate_variety_score(self, meal_plan: pd.DataFrame) -> float:
        """
        Maximize diversity of ingredients across the week.
        """
        all_ingredients = []
        meals = []
        
        for _, row in meal_plan.iterrows():
            ing_list = row['ingredients']
            if isinstance(ing_list, list):
                all_ingredients.extend(ing_list)
                meals.append(tuple(sorted(ing_list)))
            else:
                all_ingredients.append(ing_list)
                meals.append(ing_list)

        unique_ingredients = len(set(all_ingredients))
        total_ingredients = len(all_ingredients)
        diversity_score = unique_ingredients / total_ingredients if total_ingredients > 0 else 0

        unique_meals = len(set(meals))
        total_meals = len(meals)
        repetition_score = unique_meals / total_meals if total_meals > 0 else 0

        return (diversity_score + repetition_score) / 2.0

    def _calculate_cost_score(self, constraints: UserConstraints, meal_plan: pd.DataFrame) -> float:
        """
        Total price vs budget.
        """
        total_cost = meal_plan['cost'].sum() if 'cost' in meal_plan.columns else 0
        
        if constraints.budget_min <= total_cost <= constraints.budget_max:
            return 1.0
        elif total_cost < constraints.budget_min:
            return total_cost / constraints.budget_min
        else:
            overage = total_cost - constraints.budget_max
            return np.exp(-overage / constraints.budget_max)
