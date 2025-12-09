import pandas as pd
import numpy as np
from typing import Dict
from src.eval.onboarding.structs import UserConstraints, DietaryGoal

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
        self.ingredients_df = ingredients_df.copy()
        if "name_clean" not in self.ingredients_df.columns:
            self.ingredients_df["name_clean"] = self.ingredients_df["name"].str.lower().str.strip()

    def evaluate(self, user_constraints: UserConstraints, meal_plan: pd.DataFrame) -> Dict[str, float]:
        utilization_score = self._calculate_utilization_score(user_constraints, meal_plan)
        nutrition_score = self._calculate_nutrition_score(user_constraints, meal_plan)
        variety_score = self._calculate_variety_score(meal_plan)
        cost_score = self._calculate_cost_score(user_constraints, meal_plan)

        # Base Score (Utilization, Nutrition, Cost)
        base_score = (
            0.45 * utilization_score +
            0.35 * nutrition_score +
            0.20 * cost_score
        )

        total_score = base_score * variety_score

        return {
            "total_score": float(total_score),
            "base_score": float(base_score),
            "utilization_score": float(utilization_score),
            "nutrition_score": float(nutrition_score),
            "variety_score": float(variety_score),
            "cost_score": float(cost_score)
        }

    def _safe_float(self, v):
        try:
            if pd.isna(v):
                return 0.0
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v)
            filtered = ''.join(ch for ch in s if (ch.isdigit() or ch == '.'))
            return float(filtered) if filtered else 0.0
        except Exception:
            return 0.0

    def _find_ingredient_row(self, ing_name: str):
        if ing_name is None:
            return None
        ing_norm = str(ing_name).lower().strip()
        match = self.ingredients_df[self.ingredients_df["name_clean"] == ing_norm]
        if not match.empty:
            return match.iloc[0]
        match = self.ingredients_df[self.ingredients_df["name"] == ing_name]
        if not match.empty:
            return match.iloc[0]
        return None

    def _calculate_utilization_score(self, constraints: UserConstraints, meal_plan: pd.DataFrame) -> float:
        pantry = {item.name: float(item.servings_available) for item in constraints.pantry}
        total_pantry_servings = sum(pantry.values())

        if total_pantry_servings == 0:
            return 1.0

        used_servings = 0.0

        for _, row in meal_plan.iterrows():
            ing_servings_map = row.get("ingredient_servings", None)
            ingredients_list = row.get("ingredients", [])
            if isinstance(ing_servings_map, dict):
                for ing_key, servings in ing_servings_map.items():
                    if servings <= 0:
                        continue
                    for p_name in pantry:
                        if p_name.lower() == ing_key.lower() or p_name.lower() in ing_key.lower() or ing_key.lower() in p_name.lower():
                            used_servings += float(servings)
                            break
            else:
                if isinstance(ingredients_list, list):
                    for ing_name in ingredients_list:
                        for p_name in pantry:
                            if p_name.lower() == str(ing_name).lower() or p_name.lower() in str(ing_name).lower() or str(ing_name).lower() in p_name.lower():
                                used_servings += 1.0
                                break

        used_servings = min(used_servings, total_pantry_servings)
        return used_servings / total_pantry_servings if total_pantry_servings > 0 else 1.0

    def _calculate_nutrition_score(self, constraints: UserConstraints, meal_plan: pd.DataFrame) -> float:
        """
        Compute deviation from macro targets using per-serving nutrient values and serving counts.
        """
        target_calories = 1000.0
        target_protein = 25.0
        target_carbs = 135.0
        target_fat = 39.0

        if DietaryGoal.HIGH_PROTEIN == constraints.dietary_goal:
            target_protein = 50.0
        elif DietaryGoal.KETO == constraints.dietary_goal:
            target_carbs = 15.0
            target_fat = 75.0
            target_protein = 50.0
        elif DietaryGoal.LOW_CARB == constraints.dietary_goal:
            target_carbs = 50.0

        total_calories = 0.0
        total_protein = 0.0
        total_carbs = 0.0
        total_fat = 0.0

        for _, row in meal_plan.iterrows():
            ing_servings_map = row.get("ingredient_servings", None)
            ingredients_list = row.get("ingredients", [])
            if isinstance(ing_servings_map, dict):
                for ing_key, servings in ing_servings_map.items():
                    if servings <= 0:
                        continue
                    r = self._find_ingredient_row(ing_key)
                    if r is None:
                        continue
                    total_calories += self._safe_float(r.get("calories", 0)) * float(servings)
                    total_protein += self._safe_float(r.get("protein", 0)) * float(servings)
                    total_carbs += self._safe_float(r.get("carbs", 0)) * float(servings)
                    total_fat += self._safe_float(r.get("fat", 0)) * float(servings)
            else:
                if isinstance(ingredients_list, list):
                    for ing_name in ingredients_list:
                        r = self._find_ingredient_row(ing_name)
                        if r is None:
                            continue
                        total_calories += self._safe_float(r.get("calories", 0))
                        total_protein += self._safe_float(r.get("protein", 0))
                        total_carbs += self._safe_float(r.get("carbs", 0))
                        total_fat += self._safe_float(r.get("fat", 0))

        num_days = meal_plan['day'].nunique() if 'day' in meal_plan.columns else max(1, len(meal_plan))
        avg_calories = total_calories / num_days
        avg_protein = total_protein / num_days
        avg_carbs = total_carbs / num_days
        avg_fat = total_fat / num_days

        def deviation(actual, target):
            if target == 0:
                return 1.0
            return float(np.exp(-abs(actual - target) / max(target, 1e-6)))

        cal_score = deviation(avg_calories, target_calories)
        prot_score = deviation(avg_protein, target_protein)
        carb_score = deviation(avg_carbs, target_carbs)
        fat_score = deviation(avg_fat, target_fat)

        return float((cal_score + prot_score + carb_score + fat_score) / 4.0)

    def _calculate_variety_score(self, meal_plan: pd.DataFrame) -> float:
        """
        Maximize diversity of ingredients across the week.
        """
        all_ingredients = []
        meals = []

        for _, row in meal_plan.iterrows():
            ing_list = row.get('ingredients', [])
            if isinstance(ing_list, list):
                normalized = [str(x).lower().strip() for x in ing_list]
                all_ingredients.extend(normalized)
                meals.append(tuple(sorted(normalized)))
            else:
                s = str(ing_list).lower().strip()
                all_ingredients.append(s)
                meals.append(s)

        unique_ingredients = len(set(all_ingredients))
        total_ingredients = len(all_ingredients)
        diversity_score = unique_ingredients / total_ingredients if total_ingredients > 0 else 0.0

        unique_meals = len(set(meals))
        total_meals = len(meals)
        repetition_score = unique_meals / total_meals if total_meals > 0 else 0.0

        # variety acts as multiplier between (0.0, 1.0], keep a floor
        return float(max(0.1, (diversity_score + repetition_score) / 2.0))

    def _calculate_cost_score(self, constraints: UserConstraints, meal_plan: pd.DataFrame) -> float:
        """
        Total price vs budget.
        """
        total_cost = meal_plan['cost'].sum() if 'cost' in meal_plan.columns else 0.0

        bmin = float(constraints.budget_min) if hasattr(constraints, 'budget_min') else 0.0
        bmax = float(constraints.budget_max) if hasattr(constraints, 'budget_max') else 1.0

        if bmin <= total_cost <= bmax:
            return 1.0
        elif total_cost < bmin and bmin > 0:
            return float(total_cost / bmin)
        else:
            overage = max(0.0, total_cost - bmax)
            return float(np.exp(-overage / max(bmax, 1.0)))